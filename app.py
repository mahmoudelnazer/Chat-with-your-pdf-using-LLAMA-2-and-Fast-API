from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline
import uvicorn

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
""".strip()
# Initialize FastAPI app
# Model setup (Assuming these are the correct imports and initializations for your models)
class ModelManager:
    def _init_(self):
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        self.db = None
        self.qa_chain = None
        self.ready = False

    def setup(self):
        # HuggingFace embeddings setup
        self.embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large", model_kwargs={"device":"cuda:0" }
        )

        # Llama 2 13B model and tokenizer setup
        model_name_or_path =  r"D:\Personal\llama2-chat-with-documents-main\llama2-chat-with-documents-main\Llama-2-13B-chat-GPTQ"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            revision="gptq-4bit-128g-actorder_True",
            model_basename="model",
            use_safetensors=True,
            trust_remote_code=True,
            inject_fused_attention=False,
            device="cuda:0",
            quantize_config=None,
        )

        # Assuming a function to load documents
        loader = PyPDFDirectoryLoader("pdfs")
        docs = loader.load()

        # Splitting text from documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(docs)

        # Chroma database setup
        self.db = Chroma.from_documents(texts, self.embeddings, persist_directory="db")

        # Setting up the question-answering chain
        self.qa_chain = self._setup_qa_chain()

        self.ready = True

    def _setup_qa_chain(self):
        # Pipeline for text generation
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=streamer,
        )
        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

        # QA Chain setup
        SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
        template = generate_prompt(
            """
    {context}

    Question: {question}
    """,
            system_prompt=SYSTEM_PROMPT,
        )
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
    
    async def generate_answer(self, question):
        if not self.ready:
            raise RuntimeError("Model is not ready. Make sure to call setup method first.")
        
        # Generate answer using the model
        return self.qa_chain(question)

# The shared model instance
model_manager = ModelManager()

# Define a Request model for Pydantic validation
class QuestionRequest(BaseModel):
    question: str
app = FastAPI()



@app.on_event("startup")
def load_model():
    # Load the model on startup
    model_manager.setup()

@app.post("/generate")
async def generate(request_data: QuestionRequest):
    try:
        # Generate the answer
        result = await model_manager.generate_answer(request_data.question)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if "_name_" == "_main_":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)