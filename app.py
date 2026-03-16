import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama, OllamaEmbeddings

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest")

embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

if not (VECTORSTORE_DIR / "index.faiss").exists():
    raise RuntimeError(
        f"FAISS index not found at {VECTORSTORE_DIR / 'index.faiss'}. "
        "Run `./env3/bin/python ingest.py` from the project directory first."
    )

db = FAISS.load_local(
    str(VECTORSTORE_DIR),
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = db.as_retriever()

llm = ChatOllama(
    temperature=0,
    model=OLLAMA_CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "AI Document Assistant is running",
        "endpoints": ["/ask", "/health", "/docs"],
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(q: Question):
    answer = qa.run(q.question)
    return {"answer": answer}