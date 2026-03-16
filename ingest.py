import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "documents" / "sample.txt"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

loader = TextLoader(str(SOURCE_FILE))
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

db = FAISS.from_documents(docs, embeddings)

db.save_local(str(VECTORSTORE_DIR))

print("Documents ingested successfully")