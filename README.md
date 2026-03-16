# ai-code-assistant

## Project Summary

This project is a lightweight code assistant API for internal experimentation and learning.
It reads local reference files, builds a searchable vector index, and answers coding questions through a simple FastAPI service.
The goal is practical support for day-to-day development tasks, not a production-grade platform.

## How It Works

1. `ingest.py` reads `documents/sample.txt`.
2. The text is split into chunks.
3. Embeddings are created with local Ollama embedding models.
4. Chunks are stored in a FAISS vector index under `vectorstore/`.
5. `app.py` loads that index and exposes a question-answer API.

## API Endpoints

- `GET /`
	- Basic service info and endpoint list.
- `GET /health`
	- Health check endpoint.
- `POST /ask`
	- Accepts a JSON payload and returns an answer from retrieval + Ollama chat model.

Request body for `/ask`:

```json
{
	"question": "What does this project do?"
}
```

Response example:

```json
{
	"answer": "..."
}
```

## Local Setup

1. Install dependencies:

```bash
./env3/bin/python -m pip install -r requirements.txt
```

2. Make sure Ollama is running and required models are available:

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2:latest
```

3. Build vector index:

```bash
./env3/bin/python ingest.py
```

4. Start API server:

```bash
./env3/bin/python -m uvicorn app:app --reload
```

5. Open docs:

`http://127.0.0.1:8000/docs`

## Environment Variables

- `OLLAMA_BASE_URL` default: `http://localhost:11434`
- `OLLAMA_EMBED_MODEL` default: `nomic-embed-text`
- `OLLAMA_CHAT_MODEL` default: `llama3.2:latest`

## Project Structure

- `app.py`: FastAPI application and QA endpoint.
- `ingest.py`: document loading, chunking, embedding, and FAISS index creation.
- `documents/sample.txt`: source content used for indexing.
- `vectorstore/`: generated FAISS index files (`index.faiss`, `index.pkl`).

## Notes

- Run ingestion before starting the API; otherwise startup fails because the FAISS index is missing.
- This is a learning/demo project, so error handling and production hardening are intentionally minimal.