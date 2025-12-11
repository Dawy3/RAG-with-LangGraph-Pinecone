# RAG with LangGraph + Pinecone

Modern RAG API that rewrites queries, retrieves from Pinecone, grades relevance, and generates answers using LangGraph, LangChain, and OpenRouter-hosted models.

## Prerequisites
- Python 3.12
- Pip
- Access keys: `OPENROUTER_API_KEY`, `PINECONE_API_KEY`

## Setup
1) Create a virtual environment:
```
python -m venv venv
.\venv\Scripts\activate
```
2) Install dependencies:
```
pip install -r requirements.txt
```
3) Configure environment:
```
cp .env.example .env  # if you add a template
```
Set at least:
- `OPENROUTER_API_KEY`
- `MODEL_NAME` (e.g., `gpt-4o-mini` or another OpenRouter model id)
- `PINECONE_API_KEY`

## Run the API
```
uvicorn backend.main:app --reload
```

## API surface
- `POST /documents/upload` — upload a PDF, chunk, and ingest into Pinecone.
- `POST /query` — run the LangGraph pipeline (rewrite → retrieve → grade → generate).
- `GET /index/state` — inspect Pinecone index stats.

## Project layout
- `backend/main.py` — FastAPI app, LangGraph pipeline, Pinecone setup.
- `requirements.txt` — pinned dependencies.

## Notes
- The graph uses sentence-transformers CPU embeddings and creates the Pinecone index if missing.
- Update `MODEL_NAME` and region/metric as needed for your deployment.

