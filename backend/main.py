from contextlib import asynccontextmanager
import logging
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.rag import (
    get_books_status,
    import_local_books,
    initialize_chroma,
    run_chat,
    stream_chat,
)
from backend.schemas import (
    BooksStatusResponse,
    ChatRequest,
    ChatResponse,
    LocalImportRequest,
)

logger = logging.getLogger(__name__)


def _auto_import_local_books() -> None:
    try:
        if settings.auto_import_local_books:
            indexed_chunks = get_books_status()
            logger.info(
                "Starting background local import up to %s books. Current index: %s chunks.",
                settings.local_books_import_limit,
                indexed_chunks,
            )
            result = import_local_books(
                LocalImportRequest(limit=settings.local_books_import_limit)
            )
            logger.info(
                "Background local import finished: %s books scanned, %s new chunks.",
                result.books_indexed,
                result.chunks_added,
            )

    except Exception as exc:
        logger.warning("Background catalog import failed: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Tenta preparar o ChromaDB no arranque, mas deixa a API arrancar mesmo
    # quando o Ollama ainda nao tem os modelos descarregados.
    try:
        initialize_chroma()
        if settings.auto_import_local_books:
            threading.Thread(target=_auto_import_local_books, daemon=True).start()
    except Exception as exc:
        logger.warning("ChromaDB was not prepared on startup: %s", exc)
    yield


app = FastAPI(
    title="Books RAG Chatbot",
    description="Chatbot de livros com CSV local, ChromaDB e Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "RAG API running",
        "docs": "/docs",
        "main_endpoints": [
            "/chat",
            "/chat/stream",
            "/books/status",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        return run_chat(payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    return StreamingResponse(stream_chat(payload), media_type="application/x-ndjson")


@app.get("/books/status", response_model=BooksStatusResponse)
def books_status():
    try:
        return BooksStatusResponse(indexed_chunks=get_books_status())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
