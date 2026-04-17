from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.rag import ingest_file, initialize_chroma, run_chat
from backend.schemas import ChatRequest, ChatResponse, UploadResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Prepara a colecao de documentos do ChromaDB quando a API arranca.
    initialize_chroma()
    yield


app = FastAPI(
    title="Academic RAG Chatbot",
    description="Plataforma simples de chatbot com RAG, ChromaDB e Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
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
        "main_endpoints": ["/chat", "/upload"],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        return run_chat(payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        return await ingest_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
