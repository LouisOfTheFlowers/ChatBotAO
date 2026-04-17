from pydantic import BaseModel, Field

from backend.config import settings


class ChatRequest(BaseModel):
    question: str = Field(..., description="Pergunta do utilizador.")
    top_k: int = Field(
        default=settings.default_top_k,
        ge=1,
        le=10,
        description="Numero de chunks a procurar no ChromaDB.",
    )


class ChatResponse(BaseModel):
    question: str
    answer: str
    document_sources: list[str]


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int
