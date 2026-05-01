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


class ScrapeRequest(BaseModel):
    max_pages: int = Field(
        default=settings.books_scrape_max_pages,
        ge=1,
        le=50,
        description="Numero maximo de paginas do catalogo Books to Scrape a indexar.",
    )


class ScrapeResponse(BaseModel):
    message: str
    books_indexed: int
    chunks_added: int
    max_pages: int


class LocalImportRequest(BaseModel):
    limit: int = Field(
        default=settings.local_books_import_limit,
        ge=1,
        le=50000,
        description="Numero maximo de livros do ficheiro local books.csv a indexar.",
    )


class LocalImportResponse(BaseModel):
    message: str
    books_indexed: int
    chunks_added: int
    ratings_loaded: int
    limit: int


class BooksStatusResponse(BaseModel):
    indexed_chunks: int
