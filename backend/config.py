from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Configuracao principal da aplicacao.
    app_name: str = "Books RAG Chatbot"
    chroma_persist_directory: str = "./chroma_data"
    chroma_documents_collection: str = "books"
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2:1b"
    ollama_embedding_model: str = "nomic-embed-text"
    chunk_size: int = 650
    chunk_overlap: int = 80
    default_top_k: int = 3
    local_books_data_directory: str = "backend/books_data"
    local_books_cache_path: str = "backend/books_data/books_cache.sqlite3"
    book_synopsis_cache_path: str = "backend/books_data/book_synopses.sqlite3"
    local_books_import_limit: int = 12000
    auto_import_local_books: bool = True
    web_synopsis_enabled: bool = True
    web_synopsis_timeout: float = 8.0
    open_library_base_url: str = "https://openlibrary.org"
    local_books_ingest_batch_size: int = 500
    local_books_ingest_batch_delay: float = 0.25
    chroma_add_batch_size: int = 100
    chroma_add_batch_delay: float = 0.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
