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
    books_base_url: str = "https://books.toscrape.com/"
    books_scrape_max_pages: int = 5
    books_scrape_request_delay: float = 0.1
    books_scrape_workers: int = 6
    local_books_data_directory: str = "backend/books_data"
    local_books_cache_path: str = "backend/books_data/books_cache.sqlite3"
    local_books_import_limit: int = 1000
    chroma_add_batch_size: int = 100

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
