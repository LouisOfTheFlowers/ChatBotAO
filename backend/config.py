from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Configuracao principal da aplicacao.
    app_name: str = "Academic RAG Chatbot"
    chroma_persist_directory: str = "./chroma_data"
    chroma_documents_collection: str = "documents"
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3"
    ollama_embedding_model: str = "nomic-embed-text"
    chunk_size: int = 800
    chunk_overlap: int = 120
    default_top_k: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
