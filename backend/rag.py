from io import BytesIO

from fastapi import UploadFile
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from backend.config import settings
from backend.schemas import ChatRequest, ChatResponse, UploadResponse

_documents_store: Chroma | None = None


def _get_embeddings() -> OllamaEmbeddings:
    # Modelo de embeddings local servido pelo Ollama.
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def _get_llm() -> ChatOllama:
    # Modelo principal usado para gerar a resposta final.
    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )


def initialize_chroma() -> Chroma:
    global _documents_store

    if _documents_store is not None:
        return _documents_store

    embeddings = _get_embeddings()

    try:
        embeddings.embed_query("dimension probe")
    except Exception as exc:
        raise RuntimeError(
            "Nao foi possivel inicializar os embeddings no Ollama. "
            f"Confirma que o modelo '{settings.ollama_embedding_model}' esta disponivel."
        ) from exc

    _documents_store = Chroma(
        collection_name=settings.chroma_documents_collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_directory,
    )
    return _documents_store


def _extract_text(file_name: str, content: bytes) -> str:
    suffix = file_name.lower().rsplit(".", maxsplit=1)[-1]

    if suffix == "txt":
        return content.decode("utf-8", errors="ignore")

    if suffix == "pdf":
        pdf_reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in pdf_reader.pages]
        return "\n".join(pages).strip()

    raise ValueError("Formato nao suportado. Use apenas ficheiros PDF ou TXT.")


def _build_chunks(text: str, file_name: str) -> list[Document]:
    if not text.strip():
        raise ValueError("Nao foi encontrado texto no ficheiro enviado.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    documents = [Document(page_content=text, metadata={"source": file_name})]
    return splitter.split_documents(documents)


async def ingest_file(file: UploadFile) -> UploadResponse:
    if not file.filename:
        raise ValueError("O ficheiro tem de ter um nome.")

    content = await file.read()
    text = _extract_text(file.filename, content)
    chunks = _build_chunks(text, file.filename)

    documents_store = initialize_chroma()
    documents_store.add_documents(chunks)

    return UploadResponse(
        message="Documento processado com sucesso.",
        filename=file.filename,
        chunks_added=len(chunks),
    )


def _get_document_context(question: str, k: int) -> tuple[str, list[str]]:
    documents_store = initialize_chroma()
    stored_documents = documents_store.get(include=[])

    if not stored_documents.get("ids"):
        return "Nao existem documentos carregados no ChromaDB.", []

    documents = documents_store.similarity_search(question, k=k)

    if not documents:
        return "Nao foi encontrado contexto documental relevante.", []

    sources = []
    context_parts = []

    for document in documents:
        source_name = document.metadata.get("source", "desconhecido")
        sources.append(source_name)
        context_parts.append(f"Fonte: {source_name}\nConteudo: {document.page_content}")

    unique_sources = list(dict.fromkeys(sources))
    return "\n\n".join(context_parts), unique_sources


def run_chat(payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise ValueError("A pergunta nao pode estar vazia.")

    document_context, sources = _get_document_context(question, payload.top_k)

    prompt = ChatPromptTemplate.from_template(
        """
        Es um assistente academico de uma plataforma RAG.
        Responde sempre em portugues de Portugal, de forma clara e objetiva.

        Usa apenas o contexto documental fornecido.
        Se a informacao nao existir no contexto, diz isso explicitamente.
        Nao inventes dados.

        Contexto documental:
        {document_context}

        Pergunta do utilizador:
        {question}
        """
    )

    chain = prompt | _get_llm() | StrOutputParser()
    answer = chain.invoke(
        {
            "document_context": document_context,
            "question": question,
        }
    )

    return ChatResponse(
        question=question,
        answer=answer,
        document_sources=sources,
    )
