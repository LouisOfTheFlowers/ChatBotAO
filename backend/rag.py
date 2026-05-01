import csv
import html
import json
import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    LocalImportRequest,
    LocalImportResponse,
    ScrapeRequest,
    ScrapeResponse,
)

_documents_store: Chroma | None = None
_llm: ChatOllama | None = None
_csv_cache_lock = threading.Lock()
_scrape_thread_local = threading.local()
_prompt = ChatPromptTemplate.from_template(
    """
    Es um assistente especialista no catalogo do Books to Scrape.
    Responde sempre em portugues de Portugal, de forma clara e objetiva.

    Usa apenas o contexto de livros fornecido.
    Se a informacao nao existir no contexto, diz isso explicitamente.
    Nao respondas a pedidos que nao sejam sobre livros ou sobre este catalogo.
    Nao inventes dados.

    Contexto de livros:
    {document_context}

    Pergunta do utilizador:
    {question}
    """
)

CSV_CACHE_VERSION = "1"


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def _get_llm() -> ChatOllama:
    global _llm

    if _llm is None:
        _llm = ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
            keep_alive="10m",
        )

    return _llm


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


def _fetch_html(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(
        url,
        timeout=20,
        headers={"User-Agent": "ChatBotAO/1.0 educational scraper"},
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _get_scrape_session() -> requests.Session:
    session = getattr(_scrape_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=settings.books_scrape_workers,
            pool_maxsize=settings.books_scrape_workers,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _scrape_thread_local.session = session

    return session


def _clean_text(value: str) -> str:
    return html.unescape(" ".join(str(value).split()))


def _rating_from_classes(classes: list[str]) -> str:
    for class_name in classes:
        if class_name in {"One", "Two", "Three", "Four", "Five"}:
            return class_name
    return "Unknown"


def _extract_product_links(page: BeautifulSoup, page_url: str) -> list[str]:
    links = []
    for anchor in page.select("article.product_pod h3 a"):
        href = anchor.get("href")
        if href:
            links.append(urljoin(page_url, href))
    return links


def _extract_next_page(page: BeautifulSoup, page_url: str) -> str | None:
    next_anchor = page.select_one("li.next a")
    if not next_anchor:
        return None

    href = next_anchor.get("href")
    if not href:
        return None

    return urljoin(page_url, href)


def _extract_book_document(page: BeautifulSoup, url: str) -> Document:
    title = _clean_text(page.select_one("div.product_main h1").get_text())
    price = _clean_text(page.select_one(".price_color").get_text())
    availability = _clean_text(page.select_one(".availability").get_text())
    rating_node = page.select_one(".star-rating")
    rating = _rating_from_classes(rating_node.get("class", []) if rating_node else [])

    category_links = page.select(".breadcrumb li a")
    category = _clean_text(category_links[-1].get_text()) if category_links else "Books"

    description = ""
    description_header = page.select_one("#product_description")
    if description_header:
        description_node = description_header.find_next_sibling("p")
        if description_node:
            description = _clean_text(description_node.get_text())

    table_data = {}
    for row in page.select("table.table.table-striped tr"):
        key_node = row.select_one("th")
        value_node = row.select_one("td")
        if key_node and value_node:
            table_data[_clean_text(key_node.get_text())] = _clean_text(value_node.get_text())

    upc = table_data.get("UPC", "")
    reviews = table_data.get("Number of reviews", "0")

    content = "\n".join(
        [
            f"Titulo: {title}",
            f"Categoria: {category}",
            f"Preco: {price}",
            f"Disponibilidade: {availability}",
            f"Classificacao: {rating}",
            f"UPC: {upc}",
            f"Numero de reviews: {reviews}",
            f"URL: {url}",
            f"Descricao: {description or 'Sem descricao disponivel.'}",
        ]
    )

    return Document(
        page_content=content,
        metadata={
            "source": title,
            "title": title,
            "category": category,
            "price": price,
            "availability": availability,
            "rating": rating,
            "url": url,
            "site": "books.toscrape.com",
        },
    )


def _scrape_book_documents(max_pages: int) -> list[Document]:
    session = requests.Session()
    page_url = settings.books_base_url
    product_urls = []

    for _ in range(max_pages):
        catalog_page = _fetch_html(session, page_url)
        product_urls.extend(_extract_product_links(catalog_page, page_url))

        next_page = _extract_next_page(catalog_page, page_url)
        if not next_page:
            break

        page_url = next_page
        time.sleep(settings.books_scrape_request_delay)

    if not product_urls:
        return []

    def fetch_product(product_url: str) -> Document:
        if settings.books_scrape_request_delay > 0:
            time.sleep(settings.books_scrape_request_delay)

        product_page = _fetch_html(_get_scrape_session(), product_url)
        return _extract_book_document(product_page, product_url)

    worker_count = max(1, min(settings.books_scrape_workers, len(product_urls)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(fetch_product, product_urls))


def _build_chunks(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


def _document_id(document: Document, index: int) -> str:
    url = document.metadata.get("url", "")
    digest = sha1(f"{url}:{index}".encode("utf-8")).hexdigest()
    return f"book-{digest}"


def _local_document_id(document: Document, index: int) -> str:
    isbn = document.metadata.get("isbn", "")
    digest = sha1(f"{isbn}:{index}".encode("utf-8")).hexdigest()
    return f"local-book-{digest}"


def _book_count() -> int:
    documents_store = initialize_chroma()
    try:
        return documents_store._collection.count()
    except Exception:
        stored_documents = documents_store.get(include=[])
        return len(stored_documents.get("ids", []))


def get_books_status() -> int:
    return _book_count()


def _existing_chroma_ids(documents_store: Chroma, ids: list[str]) -> set[str]:
    if not ids:
        return set()

    existing_ids = set()
    for start in range(0, len(ids), settings.chroma_add_batch_size):
        batch_ids = ids[start:start + settings.chroma_add_batch_size]
        try:
            stored = documents_store.get(ids=batch_ids, include=[])
        except Exception:
            continue

        existing_ids.update(stored.get("ids", []))

    return existing_ids


def _add_new_documents_in_batches(
    documents_store: Chroma,
    chunks: list[Document],
    ids: list[str],
) -> int:
    existing_ids = _existing_chroma_ids(documents_store, ids)
    pending = [
        (chunk, document_id)
        for chunk, document_id in zip(chunks, ids, strict=True)
        if document_id not in existing_ids
    ]

    for start in range(0, len(pending), settings.chroma_add_batch_size):
        batch = pending[start:start + settings.chroma_add_batch_size]
        if not batch:
            continue

        batch_chunks, batch_ids = zip(*batch, strict=True)
        documents_store.add_documents(list(batch_chunks), ids=list(batch_ids))

    return len(pending)


def scrape_books(payload: ScrapeRequest) -> ScrapeResponse:
    documents = _scrape_book_documents(payload.max_pages)
    if not documents:
        raise RuntimeError("Nao foi encontrado nenhum livro para indexar.")

    chunks = _build_chunks(documents)
    ids = [_document_id(chunk, index) for index, chunk in enumerate(chunks)]

    documents_store = initialize_chroma()
    chunks_added = _add_new_documents_in_batches(documents_store, chunks, ids)

    return ScrapeResponse(
        message="Catalogo de livros indexado com sucesso.",
        books_indexed=len(documents),
        chunks_added=chunks_added,
        max_pages=payload.max_pages,
    )


def _books_data_path(file_name: str) -> Path:
    return Path(settings.local_books_data_directory) / file_name


def _read_local_books(limit: int) -> list[dict]:
    books_path = _books_data_path("books.csv")
    if not books_path.exists():
        raise RuntimeError(f"Nao foi encontrado o ficheiro local {books_path}.")

    books = []
    with books_path.open("r", encoding="latin-1", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            isbn = _clean_text(row.get("ISBN", ""))
            title = _clean_text(row.get("Book-Title", ""))
            if not isbn or not title:
                continue

            books.append(
                {
                    "isbn": isbn,
                    "title": title,
                    "author": _clean_text(row.get("Book-Author", "")),
                    "year": _clean_text(row.get("Year-Of-Publication", "")),
                    "publisher": _clean_text(row.get("Publisher", "")),
                    "image_url": _clean_text(row.get("Image-URL-L", "")),
                }
            )

            if len(books) >= limit:
                break

    return books


def _location_country(location: str) -> str:
    parts = [part.strip() for part in location.split(",") if part.strip()]
    return parts[-1].title() if parts else "Localizacao desconhecida"


def _csv_cache_path() -> Path:
    return Path(settings.local_books_cache_path)


def _csv_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _connect_csv_cache() -> sqlite3.Connection:
    cache_path = _csv_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _csv_cache_is_current(connection: sqlite3.Connection) -> bool:
    ratings_path = _books_data_path("ratings.csv")
    users_path = _books_data_path("users.csv")
    if not ratings_path.exists():
        return True

    try:
        metadata = dict(
            connection.execute("SELECT key, value FROM csv_cache_metadata").fetchall()
        )
    except sqlite3.Error:
        return False

    expected = {
        "version": CSV_CACHE_VERSION,
        "ratings_signature": _csv_signature(ratings_path),
        "users_signature": _csv_signature(users_path) if users_path.exists() else "",
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def _load_user_countries() -> dict[str, str]:
    users_path = _books_data_path("users.csv")
    if not users_path.exists():
        return {}

    countries = {}
    with users_path.open("r", encoding="latin-1", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            user_id = _clean_text(row.get("User-ID", ""))
            if user_id:
                countries[user_id] = _location_country(_clean_text(row.get("Location", "")))

    return countries


def _rebuild_csv_cache(connection: sqlite3.Connection) -> None:
    ratings_path = _books_data_path("ratings.csv")
    users_path = _books_data_path("users.csv")
    if not ratings_path.exists():
        return

    user_countries = _load_user_countries()
    sums = defaultdict(int)
    counts = defaultdict(int)
    location_counts = defaultdict(Counter)

    with ratings_path.open("r", encoding="latin-1", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            isbn = _clean_text(row.get("ISBN", ""))
            if not isbn:
                continue

            try:
                rating = int(row.get("Book-Rating", "0"))
            except ValueError:
                continue

            if rating <= 0:
                continue

            user_id = _clean_text(row.get("User-ID", ""))
            sums[isbn] += rating
            counts[isbn] += 1
            country = user_countries.get(user_id)
            if country:
                location_counts[isbn][country] += 1

    with connection:
        connection.execute("DROP TABLE IF EXISTS rating_summary")
        connection.execute("DROP TABLE IF EXISTS csv_cache_metadata")
        connection.execute(
            """
            CREATE TABLE rating_summary (
                isbn TEXT PRIMARY KEY,
                average REAL NOT NULL,
                rating_count INTEGER NOT NULL,
                locations_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE csv_cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        rows = [
            (
                isbn,
                round(sums[isbn] / counts[isbn], 2),
                counts[isbn],
                json.dumps(location_counts[isbn].most_common(3), ensure_ascii=False),
            )
            for isbn in counts
        ]
        connection.executemany(
            """
            INSERT INTO rating_summary (
                isbn,
                average,
                rating_count,
                locations_json
            ) VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        connection.executemany(
            "INSERT INTO csv_cache_metadata (key, value) VALUES (?, ?)",
            [
                ("version", CSV_CACHE_VERSION),
                ("ratings_signature", _csv_signature(ratings_path)),
                ("users_signature", _csv_signature(users_path) if users_path.exists() else ""),
            ],
        )


def _ensure_csv_cache() -> None:
    with _csv_cache_lock:
        connection = _connect_csv_cache()
        try:
            if not _csv_cache_is_current(connection):
                _rebuild_csv_cache(connection)
        finally:
            connection.close()


def _read_rating_summary(isbns: set[str]) -> tuple[dict[str, dict], int]:
    ratings_path = _books_data_path("ratings.csv")
    if not ratings_path.exists() or not isbns:
        return {}, 0

    _ensure_csv_cache()

    summary = {}
    ratings_loaded = 0
    isbn_list = list(isbns)

    connection = _connect_csv_cache()
    try:
        for start in range(0, len(isbn_list), 900):
            batch = isbn_list[start:start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                f"""
                SELECT isbn, average, rating_count, locations_json
                FROM rating_summary
                WHERE isbn IN ({placeholders})
                """,
                batch,
            ).fetchall()

            for isbn, average, rating_count, locations_json in rows:
                try:
                    locations = json.loads(locations_json)
                except json.JSONDecodeError:
                    locations = []

                summary[isbn] = {
                    "average": average,
                    "count": rating_count,
                    "locations": locations,
                }
                ratings_loaded += rating_count
    finally:
        connection.close()

    return summary, ratings_loaded


def _build_local_book_documents(books: list[dict]) -> tuple[list[Document], int]:
    rating_summary, ratings_loaded = _read_rating_summary(
        {book["isbn"] for book in books}
    )
    documents = []

    for book in books:
        ratings = rating_summary.get(book["isbn"], {})
        average_rating = ratings.get("average")
        rating_count = ratings.get("count", 0)
        rating_text = (
            f"{average_rating}/10 baseado em {rating_count} avaliacoes"
            if average_rating is not None
            else "Sem avaliacoes positivas no ficheiro local"
        )
        locations = ratings.get("locations", [])
        location_text = (
            ", ".join(f"{country} ({count})" for country, count in locations)
            if locations
            else "Sem localizacoes de utilizadores associadas"
        )

        content = "\n".join(
            [
                f"Titulo: {book['title']}",
                f"Autor: {book['author'] or 'Desconhecido'}",
                f"Ano de publicacao: {book['year'] or 'Desconhecido'}",
                f"Editora: {book['publisher'] or 'Desconhecida'}",
                f"ISBN: {book['isbn']}",
                f"Classificacao local: {rating_text}",
                f"Localizacoes dos avaliadores: {location_text}",
                f"Imagem: {book['image_url'] or 'Sem imagem'}",
                "Origem: backend/books_data/books.csv, ratings.csv e users.csv",
            ]
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": book["title"],
                    "title": book["title"],
                    "author": book["author"],
                    "year": book["year"],
                    "publisher": book["publisher"],
                    "isbn": book["isbn"],
                    "site": "backend/books_data",
                },
            )
        )

    return documents, ratings_loaded


def import_local_books(payload: LocalImportRequest) -> LocalImportResponse:
    books = _read_local_books(payload.limit)
    if not books:
        raise RuntimeError("Nao foi encontrado nenhum livro no ficheiro local.")

    documents, ratings_loaded = _build_local_book_documents(books)
    chunks = _build_chunks(documents)
    ids = [_local_document_id(chunk, index) for index, chunk in enumerate(chunks)]

    documents_store = initialize_chroma()
    chunks_added = _add_new_documents_in_batches(documents_store, chunks, ids)

    return LocalImportResponse(
        message="Ficheiro local de livros indexado com sucesso.",
        books_indexed=len(documents),
        chunks_added=chunks_added,
        ratings_loaded=ratings_loaded,
        limit=payload.limit,
    )


def _question_terms(question: str) -> list[str]:
    return [
        term.lower()
        for term in re.findall(r"[\w']+", question, flags=re.UNICODE)
        if len(term) >= 3
    ]


def _document_match_score(document: Document, question: str, terms: list[str]) -> int:
    title = str(document.metadata.get("title", "")).lower()
    author = str(document.metadata.get("author", "")).lower()
    content = document.page_content.lower()
    question_lower = question.lower()

    score = 0
    if title and title in question_lower:
        score += 30
    if author and author in question_lower:
        score += 20

    for term in terms:
        if term in title:
            score += 8
        if term in author:
            score += 5
        if term in content:
            score += 1

    return score


def _rerank_documents(
    documents: list[Document],
    question: str,
    k: int,
) -> list[Document]:
    terms = _question_terms(question)
    ranked_documents = sorted(
        enumerate(documents),
        key=lambda item: (
            _document_match_score(item[1], question, terms),
            -item[0],
        ),
        reverse=True,
    )
    return [document for _, document in ranked_documents[:k]]


def _get_book_context(question: str, k: int) -> tuple[str, list[str]]:
    documents_store = initialize_chroma()
    if _book_count() == 0:
        return "Ainda nao existem livros indexados no ChromaDB.", []

    candidate_count = min(max(k * 4, 12), 40)
    documents = documents_store.similarity_search(question, k=candidate_count)
    if not documents:
        return "Nao foi encontrado contexto de livros relevante.", []

    documents = _rerank_documents(documents, question, k)
    sources = []
    context_parts = []

    for document in documents:
        source_name = document.metadata.get("source", "desconhecido")
        sources.append(source_name)
        context_parts.append(f"Livro: {source_name}\n{document.page_content}")

    unique_sources = list(dict.fromkeys(sources))
    return "\n\n".join(context_parts), unique_sources


def _build_chain():
    return _prompt | _get_llm() | StrOutputParser()


def run_chat(payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise ValueError("A pergunta nao pode estar vazia.")

    document_context, sources = _get_book_context(question, payload.top_k)

    try:
        answer = _build_chain().invoke(
            {
                "document_context": document_context,
                "question": question,
            }
        )
    except Exception as exc:
        raise RuntimeError(
            "Nao foi possivel gerar a resposta no Ollama. "
            f"Confirma que o modelo '{settings.ollama_llm_model}' esta disponivel "
            "e que o Docker tem memoria suficiente."
        ) from exc

    return ChatResponse(
        question=question,
        answer=answer,
        document_sources=sources,
    )


def stream_chat(payload: ChatRequest) -> Generator[str, None, None]:
    question = payload.question.strip()
    if not question:
        yield _json_event({"type": "error", "message": "A pergunta nao pode estar vazia."})
        return

    try:
        document_context, sources = _get_book_context(question, payload.top_k)
        yield _json_event({"type": "sources", "sources": sources})

        chain = _prompt | _get_llm()
        for chunk in chain.stream(
            {
                "document_context": document_context,
                "question": question,
            }
        ):
            content = getattr(chunk, "content", "")
            if content:
                yield _json_event({"type": "token", "content": content})

        yield _json_event({"type": "done"})
    except Exception as exc:
        yield _json_event(
            {
                "type": "error",
                "message": (
                    "Nao foi possivel gerar a resposta no Ollama. "
                    f"Confirma que o modelo '{settings.ollama_llm_model}' esta disponivel."
                ),
            }
        )


def _json_event(payload: dict) -> str:
    return f"{json.dumps(payload, ensure_ascii=False)}\n"
