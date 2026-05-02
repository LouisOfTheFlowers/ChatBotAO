import csv
import html
import json
import re
import sqlite3
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Generator
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

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
)

_documents_store: Chroma | None = None
_llm: ChatOllama | None = None
_csv_cache_lock = threading.Lock()
_local_import_lock = threading.Lock()
_prompt = ChatPromptTemplate.from_template(
    """
    Es um assistente especialista no catalogo de livros indexado.
    Responde sempre em portugues de Portugal, de forma clara e objetiva.

    Usa apenas o contexto de livros fornecido.
    Se a informacao nao existir no contexto, diz apenas que nao encontraste informacao suficiente no indice.
    Nunca digas que nao podes fornecer informacoes por motivos genericos.
    Nao respondas a pedidos que nao sejam sobre livros ou sobre o catalogo indexado.
    Nao inventes dados.

    Contexto de livros:
    {document_context}

    Pergunta do utilizador:
    {question}
    """
)

CSV_CACHE_VERSION = "1"

_STOPWORDS = {
    "about",
    "and",
    "are",
    "book",
    "books",
    "com",
    "dos",
    "das",
    "de",
    "do",
    "e",
    "fala",
    "falar",
    "for",
    "from",
    "livro",
    "livros",
    "na",
    "nas",
    "no",
    "nos",
    "of",
    "os",
    "por",
    "que",
    "qual",
    "quais",
    "sobre",
    "the",
    "um",
    "uma",
}

_DESCRIPTIVE_INTENT_TERMS = {
    "descreve",
    "descricao",
    "detalha",
    "explica",
    "fala",
    "resume",
    "resumo",
    "sobre",
}


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


def _clean_text(value: str) -> str:
    return html.unescape(" ".join(str(value).split()))


def _build_chunks(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


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
        if settings.chroma_add_batch_delay > 0:
            time.sleep(settings.chroma_add_batch_delay)

    return len(pending)


def _books_data_path(file_name: str) -> Path:
    return Path(settings.local_books_data_directory) / file_name


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _parse_local_book_row(row: dict) -> dict | None:
    isbn = _clean_text(row.get("ISBN", "") or row.get("isbn", ""))
    title = _clean_text(row.get("Book-Title", "") or row.get("title", ""))
    if not isbn or not title:
        return None

    publication_date = _clean_text(row.get("publication_date", ""))
    publication_year = _clean_text(row.get("Year-Of-Publication", ""))
    if not publication_year and publication_date:
        publication_year = publication_date.split("/")[-1]

    synopsis = _clean_text(
        row.get("sinopse", "")
        or row.get("synopsis", "")
        or row.get("description", "")
        or row.get("descricao", "")
        or row.get("summary", "")
        or row.get("plot", "")
    )

    return {
        "isbn": isbn,
        "title": title,
        "author": _clean_text(row.get("Book-Author", "") or row.get("authors", "")),
        "year": publication_year,
        "publisher": _clean_text(row.get("Publisher", "") or row.get("publisher", "")),
        "image_url": _clean_text(row.get("Image-URL-L", "")),
        "average_rating": _clean_text(row.get("average_rating", "")),
        "ratings_count": _clean_text(row.get("ratings_count", "")),
        "text_reviews_count": _clean_text(row.get("text_reviews_count", "")),
        "language_code": _clean_text(row.get("language_code", "")),
        "num_pages": _clean_text(row.get("  num_pages", "") or row.get("num_pages", "")),
        "publication_date": publication_date,
        "isbn13": _clean_text(row.get("isbn13", "")),
        "synopsis": synopsis,
    }


def _local_books_csv_reader(file) -> csv.DictReader:
    sample = file.read(4096)
    file.seek(0)
    delimiter = ";" if sample.splitlines()[0].count(";") > sample.splitlines()[0].count(",") else ","
    return csv.DictReader(file, delimiter=delimiter)


def _local_books_encoding(path: Path) -> str:
    sample = path.read_bytes()[:4096]
    try:
        sample.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "latin-1"


def _synopsis_cache_path() -> Path:
    return Path(settings.book_synopsis_cache_path)


def _connect_synopsis_cache() -> sqlite3.Connection:
    cache_path = _synopsis_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS synopsis_cache (
            cache_key TEXT PRIMARY KEY,
            synopsis TEXT NOT NULL,
            source_url TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    return connection


def _synopsis_cache_key(book: dict[str, str]) -> str:
    isbn = book.get("isbn") or book.get("isbn13")
    if isbn:
        return f"isbn:{isbn}"

    identity = _normalize_text(f"{book.get('title', '')}:{book.get('author', '')}")
    return "title:" + sha1(identity.encode("utf-8")).hexdigest()


def _read_cached_synopsis(book: dict[str, str]) -> tuple[str, str] | None:
    connection = _connect_synopsis_cache()
    try:
        row = connection.execute(
            """
            SELECT synopsis, source_url
            FROM synopsis_cache
            WHERE cache_key = ?
            """,
            (_synopsis_cache_key(book),),
        ).fetchone()
    finally:
        connection.close()

    return (row[0], row[1]) if row else None


def _write_cached_synopsis(book: dict[str, str], synopsis: str, source_url: str) -> None:
    connection = _connect_synopsis_cache()
    try:
        with connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO synopsis_cache (
                    cache_key,
                    synopsis,
                    source_url,
                    created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (_synopsis_cache_key(book), synopsis, source_url, time.time()),
            )
    finally:
        connection.close()


def _fetch_json_url(url: str) -> dict:
    request = Request(
        url,
        headers={"User-Agent": "ChatBotAO/1.0 book synopsis lookup"},
    )
    with urlopen(request, timeout=settings.web_synopsis_timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _open_library_url(path: str) -> str:
    return f"{settings.open_library_base_url.rstrip('/')}{path}"


def _extract_open_library_description(payload: dict) -> str:
    description = payload.get("description")
    if isinstance(description, dict):
        description = description.get("value", "")
    if isinstance(description, str):
        return _clean_text(description)
    return ""


def _fetch_open_library_work_synopsis(work_key: str) -> tuple[str, str] | None:
    source_url = _open_library_url(f"{work_key}.json")
    payload = _fetch_json_url(source_url)
    synopsis = _extract_open_library_description(payload)
    if synopsis:
        return synopsis, source_url
    return None


def _fetch_open_library_by_isbn(isbn: str) -> tuple[str, str] | None:
    if not isbn:
        return None

    edition_url = _open_library_url(f"/isbn/{quote(isbn)}.json")
    payload = _fetch_json_url(edition_url)
    synopsis = _extract_open_library_description(payload)
    if synopsis:
        return synopsis, edition_url

    for work in payload.get("works", []):
        work_key = work.get("key")
        if work_key:
            work_result = _fetch_open_library_work_synopsis(work_key)
            if work_result:
                return work_result

    return None


def _fetch_open_library_by_title(book: dict[str, str]) -> tuple[str, str] | None:
    params = {
        "title": book.get("title", ""),
        "fields": "key,title,author_name",
        "limit": "1",
    }
    if book.get("author"):
        params["author"] = book["author"].split("/")[0]

    search_url = _open_library_url(f"/search.json?{urlencode(params)}")
    payload = _fetch_json_url(search_url)
    docs = payload.get("docs", [])
    if not docs:
        return None

    work_key = docs[0].get("key")
    if not work_key:
        return None

    return _fetch_open_library_work_synopsis(work_key)


def _lookup_web_synopsis(book: dict[str, str]) -> tuple[str, str] | None:
    if not settings.web_synopsis_enabled:
        return None

    cached = _read_cached_synopsis(book)
    if cached:
        return cached

    try:
        result = None
        for isbn in (book.get("isbn"), book.get("isbn13")):
            result = _fetch_open_library_by_isbn(isbn or "")
            if result:
                break

        if not result:
            result = _fetch_open_library_by_title(book)

        if not result:
            return None

        synopsis, source_url = result
        _write_cached_synopsis(book, synopsis, source_url)
        return synopsis, source_url
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _iter_local_book_batches(limit: int, batch_size: int) -> Generator[list[dict], None, None]:
    books_path = _books_data_path("books.csv")
    if not books_path.exists():
        raise RuntimeError(f"Nao foi encontrado o ficheiro local {books_path}.")

    books_batch = []
    books_read = 0
    with books_path.open("r", encoding=_local_books_encoding(books_path), newline="") as file:
        reader = _local_books_csv_reader(file)
        for row in reader:
            book = _parse_local_book_row(row)
            if not book:
                continue

            books_batch.append(book)
            books_read += 1

            if len(books_batch) >= batch_size:
                yield books_batch
                books_batch = []

            if books_read >= limit:
                break

    if books_batch:
        yield books_batch


def _read_local_books(limit: int) -> list[dict]:
    books = []
    for books_batch in _iter_local_book_batches(limit, limit):
        books.extend(books_batch)
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
        if book.get("average_rating"):
            rating_count = book.get("ratings_count") or "numero desconhecido de"
            rating_text = f"{book['average_rating']}/5 baseado em {rating_count} avaliacoes"
        else:
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
        review_text = (
            f"{book['text_reviews_count']} reviews de texto"
            if book.get("text_reviews_count")
            else "Sem contagem de reviews de texto"
        )
        page_text = (
            f"{book['num_pages']} paginas"
            if book.get("num_pages")
            else "Numero de paginas desconhecido"
        )
        language_text = book.get("language_code") or "Idioma desconhecido"
        publication_text = book.get("publication_date") or book["year"] or "Desconhecida"
        synopsis_text = book.get("synopsis") or "Sem sinopse disponivel"

        content = "\n".join(
            [
                f"Titulo: {book['title']}",
                f"Autor: {book['author'] or 'Desconhecido'}",
                f"Ano de publicacao: {book['year'] or 'Desconhecido'}",
                f"Data de publicacao: {publication_text}",
                f"Editora: {book['publisher'] or 'Desconhecida'}",
                f"ISBN: {book['isbn']}",
                f"Idioma: {language_text}",
                f"Paginas: {page_text}",
                f"Classificacao local: {rating_text}",
                f"Reviews: {review_text}",
                f"Localizacoes dos avaliadores: {location_text}",
                f"Imagem: {book['image_url'] or 'Sem imagem'}",
                f"Sinopse: {synopsis_text}",
                "Origem: backend/books_data/books.csv",
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
                    "average_rating": book.get("average_rating", ""),
                    "language_code": book.get("language_code", ""),
                    "synopsis": book.get("synopsis", ""),
                    "site": "backend/books_data",
                },
            )
        )

    return documents, ratings_loaded


def import_local_books(payload: LocalImportRequest) -> LocalImportResponse:
    total_books_indexed = 0
    total_chunks_added = 0
    total_ratings_loaded = 0
    batch_size = max(1, settings.local_books_ingest_batch_size)

    with _local_import_lock:
        documents_store = initialize_chroma()
        for books in _iter_local_book_batches(payload.limit, batch_size):
            documents, ratings_loaded = _build_local_book_documents(books)
            chunks = _build_chunks(documents)
            ids = [
                _local_document_id(chunk, index)
                for index, chunk in enumerate(chunks)
            ]

            total_chunks_added += _add_new_documents_in_batches(
                documents_store,
                chunks,
                ids,
            )
            total_books_indexed += len(documents)
            total_ratings_loaded += ratings_loaded

            if settings.local_books_ingest_batch_delay > 0:
                time.sleep(settings.local_books_ingest_batch_delay)

    if total_books_indexed == 0:
        raise RuntimeError("Nao foi encontrado nenhum livro no ficheiro local.")

    return LocalImportResponse(
        message="Ficheiro local de livros indexado com sucesso.",
        books_indexed=total_books_indexed,
        chunks_added=total_chunks_added,
        ratings_loaded=total_ratings_loaded,
        limit=payload.limit,
    )


def _question_terms(question: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[\w']+", _normalize_text(question), flags=re.UNICODE)
        if len(term) >= 3 and term not in _STOPWORDS
    ]


def _document_match_score(document: Document, question: str, terms: list[str]) -> int:
    title = _normalize_text(str(document.metadata.get("title", "")))
    author = _normalize_text(str(document.metadata.get("author", "")))
    content = _normalize_text(document.page_content)
    question_lower = _normalize_text(question)

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


def _document_matched_term_count(document: Document, terms: list[str]) -> int:
    haystack = _normalize_text(
        " ".join(
            [
                str(document.metadata.get("title", "")),
                str(document.metadata.get("author", "")),
                document.page_content,
            ]
        )
    )
    return sum(1 for term in terms if term in haystack)


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


def _find_local_catalog_matches(question: str, limit: int = 3) -> list[dict[str, str]]:
    terms = _question_terms(question)
    if not terms:
        return []

    books_path = _books_data_path("books.csv")
    if not books_path.exists():
        return []

    matches = []
    question_lower = _normalize_text(question)
    with books_path.open("r", encoding=_local_books_encoding(books_path), newline="") as file:
        reader = _local_books_csv_reader(file)
        for row in reader:
            book = _parse_local_book_row(row)
            if not book:
                continue

            title = _normalize_text(book["title"])
            author = _normalize_text(book["author"])
            haystack = f"{title} {author}"
            score = sum(1 for term in terms if term in haystack)
            if title and title in question_lower:
                score += 10
            if author and author in question_lower:
                score += 5
            if score == 0:
                continue

            matches.append({**book, "score": str(score)})

    matches.sort(key=lambda match: int(match["score"]), reverse=True)
    return matches[:limit]


def _build_context_from_local_matches(
    local_matches: list[dict[str, str]],
) -> tuple[str, list[str]]:
    documents, _ = _build_local_book_documents(local_matches)
    sources = []
    context_parts = []

    for document in documents:
        source_name = document.metadata.get("source", "desconhecido")
        sources.append(source_name)
        context_parts.append(f"Livro: {source_name}\n{document.page_content}")

    return "\n\n".join(context_parts), list(dict.fromkeys(sources))


def _should_answer_from_local_catalog(
    question: str,
    local_matches: list[dict[str, str]],
) -> bool:
    if not local_matches:
        return False

    question_terms = set(
        re.findall(r"[\w']+", _normalize_text(question), flags=re.UNICODE)
    )
    if question_terms & _DESCRIPTIVE_INTENT_TERMS:
        return False

    terms = _question_terms(question)
    if not terms:
        return False

    top_score = int(local_matches[0].get("score", "0"))
    minimum_score = min(2, len(terms))
    return len(terms) <= 5 and top_score >= minimum_score


def _should_describe_from_local_catalog(
    question: str,
    local_matches: list[dict[str, str]],
) -> bool:
    if not local_matches:
        return False

    question_terms = set(
        re.findall(r"[\w']+", _normalize_text(question), flags=re.UNICODE)
    )
    return bool(question_terms & _DESCRIPTIVE_INTENT_TERMS)


def _build_local_catalog_answer(local_matches: list[dict[str, str]]) -> str:
    lines = []
    for match in local_matches:
        details = [
            f"autor: {match.get('author') or 'desconhecido'}",
            f"ano: {match.get('year') or 'desconhecido'}",
            f"editora: {match.get('publisher') or 'desconhecida'}",
            f"ISBN: {match.get('isbn') or 'desconhecido'}",
        ]

        if match.get("average_rating"):
            details.append(f"avaliacao media: {match['average_rating']}/5")
        if match.get("ratings_count"):
            details.append(f"{match['ratings_count']} avaliacoes")
        if match.get("num_pages"):
            details.append(f"{match['num_pages']} paginas")
        if match.get("language_code"):
            details.append(f"idioma: {match['language_code']}")

        lines.append(f"- {match['title']} ({'; '.join(details)}).")

    return "Encontrei estes livros no catalogo local:\n" + "\n".join(lines)


def _build_book_description_answer(local_matches: list[dict[str, str]]) -> str:
    if not local_matches:
        return (
            "Nao encontrei esse livro no catalogo local. Tenta indicar o titulo "
            "ou o autor com mais detalhe."
        )

    primary = local_matches[0]
    related = local_matches[1:]
    synopsis_source = ""
    if not primary.get("synopsis"):
        web_synopsis = _lookup_web_synopsis(primary)
        if web_synopsis:
            primary["synopsis"], synopsis_source = web_synopsis

    publication_year = f" em {primary['year']}" if primary.get("year") else ""
    parts = [
        (
            f"{primary['title']} e um livro de {primary.get('author') or 'autor desconhecido'}, "
            f"publicado por {primary.get('publisher') or 'editora desconhecida'}"
            f"{publication_year}."
        )
    ]

    if primary.get("synopsis"):
        parts.append(f"Sinopse: {primary['synopsis']}")
        if synopsis_source:
            parts.append(f"Fonte da sinopse: {synopsis_source}")

    facts = []
    if primary.get("average_rating"):
        rating_text = f"tem avaliacao media de {primary['average_rating']}/5"
        if primary.get("ratings_count"):
            rating_text += f" com {primary['ratings_count']} avaliacoes"
        facts.append(rating_text)
    if primary.get("num_pages"):
        facts.append(f"tem {primary['num_pages']} paginas")
    if primary.get("language_code"):
        facts.append(f"esta registado no idioma {primary['language_code']}")
    if primary.get("isbn"):
        facts.append(f"o ISBN e {primary['isbn']}")

    if facts:
        parts.append("No catalogo, " + ", ".join(facts) + ".")

    if related:
        related_titles = ", ".join(match["title"] for match in related[:2])
        parts.append(f"Tambem encontrei entradas relacionadas: {related_titles}.")

    if not primary.get("synopsis"):
        parts.append(
            "Nota: o CSV nao inclui sinopse/enredo para este livro, por isso so consigo falar com base nos metadados disponiveis."
        )

    return " ".join(parts)


def _build_no_context_answer(local_matches: list[dict[str, str]]) -> str:
    if local_matches:
        match_lines = [
            (
                f"- {match['title']}, de {match['author']} "
                f"({match['year']}, {match['publisher']}, ISBN {match['isbn']})."
            )
            for match in local_matches
        ]
        matches_text = "\n".join(match_lines)
        return (
            "Nao encontrei contexto relevante ja indexado no ChromaDB para responder em detalhe. "
            "Mas encontrei possiveis correspondencias no CSV local:\n"
            f"{matches_text}\n\n"
            "Para o chatbot responder com base no indice, importa mais livros do CSV local "
            "e volta a fazer a pergunta."
        )

    return (
        "Nao encontrei informacao suficiente no indice para responder a essa pergunta. "
        "Confirma se ja importaste/indexaste livros suficientes e se o titulo ou autor existe no catalogo."
    )


def _get_book_context(
    question: str,
    k: int,
) -> tuple[str, list[str], bool, list[dict[str, str]]]:
    documents_store = initialize_chroma()
    if _book_count() == 0:
        local_matches = _find_local_catalog_matches(question)
        if local_matches:
            document_context, sources = _build_context_from_local_matches(local_matches)
            return document_context, sources, True, []
        return "Ainda nao existem livros indexados no ChromaDB.", [], False, []

    candidate_count = min(max(k * 4, 12), 40)
    documents = documents_store.similarity_search(question, k=candidate_count)
    if not documents:
        local_matches = _find_local_catalog_matches(question)
        if local_matches:
            document_context, sources = _build_context_from_local_matches(local_matches)
            return document_context, sources, True, []
        return "Nao foi encontrado contexto de livros relevante.", [], False, []

    documents = _rerank_documents(documents, question, k)
    terms = _question_terms(question)
    minimum_matches = 2 if len(terms) >= 2 else 1
    documents = [
        document
        for document in documents
        if _document_matched_term_count(document, terms) >= minimum_matches
    ]
    if not documents:
        local_matches = _find_local_catalog_matches(question)
        if local_matches:
            document_context, sources = _build_context_from_local_matches(local_matches)
            return document_context, sources, True, []
        return "Nao foi encontrado contexto de livros relevante.", [], False, []

    sources = []
    context_parts = []

    for document in documents:
        source_name = document.metadata.get("source", "desconhecido")
        sources.append(source_name)
        context_parts.append(f"Livro: {source_name}\n{document.page_content}")

    unique_sources = list(dict.fromkeys(sources))
    return "\n\n".join(context_parts), unique_sources, True, []


def _build_chain():
    return _prompt | _get_llm() | StrOutputParser()


def run_chat(payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise ValueError("A pergunta nao pode estar vazia.")

    local_matches = _find_local_catalog_matches(question)
    if _should_describe_from_local_catalog(question, local_matches):
        return ChatResponse(
            question=question,
            answer=_build_book_description_answer(local_matches),
            document_sources=[match["title"] for match in local_matches],
        )

    if _should_answer_from_local_catalog(question, local_matches):
        return ChatResponse(
            question=question,
            answer=_build_local_catalog_answer(local_matches),
            document_sources=[match["title"] for match in local_matches],
        )

    document_context, sources, has_relevant_context, local_matches = _get_book_context(
        question,
        payload.top_k,
    )
    if not has_relevant_context:
        fallback_sources = [match["title"] for match in local_matches]
        return ChatResponse(
            question=question,
            answer=_build_no_context_answer(local_matches),
            document_sources=fallback_sources,
        )

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
        local_matches = _find_local_catalog_matches(question)
        if _should_describe_from_local_catalog(question, local_matches):
            yield _json_event(
                {"type": "sources", "sources": [match["title"] for match in local_matches]}
            )
            yield _json_event(
                {"type": "token", "content": _build_book_description_answer(local_matches)}
            )
            yield _json_event({"type": "done"})
            return

        if _should_answer_from_local_catalog(question, local_matches):
            yield _json_event(
                {"type": "sources", "sources": [match["title"] for match in local_matches]}
            )
            yield _json_event(
                {"type": "token", "content": _build_local_catalog_answer(local_matches)}
            )
            yield _json_event({"type": "done"})
            return

        document_context, sources, has_relevant_context, local_matches = _get_book_context(
            question,
            payload.top_k,
        )
        if not has_relevant_context:
            fallback_sources = [match["title"] for match in local_matches]
            yield _json_event({"type": "sources", "sources": fallback_sources})
            yield _json_event(
                {"type": "token", "content": _build_no_context_answer(local_matches)}
            )
            yield _json_event({"type": "done"})
            return

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
