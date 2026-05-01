# Plataforma de Chatbot de Livros com RAG

Projeto simples e funcional para um trabalho academico. O sistema combina:

- catalogo de livros de `https://books.toscrape.com`
- ficheiros locais em `backend/books_data`
- ChromaDB para pesquisa semantica
- modelo local no Ollama
- API em FastAPI

## Tecnologias usadas

- FastAPI
- LangChain
- ChromaDB
- Ollama
- React

## Estrutura do projeto

```text
backend/
  books_data/    # CSVs locais com livros, ratings e utilizadores
  config.py      # Variaveis de ambiente e configuracao
  main.py        # Endpoints FastAPI
  rag.py         # Scraping, importacao local, embeddings, pesquisa e geracao
  schemas.py     # Modelos de pedidos e respostas
docker-compose.yml
requirements.txt
```

## Como correr

### 1. Correr tudo com Docker

```bash
docker compose up --build -d
```

Isto arranca:

- Backend FastAPI em `localhost:8000`
- Frontend React em `localhost:3000`
- Ollama em `localhost:11434`

### 2. Descarregar os modelos no Ollama dentro do container

```bash
docker exec -it ollama ollama pull llama3.2:1b
docker exec -it ollama ollama pull nomic-embed-text
```

Depois de descarregar os modelos, reinicia o backend:

```bash
docker compose restart backend
```

### 3. Enderecos da aplicacao

- Swagger UI: `http://127.0.0.1:8000/docs`
- Frontend React: `http://127.0.0.1:3000`

## Como funciona

### Endpoint `/books/scrape`

Pesquisa paginas do catalogo `books.toscrape.com`, extrai detalhes dos livros e guarda chunks no ChromaDB.

```bash
curl -X POST "http://127.0.0.1:8000/books/scrape" \
  -H "Content-Type: application/json" \
  -d "{\"max_pages\":5}"
```

### Endpoint `/books/import-local`

Le `backend/books_data/books.csv`, junta medias de `ratings.csv`, usa localizacoes de `users.csv` quando existem e guarda esses livros no ChromaDB.

```bash
curl -X POST "http://127.0.0.1:8000/books/import-local" \
  -H "Content-Type: application/json" \
  -d "{\"limit\":1000}"
```

### Endpoint `/chat/stream`

Recebe uma pergunta sobre livros e devolve a resposta em streaming.

```bash
curl -N -X POST "http://127.0.0.1:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Que livros de John Grisham existem?\",\"top_k\":3}"
```

## Observacoes academicas

- O chatbot responde apenas com base nos livros indexados no ChromaDB.
- O frontend permite indexar dados do site, importar CSV local e fazer perguntas sobre livros.
- O limite de importacao local evita tentar criar embeddings para centenas de milhares de livros de uma vez.
