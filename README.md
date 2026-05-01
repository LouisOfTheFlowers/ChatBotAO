# Plataforma de Chatbot de Livros com RAG

Projeto simples e funcional para um trabalho academico. O sistema combina:

- ficheiro local `backend/books_data/books.csv`
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
  rag.py         # Importacao local, embeddings, pesquisa e geracao
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

### Importacao do CSV local

Le `backend/books_data/books.csv`, junta medias de `ratings.csv`, usa localizacoes de `users.csv` quando existem e guarda esses livros no ChromaDB.

O backend faz esta importacao automaticamente no arranque quando o ChromaDB esta vazio. O Docker Compose vem com:

- `AUTO_IMPORT_LOCAL_BOOKS=true`
- `LOCAL_BOOKS_IMPORT_LIMIT=12000`

Para este CSV, `12000` carrega o ficheiro todo.
Para um arranque mais rapido e com menos uso de CPU, podes baixar o limite para `5000`.
Valores como `50000` ou `100000` fazem o Ollama gerar muitos embeddings no primeiro arranque e podem demorar bastante.
Se quiseres carregar o CSV inteiro sem picos grandes de CPU, aumenta `LOCAL_BOOKS_IMPORT_LIMIT`
e mantem os lotes/pausas ativos:

- `LOCAL_BOOKS_INGEST_BATCH_SIZE=500`
- `LOCAL_BOOKS_INGEST_BATCH_DELAY=0.25`
- `CHROMA_ADD_BATCH_SIZE=50`
- `CHROMA_ADD_BATCH_DELAY=0.1`

O `docker-compose.yml` tambem limita CPU no backend e no Ollama para evitar que
a criacao de embeddings ocupe a maquina toda durante a importacao inicial.

### Endpoint `/books/status`

Mostra quantos chunks ja foram indexados no ChromaDB.

```bash
curl "http://127.0.0.1:8000/books/status"
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
- O frontend apenas mostra o estado do indice e permite fazer perguntas sobre livros.
- O limite de importacao local evita tentar criar embeddings para centenas de milhares de livros de uma vez. O valor por defeito e 12000 para cobrir o CSV local atual.
