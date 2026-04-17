# Plataforma de Chatbot com RAG

Projeto simples e funcional para um trabalho academico. O sistema combina:

- documentos nao estruturados indexados no ChromaDB
- modelo local no Ollama
- API em FastAPI

## Tecnologias usadas

- FastAPI
- LangChain
- ChromaDB
- Ollama

## Estrutura do projeto

```text
backend/
  config.py      # Variaveis de ambiente e configuracao
  main.py        # Endpoints FastAPI
  rag.py         # Logica de upload, embeddings, pesquisa e geracao com ChromaDB
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
docker exec -it ollama ollama pull llama3
docker exec -it ollama ollama pull nomic-embed-text
```

Depois de descarregar os modelos, reinicia o backend para garantir que arranca com tudo pronto:

```bash
docker compose restart backend
```

### 3. Enderecos da aplicacao

- Swagger UI: `http://127.0.0.1:8000/docs`
- Frontend React: `http://127.0.0.1:3000`

## Execucao local sem Docker

### 1. Subir apenas o Ollama

```bash
docker compose up -d ollama
```

### 2. Instalar dependencias Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar variaveis de ambiente do backend

```bash
copy .env.example .env
```

### 4. Correr a API

```bash
uvicorn backend.main:app --reload
```

### 5. Correr o frontend React

```bash
cd Frontend
npm install
copy .env.example .env
npm run dev
```

## Como funciona

### Endpoint `/upload`

Recebe um ficheiro PDF ou TXT e faz:

1. extracao do texto
2. divisao em chunks
3. geracao de embeddings
4. armazenamento no ChromaDB

Exemplo com `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/upload" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@regras.pdf"
```

### Endpoint `/chat`

Recebe uma pergunta e:

1. pesquisa chunks relevantes no ChromaDB
2. junta o contexto encontrado
3. envia tudo para o modelo no Ollama
4. devolve a resposta final

Exemplo:

```bash
curl -X POST "http://127.0.0.1:8000/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Quais sao as regras?\"}"
```

Exemplo com resumo:

```bash
curl -X POST "http://127.0.0.1:8000/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Resume os pontos principais do regulamento.\"}"
```

## Como adicionar novos documentos

1. Envia um novo ficheiro para o endpoint `/upload`
2. O texto sera convertido em chunks
3. Os embeddings serao guardados no ChromaDB
4. O chatbot passa a poder usar esse conteudo nas respostas

Nao e necessario reiniciar a API para adicionar novos documentos.

## Observacoes academicas

- O projeto foi mantido simples de proposito.
- O frontend nao e necessario; os testes podem ser feitos por Swagger, Postman ou `curl`.
- O sistema responde apenas com base nos documentos enviados para o endpoint `/upload`.
- O frontend React permite demonstrar o fluxo completo de upload e perguntas sobre documentos.
