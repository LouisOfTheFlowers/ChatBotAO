import { useEffect, useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const initialMessages = [
  {
    id: crypto.randomUUID(),
    role: "assistant",
    content:
      "Faz uma pergunta sobre livros. Primeiro importa o CSV local ou indexa paginas do Books to Scrape.",
    sources: [],
  },
];

async function readApiError(response) {
  try {
    const payload = await response.json();
    return payload.detail || payload.message || "Pedido falhou.";
  } catch {
    return "Pedido falhou.";
  }
}

function updateMessage(messages, messageId, updater) {
  return messages.map((message) =>
    message.id === messageId ? updater(message) : message,
  );
}

export default function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [indexStatus, setIndexStatus] = useState("checking");
  const [indexError, setIndexError] = useState("");
  const [booksIndexed, setBooksIndexed] = useState(0);
  const [scrapePages, setScrapePages] = useState(5);
  const [localLimit, setLocalLimit] = useState(1000);
  const [scrapeState, setScrapeState] = useState({
    loading: false,
    message: "",
    error: "",
  });
  const [localImportState, setLocalImportState] = useState({
    loading: false,
    message: "",
    error: "",
  });
  const [messages, setMessages] = useState(initialMessages);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(3);
  const [chatLoading, setChatLoading] = useState(false);

  async function refreshBooksStatus() {
    const response = await fetch(`${API_URL}/books/status`);
    if (!response.ok) {
      throw new Error(await readApiError(response));
    }

    const payload = await response.json();
    setBooksIndexed(payload.indexed_chunks ?? 0);
    setIndexStatus("ready");
    setIndexError("");
  }

  useEffect(() => {
    let cancelled = false;

    async function pingBackend() {
      try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) {
          throw new Error("Backend unavailable");
        }

        if (!cancelled) {
          setBackendStatus("online");
        }

        try {
          await refreshBooksStatus();
        } catch (error) {
          if (!cancelled) {
            setIndexStatus("unavailable");
            setIndexError(error.message || "Indice indisponivel.");
            setBooksIndexed(0);
          }
        }
      } catch {
        if (!cancelled) {
          setBackendStatus("offline");
          setIndexStatus("unavailable");
          setIndexError("Nao foi possivel contactar a API.");
        }
      }
    }

    pingBackend();
    return () => {
      cancelled = true;
    };
  }, []);

  const statusLabel = useMemo(() => {
    if (backendStatus === "online") return "API online";
    if (backendStatus === "offline") return "API offline";
    return "A verificar API";
  }, [backendStatus]);

  const indexLabel = useMemo(() => {
    if (indexStatus === "ready") return "Indice pronto";
    if (indexStatus === "unavailable") return "Indice/Ollama indisponivel";
    return "A verificar indice";
  }, [indexStatus]);

  async function handleScrape(event) {
    event.preventDefault();
    setScrapeState({ loading: true, message: "", error: "" });

    try {
      const response = await fetch(`${API_URL}/books/scrape`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ max_pages: scrapePages }),
      });

      if (!response.ok) {
        throw new Error(await readApiError(response));
      }

      const payload = await response.json();
      setBooksIndexed(payload.chunks_added ?? 0);
      setScrapeState({
        loading: false,
        message: `${payload.message} ${payload.books_indexed} livros, ${payload.chunks_added} novos chunks, ${payload.max_pages} paginas.`,
        error: "",
      });

      await refreshBooksStatus();
    } catch (error) {
      setScrapeState({
        loading: false,
        message: "",
        error: error.message || "Indexacao do site falhou.",
      });
    }
  }

  async function handleLocalImport(event) {
    event.preventDefault();
    setLocalImportState({ loading: true, message: "", error: "" });

    try {
      const response = await fetch(`${API_URL}/books/import-local`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ limit: localLimit }),
      });

      if (!response.ok) {
        throw new Error(await readApiError(response));
      }

      const payload = await response.json();
      setBooksIndexed(payload.chunks_added ?? 0);
      setLocalImportState({
        loading: false,
        message: `${payload.message} ${payload.books_indexed} livros, ${payload.chunks_added} novos chunks, ${payload.ratings_loaded} avaliacoes.`,
        error: "",
      });

      await refreshBooksStatus();
    } catch (error) {
      setLocalImportState({
        loading: false,
        message: "",
        error: error.message || "Importacao local falhou.",
      });
    }
  }

  async function handleAsk(event) {
    event.preventDefault();

    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      return;
    }

    const assistantMessageId = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmedQuestion,
        sources: [],
      },
      {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        sources: [],
      },
    ]);

    setQuestion("");
    setChatLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
          top_k: topK,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(await readApiError(response));
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamError = "";

      function handleStreamLine(line) {
        if (!line.trim()) return;

        let eventPayload;
        try {
          eventPayload = JSON.parse(line);
        } catch {
          streamError = "A resposta em streaming veio num formato invalido.";
          return;
        }

        if (eventPayload.type === "sources") {
          setMessages((current) =>
            updateMessage(current, assistantMessageId, (message) => ({
              ...message,
              sources: eventPayload.sources ?? [],
            })),
          );
        }

        if (eventPayload.type === "token") {
          setMessages((current) =>
            updateMessage(current, assistantMessageId, (message) => ({
              ...message,
              content: `${message.content}${eventPayload.content}`,
            })),
          );
        }

        if (eventPayload.type === "error") {
          streamError = eventPayload.message || "Pedido falhou.";
        }

        if (eventPayload.type === "done") {
          streamError = "";
        }
      }

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        lines.forEach(handleStreamLine);
      }

      if (buffer) {
        handleStreamLine(buffer);
      }

      if (streamError) {
        throw new Error(streamError);
      }
    } catch (error) {
      setMessages((current) =>
        updateMessage(current, assistantMessageId, (message) => ({
          ...message,
          content: error.message || "Pedido falhou.",
          sources: [],
          isError: true,
        })),
      );
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <div className="page-shell">
      <main className="app-frame">
        <section className="hero-card">
          <div className="hero-copy">
            <p className="eyebrow">Books to Scrape + ChromaDB + Ollama</p>
            <h1>Chatbot de livros</h1>
            <p className="hero-text">
              Pesquisa o catalogo indexado por titulo, autor, categoria,
              preco, classificacao, disponibilidade e descricao.
            </p>
          </div>

          <div className="status-stack">
            <div className="status-box">
              <span className={`status-dot status-${backendStatus}`} />
              <div>
                <p className="status-label">{statusLabel}</p>
                <p className="status-subtitle">{API_URL}</p>
              </div>
            </div>

            <div className="status-box index-status-box">
              <span className={`status-dot status-${indexStatus}`} />
              <div>
                <p className="status-label">{indexLabel}</p>
                <p className="status-subtitle">
                  {indexError || `${booksIndexed} chunks indexados`}
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="workspace-grid">
          <article className="panel panel-catalog">
            <div className="panel-header">
              <p className="panel-kicker">Catalogo</p>
              <h2>Indice de livros</h2>
              <p className="panel-copy">
                Importa os CSVs locais ou recolhe paginas do Books to Scrape
                para guardar chunks no ChromaDB.
              </p>
            </div>

            <div className="metric-strip">
              <span>Chunks indexados</span>
              <strong>{booksIndexed}</strong>
            </div>

            <form className="index-form" onSubmit={handleScrape}>
              <label className="field-control" htmlFor="scrape-pages">
                <span>Paginas do site</span>
                <select
                  id="scrape-pages"
                  value={scrapePages}
                  onChange={(event) => setScrapePages(Number(event.target.value))}
                >
                  {[1, 3, 5, 10, 25, 50].map((value) => (
                    <option key={value} value={value}>
                      {value} paginas
                    </option>
                  ))}
                </select>
              </label>

              <button
                className="primary-button"
                type="submit"
                disabled={scrapeState.loading || backendStatus !== "online"}
              >
                {scrapeState.loading ? "A indexar livros..." : "Indexar site"}
              </button>
            </form>

            {scrapeState.message ? (
              <p className="feedback success-feedback">{scrapeState.message}</p>
            ) : null}

            {scrapeState.error ? (
              <p className="feedback error-feedback">{scrapeState.error}</p>
            ) : null}

            <form className="index-form local-form" onSubmit={handleLocalImport}>
              <label className="field-control" htmlFor="local-limit">
                <span>Livros do CSV local</span>
                <select
                  id="local-limit"
                  value={localLimit}
                  onChange={(event) => setLocalLimit(Number(event.target.value))}
                >
                  {[250, 500, 1000, 2500, 5000, 10000].map((value) => (
                    <option key={value} value={value}>
                      {value} livros
                    </option>
                  ))}
                </select>
              </label>

              <button
                className="secondary-button"
                type="submit"
                disabled={localImportState.loading || backendStatus !== "online"}
              >
                {localImportState.loading ? "A importar CSV..." : "Importar CSV"}
              </button>
            </form>

            {localImportState.message ? (
              <p className="feedback success-feedback">
                {localImportState.message}
              </p>
            ) : null}

            {localImportState.error ? (
              <p className="feedback error-feedback">
                {localImportState.error}
              </p>
            ) : null}

            <div className="hint-list">
              <h3>Perguntas rapidas</h3>
              <button
                type="button"
                onClick={() => setQuestion("Quais livros de Poetry estao disponiveis?")}
              >
                Livros de poesia
              </button>
              <button
                type="button"
                onClick={() => setQuestion("Que livros tem classificacao Five?")}
              >
                Classificacao maxima
              </button>
              <button
                type="button"
                onClick={() => setQuestion("Qual e o preco de A Light in the Attic?")}
              >
                Preco de um livro
              </button>
              <button
                type="button"
                onClick={() => setQuestion("Que livros de John Grisham existem no catalogo?")}
              >
                Autores no CSV
              </button>
            </div>
          </article>

          <article className="panel panel-chat">
            <div className="panel-header chat-header">
              <div>
                <p className="panel-kicker">Chat</p>
                <h2>Pergunta sobre livros</h2>
                <p className="panel-copy">
                  As respostas chegam em streaming do Ollama e usam apenas o
                  contexto indexado.
                </p>
              </div>

              <label className="field-control compact-control" htmlFor="top-k">
                <span>Chunks recuperados</span>
                <select
                  id="top-k"
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                >
                  {[2, 3, 4, 5, 6].map((value) => (
                    <option key={value} value={value}>
                      Top {value}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="messages">
              {messages.map((message) => (
                <article
                  key={message.id}
                  className={`message-bubble message-${message.role} ${
                    message.isError ? "message-error" : ""
                  }`}
                >
                  <p className="message-role">
                    {message.role === "user" ? "Tu" : "Assistente"}
                  </p>
                  <p className="message-content">
                    {message.content || "A ler o catalogo..."}
                  </p>

                  {message.sources?.length ? (
                    <div className="source-strip">
                      {message.sources.map((source) => (
                        <span key={`${message.id}-${source}`}>{source}</span>
                      ))}
                    </div>
                  ) : null}
                </article>
              ))}
            </div>

            <form className="chat-form" onSubmit={handleAsk}>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Pergunta por titulo, autor, categoria, preco, avaliacao ou disponibilidade..."
                rows={4}
              />

              <button
                className="primary-button"
                type="submit"
                disabled={chatLoading || backendStatus !== "online"}
              >
                {chatLoading ? "A responder..." : "Enviar pergunta"}
              </button>
            </form>
          </article>
        </section>
      </main>
    </div>
  );
}
