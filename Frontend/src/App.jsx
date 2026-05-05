import { useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const DEFAULT_TOP_K = 3;

const initialMessages = [
  {
    id: crypto.randomUUID(),
    role: "assistant",
    content:
      "Pergunta por titulo, autor, editora, idioma, paginas ou avaliacao.",
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
  const [messages, setMessages] = useState(initialMessages);
  const [question, setQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

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
      } catch {
        if (!cancelled) {
          setBackendStatus("offline");
        }
      }
    }

    pingBackend();
    return () => {
      cancelled = true;
    };
  }, []);

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
          top_k: DEFAULT_TOP_K,
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
            <p className="eyebrow">Pesquisa sem ruido</p>
            <h1>Chatbot de livros</h1>
            <p className="hero-text">
              Faz perguntas diretas sobre o catalogo e recebe respostas com
              contexto indexado em tempo real.
            </p>
          </div>
        </section>

        <section className="workspace-grid">
          <article className="panel panel-chat">
            <div className="panel-header">
              <p className="panel-kicker">Chat</p>
              <h2>Pergunta sobre livros</h2>
              <p className="panel-copy">
                Interface simplificada para te focares apenas na conversa.
              </p>
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
                placeholder="Pergunta por titulo, autor, avaliacao, idioma, paginas ou editora..."
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
