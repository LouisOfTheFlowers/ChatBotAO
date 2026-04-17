import { startTransition, useEffect, useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

const initialMessages = [
  {
    id: crypto.randomUUID(),
    role: "assistant",
    content:
      "Upload a PDF or TXT document and ask a question. The answer will be based only on the uploaded content.",
    sources: [],
  },
];

async function readApiError(response) {
  try {
    const payload = await response.json();
    return payload.detail || payload.message || "Request failed.";
  } catch {
    return "Request failed.";
  }
}

export default function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [documentsUploaded, setDocumentsUploaded] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadState, setUploadState] = useState({
    loading: false,
    message: "",
    error: "",
  });
  const [messages, setMessages] = useState(initialMessages);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(4);
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function pingBackend() {
      try {
        const response = await fetch(`${API_URL}/`);
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

  const statusLabel = useMemo(() => {
    if (backendStatus === "online") return "Backend online";
    if (backendStatus === "offline") return "Backend offline";
    return "Checking backend";
  }, [backendStatus]);

  async function handleUpload(event) {
    event.preventDefault();

    if (!selectedFile) {
      setUploadState({
        loading: false,
        message: "",
        error: "Choose a PDF or TXT file first.",
      });
      return;
    }

    setUploadState({ loading: true, message: "", error: "" });

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await readApiError(response));
      }

      const payload = await response.json();

      startTransition(() => {
        setDocumentsUploaded((current) => [
          {
            filename: payload.filename,
            chunksAdded: payload.chunks_added,
          },
          ...current,
        ]);
      });

      setUploadState({
        loading: false,
        message: `${payload.filename} indexed successfully with ${payload.chunks_added} chunks.`,
        error: "",
      });
      setSelectedFile(null);
    } catch (error) {
      setUploadState({
        loading: false,
        message: "",
        error: error.message || "Upload failed.",
      });
    }
  }

  async function handleAsk(event) {
    event.preventDefault();

    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      return;
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmedQuestion,
      sources: [],
    };

    startTransition(() => {
      setMessages((current) => [...current, userMessage]);
    });

    setQuestion("");
    setChatLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
          top_k: topK,
        }),
      });

      if (!response.ok) {
        throw new Error(await readApiError(response));
      }

      const payload = await response.json();

      startTransition(() => {
        setMessages((current) => [
          ...current,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: payload.answer,
            sources: payload.document_sources,
          },
        ]);
      });
    } catch (error) {
      startTransition(() => {
        setMessages((current) => [
          ...current,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: error.message || "The request failed.",
            sources: [],
            isError: true,
          },
        ]);
      });
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <div className="page-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <main className="app-frame">
        <section className="hero-card">
          <div className="hero-copy">
            <p className="eyebrow">React + FastAPI + ChromaDB</p>
            <h1>RAG chatbot platform</h1>
            <p className="hero-text">
              Upload course documents, index them in the backend, and ask
              questions against the material you provided.
            </p>
          </div>

          <div className="status-box">
            <span className={`status-dot status-${backendStatus}`} />
            <div>
              <p className="status-label">{statusLabel}</p>
              <p className="status-subtitle">{API_URL}</p>
            </div>
          </div>
        </section>

        <section className="workspace-grid">
          <article className="panel panel-upload">
            <div className="panel-header">
              <p className="panel-kicker">Step 1</p>
              <h2>Upload documents</h2>
              <p className="panel-copy">
                Send PDF or TXT files to the backend so they can be chunked and
                embedded for retrieval.
              </p>
            </div>

            <form className="upload-form" onSubmit={handleUpload}>
              <label className="file-dropzone" htmlFor="document-upload">
                <span className="file-dropzone-title">
                  {selectedFile ? selectedFile.name : "Choose a file"}
                </span>
                <span className="file-dropzone-subtitle">
                  Supported formats: PDF, TXT
                </span>
              </label>

              <input
                id="document-upload"
                className="hidden-input"
                type="file"
                accept=".pdf,.txt"
                onChange={(event) =>
                  setSelectedFile(event.target.files?.[0] ?? null)
                }
              />

              <button
                className="primary-button"
                type="submit"
                disabled={uploadState.loading}
              >
                {uploadState.loading ? "Uploading..." : "Upload document"}
              </button>
            </form>

            {uploadState.message ? (
              <p className="feedback success-feedback">{uploadState.message}</p>
            ) : null}

            {uploadState.error ? (
              <p className="feedback error-feedback">{uploadState.error}</p>
            ) : null}

            <div className="document-list">
              <div className="document-list-header">
                <h3>Indexed files</h3>
                <span>{documentsUploaded.length}</span>
              </div>

              {documentsUploaded.length === 0 ? (
                <p className="empty-state">
                  No documents uploaded in this session yet.
                </p>
              ) : (
                <ul className="document-items">
                  {documentsUploaded.map((document) => (
                    <li key={`${document.filename}-${document.chunksAdded}`}>
                      <span>{document.filename}</span>
                      <strong>{document.chunksAdded} chunks</strong>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </article>

          <article className="panel panel-chat">
            <div className="panel-header">
              <p className="panel-kicker">Step 2</p>
              <h2>Ask the chatbot</h2>
              <p className="panel-copy">
                The answer uses only the information found in the uploaded
                documents.
              </p>
            </div>

            <div className="chat-toolbar">
              <label className="topk-control" htmlFor="top-k">
                <span>Retrieved chunks</span>
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
                    {message.role === "user" ? "You" : "Assistant"}
                  </p>
                  <p className="message-content">{message.content}</p>

                  {message.sources?.length ? (
                    <div className="source-strip">
                      {message.sources.map((source) => (
                        <span key={`${message.id}-${source}`}>{source}</span>
                      ))}
                    </div>
                  ) : null}
                </article>
              ))}

              {chatLoading ? (
                <article className="message-bubble message-assistant typing-bubble">
                  <p className="message-role">Assistant</p>
                  <p className="message-content">Generating answer...</p>
                </article>
              ) : null}
            </div>

            <form className="chat-form" onSubmit={handleAsk}>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask something about the uploaded documents..."
                rows={4}
              />

              <button
                className="primary-button"
                type="submit"
                disabled={chatLoading}
              >
                {chatLoading ? "Asking..." : "Send question"}
              </button>
            </form>
          </article>
        </section>
      </main>
    </div>
  );
}
