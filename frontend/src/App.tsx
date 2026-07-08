import {
  AlertCircle,
  Bot,
  Check,
  ChevronDown,
  Circle,
  FileText,
  Loader2,
  MessageSquare,
  PanelRightOpen,
  Plus,
  RefreshCw,
  Search,
  Send,
  Settings,
  ShieldCheck,
  UploadCloud,
  User,
  X,
} from "lucide-react";
import { ChangeEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiClientSettings,
  ApiError,
  Citation,
  ContextChunk,
  QueryHistoryMessage,
  QueryResponse,
  checkReadiness,
  getIngestionJob,
  runQuery,
  uploadDocument,
} from "./api";

type IndexedDocument = {
  id: string;
  title: string;
  fileName: string;
  byteSize: number | null;
  jobId: string;
  status: "queued" | "running" | "ready" | "failed";
  stage: string;
  selected: boolean;
  errorMessage: string | null;
  createdAt: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: string;
  response?: QueryResponse;
};

type RuntimeSettings = {
  apiBaseUrl: string;
  authToken: string;
  workspaceId: string;
  retrievalLimit: number;
  maxContextChunks: number;
  maxContextTokens: number;
};

const DEFAULT_SETTINGS: RuntimeSettings = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || "/api",
  authToken: import.meta.env.VITE_AUTH_TOKEN || "local-dev-token",
  workspaceId: import.meta.env.VITE_WORKSPACE_ID || "local-workspace",
  retrievalLimit: 8,
  maxContextChunks: 5,
  maxContextTokens: 2500,
};

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

function formatBytes(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return "Unknown size";
  }

  const units = ["B", "KB", "MB", "GB"];
  let unitIndex = 0;
  let size = value;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size = size / 1024;
    unitIndex += 1;
  }

  return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function buildAssistantContent(response: QueryResponse): string {
  const defaultAnswer = response.answer.trim();
  const synthesisDisabled = defaultAnswer.toLowerCase().startsWith("llm synthesis is not enabled");
  if (!synthesisDisabled || response.context.length === 0) {
    return defaultAnswer || "I could not find relevant context in the selected documents.";
  }

  const sections: string[] = ["I found relevant context in the selected document."];
  for (const contextChunk of response.context.slice(0, 3)) {
    const title = contextChunk.citation.title || "Selected document";
    const text = contextChunk.content.replace(/\s+/g, " ").trim();
    const preview = text.length > 900 ? `${text.slice(0, 900).trim()}...` : text;
    sections.push(`Source: ${title}\n${preview}`);
  }

  return sections.join("\n\n");
}

function buildApiSettings(settings: RuntimeSettings): ApiClientSettings {
  return {
    apiBaseUrl: settings.apiBaseUrl,
    authToken: settings.authToken,
  };
}

function buildRetrievalQuestion(question: string, messages: ChatMessage[]): string {
  const recentLines: string[] = [];
  for (const message of messages.slice(-6)) {
    if (message.role !== "user" && message.role !== "assistant") {
      continue;
    }

    const content = message.content.replace(/\s+/g, " ").trim();
    if (!content) {
      continue;
    }

    const clippedContent = content.length > 700 ? `${content.slice(0, 700).trim()}...` : content;
    recentLines.push(`${message.role}: ${clippedContent}`);
  }

  if (recentLines.length === 0) {
    return question;
  }

  return [
    `User question: ${question}`,
    "Recent conversation context:",
    recentLines.join("\n"),
  ].join("\n\n");
}

function App() {
  const [settings, setSettings] = useState<RuntimeSettings>(DEFAULT_SETTINGS);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentTitle, setDocumentTitle] = useState("");
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [statusText, setStatusText] = useState("Checking API");
  const [healthState, setHealthState] = useState<"unknown" | "healthy" | "degraded" | "down">(
    "unknown",
  );
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showEvidence, setShowEvidence] = useState(true);
  const [errorText, setErrorText] = useState("");
  const [latestResponse, setLatestResponse] = useState<QueryResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const readyDocuments = useMemo(() => {
    return documents.filter((document) => document.status === "ready");
  }, [documents]);

  const selectedDocumentIds = useMemo(() => {
    const explicitlySelected = readyDocuments
      .filter((document) => document.selected)
      .map((document) => document.id);
    if (explicitlySelected.length > 0) {
      return explicitlySelected;
    }
    return readyDocuments.map((document) => document.id);
  }, [readyDocuments]);

  const latestCitations = latestResponse?.citations ?? [];
  const latestContext = latestResponse?.context ?? [];

  useEffect(() => {
    async function loadHealth() {
      try {
        const payload = await checkReadiness(buildApiSettings(settings));
        setHealthState(payload.status === "healthy" ? "healthy" : "degraded");
        setStatusText(`${payload.service} ${payload.status} on ${payload.version}`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setHealthState("down");
        setStatusText(message);
      }
    }

    void loadHealth();
  }, [settings.apiBaseUrl, settings.authToken]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isAsking]);

  function updateSetting<K extends keyof RuntimeSettings>(key: K, value: RuntimeSettings[K]) {
    setSettings((current) => {
      return {
        ...current,
        [key]: value,
      };
    });
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setDocumentTitle(file?.name ?? "");
    setErrorText("");
  }

  async function handleHealthCheck() {
    setStatusText("Checking API");
    setHealthState("unknown");
    try {
      const payload = await checkReadiness(buildApiSettings(settings));
      setHealthState(payload.status === "healthy" ? "healthy" : "degraded");
      setStatusText(`${payload.service} ${payload.status} on ${payload.version}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setHealthState("down");
      setStatusText(message);
    }
  }

  async function handleUpload() {
    if (!selectedFile) {
      setErrorText("Choose a PDF or text document first.");
      return;
    }

    setErrorText("");
    setIsUploading(true);
    const apiSettings = buildApiSettings(settings);
    const startedAt = new Date().toISOString();

    try {
      const upload = await uploadDocument(
        apiSettings,
        selectedFile,
        settings.workspaceId,
        documentTitle,
      );
      const uploadedDocument = upload.document;
      const documentId = uploadedDocument.id;
      const jobId = upload.ingestion_job_id;
      const title = uploadedDocument.title || uploadedDocument.file_name || selectedFile.name;
      const initialDocument: IndexedDocument = {
        id: documentId,
        title,
        fileName: uploadedDocument.file_name || selectedFile.name,
        byteSize: uploadedDocument.byte_size,
        jobId,
        status: "running",
        stage: upload.ingestion_stage,
        selected: true,
        errorMessage: null,
        createdAt: uploadedDocument.created_at || startedAt,
      };

      setDocuments((current) => {
        const nextDocuments = current.filter((document) => document.id !== documentId);
        return [initialDocument, ...nextDocuments];
      });

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "system",
          content: `Indexing ${title}.`,
          createdAt: new Date().toISOString(),
        },
      ]);

      let finalStatus: IndexedDocument["status"] = "running";
      let finalStage = upload.ingestion_stage;
      let finalError: string | null = null;
      const deadline = Date.now() + 180000;

      while (Date.now() < deadline) {
        const job = await getIngestionJob(apiSettings, documentId, jobId);
        finalStage = job.current_stage;
        if (job.status === "completed") {
          finalStatus = "ready";
          break;
        }
        if (job.status === "failed" || job.status === "cancelled") {
          finalStatus = "failed";
          finalError = job.error_message || `Ingestion ${job.status}.`;
          break;
        }
        await wait(2000);
      }

      if (finalStatus === "running") {
        finalStatus = "failed";
        finalError = "Ingestion did not finish before timeout.";
      }

      setDocuments((current) => {
        return current.map((document) => {
          if (document.id !== documentId) {
            return document;
          }
          return {
            ...document,
            status: finalStatus,
            stage: finalStage,
            errorMessage: finalError,
          };
        });
      });

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            finalStatus === "ready"
              ? `${title} is indexed and ready.`
              : `${title} could not be indexed. ${finalError || ""}`.trim(),
          createdAt: new Date().toISOString(),
        },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setErrorText(message);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Upload failed. ${message}`,
          createdAt: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsUploading(false);
    }
  }

  async function handleSend() {
    const question = input.trim();
    if (!question || isAsking) {
      return;
    }
    if (selectedDocumentIds.length === 0) {
      setErrorText("Index a document before asking a question.");
      return;
    }

    setInput("");
    setErrorText("");
    setIsAsking(true);

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
      createdAt: new Date().toISOString(),
    };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);

    const history: QueryHistoryMessage[] = [];
    for (const message of nextMessages.slice(-10, -1)) {
      if (message.role !== "user" && message.role !== "assistant") {
        continue;
      }
      const content = message.content.trim();
      if (!content) {
        continue;
      }
      history.push({
        role: message.role,
        content,
      });
    }

    try {
      const retrievalQuestion = buildRetrievalQuestion(question, messages);
      const response = await runQuery(buildApiSettings(settings), {
        query: retrievalQuestion,
        workspace_id: settings.workspaceId,
        filters: {
          workspace_id: settings.workspaceId,
          document_ids: selectedDocumentIds,
        },
        history,
        retrieval_limit: settings.retrievalLimit,
        max_context_chunks: settings.maxContextChunks,
        max_context_tokens: settings.maxContextTokens,
      });
      setLatestResponse(response);

      const assistantContent = buildAssistantContent(response);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: assistantContent,
          createdAt: new Date().toISOString(),
          response,
        },
      ]);
    } catch (error) {
      let message = error instanceof Error ? error.message : String(error);
      if (error instanceof ApiError && error.status === 401) {
        message = "Authorization failed. Check the bearer token.";
      }
      setErrorText(message);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Query failed. ${message}`,
          createdAt: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsAsking(false);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }
    event.preventDefault();
    void handleSend();
  }

  function toggleDocumentSelection(documentId: string) {
    setDocuments((current) => {
      return current.map((document) => {
        if (document.id !== documentId) {
          return document;
        }
        return {
          ...document,
          selected: !document.selected,
        };
      });
    });
  }

  function resetChat() {
    setMessages([]);
    setLatestResponse(null);
    setErrorText("");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">
            <MessageSquare size={20} />
          </div>
          <div>
            <h1>Agentic RAG</h1>
            <p>Document chat</p>
          </div>
        </div>

        <div className={`health-pill health-${healthState}`}>
          <Circle size={10} fill="currentColor" />
          <span>{statusText}</span>
          <button type="button" aria-label="Check API" onClick={() => void handleHealthCheck()}>
            <RefreshCw size={15} />
          </button>
        </div>

        <section className="upload-panel">
          <button
            type="button"
            className="upload-target"
            onClick={() => fileInputRef.current?.click()}
          >
            <UploadCloud size={22} />
            <span>{selectedFile ? selectedFile.name : "Choose document"}</span>
            <small>{selectedFile ? formatBytes(selectedFile.size) : "PDF, text, markdown, CSV, JSON"}</small>
          </button>
          <input
            ref={fileInputRef}
            className="hidden-input"
            type="file"
            accept=".pdf,.txt,.md,.markdown,.json,.jsonl,.csv,.tsv,text/*,application/pdf,application/json"
            onChange={handleFileChange}
          />
          <label className="field-label" htmlFor="document-title">
            Title
          </label>
          <input
            id="document-title"
            className="text-field"
            value={documentTitle}
            onChange={(event) => setDocumentTitle(event.target.value)}
            placeholder="Document title"
          />
          <button
            type="button"
            className="primary-button"
            disabled={isUploading || !selectedFile}
            onClick={() => void handleUpload()}
          >
            {isUploading ? <Loader2 className="spin" size={18} /> : <Plus size={18} />}
            <span>{isUploading ? "Indexing" : "Upload & Index"}</span>
          </button>
        </section>

        <section className="document-panel">
          <div className="panel-heading">
            <span>Documents</span>
            <small>{readyDocuments.length} ready</small>
          </div>
          <div className="document-list">
            {documents.length === 0 ? (
              <div className="empty-state">No indexed documents.</div>
            ) : (
              documents.map((document) => (
                <button
                  type="button"
                  className={`document-row document-${document.status}`}
                  key={document.id}
                  onClick={() => toggleDocumentSelection(document.id)}
                >
                  <span className="document-icon">
                    {document.status === "ready" ? <Check size={15} /> : <FileText size={15} />}
                  </span>
                  <span className="document-copy">
                    <strong>{document.title}</strong>
                    <small>
                      {document.status} · {document.stage} · {formatBytes(document.byteSize)}
                    </small>
                  </span>
                  <span className={`select-box ${document.selected ? "selected" : ""}`} />
                </button>
              ))
            )}
          </div>
        </section>

        <section className="settings-panel">
          <button
            type="button"
            className="section-toggle"
            onClick={() => setShowSettings((current) => !current)}
          >
            <Settings size={16} />
            <span>Settings</span>
            <ChevronDown className={showSettings ? "open" : ""} size={16} />
          </button>
          {showSettings ? (
            <div className="settings-fields">
              <label className="field-label" htmlFor="api-base-url">
                API URL
              </label>
              <input
                id="api-base-url"
                className="text-field"
                value={settings.apiBaseUrl}
                onChange={(event) => updateSetting("apiBaseUrl", event.target.value)}
              />
              <label className="field-label" htmlFor="auth-token">
                Bearer token
              </label>
              <input
                id="auth-token"
                className="text-field"
                type="password"
                value={settings.authToken}
                onChange={(event) => updateSetting("authToken", event.target.value)}
              />
              <label className="field-label" htmlFor="workspace-id">
                Workspace
              </label>
              <input
                id="workspace-id"
                className="text-field"
                value={settings.workspaceId}
                onChange={(event) => updateSetting("workspaceId", event.target.value)}
              />
              <div className="range-grid">
                <label>
                  Retrieval
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={settings.retrievalLimit}
                    onChange={(event) => updateSetting("retrievalLimit", Number(event.target.value))}
                  />
                </label>
                <label>
                  Chunks
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={settings.maxContextChunks}
                    onChange={(event) =>
                      updateSetting("maxContextChunks", Number(event.target.value))
                    }
                  />
                </label>
              </div>
            </div>
          ) : null}
        </section>
      </aside>

      <main className="chat-layout">
        <header className="chat-header">
          <div>
            <h2>Chat</h2>
            <p>{selectedDocumentIds.length} selected document scope</p>
          </div>
          <div className="header-actions">
            <button type="button" className="ghost-button" onClick={resetChat}>
              <X size={17} />
              <span>Clear</span>
            </button>
            <button
              type="button"
              className="ghost-button"
              onClick={() => setShowEvidence((current) => !current)}
            >
              <PanelRightOpen size={17} />
              <span>Evidence</span>
            </button>
          </div>
        </header>

        {errorText ? (
          <div className="error-banner">
            <AlertCircle size={18} />
            <span>{errorText}</span>
          </div>
        ) : null}

        <div className={`workspace ${showEvidence ? "with-evidence" : ""}`}>
          <section className="conversation">
            <div className="message-stream" ref={scrollRef}>
              {messages.length === 0 ? (
                <div className="welcome-panel">
                  <Bot size={30} />
                  <h3>Ask against indexed documents.</h3>
                  <p>Answers are grounded in authorized retrieved context.</p>
                </div>
              ) : (
                messages.map((message) => (
                  <article className={`message message-${message.role}`} key={message.id}>
                    <div className="avatar">
                      {message.role === "user" ? <User size={17} /> : <Bot size={17} />}
                    </div>
                    <div className="message-body">
                      <p>{message.content}</p>
                      {message.response ? (
                        <div className="message-meta">
                          <span>Confidence {message.response.confidence_score.toFixed(2)}</span>
                          <span>{message.response.latency_ms} ms</span>
                          <span>{message.response.retrieval_strategy}</span>
                        </div>
                      ) : null}
                    </div>
                  </article>
                ))
              )}
              {isAsking ? (
                <article className="message message-assistant">
                  <div className="avatar">
                    <Bot size={17} />
                  </div>
                  <div className="message-body thinking">
                    <Loader2 className="spin" size={18} />
                    <span>Searching authorized context</span>
                  </div>
                </article>
              ) : null}
            </div>

            <div className="composer">
              <div className="composer-input">
                <Search size={18} />
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={handleComposerKeyDown}
                  placeholder="Ask about the selected documents"
                  rows={1}
                />
              </div>
              <button
                type="button"
                className="send-button"
                disabled={!input.trim() || isAsking}
                onClick={() => void handleSend()}
              >
                {isAsking ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              </button>
            </div>
          </section>

          {showEvidence ? (
            <aside className="evidence-panel">
              <div className="evidence-heading">
                <div>
                  <h3>Evidence</h3>
                  <p>{latestContext.length} context chunks</p>
                </div>
                <ShieldCheck size={20} />
              </div>

              <EvidenceList citations={latestCitations} context={latestContext} />
            </aside>
          ) : null}
        </div>
      </main>
    </div>
  );
}

function EvidenceList({
  citations,
  context,
}: {
  citations: Citation[];
  context: ContextChunk[];
}) {
  if (context.length === 0 && citations.length === 0) {
    return <div className="empty-state">No evidence returned yet.</div>;
  }

  return (
    <div className="evidence-list">
      {context.map((contextChunk, index) => {
        const citation = contextChunk.citation || {};
        const title = citation.title || "Document";
        const content = contextChunk.content.replace(/\s+/g, " ").trim();
        const preview = content.length > 760 ? `${content.slice(0, 760).trim()}...` : content;
        return (
          <article className="evidence-item" key={`${contextChunk.document_id}-${contextChunk.chunk_id}`}>
            <div className="evidence-title">
              <span>{index + 1}</span>
              <strong>{title}</strong>
            </div>
            <p>{preview}</p>
            <small>
              Chunk {contextChunk.chunk_id.slice(0, 8)} · {contextChunk.token_count} tokens
            </small>
          </article>
        );
      })}

      {citations.length > 0 ? (
        <div className="citation-strip">
          {citations.slice(0, 6).map((citation, index) => (
            <span key={`${citation.document_id}-${citation.chunk_id}-${index}`}>
              {citation.title || "Source"} {citation.score ? `· ${citation.score.toFixed(2)}` : ""}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export default App;
