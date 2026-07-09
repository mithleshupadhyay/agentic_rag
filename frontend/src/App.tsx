import {
  Activity,
  AlertCircle,
  Bot,
  Check,
  ChevronDown,
  Circle,
  ClipboardList,
  Database,
  Eye,
  FileSearch,
  FileText,
  History,
  Layers,
  Loader2,
  MessageSquare,
  PanelRightOpen,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  Send,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Trash2,
  UploadCloud,
  User,
  X,
} from "lucide-react";
import { ChangeEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiClientSettings,
  ApiError,
  CandidateChunk,
  Citation,
  ClassificationLevel,
  ContextChunk,
  DocumentListItem,
  DocumentRead,
  DocumentSourceType,
  DocumentStatus,
  HealthResponse,
  IngestionJobRead,
  IngestionJobStatus,
  JsonObject,
  QueryHistoryMessage,
  QueryResponse,
  QueryRunListItem,
  QueryRunRead,
  QueryRunStatus,
  RetrievalResponse,
  RetrievalStrategy,
  cancelQueryRun,
  checkReadiness,
  createDocument,
  deleteDocument,
  getDocument,
  getIngestionJob,
  getQueryRun,
  listIngestionJobs,
  listQueryRuns,
  rerankCandidates,
  restoreDocument,
  runQuery,
  runRetrieval,
  searchDocuments,
  updateDocument,
  uploadDocument,
} from "./api";

type AppView = "chat" | "documents" | "retrieval" | "runs" | "operations";

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
  queryStrategy: RetrievalStrategy;
  retrievalLimit: number;
  maxContextChunks: number;
  maxContextTokens: number;
};

type ParsedJsonResult = {
  ok: boolean;
  value: JsonObject;
  error: string;
};

const DEFAULT_SETTINGS: RuntimeSettings = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || "/api",
  authToken: import.meta.env.VITE_AUTH_TOKEN || "local-dev-token",
  workspaceId: import.meta.env.VITE_WORKSPACE_ID || "local-workspace",
  queryStrategy: (import.meta.env.VITE_QUERY_STRATEGY || "hybrid") as RetrievalStrategy,
  retrievalLimit: 8,
  maxContextChunks: 5,
  maxContextTokens: 2500,
};

const EMPTY_PAGE = {
  page: 1,
  size: 50,
  total: 0,
};

const DEFAULT_METADATA = {
  source: "frontend",
  uploaded_from: "agentic_rag_frontend",
};

const STATUS_OPTIONS: Array<DocumentStatus | ""> = [
  "",
  "queued",
  "parsing",
  "indexing",
  "ready",
  "failed",
  "deleted",
];

const SOURCE_OPTIONS: Array<DocumentSourceType | ""> = ["", "upload", "s3", "url", "connector"];

const QUERY_RUN_STATUS_OPTIONS: Array<QueryRunStatus | ""> = [
  "",
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
];

const CLASSIFICATION_OPTIONS: ClassificationLevel[] = [
  "public",
  "internal",
  "confidential",
  "restricted",
];

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
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

function formatDate(value: string | null | undefined): string {
  if (!value) {
    return "Not available";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function compactIdentifier(value: string | null | undefined): string {
  if (!value) {
    return "None";
  }

  if (value.length <= 14) {
    return value;
  }

  return `${value.slice(0, 8)}...${value.slice(-4)}`;
}

function parseJsonObject(value: string, fieldLabel: string): ParsedJsonResult {
  const cleanValue = value.trim();
  if (!cleanValue) {
    return {
      ok: true,
      value: {},
      error: "",
    };
  }

  try {
    const parsedValue = JSON.parse(cleanValue) as unknown;
    if (
      !parsedValue ||
      typeof parsedValue !== "object" ||
      Array.isArray(parsedValue)
    ) {
      return {
        ok: false,
        value: {},
        error: `${fieldLabel} must be a JSON object.`,
      };
    }

    return {
      ok: true,
      value: parsedValue as JsonObject,
      error: "",
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      value: {},
      error: `${fieldLabel} is invalid JSON: ${message}`,
    };
  }
}

function formatJson(value: unknown): string {
  return JSON.stringify(value ?? {}, null, 2);
}

function buildApiSettings(settings: RuntimeSettings): ApiClientSettings {
  return {
    apiBaseUrl: settings.apiBaseUrl,
    authToken: settings.authToken,
  };
}

function buildAssistantContent(response: QueryResponse): string {
  const answer = response.answer.trim();
  if (answer) {
    return answer;
  }

  return "I could not find relevant context in the selected documents.";
}

function buildRetrievalQuestion(question: string, messages: ChatMessage[]): string {
  const recentUserQuestions: string[] = [];
  for (const message of messages.slice(-8)) {
    if (message.role !== "user") {
      continue;
    }

    const content = message.content.replace(/\s+/g, " ").trim();
    if (!content) {
      continue;
    }

    const clippedContent = content.length > 180 ? content.slice(0, 180).trim() : content;
    recentUserQuestions.push(clippedContent);
  }

  const pronounFollowUpPattern = /\b(he|his|him|she|her|they|their|it|this|that)\b/i;
  if (!pronounFollowUpPattern.test(question) || recentUserQuestions.length === 0) {
    return question;
  }

  const recentContext = recentUserQuestions.slice(-2).join("\n");
  return `${question}\n${recentContext}`;
}

function statusTone(value: string | null | undefined): string {
  if (value === "ready" || value === "completed" || value === "healthy" || value === "passed") {
    return "success";
  }

  if (value === "failed" || value === "deleted" || value === "unhealthy") {
    return "danger";
  }

  if (value === "running" || value === "queued" || value === "indexing" || value === "parsing") {
    return "warning";
  }

  return "neutral";
}

function defaultAclPolicy() {
  return {
    visibility: "tenant" as const,
    allowed_user_ids: [],
    allowed_group_ids: [],
    allowed_roles: ["admin", "user"],
    denied_user_ids: [],
    denied_group_ids: [],
    acl_version: 1,
  };
}

function App() {
  const [activeView, setActiveView] = useState<AppView>("chat");
  const [settings, setSettings] = useState<RuntimeSettings>(DEFAULT_SETTINGS);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthState, setHealthState] = useState<"unknown" | "healthy" | "degraded" | "down">(
    "unknown",
  );
  const [statusText, setStatusText] = useState("Checking API");
  const [errorText, setErrorText] = useState("");

  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [documentsPage, setDocumentsPage] = useState(EMPTY_PAGE);
  const [documentPage, setDocumentPage] = useState(1);
  const [documentStatusFilter, setDocumentStatusFilter] = useState<DocumentStatus | "">("");
  const [documentSourceFilter, setDocumentSourceFilter] = useState<DocumentSourceType | "">("");
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [activeDocument, setActiveDocument] = useState<DocumentRead | null>(null);
  const [isLoadingDocumentDetail, setIsLoadingDocumentDetail] = useState(false);
  const [documentJobs, setDocumentJobs] = useState<IngestionJobRead[]>([]);
  const [isLoadingJobs, setIsLoadingJobs] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [editClassification, setEditClassification] =
    useState<ClassificationLevel>("internal");
  const [editMetadataDraft, setEditMetadataDraft] = useState("{}");
  const [isSavingDocument, setIsSavingDocument] = useState(false);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentTitle, setDocumentTitle] = useState("");
  const [uploadMetadataDraft, setUploadMetadataDraft] = useState(
    formatJson(DEFAULT_METADATA),
  );
  const [isUploading, setIsUploading] = useState(false);

  const [sourceType, setSourceType] = useState<DocumentSourceType>("url");
  const [sourceUri, setSourceUri] = useState("");
  const [sourceTitle, setSourceTitle] = useState("");
  const [sourceMetadataDraft, setSourceMetadataDraft] = useState(
    formatJson({ source: "frontend_source_registration" }),
  );
  const [isCreatingSource, setIsCreatingSource] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [showEvidence, setShowEvidence] = useState(true);
  const [latestResponse, setLatestResponse] = useState<QueryResponse | null>(null);

  const [retrievalStrategy, setRetrievalStrategy] = useState<RetrievalStrategy>("bm25");
  const [retrievalQuery, setRetrievalQuery] = useState("");
  const [retrievalLimit, setRetrievalLimit] = useState(8);
  const [retrievalMinSimilarity, setRetrievalMinSimilarity] = useState(0);
  const [retrievalTags, setRetrievalTags] = useState("");
  const [retrievalSourceTypes, setRetrievalSourceTypes] = useState("");
  const [retrievalMetadataDraft, setRetrievalMetadataDraft] = useState("{}");
  const [retrievalResponse, setRetrievalResponse] = useState<RetrievalResponse | null>(null);
  const [rerankedCandidates, setRerankedCandidates] = useState<CandidateChunk[]>([]);
  const [isRunningRetrieval, setIsRunningRetrieval] = useState(false);

  const [queryRuns, setQueryRuns] = useState<QueryRunListItem[]>([]);
  const [queryRunsPage, setQueryRunsPage] = useState(EMPTY_PAGE);
  const [queryRunPage, setQueryRunPage] = useState(1);
  const [queryRunStatusFilter, setQueryRunStatusFilter] = useState<QueryRunStatus | "">("");
  const [queryRunVerificationFilter, setQueryRunVerificationFilter] = useState("");
  const [isLoadingQueryRuns, setIsLoadingQueryRuns] = useState(false);
  const [activeQueryRun, setActiveQueryRun] = useState<QueryRunRead | null>(null);
  const [isLoadingQueryRunDetail, setIsLoadingQueryRunDetail] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const apiSettings = useMemo(() => {
    return buildApiSettings(settings);
  }, [settings.apiBaseUrl, settings.authToken]);

  const readyDocuments = useMemo(() => {
    return documents.filter((document) => document.status === "ready");
  }, [documents]);

  const selectedReadyDocuments = useMemo(() => {
    const selectedIdSet = new Set(selectedDocumentIds);
    return documents.filter((document) => {
      return selectedIdSet.has(document.id) && document.status === "ready";
    });
  }, [documents, selectedDocumentIds]);

  const selectedScopeIds = useMemo(() => {
    return selectedReadyDocuments.map((document) => document.id);
  }, [selectedReadyDocuments]);

  const latestCitations = latestResponse?.citations ?? [];
  const latestContext = latestResponse?.context ?? [];

  useEffect(() => {
    void handleHealthCheck();
  }, [settings.apiBaseUrl, settings.authToken]);

  useEffect(() => {
    void refreshDocuments();
  }, [
    settings.apiBaseUrl,
    settings.authToken,
    settings.workspaceId,
    documentPage,
    documentStatusFilter,
    documentSourceFilter,
  ]);

  useEffect(() => {
    void refreshQueryRuns();
  }, [
    settings.apiBaseUrl,
    settings.authToken,
    settings.workspaceId,
    queryRunPage,
    queryRunStatusFilter,
    queryRunVerificationFilter,
  ]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isAsking, activeView]);

  function updateSetting<K extends keyof RuntimeSettings>(key: K, value: RuntimeSettings[K]) {
    setSettings((current) => {
      return {
        ...current,
        [key]: value,
      };
    });
  }

  async function handleHealthCheck() {
    setStatusText("Checking API");
    setHealthState("unknown");
    try {
      const payload = await checkReadiness(apiSettings);
      setHealth(payload);
      setHealthState(payload.status === "healthy" ? "healthy" : "degraded");
      setStatusText(`${payload.service} ${payload.status} on ${payload.version}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setHealth(null);
      setHealthState("down");
      setStatusText(message);
    }
  }

  async function refreshDocuments() {
    setIsLoadingDocuments(true);
    try {
      const response = await searchDocuments(apiSettings, {
        page: {
          page: documentPage,
          size: 50,
        },
        workspace_id: settings.workspaceId.trim() || null,
        source_type: documentSourceFilter || null,
        status: documentStatusFilter || null,
        metadata_filters: {},
        tags: [],
      });
      setDocuments(response.items);
      setDocumentsPage(response.page);
      setSelectedDocumentIds((current) => {
        const currentSet = new Set(current);
        const visibleReadyIds = response.items
          .filter((document) => document.status === "ready")
          .map((document) => document.id);
        const retainedSelection = visibleReadyIds.filter((id) => currentSet.has(id));
        if (retainedSelection.length > 0) {
          return retainedSelection;
        }
        return visibleReadyIds.slice(0, 1);
      });
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoadingDocuments(false);
    }
  }

  async function refreshQueryRuns() {
    setIsLoadingQueryRuns(true);
    try {
      const response = await listQueryRuns(apiSettings, {
        page: queryRunPage,
        size: 50,
        workspace_id: settings.workspaceId.trim() || undefined,
        status: queryRunStatusFilter,
        verification_status: queryRunVerificationFilter.trim() || undefined,
      });
      setQueryRuns(response.items);
      setQueryRunsPage(response.page);
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoadingQueryRuns(false);
    }
  }

  async function openDocument(documentId: string) {
    setActiveView("documents");
    setIsLoadingDocumentDetail(true);
    setErrorText("");
    try {
      const document = await getDocument(apiSettings, documentId);
      setActiveDocument(document);
      setEditTitle(document.title || "");
      setEditClassification(document.classification_level);
      setEditMetadataDraft(formatJson(document.metadata));
      await refreshIngestionJobs(document.id);
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoadingDocumentDetail(false);
    }
  }

  async function refreshIngestionJobs(documentId: string) {
    setIsLoadingJobs(true);
    try {
      const response = await listIngestionJobs(apiSettings, documentId, 1, 25);
      setDocumentJobs(response.items);
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoadingJobs(false);
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setDocumentTitle(file?.name ?? "");
    setErrorText("");
  }

  async function handleUpload() {
    if (!selectedFile) {
      setErrorText("Choose a document before uploading.");
      return;
    }

    const metadataResult = parseJsonObject(uploadMetadataDraft, "Upload metadata");
    if (!metadataResult.ok) {
      setErrorText(metadataResult.error);
      return;
    }

    setErrorText("");
    setIsUploading(true);
    try {
      const upload = await uploadDocument(
        apiSettings,
        selectedFile,
        settings.workspaceId,
        documentTitle,
        {
          ...metadataResult.value,
          uploaded_at: new Date().toISOString(),
        },
      );

      const uploadedDocument = upload.document;
      setSelectedDocumentIds([uploadedDocument.id]);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "system",
          content: `Indexing ${uploadedDocument.title || uploadedDocument.file_name || selectedFile.name}.`,
          createdAt: new Date().toISOString(),
        },
      ]);

      let completedJob: IngestionJobRead | null = null;
      const deadline = Date.now() + 180000;
      while (Date.now() < deadline) {
        const job = await getIngestionJob(
          apiSettings,
          uploadedDocument.id,
          upload.ingestion_job_id,
        );
        completedJob = job;
        if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
          break;
        }
        await wait(2000);
      }

      await refreshDocuments();
      await openDocument(uploadedDocument.id);

      const finalStatus = completedJob?.status || "running";
      const finalMessage =
        finalStatus === "completed"
          ? `${uploadedDocument.title || uploadedDocument.file_name || selectedFile.name} is indexed and ready.`
          : `${uploadedDocument.title || uploadedDocument.file_name || selectedFile.name} ingestion status is ${finalStatus}.`;

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: finalMessage,
          createdAt: new Date().toISOString(),
        },
      ]);
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    } finally {
      setIsUploading(false);
    }
  }

  async function handleCreateSourceDocument() {
    const metadataResult = parseJsonObject(sourceMetadataDraft, "Source metadata");
    if (!metadataResult.ok) {
      setErrorText(metadataResult.error);
      return;
    }

    if (!sourceUri.trim()) {
      setErrorText("Source URI is required.");
      return;
    }

    setIsCreatingSource(true);
    setErrorText("");
    try {
      const document = await createDocument(apiSettings, {
        workspace_id: settings.workspaceId.trim() || null,
        source_type: sourceType,
        source_uri: sourceUri.trim(),
        title: sourceTitle.trim() || sourceUri.trim(),
        metadata: metadataResult.value,
        acl: defaultAclPolicy(),
        idempotency_key: `frontend-source-${crypto.randomUUID()}`,
      });
      setSourceUri("");
      setSourceTitle("");
      await refreshDocuments();
      await openDocument(document.id);
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    } finally {
      setIsCreatingSource(false);
    }
  }

  async function handleSaveDocument() {
    if (!activeDocument) {
      return;
    }

    const metadataResult = parseJsonObject(editMetadataDraft, "Document metadata");
    if (!metadataResult.ok) {
      setErrorText(metadataResult.error);
      return;
    }

    setIsSavingDocument(true);
    setErrorText("");
    try {
      const document = await updateDocument(apiSettings, activeDocument.id, {
        title: editTitle.trim() || null,
        classification_level: editClassification,
        metadata: metadataResult.value,
      });
      setActiveDocument(document);
      await refreshDocuments();
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    } finally {
      setIsSavingDocument(false);
    }
  }

  async function handleDeleteDocument(documentId: string) {
    const confirmed = window.confirm("Delete this document and hide it from retrieval?");
    if (!confirmed) {
      return;
    }

    setErrorText("");
    try {
      await deleteDocument(apiSettings, documentId);
      setSelectedDocumentIds((current) => current.filter((id) => id !== documentId));
      if (activeDocument?.id === documentId) {
        setActiveDocument(null);
        setDocumentJobs([]);
      }
      await refreshDocuments();
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleRestoreDocument(documentId: string) {
    setErrorText("");
    try {
      const document = await restoreDocument(apiSettings, documentId);
      await refreshDocuments();
      await openDocument(document.id);
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  function toggleDocumentSelection(documentId: string) {
    setSelectedDocumentIds((current) => {
      if (current.includes(documentId)) {
        return current.filter((id) => id !== documentId);
      }
      return [...current, documentId];
    });
  }

  function selectAllReadyDocuments() {
    setSelectedDocumentIds(readyDocuments.map((document) => document.id));
  }

  function clearDocumentSelection() {
    setSelectedDocumentIds([]);
  }

  async function handleSend() {
    const question = input.trim();
    if (!question || isAsking) {
      return;
    }
    if (selectedScopeIds.length === 0) {
      setErrorText("Select at least one ready document before asking.");
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
      const response = await runQuery(apiSettings, {
        query: retrievalQuestion,
        workspace_id: settings.workspaceId,
        conversation_id: "frontend-session",
        filters: {
          workspace_id: settings.workspaceId,
          document_ids: selectedScopeIds,
        },
        history,
        retrieval_strategy: settings.queryStrategy,
        retrieval_limit: settings.retrievalLimit,
        max_context_chunks: settings.maxContextChunks,
        max_context_tokens: settings.maxContextTokens,
      });
      setLatestResponse(response);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: buildAssistantContent(response),
          createdAt: new Date().toISOString(),
          response,
        },
      ]);
      await refreshQueryRuns();
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

  function resetChat() {
    setMessages([]);
    setLatestResponse(null);
    setErrorText("");
  }

  async function handleRunRetrieval() {
    const query = retrievalQuery.trim();
    if (!query) {
      setErrorText("Search text is required.");
      return;
    }

    const metadataResult = parseJsonObject(retrievalMetadataDraft, "Retrieval metadata filters");
    if (!metadataResult.ok) {
      setErrorText(metadataResult.error);
      return;
    }

    setErrorText("");
    setIsRunningRetrieval(true);
    setRerankedCandidates([]);
    try {
      const sourceTypes = retrievalSourceTypes
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean);
      const tags = retrievalTags
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean);
      const response = await runRetrieval(apiSettings, retrievalStrategy, {
        query,
        filters: {
          workspace_id: settings.workspaceId,
          document_ids: selectedDocumentIds,
          source_types: sourceTypes,
          tags,
          metadata: metadataResult.value,
          date_range: {},
        },
        limit: retrievalLimit,
        min_similarity: retrievalMinSimilarity,
      });
      setRetrievalResponse(response);
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunningRetrieval(false);
    }
  }

  async function handleRerankCandidates() {
    if (!retrievalResponse || retrievalResponse.candidates.length === 0) {
      setErrorText("Run retrieval before reranking.");
      return;
    }

    setErrorText("");
    setIsRunningRetrieval(true);
    try {
      const response = await rerankCandidates(
        apiSettings,
        retrievalQuery,
        retrievalResponse.candidates,
        retrievalLimit,
      );
      setRerankedCandidates(response.chunks);
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunningRetrieval(false);
    }
  }

  async function openQueryRun(agentRunId: string) {
    setActiveView("runs");
    setIsLoadingQueryRunDetail(true);
    setErrorText("");
    try {
      const run = await getQueryRun(apiSettings, agentRunId);
      setActiveQueryRun(run);
      if (run.response) {
        setLatestResponse(run.response);
      }
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoadingQueryRunDetail(false);
    }
  }

  async function handleCancelQueryRun(agentRunId: string) {
    setErrorText("");
    try {
      const run = await cancelQueryRun(apiSettings, agentRunId);
      setActiveQueryRun(run);
      await refreshQueryRuns();
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    }
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
            <p>Multi-tenant document intelligence</p>
          </div>
        </div>

        <div className={`health-pill health-${healthState}`}>
          <Circle size={10} fill="currentColor" />
          <span>{statusText}</span>
          <button type="button" aria-label="Check API" onClick={() => void handleHealthCheck()}>
            <RefreshCw size={15} />
          </button>
        </div>

        <nav className="nav-list" aria-label="Application sections">
          <NavButton activeView={activeView} view="chat" label="Chat" icon={<Bot size={17} />} onClick={setActiveView} />
          <NavButton activeView={activeView} view="documents" label="Documents" icon={<FileText size={17} />} onClick={setActiveView} />
          <NavButton activeView={activeView} view="retrieval" label="Search Lab" icon={<FileSearch size={17} />} onClick={setActiveView} />
          <NavButton activeView={activeView} view="runs" label="Query Runs" icon={<History size={17} />} onClick={setActiveView} />
          <NavButton activeView={activeView} view="operations" label="Operations" icon={<Activity size={17} />} onClick={setActiveView} />
        </nav>

        <section className="upload-panel">
          <div className="panel-heading">
            <span>Upload</span>
            {isUploading ? <Loader2 className="spin" size={15} /> : null}
          </div>
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
            <span>Selected Scope</span>
            <small>{selectedScopeIds.length} ready</small>
          </div>
          <div className="scope-actions">
            <button type="button" onClick={selectAllReadyDocuments}>All ready</button>
            <button type="button" onClick={clearDocumentSelection}>Clear</button>
          </div>
          <div className="document-list compact">
            {documents.length === 0 ? (
              <div className="empty-state">No documents loaded.</div>
            ) : (
              documents.slice(0, 8).map((document) => (
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
                    <strong>{document.title || document.file_name || document.id}</strong>
                    <small>
                      {document.status} · {formatDate(document.updated_at)}
                    </small>
                  </span>
                  <span className={`select-box ${selectedDocumentIds.includes(document.id) ? "selected" : ""}`} />
                </button>
              ))
            )}
          </div>
        </section>

        <section className="settings-panel">
          <button
            type="button"
            className="section-toggle"
            onClick={() => setActiveView("operations")}
          >
            <Settings size={16} />
            <span>Connection</span>
            <ChevronDown size={16} />
          </button>
        </section>
      </aside>

      <main className="app-main">
        <header className="app-header">
          <div>
            <h2>{viewTitle(activeView)}</h2>
            <p>{viewSubtitle(activeView, selectedScopeIds.length, documentsPage.total)}</p>
          </div>
          <div className="header-actions">
            <button type="button" className="ghost-button" onClick={() => void refreshDocuments()}>
              <RefreshCw size={17} />
              <span>Documents</span>
            </button>
            <button type="button" className="ghost-button" onClick={() => void refreshQueryRuns()}>
              <ClipboardList size={17} />
              <span>Runs</span>
            </button>
          </div>
        </header>

        {errorText ? (
          <div className="error-banner">
            <AlertCircle size={18} />
            <span>{errorText}</span>
            <button type="button" onClick={() => setErrorText("")}>
              <X size={16} />
            </button>
          </div>
        ) : null}

        {activeView === "chat" ? (
          <ChatView
            input={input}
            isAsking={isAsking}
            latestCitations={latestCitations}
            latestContext={latestContext}
            messages={messages}
            scrollRef={scrollRef}
            selectedScopeIds={selectedScopeIds}
            showEvidence={showEvidence}
            strategy={settings.queryStrategy}
            onInputChange={setInput}
            onKeyDown={handleComposerKeyDown}
            onReset={resetChat}
            onSend={() => void handleSend()}
            onToggleEvidence={() => setShowEvidence((current) => !current)}
          />
        ) : null}

        {activeView === "documents" ? (
          <DocumentsView
            activeDocument={activeDocument}
            documentJobs={documentJobs}
            documentPage={documentPage}
            documentSourceFilter={documentSourceFilter}
            documentStatusFilter={documentStatusFilter}
            documents={documents}
            documentsPage={documentsPage}
            editClassification={editClassification}
            editMetadataDraft={editMetadataDraft}
            editTitle={editTitle}
            isCreatingSource={isCreatingSource}
            isLoadingDocumentDetail={isLoadingDocumentDetail}
            isLoadingDocuments={isLoadingDocuments}
            isLoadingJobs={isLoadingJobs}
            isSavingDocument={isSavingDocument}
            selectedDocumentIds={selectedDocumentIds}
            sourceMetadataDraft={sourceMetadataDraft}
            sourceTitle={sourceTitle}
            sourceType={sourceType}
            sourceUri={sourceUri}
            onCreateSource={() => void handleCreateSourceDocument()}
            onDeleteDocument={(documentId) => void handleDeleteDocument(documentId)}
            onDocumentPageChange={setDocumentPage}
            onEditClassificationChange={setEditClassification}
            onEditMetadataChange={setEditMetadataDraft}
            onEditTitleChange={setEditTitle}
            onOpenDocument={(documentId) => void openDocument(documentId)}
            onRefreshDocuments={() => void refreshDocuments()}
            onRefreshJobs={() => activeDocument ? void refreshIngestionJobs(activeDocument.id) : undefined}
            onRestoreDocument={(documentId) => void handleRestoreDocument(documentId)}
            onSaveDocument={() => void handleSaveDocument()}
            onSourceMetadataChange={setSourceMetadataDraft}
            onSourceTitleChange={setSourceTitle}
            onSourceTypeChange={setSourceType}
            onSourceUriChange={setSourceUri}
            onSourceFilterChange={setDocumentSourceFilter}
            onStatusFilterChange={setDocumentStatusFilter}
            onToggleSelection={toggleDocumentSelection}
          />
        ) : null}

        {activeView === "retrieval" ? (
          <RetrievalView
            isRunningRetrieval={isRunningRetrieval}
            metadataDraft={retrievalMetadataDraft}
            minSimilarity={retrievalMinSimilarity}
            query={retrievalQuery}
            rerankedCandidates={rerankedCandidates}
            response={retrievalResponse}
            selectedScopeIds={selectedScopeIds}
            sourceTypes={retrievalSourceTypes}
            strategy={retrievalStrategy}
            tags={retrievalTags}
            limit={retrievalLimit}
            onMetadataChange={setRetrievalMetadataDraft}
            onMinSimilarityChange={setRetrievalMinSimilarity}
            onQueryChange={setRetrievalQuery}
            onRerank={() => void handleRerankCandidates()}
            onRun={() => void handleRunRetrieval()}
            onSourceTypesChange={setRetrievalSourceTypes}
            onStrategyChange={setRetrievalStrategy}
            onTagsChange={setRetrievalTags}
            onLimitChange={setRetrievalLimit}
          />
        ) : null}

        {activeView === "runs" ? (
          <QueryRunsView
            activeQueryRun={activeQueryRun}
            isLoadingQueryRunDetail={isLoadingQueryRunDetail}
            isLoadingQueryRuns={isLoadingQueryRuns}
            queryRunPage={queryRunPage}
            queryRuns={queryRuns}
            queryRunsPage={queryRunsPage}
            statusFilter={queryRunStatusFilter}
            verificationFilter={queryRunVerificationFilter}
            onCancelRun={(agentRunId) => void handleCancelQueryRun(agentRunId)}
            onOpenRun={(agentRunId) => void openQueryRun(agentRunId)}
            onPageChange={setQueryRunPage}
            onRefresh={() => void refreshQueryRuns()}
            onStatusFilterChange={setQueryRunStatusFilter}
            onVerificationFilterChange={setQueryRunVerificationFilter}
          />
        ) : null}

        {activeView === "operations" ? (
          <OperationsView
            health={health}
            healthState={healthState}
            settings={settings}
            onHealthCheck={() => void handleHealthCheck()}
            onSettingChange={updateSetting}
          />
        ) : null}
      </main>
    </div>
  );
}

function viewTitle(view: AppView): string {
  if (view === "chat") {
    return "Chat";
  }
  if (view === "documents") {
    return "Documents";
  }
  if (view === "retrieval") {
    return "Search Lab";
  }
  if (view === "runs") {
    return "Query Runs";
  }
  return "Operations";
}

function viewSubtitle(view: AppView, selectedCount: number, documentTotal: number): string {
  if (view === "chat") {
    return `${selectedCount} selected ready document${selectedCount === 1 ? "" : "s"}`;
  }
  if (view === "documents") {
    return `${documentTotal} readable document${documentTotal === 1 ? "" : "s"}`;
  }
  if (view === "retrieval") {
    return "BM25, vector, hybrid, and rerank";
  }
  if (view === "runs") {
    return "Persisted query execution history";
  }
  return "API health, auth token, workspace, and retrieval defaults";
}

function NavButton({
  activeView,
  view,
  label,
  icon,
  onClick,
}: {
  activeView: AppView;
  view: AppView;
  label: string;
  icon: React.ReactNode;
  onClick: (view: AppView) => void;
}) {
  return (
    <button
      type="button"
      className={`nav-button ${activeView === view ? "active" : ""}`}
      onClick={() => onClick(view)}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

function ChatView({
  input,
  isAsking,
  latestCitations,
  latestContext,
  messages,
  scrollRef,
  selectedScopeIds,
  showEvidence,
  strategy,
  onInputChange,
  onKeyDown,
  onReset,
  onSend,
  onToggleEvidence,
}: {
  input: string;
  isAsking: boolean;
  latestCitations: Citation[];
  latestContext: ContextChunk[];
  messages: ChatMessage[];
  scrollRef: React.RefObject<HTMLDivElement | null>;
  selectedScopeIds: string[];
  showEvidence: boolean;
  strategy: RetrievalStrategy;
  onInputChange: (value: string) => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  onReset: () => void;
  onSend: () => void;
  onToggleEvidence: () => void;
}) {
  return (
    <div className={`workspace ${showEvidence ? "with-evidence" : ""}`}>
      <section className="conversation">
        <div className="chat-toolbar">
          <div className="metric-row">
            <Metric label="Scope" value={`${selectedScopeIds.length} docs`} />
            <Metric label="Strategy" value={strategy} />
            <Metric label="Messages" value={String(messages.length)} />
          </div>
          <div className="header-actions">
            <button type="button" className="ghost-button" onClick={onReset}>
              <X size={17} />
              <span>Clear</span>
            </button>
            <button type="button" className="ghost-button" onClick={onToggleEvidence}>
              <PanelRightOpen size={17} />
              <span>Evidence</span>
            </button>
          </div>
        </div>

        <div className="message-stream" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="welcome-panel">
              <Bot size={30} />
              <h3>Ask against selected documents.</h3>
              <p>Answers are grounded in authorized retrieved context and persisted as query runs.</p>
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
                      <span>{message.response.verification_status || "not_required"}</span>
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
              onChange={(event) => onInputChange(event.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask about the selected documents"
              rows={1}
            />
          </div>
          <button
            type="button"
            className="send-button"
            disabled={!input.trim() || isAsking}
            onClick={onSend}
            aria-label="Send message"
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
  );
}

function DocumentsView({
  activeDocument,
  documentJobs,
  documentPage,
  documentSourceFilter,
  documentStatusFilter,
  documents,
  documentsPage,
  editClassification,
  editMetadataDraft,
  editTitle,
  isCreatingSource,
  isLoadingDocumentDetail,
  isLoadingDocuments,
  isLoadingJobs,
  isSavingDocument,
  selectedDocumentIds,
  sourceMetadataDraft,
  sourceTitle,
  sourceType,
  sourceUri,
  onCreateSource,
  onDeleteDocument,
  onDocumentPageChange,
  onEditClassificationChange,
  onEditMetadataChange,
  onEditTitleChange,
  onOpenDocument,
  onRefreshDocuments,
  onRefreshJobs,
  onRestoreDocument,
  onSaveDocument,
  onSourceMetadataChange,
  onSourceTitleChange,
  onSourceTypeChange,
  onSourceUriChange,
  onSourceFilterChange,
  onStatusFilterChange,
  onToggleSelection,
}: {
  activeDocument: DocumentRead | null;
  documentJobs: IngestionJobRead[];
  documentPage: number;
  documentSourceFilter: DocumentSourceType | "";
  documentStatusFilter: DocumentStatus | "";
  documents: DocumentListItem[];
  documentsPage: typeof EMPTY_PAGE;
  editClassification: ClassificationLevel;
  editMetadataDraft: string;
  editTitle: string;
  isCreatingSource: boolean;
  isLoadingDocumentDetail: boolean;
  isLoadingDocuments: boolean;
  isLoadingJobs: boolean;
  isSavingDocument: boolean;
  selectedDocumentIds: string[];
  sourceMetadataDraft: string;
  sourceTitle: string;
  sourceType: DocumentSourceType;
  sourceUri: string;
  onCreateSource: () => void;
  onDeleteDocument: (documentId: string) => void;
  onDocumentPageChange: (page: number) => void;
  onEditClassificationChange: (value: ClassificationLevel) => void;
  onEditMetadataChange: (value: string) => void;
  onEditTitleChange: (value: string) => void;
  onOpenDocument: (documentId: string) => void;
  onRefreshDocuments: () => void;
  onRefreshJobs: () => void | undefined;
  onRestoreDocument: (documentId: string) => void;
  onSaveDocument: () => void;
  onSourceMetadataChange: (value: string) => void;
  onSourceTitleChange: (value: string) => void;
  onSourceTypeChange: (value: DocumentSourceType) => void;
  onSourceUriChange: (value: string) => void;
  onSourceFilterChange: (value: DocumentSourceType | "") => void;
  onStatusFilterChange: (value: DocumentStatus | "") => void;
  onToggleSelection: (documentId: string) => void;
}) {
  return (
    <div className="content-grid split">
      <section className="panel">
        <div className="panel-title-row">
          <div>
            <h3>Document Inventory</h3>
            <p>{documentsPage.total} readable documents on this page scope</p>
          </div>
          <button type="button" className="ghost-button" onClick={onRefreshDocuments}>
            {isLoadingDocuments ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
            <span>Refresh</span>
          </button>
        </div>

        <div className="filter-row">
          <label>
            Status
            <select value={documentStatusFilter} onChange={(event) => onStatusFilterChange(event.target.value as DocumentStatus | "")}>
              {STATUS_OPTIONS.map((option) => (
                <option value={option} key={option || "all"}>{option || "all"}</option>
              ))}
            </select>
          </label>
          <label>
            Source
            <select value={documentSourceFilter} onChange={(event) => onSourceFilterChange(event.target.value as DocumentSourceType | "")}>
              {SOURCE_OPTIONS.map((option) => (
                <option value={option} key={option || "all"}>{option || "all"}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Use</th>
                <th>Document</th>
                <th>Status</th>
                <th>Source</th>
                <th>Updated</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.length === 0 ? (
                <tr>
                  <td colSpan={6}>No documents match the current filters.</td>
                </tr>
              ) : (
                documents.map((document) => (
                  <tr key={document.id}>
                    <td>
                      <button type="button" className="icon-button" onClick={() => onToggleSelection(document.id)}>
                        <span className={`select-box ${selectedDocumentIds.includes(document.id) ? "selected" : ""}`} />
                      </button>
                    </td>
                    <td>
                      <button type="button" className="link-button" onClick={() => onOpenDocument(document.id)}>
                        {document.title || document.file_name || document.id}
                      </button>
                      <small>{compactIdentifier(document.id)}</small>
                    </td>
                    <td><StatusBadge value={document.status} /></td>
                    <td>{document.source_type}</td>
                    <td>{formatDate(document.updated_at)}</td>
                    <td>
                      <div className="row-actions">
                        <button type="button" className="icon-button" onClick={() => onOpenDocument(document.id)} title="Open document">
                          <Eye size={16} />
                        </button>
                        <button type="button" className="icon-button danger" onClick={() => onDeleteDocument(document.id)} title="Delete document">
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <PaginationControls
          page={documentPage}
          pageInfo={documentsPage}
          onPageChange={onDocumentPageChange}
        />

        <div className="sub-panel">
          <div className="panel-title-row">
            <div>
              <h3>Register External Source</h3>
              <p>Create a tenant-scoped source record for URL, S3, or connector inputs.</p>
            </div>
          </div>
          <div className="form-grid">
            <label>
              Source type
              <select value={sourceType} onChange={(event) => onSourceTypeChange(event.target.value as DocumentSourceType)}>
                <option value="url">url</option>
                <option value="s3">s3</option>
                <option value="connector">connector</option>
              </select>
            </label>
            <label>
              Title
              <input value={sourceTitle} onChange={(event) => onSourceTitleChange(event.target.value)} placeholder="Optional title" />
            </label>
          </div>
          <label className="stacked-field">
            Source URI
            <input value={sourceUri} onChange={(event) => onSourceUriChange(event.target.value)} placeholder="s3://bucket/key or https://..." />
          </label>
          <label className="stacked-field">
            Metadata JSON
            <textarea value={sourceMetadataDraft} onChange={(event) => onSourceMetadataChange(event.target.value)} rows={5} />
          </label>
          <button type="button" className="primary-button inline" disabled={isCreatingSource} onClick={onCreateSource}>
            {isCreatingSource ? <Loader2 className="spin" size={17} /> : <Plus size={17} />}
            <span>Register Source</span>
          </button>
        </div>
      </section>

      <section className="panel detail-panel">
        {isLoadingDocumentDetail ? (
          <div className="loading-state">
            <Loader2 className="spin" size={20} />
            <span>Loading document</span>
          </div>
        ) : activeDocument ? (
          <>
            <div className="panel-title-row">
              <div>
                <h3>Document Detail</h3>
                <p>{activeDocument.file_name || activeDocument.source_uri || activeDocument.id}</p>
              </div>
              <StatusBadge value={activeDocument.status} />
            </div>
            <div className="form-grid">
              <label>
                Title
                <input value={editTitle} onChange={(event) => onEditTitleChange(event.target.value)} />
              </label>
              <label>
                Classification
                <select value={editClassification} onChange={(event) => onEditClassificationChange(event.target.value as ClassificationLevel)}>
                  {CLASSIFICATION_OPTIONS.map((option) => (
                    <option value={option} key={option}>{option}</option>
                  ))}
                </select>
              </label>
            </div>
            <div className="detail-grid">
              <Metric label="Document ID" value={compactIdentifier(activeDocument.id)} />
              <Metric label="Workspace" value={activeDocument.workspace_id || "none"} />
              <Metric label="Source" value={activeDocument.source_type} />
              <Metric label="Size" value={formatBytes(activeDocument.byte_size)} />
              <Metric label="ACL version" value={String(activeDocument.acl_version)} />
              <Metric label="Updated" value={formatDate(activeDocument.updated_at)} />
            </div>
            <label className="stacked-field">
              Metadata JSON
              <textarea value={editMetadataDraft} onChange={(event) => onEditMetadataChange(event.target.value)} rows={8} />
            </label>
            <div className="button-row">
              <button type="button" className="primary-button inline" disabled={isSavingDocument} onClick={onSaveDocument}>
                {isSavingDocument ? <Loader2 className="spin" size={17} /> : <Save size={17} />}
                <span>Save</span>
              </button>
              <button type="button" className="ghost-button" onClick={() => onRestoreDocument(activeDocument.id)}>
                <RotateCcw size={17} />
                <span>Restore</span>
              </button>
              <button type="button" className="ghost-button danger" onClick={() => onDeleteDocument(activeDocument.id)}>
                <Trash2 size={17} />
                <span>Delete</span>
              </button>
            </div>

            <div className="panel-title-row section-divider">
              <div>
                <h3>Ingestion Jobs</h3>
                <p>{documentJobs.length} jobs for this document</p>
              </div>
              <button type="button" className="ghost-button" onClick={onRefreshJobs}>
                {isLoadingJobs ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
                <span>Jobs</span>
              </button>
            </div>
            <div className="job-list">
              {documentJobs.length === 0 ? (
                <div className="empty-state">No ingestion jobs returned.</div>
              ) : (
                documentJobs.map((job) => (
                  <article className="job-card" key={job.id}>
                    <div>
                      <strong>{compactIdentifier(job.id)}</strong>
                      <small>{formatDate(job.updated_at)}</small>
                    </div>
                    <StatusBadge value={job.status} />
                    <span>{job.current_stage}</span>
                    {job.error_message ? <p>{job.error_message}</p> : null}
                  </article>
                ))
              )}
            </div>
          </>
        ) : (
          <div className="empty-state">Select a document to inspect metadata, ACL version, and ingestion jobs.</div>
        )}
      </section>
    </div>
  );
}

function RetrievalView({
  isRunningRetrieval,
  metadataDraft,
  minSimilarity,
  query,
  rerankedCandidates,
  response,
  selectedScopeIds,
  sourceTypes,
  strategy,
  tags,
  limit,
  onMetadataChange,
  onMinSimilarityChange,
  onQueryChange,
  onRerank,
  onRun,
  onSourceTypesChange,
  onStrategyChange,
  onTagsChange,
  onLimitChange,
}: {
  isRunningRetrieval: boolean;
  metadataDraft: string;
  minSimilarity: number;
  query: string;
  rerankedCandidates: CandidateChunk[];
  response: RetrievalResponse | null;
  selectedScopeIds: string[];
  sourceTypes: string;
  strategy: RetrievalStrategy;
  tags: string;
  limit: number;
  onMetadataChange: (value: string) => void;
  onMinSimilarityChange: (value: number) => void;
  onQueryChange: (value: string) => void;
  onRerank: () => void;
  onRun: () => void;
  onSourceTypesChange: (value: string) => void;
  onStrategyChange: (value: RetrievalStrategy) => void;
  onTagsChange: (value: string) => void;
  onLimitChange: (value: number) => void;
}) {
  const displayedCandidates = rerankedCandidates.length > 0 ? rerankedCandidates : response?.candidates ?? [];

  return (
    <div className="content-grid split">
      <section className="panel">
        <div className="panel-title-row">
          <div>
            <h3>Search Lab</h3>
            <p>{selectedScopeIds.length} selected document filters are applied.</p>
          </div>
          <SlidersHorizontal size={20} />
        </div>
        <label className="stacked-field">
          Search text
          <textarea value={query} onChange={(event) => onQueryChange(event.target.value)} rows={5} placeholder="Question or keywords" />
        </label>
        <div className="form-grid">
          <label>
            Strategy
            <select value={strategy} onChange={(event) => onStrategyChange(event.target.value as RetrievalStrategy)}>
              <option value="bm25">bm25</option>
              <option value="vector">vector</option>
              <option value="hybrid">hybrid</option>
            </select>
          </label>
          <label>
            Limit
            <input type="number" min={1} max={200} value={limit} onChange={(event) => onLimitChange(Number(event.target.value))} />
          </label>
          <label>
            Min similarity
            <input type="number" min={0} max={1} step={0.05} value={minSimilarity} onChange={(event) => onMinSimilarityChange(Number(event.target.value))} />
          </label>
        </div>
        <div className="form-grid">
          <label>
            Source types
            <input value={sourceTypes} onChange={(event) => onSourceTypesChange(event.target.value)} placeholder="upload,s3" />
          </label>
          <label>
            Tags
            <input value={tags} onChange={(event) => onTagsChange(event.target.value)} placeholder="security,policy" />
          </label>
        </div>
        <label className="stacked-field">
          Metadata filters JSON
          <textarea value={metadataDraft} onChange={(event) => onMetadataChange(event.target.value)} rows={6} />
        </label>
        <div className="button-row">
          <button type="button" className="primary-button inline" disabled={isRunningRetrieval} onClick={onRun}>
            {isRunningRetrieval ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
            <span>Search</span>
          </button>
          <button type="button" className="ghost-button" disabled={!response || response.candidates.length === 0 || isRunningRetrieval} onClick={onRerank}>
            <Layers size={17} />
            <span>Rerank</span>
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="panel-title-row">
          <div>
            <h3>Retrieved Chunks</h3>
            <p>{displayedCandidates.length} candidates {rerankedCandidates.length ? "after rerank" : "from retrieval"}</p>
          </div>
          {response ? <Metric label="Latency" value={`${response.latency_ms} ms`} /> : null}
        </div>
        <CandidateList candidates={displayedCandidates} />
      </section>
    </div>
  );
}

function QueryRunsView({
  activeQueryRun,
  isLoadingQueryRunDetail,
  isLoadingQueryRuns,
  queryRunPage,
  queryRuns,
  queryRunsPage,
  statusFilter,
  verificationFilter,
  onCancelRun,
  onOpenRun,
  onPageChange,
  onRefresh,
  onStatusFilterChange,
  onVerificationFilterChange,
}: {
  activeQueryRun: QueryRunRead | null;
  isLoadingQueryRunDetail: boolean;
  isLoadingQueryRuns: boolean;
  queryRunPage: number;
  queryRuns: QueryRunListItem[];
  queryRunsPage: typeof EMPTY_PAGE;
  statusFilter: QueryRunStatus | "";
  verificationFilter: string;
  onCancelRun: (agentRunId: string) => void;
  onOpenRun: (agentRunId: string) => void;
  onPageChange: (page: number) => void;
  onRefresh: () => void;
  onStatusFilterChange: (value: QueryRunStatus | "") => void;
  onVerificationFilterChange: (value: string) => void;
}) {
  return (
    <div className="content-grid split">
      <section className="panel">
        <div className="panel-title-row">
          <div>
            <h3>Query Run History</h3>
            <p>{queryRunsPage.total} runs in this filter scope</p>
          </div>
          <button type="button" className="ghost-button" onClick={onRefresh}>
            {isLoadingQueryRuns ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
            <span>Refresh</span>
          </button>
        </div>
        <div className="filter-row">
          <label>
            Status
            <select value={statusFilter} onChange={(event) => onStatusFilterChange(event.target.value as QueryRunStatus | "")}>
              {QUERY_RUN_STATUS_OPTIONS.map((option) => (
                <option value={option} key={option || "all"}>{option || "all"}</option>
              ))}
            </select>
          </label>
          <label>
            Verification
            <input value={verificationFilter} onChange={(event) => onVerificationFilterChange(event.target.value)} placeholder="passed, failed, skipped" />
          </label>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>Strategy</th>
                <th>Latency</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {queryRuns.length === 0 ? (
                <tr>
                  <td colSpan={6}>No query runs match the current filters.</td>
                </tr>
              ) : (
                queryRuns.map((run) => (
                  <tr key={run.agent_run_id}>
                    <td>
                      <button type="button" className="link-button" onClick={() => onOpenRun(run.agent_run_id)}>
                        {compactIdentifier(run.agent_run_id)}
                      </button>
                      <small>{run.query.slice(0, 80)}</small>
                    </td>
                    <td><StatusBadge value={run.status} /></td>
                    <td>{run.retrieval_strategy || "pending"}</td>
                    <td>{run.latency_ms === null ? "pending" : `${run.latency_ms} ms`}</td>
                    <td>{formatDate(run.created_at)}</td>
                    <td>
                      <div className="row-actions">
                        <button type="button" className="icon-button" onClick={() => onOpenRun(run.agent_run_id)}>
                          <Eye size={16} />
                        </button>
                        {(run.status === "queued" || run.status === "running") ? (
                          <button type="button" className="icon-button danger" onClick={() => onCancelRun(run.agent_run_id)}>
                            <X size={16} />
                          </button>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        <PaginationControls page={queryRunPage} pageInfo={queryRunsPage} onPageChange={onPageChange} />
      </section>

      <section className="panel detail-panel">
        {isLoadingQueryRunDetail ? (
          <div className="loading-state">
            <Loader2 className="spin" size={20} />
            <span>Loading query run</span>
          </div>
        ) : activeQueryRun ? (
          <>
            <div className="panel-title-row">
              <div>
                <h3>Run Detail</h3>
                <p>{activeQueryRun.query}</p>
              </div>
              <StatusBadge value={activeQueryRun.status} />
            </div>
            <div className="detail-grid">
              <Metric label="Run ID" value={compactIdentifier(activeQueryRun.agent_run_id)} />
              <Metric label="Workspace" value={activeQueryRun.workspace_id || "none"} />
              <Metric label="Strategy" value={activeQueryRun.retrieval_strategy || "pending"} />
              <Metric label="Confidence" value={activeQueryRun.confidence_score === null ? "pending" : activeQueryRun.confidence_score.toFixed(2)} />
              <Metric label="Verification" value={activeQueryRun.verification_status} />
              <Metric label="Latency" value={activeQueryRun.latency_ms === null ? "pending" : `${activeQueryRun.latency_ms} ms`} />
            </div>
            {activeQueryRun.answer ? (
              <div className="answer-box">
                <strong>Answer</strong>
                <p>{activeQueryRun.answer}</p>
              </div>
            ) : null}
            {activeQueryRun.error_message ? (
              <div className="error-box">
                <strong>{activeQueryRun.error_type || "Error"}</strong>
                <p>{activeQueryRun.error_message}</p>
              </div>
            ) : null}
            <label className="stacked-field">
              Response payload
              <textarea readOnly value={formatJson(activeQueryRun.response_payload)} rows={12} />
            </label>
          </>
        ) : (
          <div className="empty-state">Select a query run to inspect persisted answer, filters, and payload.</div>
        )}
      </section>
    </div>
  );
}

function OperationsView({
  health,
  healthState,
  settings,
  onHealthCheck,
  onSettingChange,
}: {
  health: HealthResponse | null;
  healthState: string;
  settings: RuntimeSettings;
  onHealthCheck: () => void;
  onSettingChange: <K extends keyof RuntimeSettings>(key: K, value: RuntimeSettings[K]) => void;
}) {
  const dependencies = Object.entries(health?.dependencies ?? {});

  return (
    <div className="content-grid operations-grid">
      <section className="panel">
        <div className="panel-title-row">
          <div>
            <h3>API Health</h3>
            <p>{health ? `${health.service} ${health.version}` : "No health payload loaded"}</p>
          </div>
          <button type="button" className="ghost-button" onClick={onHealthCheck}>
            <RefreshCw size={16} />
            <span>Check</span>
          </button>
        </div>
        <div className="detail-grid">
          <Metric label="State" value={healthState} />
          <Metric label="Service" value={health?.service || "unknown"} />
          <Metric label="Version" value={health?.version || "unknown"} />
          <Metric label="Dependencies" value={String(dependencies.length)} />
        </div>
        <div className="job-list">
          {dependencies.length === 0 ? (
            <div className="empty-state">No dependency details returned.</div>
          ) : (
            dependencies.map(([key, dependency]) => (
              <article className="job-card" key={key}>
                <div>
                  <strong>{dependency.name || key}</strong>
                  <small>{dependency.detail || "No detail"}</small>
                </div>
                <StatusBadge value={dependency.status || "unknown"} />
              </article>
            ))
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel-title-row">
          <div>
            <h3>Runtime Settings</h3>
            <p>Used by all frontend API calls in this browser session.</p>
          </div>
          <Database size={20} />
        </div>
        <label className="stacked-field">
          API URL
          <input value={settings.apiBaseUrl} onChange={(event) => onSettingChange("apiBaseUrl", event.target.value)} />
        </label>
        <label className="stacked-field">
          Bearer token
          <input type="password" value={settings.authToken} onChange={(event) => onSettingChange("authToken", event.target.value)} />
        </label>
        <label className="stacked-field">
          Workspace
          <input value={settings.workspaceId} onChange={(event) => onSettingChange("workspaceId", event.target.value)} />
        </label>
        <label className="stacked-field">
          Chat retrieval strategy
          <select
            value={settings.queryStrategy}
            onChange={(event) => onSettingChange("queryStrategy", event.target.value as RetrievalStrategy)}
          >
            <option value="hybrid">hybrid</option>
            <option value="vector">vector</option>
            <option value="bm25">bm25</option>
          </select>
        </label>
        <div className="form-grid">
          <label>
            Retrieval limit
            <input type="number" min={1} max={200} value={settings.retrievalLimit} onChange={(event) => onSettingChange("retrievalLimit", Number(event.target.value))} />
          </label>
          <label>
            Context chunks
            <input type="number" min={1} max={50} value={settings.maxContextChunks} onChange={(event) => onSettingChange("maxContextChunks", Number(event.target.value))} />
          </label>
          <label>
            Context tokens
            <input type="number" min={500} value={settings.maxContextTokens} onChange={(event) => onSettingChange("maxContextTokens", Number(event.target.value))} />
          </label>
        </div>
      </section>
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

function CandidateList({ candidates }: { candidates: CandidateChunk[] }) {
  if (candidates.length === 0) {
    return <div className="empty-state">No candidates returned yet.</div>;
  }

  return (
    <div className="candidate-list">
      {candidates.map((candidate, index) => {
        const title = candidate.citation?.title || "Document";
        const content = candidate.content || candidate.citation?.quote || "";
        const preview = content.replace(/\s+/g, " ").trim();
        return (
          <article className="candidate-card" key={`${candidate.document_id}-${candidate.chunk_id}-${index}`}>
            <div className="candidate-heading">
              <span>{index + 1}</span>
              <div>
                <strong>{title}</strong>
                <small>{candidate.source} · score {candidate.score.toFixed(3)}</small>
              </div>
            </div>
            <p>{preview.length > 900 ? `${preview.slice(0, 900).trim()}...` : preview}</p>
            <div className="message-meta">
              <span>document {compactIdentifier(candidate.document_id)}</span>
              <span>chunk {compactIdentifier(candidate.chunk_id)}</span>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function StatusBadge({ value }: { value: string | null | undefined }) {
  const displayValue = value || "unknown";
  return <span className={`status-badge ${statusTone(displayValue)}`}>{displayValue}</span>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PaginationControls({
  page,
  pageInfo,
  onPageChange,
}: {
  page: number;
  pageInfo: typeof EMPTY_PAGE;
  onPageChange: (page: number) => void;
}) {
  const hasNextPage = page * pageInfo.size < pageInfo.total;
  return (
    <div className="pagination-row">
      <button type="button" className="ghost-button" disabled={page <= 1} onClick={() => onPageChange(Math.max(1, page - 1))}>
        Previous
      </button>
      <span>
        Page {pageInfo.page} · {pageInfo.total} total
      </span>
      <button type="button" className="ghost-button" disabled={!hasNextPage} onClick={() => onPageChange(page + 1)}>
        Next
      </button>
    </div>
  );
}

export default App;
