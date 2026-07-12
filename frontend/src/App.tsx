import {
  Activity,
  AlertCircle,
  Bot,
  Building2,
  Check,
  ChevronDown,
  Circle,
  ClipboardList,
  Code2,
  Cpu,
  Database,
  Eye,
  FileSearch,
  FileText,
  Globe2,
  History,
  KeyRound,
  Layers,
  Loader2,
  LogIn,
  LogOut,
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
  UserPlus,
  Users,
  X,
} from "lucide-react";
import { ChangeEvent, FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiClientSettings,
  ApiError,
  AuthSession,
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
  LLMProvider,
  LLMProviderCreateRequest,
  LLMProviderType,
  LLMProviderUpdateRequest,
  LLMProviderValidationResponse,
  QueryHistoryMessage,
  QueryResponse,
  QueryRunListItem,
  QueryRunRead,
  QueryRunStatus,
  RetrievalResponse,
  RetrievalStrategy,
  TenantUser,
  TenantUserRole,
  UserInvitationRequest,
  cancelQueryRun,
  checkReadiness,
  acceptInvitation,
  createLLMProvider,
  createDocument,
  createTenant,
  deleteLLMProvider,
  deleteDocument,
  getDocument,
  getAuthSession,
  getIngestionJob,
  getQueryRun,
  inviteTenantUser,
  listIngestionJobs,
  listLLMProviders,
  listQueryRuns,
  listTenantUsers,
  rerankCandidates,
  resolveInvitation,
  restoreDocument,
  runQuery,
  runRetrieval,
  searchDocuments,
  updateDocument,
  updateLLMProvider,
  uploadDocument,
  validateLLMProvider,
} from "./api";
import {
  AuthConfiguration,
  getAuthenticationErrorMessage,
  loadAuthConfiguration,
  restoreAuth0Session,
  restoreSuperTokensSession,
  sendPasswordReset,
  signInWithEmailPassword,
  signOutSuperTokens,
  signUpWithEmailPassword,
  startAuth0Login,
  startAuth0Logout,
  startSuperTokensSocialLogin,
  submitPasswordReset,
  subscribeToAuth0Session,
} from "./auth";

type AppView =
  | "chat"
  | "documents"
  | "retrieval"
  | "runs"
  | "admin"
  | "models"
  | "operations";

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

type AuthenticationState = "loading" | "authenticated" | "anonymous" | "error";

type IdentitySummary = {
  displayName: string;
  email: string;
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

  if (
    value === "running" ||
    value === "queued" ||
    value === "indexing" ||
    value === "parsing" ||
    value === "invited"
  ) {
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
  const [authConfiguration, setAuthConfiguration] =
    useState<AuthConfiguration | null>(null);
  const [authenticationState, setAuthenticationState] =
    useState<AuthenticationState>("loading");
  const [authenticationError, setAuthenticationError] = useState("");
  const [authSession, setAuthSession] = useState<AuthSession | null>(null);
  const [identity, setIdentity] = useState<IdentitySummary>({
    displayName: "",
    email: "",
  });
  const [authenticationRetry, setAuthenticationRetry] = useState(0);
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

  const [tenantUsers, setTenantUsers] = useState<TenantUser[]>([]);
  const [tenantUsersPage, setTenantUsersPage] = useState(EMPTY_PAGE);
  const [tenantUserPage, setTenantUserPage] = useState(1);
  const [isLoadingTenantUsers, setIsLoadingTenantUsers] = useState(false);
  const [isInvitingTenantUser, setIsInvitingTenantUser] = useState(false);
  const [adminNotice, setAdminNotice] = useState("");
  const [adminNoticeTone, setAdminNoticeTone] = useState<"success" | "danger">("success");

  const [llmProviders, setLLMProviders] = useState<LLMProvider[]>([]);
  const [llmProvidersPage, setLLMProvidersPage] = useState(EMPTY_PAGE);
  const [isLoadingLLMProviders, setIsLoadingLLMProviders] = useState(false);
  const [isSavingLLMProvider, setIsSavingLLMProvider] = useState(false);
  const [validatingLLMProviderId, setValidatingLLMProviderId] = useState("");
  const [llmProviderNotice, setLLMProviderNotice] = useState("");
  const [llmProviderNoticeTone, setLLMProviderNoticeTone] =
    useState<"success" | "danger">("success");

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const apiSettings = useMemo(() => {
    return {
      ...buildApiSettings(settings),
      tenantId: authSession?.tenant_uuid,
      departmentId: authSession?.department_id,
      workspaceId: authSession?.workspace_id,
    };
  }, [settings, authSession]);

  useEffect(() => {
    let cancelled = false;
    let unsubscribeFromAuth0Session: (() => void) | null = null;

    async function loadAuthorizedSession(
      accessToken: string,
      configuration: AuthConfiguration,
      profile?: Record<string, unknown>,
    ) {
      const session = await getAuthSession({
        apiBaseUrl: settings.apiBaseUrl,
        authToken: accessToken,
      });
      if (cancelled) {
        return;
      }

      let displayName = session.user_id;
      let email = session.email || "";
      if (profile) {
        const profileName = profile.name || profile.preferred_username;
        if (typeof profileName === "string" && profileName.trim()) {
          displayName = profileName.trim();
        }
        if (typeof profile.email === "string") {
          email = profile.email;
        }
      }

      setSettings((current) => {
        return {
          ...current,
          authToken: accessToken,
          workspaceId:
            session.workspace_id ||
            (configuration.mode === "local" ? current.workspaceId : ""),
        };
      });
      setAuthSession(session);
      setIdentity({ displayName, email });
      setAuthenticationError("");
      setAuthenticationState("authenticated");
    }

    async function initializeAuthentication() {
      setAuthenticationState("loading");
      setAuthenticationError("");
      setAuthSession(null);

      try {
        const configuration = await loadAuthConfiguration(settings.apiBaseUrl);
        if (cancelled) {
          return;
        }
        setAuthConfiguration(configuration);

        if (configuration.mode === "local") {
          await loadAuthorizedSession(settings.authToken, configuration);
          return;
        }

        if (configuration.mode === "supertokens") {
          const hasSession = await restoreSuperTokensSession(
            settings.apiBaseUrl,
            configuration,
          );
          const invitationToken =
            window.location.pathname === "/invite"
              ? new URLSearchParams(window.location.search).get("token")
              : null;
          if (invitationToken) {
            await resolveInvitation(buildApiSettings(settings), invitationToken);
            window.sessionStorage.setItem("agentic_rag_pending_invitation", "true");
          }
          if (!hasSession) {
            setSettings((current) => ({ ...current, authToken: "" }));
            setAuthenticationState("anonymous");
            return;
          }
          if (
            invitationToken ||
            window.sessionStorage.getItem("agentic_rag_pending_invitation") === "true"
          ) {
            await acceptInvitation(buildApiSettings(settings));
            window.sessionStorage.removeItem("agentic_rag_pending_invitation");
            window.history.replaceState({}, document.title, "/");
          }
          await loadAuthorizedSession("", configuration);
          return;
        }

        const auth0Session = await restoreAuth0Session(configuration);
        if (cancelled) {
          return;
        }
        if (!auth0Session) {
          setSettings((current) => {
            return {
              ...current,
              authToken: "",
            };
          });
          setAuthenticationState("anonymous");
          return;
        }

        await loadAuthorizedSession(
          auth0Session.accessToken,
          configuration,
          auth0Session.profile as Record<string, unknown>,
        );

        unsubscribeFromAuth0Session = await subscribeToAuth0Session(
          configuration,
          (updatedSession) => {
            if (cancelled) {
              return;
            }
            if (!updatedSession) {
              setAuthSession(null);
              setAuthenticationState("anonymous");
              return;
            }
            void loadAuthorizedSession(
              updatedSession.accessToken,
              configuration,
              updatedSession.profile as Record<string, unknown>,
            ).catch(async (error) => {
              if (!cancelled) {
                setAuthenticationError(await getAuthenticationErrorMessage(error));
                setAuthenticationState("error");
              }
            });
          },
        );
      } catch (error) {
        if (cancelled) {
          return;
        }
        setAuthenticationError(await getAuthenticationErrorMessage(error));
        setAuthenticationState("error");
      }
    }

    void initializeAuthentication();

    return () => {
      cancelled = true;
      if (unsubscribeFromAuth0Session) {
        unsubscribeFromAuth0Session();
      }
    };
  }, [settings.apiBaseUrl, authenticationRetry]);

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
  const canReadDocuments = authSession?.scopes.includes("documents:read") ?? false;
  const canWriteDocuments = authSession?.scopes.includes("documents:write") ?? false;
  const canDeleteDocuments = authSession?.scopes.includes("documents:delete") ?? false;
  const canRunQueries = authSession?.scopes.includes("query:run") ?? false;
  const canAdministerTenant =
    authSession?.tenant_permissions.includes("tenant.members.view") ?? false;

  useEffect(() => {
    if (!authSession) {
      return;
    }
    if (
      !canRunQueries &&
      (activeView === "chat" || activeView === "retrieval" || activeView === "runs")
    ) {
      setActiveView(canReadDocuments ? "documents" : "operations");
    }
    if (
      !canAdministerTenant &&
      (activeView === "admin" || activeView === "models")
    ) {
      setActiveView(canReadDocuments ? "documents" : "operations");
    }
  }, [
    authSession,
    activeView,
    canAdministerTenant,
    canReadDocuments,
    canRunQueries,
  ]);

  useEffect(() => {
    if (authenticationState !== "authenticated") {
      return;
    }
    void handleHealthCheck();
  }, [settings.apiBaseUrl, settings.authToken, authenticationState]);

  useEffect(() => {
    if (authenticationState !== "authenticated") {
      return;
    }
    void refreshDocuments();
  }, [
    settings.apiBaseUrl,
    settings.authToken,
    settings.workspaceId,
    authenticationState,
    documentPage,
    documentStatusFilter,
    documentSourceFilter,
  ]);

  useEffect(() => {
    if (authenticationState !== "authenticated") {
      return;
    }
    void refreshQueryRuns();
  }, [
    settings.apiBaseUrl,
    settings.authToken,
    settings.workspaceId,
    authenticationState,
    queryRunPage,
    queryRunStatusFilter,
    queryRunVerificationFilter,
  ]);

  useEffect(() => {
    if (
      authenticationState !== "authenticated" ||
      !canAdministerTenant ||
      activeView !== "admin"
    ) {
      return;
    }
    void refreshTenantUsers();
  }, [
    settings.apiBaseUrl,
    settings.authToken,
    authenticationState,
    canAdministerTenant,
    activeView,
    tenantUserPage,
  ]);

  useEffect(() => {
    if (
      authenticationState !== "authenticated" ||
      !canAdministerTenant ||
      activeView !== "models"
    ) {
      return;
    }
    void refreshLLMProviders();
  }, [
    settings.apiBaseUrl,
    settings.authToken,
    authenticationState,
    canAdministerTenant,
    activeView,
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

  async function refreshTenantUsers() {
    if (!canAdministerTenant) {
      return;
    }

    setIsLoadingTenantUsers(true);
    try {
      const response = await listTenantUsers(apiSettings, tenantUserPage, 50);
      setTenantUsers(response.items);
      setTenantUsersPage(response.page);
    } catch (error) {
      setAdminNoticeTone("danger");
      setAdminNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoadingTenantUsers(false);
    }
  }

  async function handleInviteTenantUser(
    invitation: UserInvitationRequest,
  ): Promise<boolean> {
    if (!canAdministerTenant) {
      setAdminNoticeTone("danger");
      setAdminNotice("Only tenant administrators can invite users.");
      return false;
    }

    setIsInvitingTenantUser(true);
    setAdminNotice("");
    try {
      const response = await inviteTenantUser(apiSettings, invitation);
      setAdminNoticeTone("success");
      setAdminNotice(
        `Invitation sent to ${response.user.email || invitation.email} as ${response.user.roles.join(", ")}.`,
      );
      await refreshTenantUsers();
      return true;
    } catch (error) {
      setAdminNoticeTone("danger");
      setAdminNotice(error instanceof Error ? error.message : String(error));
      return false;
    } finally {
      setIsInvitingTenantUser(false);
    }
  }

  async function refreshLLMProviders() {
    if (!canAdministerTenant) {
      return;
    }

    setIsLoadingLLMProviders(true);
    try {
      const response = await listLLMProviders(apiSettings, 1, 100);
      setLLMProviders(response.items);
      setLLMProvidersPage(response.page);
    } catch (error) {
      setLLMProviderNoticeTone("danger");
      setLLMProviderNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoadingLLMProviders(false);
    }
  }

  async function handleSaveLLMProvider(
    providerId: string | null,
    request: LLMProviderCreateRequest | LLMProviderUpdateRequest,
  ): Promise<boolean> {
    setIsSavingLLMProvider(true);
    setLLMProviderNotice("");
    try {
      const provider = providerId
        ? await updateLLMProvider(
            apiSettings,
            providerId,
            request as LLMProviderUpdateRequest,
          )
        : await createLLMProvider(
            apiSettings,
            request as LLMProviderCreateRequest,
          );
      setLLMProviderNoticeTone("success");
      setLLMProviderNotice(`${provider.name} saved successfully.`);
      await refreshLLMProviders();
      return true;
    } catch (error) {
      setLLMProviderNoticeTone("danger");
      setLLMProviderNotice(error instanceof Error ? error.message : String(error));
      return false;
    } finally {
      setIsSavingLLMProvider(false);
    }
  }

  async function handleDeleteLLMProvider(providerId: string): Promise<boolean> {
    setIsSavingLLMProvider(true);
    setLLMProviderNotice("");
    try {
      await deleteLLMProvider(apiSettings, providerId);
      setLLMProviderNoticeTone("success");
      setLLMProviderNotice("Provider removed. A compatible fallback was selected when available.");
      await refreshLLMProviders();
      return true;
    } catch (error) {
      setLLMProviderNoticeTone("danger");
      setLLMProviderNotice(error instanceof Error ? error.message : String(error));
      return false;
    } finally {
      setIsSavingLLMProvider(false);
    }
  }

  async function handleValidateLLMProvider(
    providerId: string,
    capability: "chat" | "embedding",
  ): Promise<LLMProviderValidationResponse | null> {
    setValidatingLLMProviderId(providerId);
    setLLMProviderNotice("");
    try {
      const result = await validateLLMProvider(
        apiSettings,
        providerId,
        capability,
      );
      setLLMProviderNoticeTone("success");
      setLLMProviderNotice(
        `${capability === "chat" ? "Chat" : "Embedding"} route is healthy: ${result.model} in ${result.latency_ms} ms.`,
      );
      return result;
    } catch (error) {
      setLLMProviderNoticeTone("danger");
      setLLMProviderNotice(error instanceof Error ? error.message : String(error));
      return null;
    } finally {
      setValidatingLLMProviderId("");
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
    if (file && window.matchMedia("(max-width: 820px)").matches) {
      void handleUpload(file, file.name);
    }
  }

  async function handleUpload(fileOverride?: File, titleOverride?: string) {
    if (!canWriteDocuments) {
      setErrorText("Your role does not allow document uploads.");
      return;
    }
    const fileToUpload = fileOverride || selectedFile;
    if (!fileToUpload) {
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
        fileToUpload,
        settings.workspaceId,
        titleOverride || documentTitle,
        {
          ...metadataResult.value,
          uploaded_at: new Date().toISOString(),
        },
        authSession?.department_id,
      );

      const uploadedDocument = upload.document;
      setSelectedDocumentIds([uploadedDocument.id]);
      setLatestResponse(null);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "system",
          content: `Indexing ${uploadedDocument.title || uploadedDocument.file_name || fileToUpload.name}.`,
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
      setSelectedDocumentIds([uploadedDocument.id]);
      setLatestResponse(null);

      const finalStatus = completedJob?.status || "running";
      const finalMessage =
        finalStatus === "completed"
          ? `${uploadedDocument.title || uploadedDocument.file_name || fileToUpload.name} is indexed and ready.`
          : `${uploadedDocument.title || uploadedDocument.file_name || fileToUpload.name} ingestion status is ${finalStatus}.`;

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
    if (!canWriteDocuments) {
      setErrorText("Your role does not allow document creation.");
      return;
    }
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
      setSelectedDocumentIds([document.id]);
      setLatestResponse(null);
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
    if (!canWriteDocuments) {
      setErrorText("Your role does not allow document updates.");
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
    if (!canDeleteDocuments) {
      setErrorText("Your role does not allow document deletion.");
      return;
    }
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
    if (!canWriteDocuments) {
      setErrorText("Your role does not allow document restoration.");
      return;
    }
    setErrorText("");
    try {
      const document = await restoreDocument(apiSettings, documentId);
      await refreshDocuments();
      await openDocument(document.id);
      setSelectedDocumentIds([document.id]);
      setLatestResponse(null);
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  function toggleDocumentSelection(documentId: string) {
    setLatestResponse(null);
    setSelectedDocumentIds((current) => {
      if (current.includes(documentId)) {
        return current.filter((id) => id !== documentId);
      }
      return [...current, documentId];
    });
  }

  function selectAllReadyDocuments() {
    setLatestResponse(null);
    setSelectedDocumentIds(readyDocuments.map((document) => document.id));
  }

  function clearDocumentSelection() {
    setLatestResponse(null);
    setSelectedDocumentIds([]);
  }

  async function handleSend() {
    const question = input.trim();
    if (!question || isAsking) {
      return;
    }
    if (!canRunQueries) {
      setErrorText("Your role does not allow document queries.");
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
    if (!canRunQueries) {
      setErrorText("Your role does not allow retrieval searches.");
      return;
    }
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
    if (!canRunQueries) {
      setErrorText("Your role does not allow reranking.");
      return;
    }
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
    if (!canRunQueries) {
      setErrorText("Your role does not allow query run cancellation.");
      return;
    }
    setErrorText("");
    try {
      const run = await cancelQueryRun(apiSettings, agentRunId);
      setActiveQueryRun(run);
      await refreshQueryRuns();
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleLogin(identityProvider?: string) {
    if (!authConfiguration) {
      setAuthenticationError("Authentication is not configured.");
      setAuthenticationState("error");
      return;
    }

    setAuthenticationError("");
    setAuthenticationState("loading");
    try {
      if (authConfiguration.mode === "supertokens") {
        if (!identityProvider) {
          throw new Error("Choose an identity provider.");
        }
        await startSuperTokensSocialLogin(identityProvider);
      } else if (authConfiguration.mode === "auth0") {
        await startAuth0Login(authConfiguration, identityProvider);
      }
    } catch (error) {
      setAuthenticationError(await getAuthenticationErrorMessage(error));
      setAuthenticationState("anonymous");
    }
  }

  async function handleEmailAuthentication(
    email: string,
    password: string,
    createAccount: boolean,
  ) {
    setAuthenticationError("");
    setAuthenticationState("loading");
    try {
      if (createAccount) {
        await signUpWithEmailPassword(email, password);
      } else {
        await signInWithEmailPassword(email, password);
      }
      setAuthenticationRetry((current) => current + 1);
    } catch (error) {
      setAuthenticationError(await getAuthenticationErrorMessage(error));
      setAuthenticationState("anonymous");
    }
  }

  async function handleForgotPassword(email: string) {
    setAuthenticationError("");
    try {
      await sendPasswordReset(email);
      setAuthenticationError("Password reset instructions were sent if the account exists.");
    } catch (error) {
      setAuthenticationError(await getAuthenticationErrorMessage(error));
    }
  }

  async function handlePasswordReset(password: string) {
    setAuthenticationError("");
    try {
      await submitPasswordReset(password);
      window.history.replaceState({}, document.title, "/");
      setAuthenticationError("Password updated. Sign in with your new password.");
      setAuthenticationState("anonymous");
    } catch (error) {
      setAuthenticationError(await getAuthenticationErrorMessage(error));
    }
  }

  async function handleLogout() {
    if (!authConfiguration) {
      return;
    }

    setAuthenticationError("");
    try {
      if (authConfiguration.mode === "supertokens") {
        await signOutSuperTokens();
        setAuthSession(null);
        setAuthenticationState("anonymous");
      } else if (authConfiguration.mode === "auth0") {
        await startAuth0Logout(authConfiguration);
      }
    } catch (error) {
      setAuthenticationError(await getAuthenticationErrorMessage(error));
      setErrorText("Sign out could not be completed. Please try again.");
    }
  }

  async function handleCreateTenant(name: string, slug: string) {
    setAuthenticationError("");
    try {
      await createTenant(apiSettings, name, slug);
      setAuthenticationRetry((current) => current + 1);
    } catch (error) {
      setAuthenticationError(error instanceof Error ? error.message : String(error));
    }
  }

  if (authenticationState !== "authenticated" || !authSession) {
    return (
      <LoginScreen
        authenticationState={authenticationState}
        configuration={authConfiguration}
        error={authenticationError}
        onLogin={(provider) => void handleLogin(provider)}
        onEmailAuthentication={(email, password, createAccount) =>
          void handleEmailAuthentication(email, password, createAccount)
        }
        onForgotPassword={(email) => void handleForgotPassword(email)}
        onPasswordReset={(password) => void handlePasswordReset(password)}
        onRetry={() => setAuthenticationRetry((current) => current + 1)}
      />
    );
  }

  if (authConfiguration?.mode === "supertokens" && !authSession.tenant_uuid) {
    return (
      <TenantOnboardingScreen
        email={identity.email}
        error={authenticationError}
        onCreate={(name, slug) => void handleCreateTenant(name, slug)}
        onLogout={() => void handleLogout()}
      />
    );
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

        <div className="user-context-card">
          <div className="user-context-avatar">
            <User size={17} />
          </div>
          <div className="user-context-copy">
            <strong>{identity.displayName}</strong>
            <span>{identity.email || authSession.roles.join(", ") || "Authenticated user"}</span>
            <small>
              {authSession.workspace_id
                ? `${authSession.tenant_id} / ${authSession.workspace_id}`
                : authSession.tenant_id}
            </small>
          </div>
          {authConfiguration?.mode === "auth0" || authConfiguration?.mode === "supertokens" ? (
            <button
              type="button"
              className="icon-button"
              aria-label="Sign out"
              title="Sign out"
              onClick={() => void handleLogout()}
            >
              <LogOut size={16} />
            </button>
          ) : null}
        </div>

        <nav className="nav-list" aria-label="Application sections">
          {canRunQueries ? (
            <NavButton activeView={activeView} view="chat" label="Chat" icon={<Bot size={17} />} onClick={setActiveView} />
          ) : null}
          {canReadDocuments ? (
            <NavButton activeView={activeView} view="documents" label="Documents" icon={<FileText size={17} />} onClick={setActiveView} />
          ) : null}
          {canRunQueries ? (
            <NavButton activeView={activeView} view="retrieval" label="Search Lab" icon={<FileSearch size={17} />} onClick={setActiveView} />
          ) : null}
          {canRunQueries ? (
            <NavButton activeView={activeView} view="runs" label="Query Runs" icon={<History size={17} />} onClick={setActiveView} />
          ) : null}
          {canAdministerTenant ? (
            <NavButton activeView={activeView} view="admin" label="Users" icon={<Users size={17} />} onClick={setActiveView} />
          ) : null}
          {canAdministerTenant ? (
            <NavButton activeView={activeView} view="models" label="AI Models" icon={<Cpu size={17} />} onClick={setActiveView} />
          ) : null}
          <NavButton activeView={activeView} view="operations" label="Operations" icon={<Activity size={17} />} onClick={setActiveView} />
        </nav>

        {canWriteDocuments ? <section className="upload-panel">
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
        </section> : null}

        {canReadDocuments ? <section className="document-panel">
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
                  disabled={document.status !== "ready"}
                  onClick={() => {
                    setSelectedDocumentIds([document.id]);
                    setLatestResponse(null);
                  }}
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
        </section> : null}

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
            {canWriteDocuments ? (
              <button
                type="button"
                className="ghost-button mobile-only"
                disabled={isUploading}
                onClick={() => fileInputRef.current?.click()}
              >
                {isUploading ? <Loader2 className="spin" size={17} /> : <UploadCloud size={17} />}
                <span>{isUploading ? "Indexing" : "Upload"}</span>
              </button>
            ) : null}
            {canReadDocuments ? (
              <button type="button" className="ghost-button" onClick={() => void refreshDocuments()}>
                <RefreshCw size={17} />
                <span>Documents</span>
              </button>
            ) : null}
            {canRunQueries ? (
              <button type="button" className="ghost-button" onClick={() => void refreshQueryRuns()}>
                <ClipboardList size={17} />
                <span>Runs</span>
              </button>
            ) : null}
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
            canDeleteDocuments={canDeleteDocuments}
            canWriteDocuments={canWriteDocuments}
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

        {activeView === "admin" && canAdministerTenant ? (
          <AdminUsersView
            authProvider={authConfiguration?.provider || "local"}
            isInviting={isInvitingTenantUser}
            isLoading={isLoadingTenantUsers}
            notice={adminNotice}
            noticeTone={adminNoticeTone}
            page={tenantUserPage}
            pageInfo={tenantUsersPage}
            users={tenantUsers}
            onInvite={handleInviteTenantUser}
            onPageChange={setTenantUserPage}
            onRefresh={() => void refreshTenantUsers()}
          />
        ) : null}

        {activeView === "models" && canAdministerTenant ? (
          <LLMProvidersView
            isLoading={isLoadingLLMProviders}
            isSaving={isSavingLLMProvider}
            notice={llmProviderNotice}
            noticeTone={llmProviderNoticeTone}
            pageInfo={llmProvidersPage}
            providers={llmProviders}
            validatingProviderId={validatingLLMProviderId}
            onDelete={handleDeleteLLMProvider}
            onRefresh={() => void refreshLLMProviders()}
            onSave={handleSaveLLMProvider}
            onValidate={handleValidateLLMProvider}
          />
        ) : null}

        {activeView === "operations" ? (
          <OperationsView
            authMode={authConfiguration?.mode || "local"}
            authSession={authSession}
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

function LoginScreen({
  authenticationState,
  configuration,
  error,
  onLogin,
  onEmailAuthentication,
  onForgotPassword,
  onPasswordReset,
  onRetry,
}: {
  authenticationState: AuthenticationState;
  configuration: AuthConfiguration | null;
  error: string;
  onLogin: (provider?: string) => void;
  onEmailAuthentication: (email: string, password: string, createAccount: boolean) => void;
  onForgotPassword: (email: string) => void;
  onPasswordReset: (password: string) => void;
  onRetry: () => void;
}) {
  const isLoading = authenticationState === "loading";
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [createAccount, setCreateAccount] = useState(false);
  const isPasswordReset = window.location.pathname.endsWith("/reset-password");
  const providers = configuration?.mode === "supertokens"
    ? configuration.social_providers
    : configuration?.identity_connections ?? [];
  const showAuthenticationForm =
    configuration?.mode === "supertokens" && authenticationState === "anonymous";

  return (
    <main className="login-shell">
      <header className="login-topbar">
        <div className="login-brand">
          <div className="brand-mark">
            <MessageSquare size={22} />
          </div>
          <div>
            <strong>Agentic RAG</strong>
            <span>Multi-tenant document intelligence</span>
          </div>
        </div>

        {(configuration?.mode === "auth0" || configuration?.mode === "supertokens") &&
        authenticationState === "anonymous" ? (
          <div className="login-provider-actions" aria-label="Social sign in">
            {providers.map((provider) => {
              const normalizedProvider = provider.toLowerCase();
              const displayName = normalizedProvider.includes("google")
                ? "Google"
                : normalizedProvider.includes("github")
                  ? "GitHub"
                  : provider
                      .split(/[-_]/)
                      .filter(Boolean)
                      .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
                      .join(" ");
              return (
                <button
                  type="button"
                  className="identity-button compact"
                  key={provider}
                  onClick={() => onLogin(provider)}
                >
                  {normalizedProvider.includes("github") ? (
                    <Code2 size={17} />
                  ) : normalizedProvider.includes("google") ? (
                    <Globe2 size={17} />
                  ) : (
                    <LogIn size={17} />
                  )}
                  <span>{displayName}</span>
                </button>
              );
            })}
          </div>
        ) : null}
      </header>

      <section className="login-panel" aria-labelledby="login-heading">
        <div className="login-heading">
          <div className="login-heading-icon">
            <Building2 size={22} />
          </div>
          <h1 id="login-heading">Sign in to your workspace</h1>
          <p>Use your organization identity to access authorized documents and conversations.</p>
        </div>

        {error ? (
          <div className="login-error" role="alert">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        ) : null}

        {isLoading ? (
          <div className="login-loading">
            <Loader2 className="spin" size={22} />
            <span>Checking your session</span>
          </div>
        ) : isPasswordReset && configuration?.mode === "supertokens" ? (
          <form
            className="login-actions"
            onSubmit={(event) => {
              event.preventDefault();
              onPasswordReset(password);
            }}
          >
            <label htmlFor="new-password">
              <span>New password</span>
              <input
                id="new-password"
                name="new-password"
                type="password"
                autoComplete="new-password"
                minLength={12}
                required
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </label>
            <button type="submit" className="primary-button login-primary">
              <KeyRound size={18} />
              <span>Set new password</span>
            </button>
          </form>
        ) : showAuthenticationForm ? (
          <form
            className="login-actions"
            onSubmit={(event) => {
              event.preventDefault();
              onEmailAuthentication(email, password, createAccount);
            }}
          >
            <label htmlFor="login-email">
              <span>Email</span>
              <input
                id="login-email"
                name="email"
                type="email"
                autoComplete="email"
                autoCapitalize="none"
                spellCheck={false}
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
            </label>
            <label htmlFor="login-password">
              <span>Password</span>
              <input
                id="login-password"
                name="password"
                type="password"
                autoComplete={createAccount ? "new-password" : "current-password"}
                minLength={createAccount ? 12 : 1}
                required
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </label>
            <button type="submit" className="primary-button login-primary">
              <KeyRound size={18} />
              <span>{createAccount ? "Create account" : "Sign in"}</span>
            </button>
            <div className="login-form-links">
              <button type="button" onClick={() => onForgotPassword(email)}>
                Forgot password
              </button>
              {configuration.public_tenant_signup_enabled ? (
                <button type="button" onClick={() => setCreateAccount((current) => !current)}>
                  {createAccount ? "Use existing account" : "Create a company account"}
                </button>
              ) : null}
            </div>
            <p className="login-password-note">
              Invited members must sign in with the email address that received the invitation.
            </p>
          </form>
        ) : configuration?.mode === "auth0" && authenticationState === "anonymous" ? (
          <div className="login-actions">
            <button type="button" className="primary-button login-primary" onClick={() => onLogin()}>
              <KeyRound size={18} />
              <span>Continue with email or username</span>
            </button>
            <p className="login-password-note">
              Invited members accept their organization invitation before signing in.
            </p>
          </div>
        ) : (
          <button type="button" className="ghost-button login-retry" onClick={onRetry}>
            <RefreshCw size={17} />
            <span>Retry</span>
          </button>
        )}

        <div className="login-security-note">
          <ShieldCheck size={17} />
          <span>Tenant and document permissions are verified by the API on every request.</span>
        </div>
      </section>
    </main>
  );
}

function TenantOnboardingScreen({
  email,
  error,
  onCreate,
  onLogout,
}: {
  email: string;
  error: string;
  onCreate: (name: string, slug: string) => void;
  onLogout: () => void;
}) {
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");

  return (
    <main className="login-shell">
      <header className="login-topbar">
        <div className="login-brand">
          <div className="brand-mark"><MessageSquare size={22} /></div>
          <div><strong>Agentic RAG</strong><span>{email}</span></div>
        </div>
        <button type="button" className="ghost-button" onClick={onLogout}>
          <LogOut size={17} /><span>Sign out</span>
        </button>
      </header>
      <section className="login-panel" aria-labelledby="company-heading">
        <div className="login-heading">
          <div className="login-heading-icon"><Building2 size={22} /></div>
          <h1 id="company-heading">Create your company</h1>
          <p>Your first knowledge department and owner access are created automatically.</p>
        </div>
        {error ? <div className="login-error" role="alert"><AlertCircle size={18} /><span>{error}</span></div> : null}
        <form className="login-actions" onSubmit={(event) => { event.preventDefault(); onCreate(name.trim(), slug.trim()); }}>
          <label>Company name<input required minLength={2} value={name} onChange={(event) => {
            const nextName = event.target.value;
            setName(nextName);
            setSlug(nextName.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""));
          }} /></label>
          <label>Company slug<input required minLength={2} value={slug} onChange={(event) => setSlug(event.target.value.toLowerCase().replace(/[^a-z0-9-]/g, ""))} /></label>
          <button type="submit" className="primary-button login-primary" disabled={!name.trim() || !slug.trim()}>
            <Building2 size={18} /><span>Create company</span>
          </button>
        </form>
      </section>
    </main>
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
  if (view === "admin") {
    return "Users";
  }
  if (view === "models") {
    return "AI Models";
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
  if (view === "admin") {
    return "Tenant membership, invitations, and roles";
  }
  if (view === "models") {
    return "Tenant chat and embedding provider routes";
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
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const textarea = inputRef.current;
    if (!textarea) {
      return;
    }

    textarea.style.height = "auto";

    const computedStyle = window.getComputedStyle(textarea);
    const parsedLineHeight = Number.parseFloat(computedStyle.lineHeight);
    const lineHeight = Number.isFinite(parsedLineHeight) ? parsedLineHeight : 24;
    const maxHeight = Math.ceil(lineHeight * 5);
    const nextHeight = Math.min(textarea.scrollHeight, maxHeight);

    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [input]);

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
              ref={inputRef}
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
              <p>{latestContext.length} latest answer context chunks</p>
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
  canDeleteDocuments,
  canWriteDocuments,
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
  canDeleteDocuments: boolean;
  canWriteDocuments: boolean;
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
                        {canDeleteDocuments ? (
                          <button type="button" className="icon-button danger" onClick={() => onDeleteDocument(document.id)} title="Delete document">
                            <Trash2 size={16} />
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
          <button type="button" className="primary-button inline" disabled={isCreatingSource || !canWriteDocuments} onClick={onCreateSource}>
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
              <button type="button" className="primary-button inline" disabled={isSavingDocument || !canWriteDocuments} onClick={onSaveDocument}>
                {isSavingDocument ? <Loader2 className="spin" size={17} /> : <Save size={17} />}
                <span>Save</span>
              </button>
              {canWriteDocuments ? (
                <button type="button" className="ghost-button" onClick={() => onRestoreDocument(activeDocument.id)}>
                  <RotateCcw size={17} />
                  <span>Restore</span>
                </button>
              ) : null}
              {canDeleteDocuments ? (
                <button type="button" className="ghost-button danger" onClick={() => onDeleteDocument(activeDocument.id)}>
                  <Trash2 size={17} />
                  <span>Delete</span>
                </button>
              ) : null}
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

function AdminUsersView({
  authProvider,
  isInviting,
  isLoading,
  notice,
  noticeTone,
  page,
  pageInfo,
  users,
  onInvite,
  onPageChange,
  onRefresh,
}: {
  authProvider: string;
  isInviting: boolean;
  isLoading: boolean;
  notice: string;
  noticeTone: "success" | "danger";
  page: number;
  pageInfo: typeof EMPTY_PAGE;
  users: TenantUser[];
  onInvite: (invitation: UserInvitationRequest) => Promise<boolean>;
  onPageChange: (page: number) => void;
  onRefresh: () => void;
}) {
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [role, setRole] = useState<TenantUserRole>("user");
  const [workspaceId, setWorkspaceId] = useState("");
  const invitationsEnabled = authProvider === "auth0" || authProvider === "supertokens";

  async function submitInvitation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!email.trim() || !invitationsEnabled || isInviting) {
      return;
    }

    const invitationSent = await onInvite({
      email: email.trim(),
      display_name: displayName.trim() || null,
      role,
      workspace_id: workspaceId.trim() || null,
    });
    if (invitationSent) {
      setEmail("");
      setDisplayName("");
      setWorkspaceId("");
      setRole("user");
    }
  }

  return (
    <div className="content-grid split admin-users-grid">
      <section className="panel invite-user-panel">
        <div className="panel-title-row">
          <div>
            <h3>Invite User</h3>
            <p>Create a tenant membership and send secure account setup email.</p>
          </div>
          <div className="panel-icon" aria-hidden="true">
            <UserPlus size={19} />
          </div>
        </div>

        {notice ? (
          <div className={`admin-notice ${noticeTone}`} role={noticeTone === "danger" ? "alert" : "status"}>
            {noticeTone === "danger" ? <AlertCircle size={17} /> : <Check size={17} />}
            <span>{notice}</span>
          </div>
        ) : null}

        {!invitationsEnabled ? (
          <div className="admin-notice warning">
            <AlertCircle size={17} />
            <span>Enable SuperTokens or Auth0 to send secure invitations.</span>
          </div>
        ) : null}

        <form className="invite-user-form" onSubmit={(event) => void submitInvitation(event)}>
          <label className="stacked-field">
            Email address
            <input
              type="email"
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="name@company.com"
              required
            />
          </label>
          <label className="stacked-field">
            Display name
            <input
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              placeholder="Optional"
            />
          </label>
          <div className="form-grid">
            <label>
              Role
              <select value={role} onChange={(event) => setRole(event.target.value as TenantUserRole)}>
                <option value="viewer">Viewer</option>
                <option value="user">User</option>
                <option value="admin">Admin</option>
              </select>
            </label>
            <label>
              Workspace
              <input
                value={workspaceId}
                onChange={(event) => setWorkspaceId(event.target.value)}
                placeholder="All workspaces"
              />
            </label>
          </div>
          <button
            type="submit"
            className="primary-button inline invite-submit"
            disabled={!invitationsEnabled || !email.trim() || isInviting}
          >
            {isInviting ? <Loader2 className="spin" size={17} /> : <UserPlus size={17} />}
            <span>{isInviting ? "Sending invitation" : "Send invitation"}</span>
          </button>
        </form>

        <div className="credential-boundary-note">
          <ShieldCheck size={18} />
          <p>
            Passwords and social identities are managed by the authentication service. Agentic RAG stores company membership, roles, and department access.
          </p>
        </div>
      </section>

      <section className="panel tenant-users-panel">
        <div className="panel-title-row">
          <div>
            <h3>Tenant Users</h3>
            <p>{pageInfo.total} members in this tenant</p>
          </div>
          <button type="button" className="ghost-button" onClick={onRefresh}>
            {isLoading ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
            <span>Refresh</span>
          </button>
        </div>

        <div className="table-scroll">
          <table className="data-table tenant-users-table">
            <thead>
              <tr>
                <th>User</th>
                <th>Role</th>
                <th>Status</th>
                <th>Workspace</th>
                <th>Added</th>
              </tr>
            </thead>
            <tbody>
              {users.length === 0 ? (
                <tr>
                  <td colSpan={5}>{isLoading ? "Loading tenant users..." : "No tenant users found."}</td>
                </tr>
              ) : (
                users.map((tenantUser) => (
                  <tr key={tenantUser.id}>
                    <td>
                      <strong>{tenantUser.display_name || tenantUser.email || tenantUser.external_subject}</strong>
                      <small>{tenantUser.email || compactIdentifier(tenantUser.external_subject)}</small>
                    </td>
                    <td>{tenantUser.roles.join(", ") || "unassigned"}</td>
                    <td><StatusBadge value={tenantUser.status} /></td>
                    <td>{tenantUser.workspace_id || "All"}</td>
                    <td>{formatDate(tenantUser.created_at)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        <PaginationControls page={page} pageInfo={pageInfo} onPageChange={onPageChange} />
      </section>
    </div>
  );
}

function LLMProvidersView({
  isLoading,
  isSaving,
  notice,
  noticeTone,
  pageInfo,
  providers,
  validatingProviderId,
  onDelete,
  onRefresh,
  onSave,
  onValidate,
}: {
  isLoading: boolean;
  isSaving: boolean;
  notice: string;
  noticeTone: "success" | "danger";
  pageInfo: typeof EMPTY_PAGE;
  providers: LLMProvider[];
  validatingProviderId: string;
  onDelete: (providerId: string) => Promise<boolean>;
  onRefresh: () => void;
  onSave: (
    providerId: string | null,
    request: LLMProviderCreateRequest | LLMProviderUpdateRequest,
  ) => Promise<boolean>;
  onValidate: (
    providerId: string,
    capability: "chat" | "embedding",
  ) => Promise<LLMProviderValidationResponse | null>;
}) {
  const [selectedProviderId, setSelectedProviderId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [providerType, setProviderType] = useState<LLMProviderType>("google");
  const [chatModel, setChatModel] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [embeddingDimension, setEmbeddingDimension] = useState("768");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [clearApiKey, setClearApiKey] = useState(false);
  const [configDraft, setConfigDraft] = useState(
    formatJson({ temperature: 0.1, max_tokens: 700, timeout_seconds: 30 }),
  );
  const [isActive, setIsActive] = useState(true);
  const [isDefaultChat, setIsDefaultChat] = useState(false);
  const [isDefaultEmbedding, setIsDefaultEmbedding] = useState(false);
  const [formError, setFormError] = useState("");

  const selectedProvider = useMemo(() => {
    return providers.find((provider) => provider.id === selectedProviderId) || null;
  }, [providers, selectedProviderId]);

  useEffect(() => {
    if (!selectedProvider) {
      return;
    }

    setName(selectedProvider.name);
    setProviderType(selectedProvider.provider_type);
    setChatModel(selectedProvider.chat_model || "");
    setEmbeddingModel(selectedProvider.embedding_model || "");
    setEmbeddingDimension(
      selectedProvider.embedding_dimension
        ? String(selectedProvider.embedding_dimension)
        : "768",
    );
    setBaseUrl(selectedProvider.base_url || "");
    setApiKey("");
    setClearApiKey(false);
    setConfigDraft(formatJson(selectedProvider.config));
    setIsActive(selectedProvider.is_active);
    setIsDefaultChat(selectedProvider.is_default_chat);
    setIsDefaultEmbedding(selectedProvider.is_default_embedding);
    setFormError("");
  }, [selectedProvider]);

  function startNewProvider() {
    setSelectedProviderId(null);
    setName("");
    setProviderType("google");
    setChatModel("");
    setEmbeddingModel("");
    setEmbeddingDimension("768");
    setBaseUrl("");
    setApiKey("");
    setClearApiKey(false);
    setConfigDraft(
      formatJson({ temperature: 0.1, max_tokens: 700, timeout_seconds: 30 }),
    );
    setIsActive(true);
    setIsDefaultChat(false);
    setIsDefaultEmbedding(false);
    setFormError("");
  }

  async function submitProvider(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError("");

    if (!name.trim()) {
      setFormError("Provider name is required.");
      return;
    }
    if (!chatModel.trim() && !embeddingModel.trim()) {
      setFormError("Configure at least one chat or embedding model.");
      return;
    }

    const configResult = parseJsonObject(configDraft, "Provider config");
    if (!configResult.ok) {
      setFormError(configResult.error);
      return;
    }

    let parsedEmbeddingDimension: number | null = null;
    if (embeddingModel.trim()) {
      parsedEmbeddingDimension = Number.parseInt(embeddingDimension, 10);
      if (!Number.isInteger(parsedEmbeddingDimension) || parsedEmbeddingDimension < 1) {
        setFormError("Embedding dimension must be a positive integer.");
        return;
      }
    }

    if (selectedProvider) {
      const updateRequest: LLMProviderUpdateRequest = {
        name: name.trim(),
        chat_model: chatModel.trim() || null,
        embedding_model: embeddingModel.trim() || null,
        embedding_dimension: parsedEmbeddingDimension,
        base_url: baseUrl.trim() || null,
        config: configResult.value,
        is_active: isActive,
        is_default_chat: Boolean(chatModel.trim()) && isActive && isDefaultChat,
        is_default_embedding:
          Boolean(embeddingModel.trim()) && isActive && isDefaultEmbedding,
        clear_api_key: clearApiKey,
      };
      if (apiKey.trim()) {
        updateRequest.api_key = apiKey.trim();
      }
      const saved = await onSave(selectedProvider.id, updateRequest);
      if (saved) {
        setApiKey("");
        setClearApiKey(false);
      }
      return;
    }

    const createRequest: LLMProviderCreateRequest = {
      name: name.trim(),
      provider_type: providerType,
      chat_model: chatModel.trim() || null,
      embedding_model: embeddingModel.trim() || null,
      embedding_dimension: parsedEmbeddingDimension,
      base_url: baseUrl.trim() || null,
      config: configResult.value,
      is_active: isActive,
      is_default_chat: Boolean(chatModel.trim()) && isActive && isDefaultChat,
      is_default_embedding:
        Boolean(embeddingModel.trim()) && isActive && isDefaultEmbedding,
    };
    if (apiKey.trim()) {
      createRequest.api_key = apiKey.trim();
    }
    await onSave(null, createRequest);
  }

  async function removeSelectedProvider() {
    if (!selectedProvider) {
      return;
    }
    const confirmed = window.confirm(
      `Remove ${selectedProvider.name}? Existing query-run history is retained.`,
    );
    if (!confirmed) {
      return;
    }

    const removed = await onDelete(selectedProvider.id);
    if (removed) {
      startNewProvider();
    }
  }

  return (
    <div className="content-grid split llm-providers-grid">
      <section className="panel llm-provider-inventory">
        <div className="panel-title-row">
          <div>
            <h3>Provider Inventory</h3>
            <p>{pageInfo.total} tenant provider{pageInfo.total === 1 ? "" : "s"}</p>
          </div>
          <div className="button-row compact">
            <button type="button" className="icon-button" title="Refresh providers" onClick={onRefresh}>
              {isLoading ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
            </button>
            <button type="button" className="primary-button inline" onClick={startNewProvider}>
              <Plus size={16} />
              <span>New</span>
            </button>
          </div>
        </div>

        <div className="llm-provider-list">
          {providers.length === 0 ? (
            <div className="empty-state compact">
              {isLoading ? "Loading providers..." : "No tenant providers configured."}
            </div>
          ) : (
            providers.map((provider) => (
              <button
                type="button"
                className={`llm-provider-item ${selectedProviderId === provider.id ? "active" : ""}`}
                key={provider.id}
                onClick={() => setSelectedProviderId(provider.id)}
              >
                <div className="llm-provider-item-heading">
                  <span className="provider-glyph"><Cpu size={16} /></span>
                  <strong>{provider.name}</strong>
                  <StatusBadge value={provider.is_active ? "active" : "inactive"} />
                </div>
                <span>{provider.provider_type.replace("_", " ")}</span>
                <small>{provider.chat_model || provider.embedding_model || "No model"}</small>
                <div className="provider-route-badges">
                  {provider.is_default_chat ? <span>Chat default</span> : null}
                  {provider.is_default_embedding ? <span>Embedding default</span> : null}
                  {provider.has_api_key ? <span>Key stored</span> : null}
                </div>
              </button>
            ))
          )}
        </div>
      </section>

      <section className="panel llm-provider-editor">
        <div className="panel-title-row">
          <div>
            <h3>{selectedProvider ? "Edit Provider" : "Add Provider"}</h3>
            <p>{selectedProvider ? "Update routing and credential settings" : "Configure a tenant model route"}</p>
          </div>
          {selectedProvider ? (
            <button
              type="button"
              className="icon-button danger"
              title="Remove provider"
              aria-label="Remove provider"
              disabled={isSaving}
              onClick={() => void removeSelectedProvider()}
            >
              <Trash2 size={17} />
            </button>
          ) : null}
        </div>

        {notice ? (
          <div className={`admin-notice ${noticeTone}`} role={noticeTone === "danger" ? "alert" : "status"}>
            {noticeTone === "danger" ? <AlertCircle size={17} /> : <Check size={17} />}
            <span>{notice}</span>
          </div>
        ) : null}
        {formError ? (
          <div className="admin-notice danger" role="alert">
            <AlertCircle size={17} />
            <span>{formError}</span>
          </div>
        ) : null}

        <form className="llm-provider-form" onSubmit={(event) => void submitProvider(event)}>
          <div className="form-grid">
            <label>
              Provider name
              <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Production Gemini" required />
            </label>
            <label>
              Provider type
              <select
                value={providerType}
                disabled={Boolean(selectedProvider)}
                onChange={(event) => setProviderType(event.target.value as LLMProviderType)}
              >
                <option value="google">Google Gemini</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="azure">Azure OpenAI</option>
                <option value="ollama">Ollama</option>
                <option value="litellm">LiteLLM Proxy</option>
                <option value="openai_compatible">OpenAI Compatible</option>
              </select>
            </label>
          </div>

          <div className="form-grid">
            <label>
              Chat model
              <input value={chatModel} onChange={(event) => setChatModel(event.target.value)} placeholder="gemini-2.5-flash" />
            </label>
            <label>
              Embedding model
              <input value={embeddingModel} onChange={(event) => setEmbeddingModel(event.target.value)} placeholder="gemini-embedding-001" />
            </label>
          </div>

          <div className="form-grid">
            <label>
              Embedding dimension
              <input
                type="number"
                min={1}
                value={embeddingDimension}
                disabled={!embeddingModel.trim()}
                onChange={(event) => setEmbeddingDimension(event.target.value)}
              />
            </label>
            <label>
              Base URL
              <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} placeholder="Optional provider or proxy URL" />
            </label>
          </div>

          <label className="stacked-field">
            API key
            <input
              type="password"
              autoComplete="new-password"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder={selectedProvider?.has_api_key ? "Stored - enter to replace" : "Optional for local providers"}
            />
          </label>
          {selectedProvider?.has_api_key ? (
            <label className="checkbox-field">
              <input type="checkbox" checked={clearApiKey} onChange={(event) => setClearApiKey(event.target.checked)} />
              <span>Remove stored API key</span>
            </label>
          ) : null}

          <label className="stacked-field">
            Provider config JSON
            <textarea rows={6} value={configDraft} onChange={(event) => setConfigDraft(event.target.value)} spellCheck={false} />
          </label>

          <div className="provider-toggle-grid">
            <label className="checkbox-field">
              <input type="checkbox" checked={isActive} onChange={(event) => setIsActive(event.target.checked)} />
              <span>Active</span>
            </label>
            <label className="checkbox-field">
              <input
                type="checkbox"
                checked={isDefaultChat}
                disabled={!isActive || !chatModel.trim()}
                onChange={(event) => setIsDefaultChat(event.target.checked)}
              />
              <span>Default chat</span>
            </label>
            <label className="checkbox-field">
              <input
                type="checkbox"
                checked={isDefaultEmbedding}
                disabled={!isActive || !embeddingModel.trim()}
                onChange={(event) => setIsDefaultEmbedding(event.target.checked)}
              />
              <span>Default embedding</span>
            </label>
          </div>

          <div className="provider-form-actions">
            <button type="submit" className="primary-button inline" disabled={isSaving}>
              {isSaving ? <Loader2 className="spin" size={17} /> : <Save size={17} />}
              <span>{isSaving ? "Saving" : "Save provider"}</span>
            </button>
            {selectedProvider?.chat_model ? (
              <button
                type="button"
                className="ghost-button"
                disabled={Boolean(validatingProviderId)}
                onClick={() => void onValidate(selectedProvider.id, "chat")}
              >
                {validatingProviderId === selectedProvider.id ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
                <span>Test chat</span>
              </button>
            ) : null}
            {selectedProvider?.embedding_model ? (
              <button
                type="button"
                className="ghost-button"
                disabled={Boolean(validatingProviderId)}
                onClick={() => void onValidate(selectedProvider.id, "embedding")}
              >
                {validatingProviderId === selectedProvider.id ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
                <span>Test embedding</span>
              </button>
            ) : null}
          </div>
        </form>
      </section>
    </div>
  );
}

function OperationsView({
  authMode,
  authSession,
  health,
  healthState,
  settings,
  onHealthCheck,
  onSettingChange,
}: {
  authMode: AuthConfiguration["mode"];
  authSession: AuthSession;
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
        <div className="detail-grid">
          <Metric label="Auth" value={authSession.auth_provider} />
          <Metric label="Roles" value={authSession.roles.join(", ") || "none"} />
          <Metric label="ACL version" value={String(authSession.acl_version)} />
        </div>
        <label className="stacked-field">
          API URL
          <input value={settings.apiBaseUrl} onChange={(event) => onSettingChange("apiBaseUrl", event.target.value)} />
        </label>
        {authMode === "local" ? (
          <label className="stacked-field">
            Local development token
            <input type="password" value={settings.authToken} onChange={(event) => onSettingChange("authToken", event.target.value)} />
          </label>
        ) : null}
        <label className="stacked-field">
          Tenant
          <input readOnly value={authSession.tenant_id} />
        </label>
        <label className="stacked-field">
          Workspace
          <input
            readOnly={Boolean(authSession.workspace_id)}
            value={settings.workspaceId}
            onChange={(event) => onSettingChange("workspaceId", event.target.value)}
            placeholder="All authorized workspaces"
          />
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
