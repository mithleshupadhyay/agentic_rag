export type JsonObject = Record<string, unknown>;

export type PageResponse = {
  page: number;
  size: number;
  total: number;
};

export type HealthResponse = {
  service: string;
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  dependencies?: Record<
    string,
    {
      name?: string;
      status?: string;
      detail?: string | null;
      latency_ms?: number | null;
    }
  >;
};

export type DocumentStatus = "queued" | "parsing" | "indexing" | "ready" | "failed" | "deleted";
export type DocumentSourceType = "upload" | "s3" | "url" | "connector";
export type ClassificationLevel = "public" | "internal" | "confidential" | "restricted";

export type AclPolicy = {
  visibility: "private" | "group" | "tenant" | "public";
  allowed_user_ids: string[];
  allowed_group_ids: string[];
  allowed_roles: string[];
  denied_user_ids: string[];
  denied_group_ids: string[];
  acl_version: number;
};

export type DocumentRead = {
  id: string;
  tenant_id: string;
  workspace_id: string | null;
  source_type: DocumentSourceType;
  source_uri: string | null;
  object_key?: string | null;
  title: string | null;
  file_name: string | null;
  mime_type: string | null;
  byte_size: number | null;
  content_hash?: string | null;
  status: DocumentStatus;
  owner_user_id?: string | null;
  acl_version: number;
  classification_level: ClassificationLevel;
  metadata: JsonObject;
  created_by?: string | null;
  created_at: string;
  updated_at: string;
  is_deleted?: boolean;
  deleted_at?: string | null;
};

export type DocumentListItem = {
  id: string;
  tenant_id: string;
  workspace_id: string | null;
  title: string | null;
  file_name: string | null;
  source_type: DocumentSourceType;
  status: DocumentStatus;
  classification_level: ClassificationLevel;
  created_at: string;
  updated_at: string;
};

export type DocumentSearchRequest = {
  page: {
    page: number;
    size: number;
  };
  workspace_id?: string | null;
  source_type?: DocumentSourceType | null;
  status?: DocumentStatus | null;
  owner_user_id?: string | null;
  tags?: string[];
  metadata_filters?: JsonObject;
};

export type DocumentCreateRequest = {
  workspace_id?: string | null;
  source_type: DocumentSourceType;
  source_uri?: string | null;
  title?: string | null;
  file?: {
    file_name?: string | null;
    mime_type?: string | null;
    byte_size?: number | null;
    content_hash?: string | null;
  } | null;
  metadata: JsonObject;
  acl: AclPolicy;
  idempotency_key?: string | null;
};

export type DocumentSearchResponse = {
  items: DocumentListItem[];
  page: PageResponse;
};

export type DocumentUpdateRequest = {
  title?: string | null;
  metadata?: JsonObject | null;
  acl?: AclPolicy | null;
  classification_level?: ClassificationLevel | null;
};

export type DocumentActionResponse = {
  id: string;
  status: string;
};

export type DocumentUploadResponse = {
  document: DocumentRead;
  ingestion_job_id: string;
  ingestion_status: string;
  ingestion_stage: string;
  bucket: string;
  object_key: string;
  content_hash: string;
  byte_size: number;
};

export type IngestionJobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type IngestionJobRead = {
  id: string;
  tenant_id?: string;
  workspace_id?: string | null;
  document_id: string | null;
  source_type?: DocumentSourceType;
  source_uri?: string | null;
  object_key?: string | null;
  status: IngestionJobStatus;
  current_stage: string;
  retry_count: number;
  max_retries: number;
  error_type: string | null;
  error_message: string | null;
  locked_by?: string | null;
  locked_at?: string | null;
  lease_expires_at?: string | null;
  next_retry_at: string | null;
  idempotency_key?: string | null;
  metadata?: JsonObject;
  created_by?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  created_at: string;
  updated_at: string;
};

export type IngestionJobSearchResponse = {
  items: IngestionJobRead[];
  page: PageResponse;
};

export type Citation = {
  document_id?: string;
  chunk_id?: string;
  title?: string | null;
  source_uri?: string | null;
  page_number?: number | null;
  section_path?: string | null;
  score?: number | null;
  quote?: string | null;
};

export type CandidateChunk = {
  chunk_id: string;
  document_id: string;
  content?: string | null;
  score: number;
  source: string;
  metadata: JsonObject;
  citation?: Citation | null;
};

export type ContextChunk = {
  document_id: string;
  chunk_id: string;
  content: string;
  token_count: number;
  citation: Citation;
  metadata?: JsonObject;
};

export type RetrievalFilters = {
  workspace_id?: string | null;
  document_ids?: string[];
  source_types?: string[];
  tags?: string[];
  metadata?: JsonObject;
  date_range?: JsonObject;
};

export type RetrievalStrategy = "bm25" | "vector" | "hybrid";

export type RetrievalRequest = {
  query: string;
  filters: RetrievalFilters;
  limit: number;
  min_similarity?: number;
  deadline_ms?: number;
};

export type RetrievalResponse = {
  strategy: string;
  candidates: CandidateChunk[];
  latency_ms: number;
};

export type RerankResponse = {
  chunks: CandidateChunk[];
  latency_ms: number;
};

export type QueryHistoryMessage = {
  role: "user" | "assistant";
  content: string;
};

export type QueryRequest = {
  query: string;
  workspace_id?: string;
  conversation_id?: string;
  filters: RetrievalFilters;
  history: QueryHistoryMessage[];
  retrieval_strategy: RetrievalStrategy;
  retrieval_limit: number;
  max_context_chunks: number;
  max_context_tokens: number;
};

export type QueryResponse = {
  agent_run_id: string;
  answer: string;
  citations: Citation[];
  candidates: CandidateChunk[];
  context: ContextChunk[];
  context_token_count: number;
  confidence_score: number;
  retrieval_strategy: string;
  latency_ms: number;
  synthesis_enabled: boolean;
  llm_provider?: string | null;
  llm_model?: string | null;
  llm_input_tokens?: number;
  llm_output_tokens?: number;
  llm_cost_estimate?: number;
  synthesis_error?: string | null;
  verification_status?: string;
  verification_reason?: string | null;
  cache_lookup_status?: string;
  cache_write_status?: string;
  cache_ttl_seconds?: number | null;
};

export type QueryRunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type QueryRunListItem = {
  agent_run_id: string;
  status: QueryRunStatus;
  workspace_id: string | null;
  user_id: string;
  request_id: string | null;
  conversation_id?: string | null;
  query: string;
  retrieval_strategy: string | null;
  synthesis_enabled: boolean;
  llm_provider: string | null;
  llm_model: string | null;
  verification_status: string;
  verification_reason: string | null;
  latency_ms: number | null;
  created_at: string;
  completed_at: string | null;
};

export type QueryRunRead = QueryRunListItem & {
  tenant_id: string;
  filters: RetrievalFilters;
  retrieval_limit: number;
  max_context_chunks: number;
  max_context_tokens: number;
  answer: string | null;
  citations: Citation[];
  context_token_count: number;
  confidence_score: number | null;
  llm_input_tokens: number;
  llm_output_tokens: number;
  llm_cost_estimate: number;
  error_type: string | null;
  error_message: string | null;
  response_payload: JsonObject;
  updated_at: string;
  response?: QueryResponse | null;
};

export type QueryRunSearchResponse = {
  items: QueryRunListItem[];
  page: PageResponse;
};

export type ApiClientSettings = {
  apiBaseUrl: string;
  authToken: string;
};

export class ApiError extends Error {
  status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function buildSearchParams(values: Record<string, string | number | null | undefined>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    if (value === null || value === undefined || value === "") {
      continue;
    }
    params.set(key, String(value));
  }

  const query = params.toString();
  if (!query) {
    return "";
  }

  return `?${query}`;
}

export async function requestJson<T>(
  settings: ApiClientSettings,
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const baseUrl = settings.apiBaseUrl.trim().replace(/\/$/, "");
  if (!baseUrl) {
    throw new ApiError("API URL is required.");
  }

  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (settings.authToken.trim()) {
    headers.set("Authorization", `Bearer ${settings.authToken.trim()}`);
  }

  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`, {
      ...init,
      headers,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new ApiError(`API request failed: ${message}`);
  }

  if (!response.ok) {
    let detail = response.statusText || "Request failed";
    try {
      const payload = (await response.json()) as unknown;
      if (payload && typeof payload === "object" && "detail" in payload) {
        const detailValue = (payload as { detail?: unknown }).detail;
        detail = typeof detailValue === "string" ? detailValue : JSON.stringify(detailValue);
      } else {
        detail = JSON.stringify(payload);
      }
    } catch {
      const text = await response.text();
      if (text.trim()) {
        detail = text.trim().slice(0, 1000);
      }
    }
    throw new ApiError(`API returned ${response.status}: ${detail}`, response.status);
  }

  try {
    return (await response.json()) as T;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new ApiError(`API returned invalid JSON: ${message}`, response.status);
  }
}

export async function checkReadiness(settings: ApiClientSettings): Promise<HealthResponse> {
  return requestJson<HealthResponse>(settings, "/readiness");
}

export async function uploadDocument(
  settings: ApiClientSettings,
  file: File,
  workspaceId: string,
  title: string,
  metadata: JsonObject,
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("workspace_id", workspaceId.trim());
  formData.append("title", title.trim() || file.name);
  formData.append("metadata_json", JSON.stringify(metadata));
  formData.append("idempotency_key", `frontend-upload-${crypto.randomUUID()}`);

  return requestJson<DocumentUploadResponse>(settings, "/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export async function createDocument(
  settings: ApiClientSettings,
  request: DocumentCreateRequest,
): Promise<DocumentRead> {
  return requestJson<DocumentRead>(settings, "/documents", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
}

export async function listDocuments(
  settings: ApiClientSettings,
  page: number,
  size: number,
): Promise<DocumentSearchResponse> {
  const query = buildSearchParams({ page, size });
  return requestJson<DocumentSearchResponse>(settings, `/documents${query}`);
}

export async function searchDocuments(
  settings: ApiClientSettings,
  request: DocumentSearchRequest,
): Promise<DocumentSearchResponse> {
  return requestJson<DocumentSearchResponse>(settings, "/documents/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
}

export async function getDocument(
  settings: ApiClientSettings,
  documentId: string,
): Promise<DocumentRead> {
  return requestJson<DocumentRead>(settings, `/documents/${documentId}`);
}

export async function updateDocument(
  settings: ApiClientSettings,
  documentId: string,
  request: DocumentUpdateRequest,
): Promise<DocumentRead> {
  return requestJson<DocumentRead>(settings, `/documents/${documentId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
}

export async function deleteDocument(
  settings: ApiClientSettings,
  documentId: string,
): Promise<DocumentActionResponse> {
  return requestJson<DocumentActionResponse>(settings, `/documents/${documentId}`, {
    method: "DELETE",
  });
}

export async function restoreDocument(
  settings: ApiClientSettings,
  documentId: string,
): Promise<DocumentRead> {
  return requestJson<DocumentRead>(settings, `/documents/${documentId}/restore`, {
    method: "POST",
  });
}

export async function listIngestionJobs(
  settings: ApiClientSettings,
  documentId: string,
  page: number,
  size: number,
  status?: IngestionJobStatus | "",
  currentStage?: string,
): Promise<IngestionJobSearchResponse> {
  const query = buildSearchParams({
    page,
    size,
    status,
    current_stage: currentStage,
  });
  return requestJson<IngestionJobSearchResponse>(
    settings,
    `/documents/${documentId}/ingestion-jobs${query}`,
  );
}

export async function getIngestionJob(
  settings: ApiClientSettings,
  documentId: string,
  jobId: string,
): Promise<IngestionJobRead> {
  return requestJson<IngestionJobRead>(
    settings,
    `/documents/${documentId}/ingestion-jobs/${jobId}`,
  );
}

export async function runQuery(
  settings: ApiClientSettings,
  request: QueryRequest,
): Promise<QueryResponse> {
  return requestJson<QueryResponse>(settings, "/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
}

export async function listQueryRuns(
  settings: ApiClientSettings,
  values: {
    page: number;
    size: number;
    workspace_id?: string;
    status?: QueryRunStatus | "";
    verification_status?: string;
  },
): Promise<QueryRunSearchResponse> {
  const query = buildSearchParams(values);
  return requestJson<QueryRunSearchResponse>(settings, `/query${query}`);
}

export async function getQueryRun(
  settings: ApiClientSettings,
  agentRunId: string,
): Promise<QueryRunRead> {
  return requestJson<QueryRunRead>(settings, `/query/${agentRunId}`);
}

export async function cancelQueryRun(
  settings: ApiClientSettings,
  agentRunId: string,
): Promise<QueryRunRead> {
  return requestJson<QueryRunRead>(settings, `/query/${agentRunId}/cancel`, {
    method: "POST",
  });
}

export async function runRetrieval(
  settings: ApiClientSettings,
  strategy: RetrievalStrategy,
  request: RetrievalRequest,
): Promise<RetrievalResponse> {
  const endpointByStrategy: Record<RetrievalStrategy, string> = {
    bm25: "/retrieval/bm25-search",
    vector: "/retrieval/vector-search",
    hybrid: "/retrieval/hybrid-search",
  };
  return requestJson<RetrievalResponse>(settings, endpointByStrategy[strategy], {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
}

export async function rerankCandidates(
  settings: ApiClientSettings,
  query: string,
  candidates: CandidateChunk[],
  topK: number,
): Promise<RerankResponse> {
  return requestJson<RerankResponse>(settings, "/retrieval/rerank", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      candidates,
      top_k: topK,
    }),
  });
}
