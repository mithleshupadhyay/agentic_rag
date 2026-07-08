export type HealthResponse = {
  service: string;
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  dependencies?: Record<string, unknown>;
};

export type DocumentRead = {
  id: string;
  tenant_id: string;
  workspace_id: string | null;
  title: string | null;
  file_name: string | null;
  mime_type: string | null;
  byte_size: number | null;
  status: "queued" | "parsing" | "indexing" | "ready" | "failed" | "deleted";
  source_type: string;
  source_uri: string | null;
  created_at: string;
  updated_at: string;
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

export type IngestionJobRead = {
  id: string;
  document_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  current_stage: string;
  retry_count: number;
  max_retries: number;
  error_type: string | null;
  error_message: string | null;
  next_retry_at: string | null;
  created_at: string;
  updated_at: string;
};

export type Citation = {
  document_id?: string;
  chunk_id?: string;
  title?: string | null;
  source_uri?: string | null;
  section_path?: string | null;
  score?: number | null;
  quote?: string | null;
};

export type ContextChunk = {
  document_id: string;
  chunk_id: string;
  content: string;
  token_count: number;
  citation: Citation;
};

export type QueryResponse = {
  agent_run_id: string;
  answer: string;
  citations: Citation[];
  context: ContextChunk[];
  context_token_count: number;
  confidence_score: number;
  retrieval_strategy: string;
  latency_ms: number;
  synthesis_enabled: boolean;
  synthesis_error?: string | null;
  verification_status?: string;
  verification_reason?: string | null;
};

export type QueryHistoryMessage = {
  role: "user" | "assistant";
  content: string;
};

export type QueryRequest = {
  query: string;
  workspace_id?: string;
  filters: {
    workspace_id?: string;
    document_ids?: string[];
  };
  history: QueryHistoryMessage[];
  retrieval_limit: number;
  max_context_chunks: number;
  max_context_tokens: number;
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
): Promise<DocumentUploadResponse> {
  const metadata = {
    uploaded_from: "agentic_rag_frontend",
    uploaded_at: new Date().toISOString(),
  };

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
