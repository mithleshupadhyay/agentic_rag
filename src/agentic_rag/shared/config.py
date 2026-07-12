from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    auth_provider: Literal[
        "local",
        "supertokens",
        "auth0",
        "oidc",
        "keycloak",
    ] = Field(default="local", validation_alias="AUTH_PROVIDER")
    app_name: str = Field(default="Agentic RAG", validation_alias="APP_NAME")
    app_version: str = Field(default="0.1.0", validation_alias="APP_VERSION")
    allowed_origins_csv: str = Field(default="*", validation_alias="ALLOWED_ORIGINS")
    supertokens_connection_uri: str = Field(
        default="http://localhost:3567",
        validation_alias="SUPERTOKENS_CONNECTION_URI",
    )
    supertokens_api_key: str = Field(
        default="",
        validation_alias="SUPERTOKENS_API_KEY",
    )
    supertokens_app_name: str = Field(
        default="Agentic RAG",
        validation_alias="SUPERTOKENS_APP_NAME",
    )
    supertokens_api_domain: str = Field(
        default="http://localhost:8100",
        validation_alias="SUPERTOKENS_API_DOMAIN",
    )
    supertokens_website_domain: str = Field(
        default="http://localhost:5173",
        validation_alias="SUPERTOKENS_WEBSITE_DOMAIN",
    )
    supertokens_api_base_path: str = Field(
        default="/auth",
        validation_alias="SUPERTOKENS_API_BASE_PATH",
    )
    supertokens_website_base_path: str = Field(
        default="/auth",
        validation_alias="SUPERTOKENS_WEBSITE_BASE_PATH",
    )
    supertokens_cookie_domain: str = Field(
        default="",
        validation_alias="SUPERTOKENS_COOKIE_DOMAIN",
    )
    supertokens_cookie_secure: bool = Field(
        default=False,
        validation_alias="SUPERTOKENS_COOKIE_SECURE",
    )
    supertokens_cookie_same_site: Literal["lax", "none", "strict"] = Field(
        default="lax",
        validation_alias="SUPERTOKENS_COOKIE_SAME_SITE",
    )
    google_client_id: str = Field(default="", validation_alias="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(
        default="",
        validation_alias="GOOGLE_CLIENT_SECRET",
    )
    github_client_id: str = Field(default="", validation_alias="GITHUB_CLIENT_ID")
    github_client_secret: str = Field(
        default="",
        validation_alias="GITHUB_CLIENT_SECRET",
    )
    public_tenant_signup_enabled: bool = Field(
        default=True,
        validation_alias="PUBLIC_TENANT_SIGNUP_ENABLED",
    )
    invite_expiry_hours: int = Field(
        default=48,
        ge=1,
        le=720,
        validation_alias="INVITE_EXPIRY_HOURS",
    )
    invitation_context_secret: str = Field(
        default="",
        validation_alias="INVITATION_CONTEXT_SECRET",
    )
    invitation_context_ttl_seconds: int = Field(
        default=900,
        ge=60,
        le=3600,
        validation_alias="INVITATION_CONTEXT_TTL_SECONDS",
    )
    temporary_password_provisioning_enabled: bool = Field(
        default=True,
        validation_alias="TEMPORARY_PASSWORD_PROVISIONING_ENABLED",
    )
    email_delivery_provider: Literal["smtp", "console"] = Field(
        default="console",
        validation_alias="EMAIL_DELIVERY_PROVIDER",
    )
    smtp_host: str = Field(default="localhost", validation_alias="SMTP_HOST")
    smtp_port: int = Field(default=1025, ge=1, le=65535, validation_alias="SMTP_PORT")
    smtp_username: str = Field(default="", validation_alias="SMTP_USERNAME")
    smtp_password: str = Field(default="", validation_alias="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=False, validation_alias="SMTP_USE_TLS")
    email_from_address: str = Field(
        default="noreply@agentic-rag.local",
        validation_alias="EMAIL_FROM_ADDRESS",
    )
    email_from_name: str = Field(
        default="Agentic RAG",
        validation_alias="EMAIL_FROM_NAME",
    )
    support_email: str = Field(
        default="support@agentic-rag.local",
        validation_alias="SUPPORT_EMAIL",
    )
    email_worker_poll_seconds: float = Field(
        default=5.0,
        validation_alias="EMAIL_WORKER_POLL_SECONDS",
        gt=0,
    )
    email_worker_max_attempts: int = Field(
        default=5,
        validation_alias="EMAIL_WORKER_MAX_ATTEMPTS",
        ge=1,
        le=20,
    )
    email_worker_lease_seconds: int = Field(
        default=60,
        validation_alias="EMAIL_WORKER_LEASE_SECONDS",
        ge=15,
        le=900,
    )
    auth0_domain: str = Field(default="", validation_alias="AUTH0_DOMAIN")
    auth0_api_audience: str = Field(
        default="",
        validation_alias="AUTH0_API_AUDIENCE",
    )
    auth0_frontend_client_id: str = Field(
        default="",
        validation_alias="AUTH0_FRONTEND_CLIENT_ID",
    )
    auth0_frontend_scope: str = Field(
        default="openid profile email",
        validation_alias="AUTH0_FRONTEND_SCOPE",
    )
    auth0_identity_connections_csv: str = Field(
        default="google-oauth2,github",
        validation_alias="AUTH0_IDENTITY_CONNECTIONS",
    )
    auth0_organization_connection_ids_csv: str = Field(
        default="",
        validation_alias="AUTH0_ORGANIZATION_CONNECTION_IDS",
    )
    auth0_jwks_cache_seconds: int = Field(
        default=600,
        ge=60,
        le=86400,
        validation_alias="AUTH0_JWKS_CACHE_SECONDS",
    )
    auth0_jwks_timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        validation_alias="AUTH0_JWKS_TIMEOUT_SECONDS",
    )
    auth0_management_client_id: str = Field(
        default="",
        validation_alias="AUTH0_MANAGEMENT_CLIENT_ID",
    )
    auth0_management_client_secret: str = Field(
        default="",
        validation_alias="AUTH0_MANAGEMENT_CLIENT_SECRET",
    )
    auth0_database_connection_id: str = Field(
        default="",
        validation_alias="AUTH0_DATABASE_CONNECTION_ID",
    )
    auth0_management_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        validation_alias="AUTH0_MANAGEMENT_TIMEOUT_SECONDS",
    )
    auth0_inviter_name: str = Field(
        default="Agentic RAG",
        min_length=1,
        max_length=300,
        validation_alias="AUTH0_INVITER_NAME",
    )
    auth0_invitation_ttl_seconds: int = Field(
        default=604800,
        ge=300,
        le=2592000,
        validation_alias="AUTH0_INVITATION_TTL_SECONDS",
    )
    frontend_public_url: str = Field(
        default="http://localhost:5173",
        validation_alias="FRONTEND_PUBLIC_URL",
    )
    local_auth_token: str = Field(
        default="local-dev-token", validation_alias="LOCAL_AUTH_TOKEN"
    )
    local_user_id: str = Field(default="local-user", validation_alias="LOCAL_USER_ID")
    local_tenant_id: str = Field(
        default="local-tenant", validation_alias="LOCAL_TENANT_ID"
    )
    local_workspace_id: str | None = Field(
        default=None, validation_alias="LOCAL_WORKSPACE_ID"
    )
    local_roles_csv: str = Field(default="admin,user", validation_alias="LOCAL_ROLES")
    local_groups_csv: str = Field(default="", validation_alias="LOCAL_GROUPS")
    local_scopes_csv: str = Field(
        default="documents:read,documents:write,documents:delete,query:run,ingestion:write",
        validation_alias="LOCAL_SCOPES",
    )
    local_acl_version: int = Field(
        default=1, ge=1, validation_alias="LOCAL_ACL_VERSION"
    )
    database_url: str = Field(
        default="postgresql+asyncpg://agentic_rag:agentic_rag@localhost:5432/agentic_rag",
        validation_alias="DATABASE_URL",
    )
    database_echo: bool = Field(default=False, validation_alias="DATABASE_ECHO")
    database_pool_size: int = Field(
        default=10,
        ge=1,
        validation_alias="DATABASE_POOL_SIZE",
    )
    database_max_overflow: int = Field(
        default=20,
        ge=0,
        validation_alias="DATABASE_MAX_OVERFLOW",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias="REDIS_URL",
    )
    redis_socket_timeout_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        validation_alias="REDIS_SOCKET_TIMEOUT_SECONDS",
    )
    query_cache_enabled: bool = Field(
        default=False,
        validation_alias="QUERY_CACHE_ENABLED",
    )
    query_cache_ttl_seconds: int = Field(
        default=300,
        ge=1,
        le=86400,
        validation_alias="QUERY_CACHE_TTL_SECONDS",
    )
    query_cache_key_prefix: str = Field(
        default="agentic-rag:query",
        validation_alias="QUERY_CACHE_KEY_PREFIX",
    )
    kafka_publishing_enabled: bool = Field(
        default=False,
        validation_alias="KAFKA_PUBLISHING_ENABLED",
    )
    kafka_consuming_enabled: bool = Field(
        default=False,
        validation_alias="KAFKA_CONSUMING_ENABLED",
    )
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        validation_alias="KAFKA_BOOTSTRAP_SERVERS",
    )
    kafka_client_id: str = Field(
        default="agentic-rag-worker",
        validation_alias="KAFKA_CLIENT_ID",
    )
    kafka_producer_flush_timeout_seconds: float = Field(
        default=5.0,
        ge=0.0,
        validation_alias="KAFKA_PRODUCER_FLUSH_TIMEOUT_SECONDS",
    )
    kafka_ingestion_consumer_group: str = Field(
        default="agentic-rag-ingestion",
        validation_alias="KAFKA_INGESTION_CONSUMER_GROUP",
    )
    kafka_embedding_consumer_group: str = Field(
        default="agentic-rag-embedding",
        validation_alias="KAFKA_EMBEDDING_CONSUMER_GROUP",
    )
    kafka_indexing_consumer_group: str = Field(
        default="agentic-rag-indexing",
        validation_alias="KAFKA_INDEXING_CONSUMER_GROUP",
    )
    s3_endpoint_url: str = Field(
        default="http://localhost:9000", validation_alias="S3_ENDPOINT_URL"
    )
    s3_access_key_id: str = Field(
        default="agentic_rag", validation_alias="S3_ACCESS_KEY_ID"
    )
    s3_secret_access_key: str = Field(
        default="agentic_rag_password",
        validation_alias="S3_SECRET_ACCESS_KEY",
    )
    s3_bucket_name: str = Field(
        default="agentic-rag", validation_alias="S3_BUCKET_NAME"
    )
    s3_region: str = Field(default="us-east-1", validation_alias="S3_REGION")
    document_upload_max_bytes: int = Field(
        default=25 * 1024 * 1024,
        ge=1,
        validation_alias="DOCUMENT_UPLOAD_MAX_BYTES",
    )
    ingestion_worker_poll_seconds: int = Field(
        default=5,
        ge=1,
        validation_alias="INGESTION_WORKER_POLL_SECONDS",
    )
    ingestion_worker_lease_seconds: int = Field(
        default=300,
        ge=30,
        validation_alias="INGESTION_WORKER_LEASE_SECONDS",
    )
    ingestion_retry_base_seconds: int = Field(
        default=30,
        ge=1,
        validation_alias="INGESTION_RETRY_BASE_SECONDS",
    )
    ingestion_retry_max_seconds: int = Field(
        default=900,
        ge=1,
        validation_alias="INGESTION_RETRY_MAX_SECONDS",
    )
    ingestion_chunk_size: int = Field(
        default=2000,
        ge=100,
        validation_alias="INGESTION_CHUNK_SIZE",
    )
    ingestion_chunk_overlap: int = Field(
        default=200,
        ge=0,
        validation_alias="INGESTION_CHUNK_OVERLAP",
    )
    opensearch_url: str = Field(
        default="http://localhost:9200",
        validation_alias="OPENSEARCH_URL",
    )
    opensearch_username: str = Field(default="", validation_alias="OPENSEARCH_USERNAME")
    opensearch_password: str = Field(default="", validation_alias="OPENSEARCH_PASSWORD")
    opensearch_document_index: str = Field(
        default="documents-v1",
        validation_alias="OPENSEARCH_DOCUMENT_INDEX",
    )
    opensearch_chunk_index: str = Field(
        default="chunks-v1",
        validation_alias="OPENSEARCH_CHUNK_INDEX",
    )
    opensearch_chunk_read_alias: str = Field(
        default="chunks-read",
        min_length=1,
        validation_alias="OPENSEARCH_CHUNK_READ_ALIAS",
    )
    opensearch_chunk_write_alias: str = Field(
        default="chunks-write",
        min_length=1,
        validation_alias="OPENSEARCH_CHUNK_WRITE_ALIAS",
    )
    opensearch_request_timeout_seconds: int = Field(
        default=10,
        ge=1,
        validation_alias="OPENSEARCH_REQUEST_TIMEOUT_SECONDS",
    )
    opensearch_search_retry_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        validation_alias="OPENSEARCH_SEARCH_RETRY_ATTEMPTS",
    )
    opensearch_search_retry_backoff_seconds: float = Field(
        default=0.25,
        ge=0.0,
        validation_alias="OPENSEARCH_SEARCH_RETRY_BACKOFF_SECONDS",
    )
    opensearch_index_shards: int = Field(
        default=1,
        ge=1,
        validation_alias="OPENSEARCH_INDEX_SHARDS",
    )
    opensearch_index_replicas: int = Field(
        default=0,
        ge=0,
        validation_alias="OPENSEARCH_INDEX_REPLICAS",
    )
    indexing_worker_poll_seconds: int = Field(
        default=5,
        ge=1,
        validation_alias="INDEXING_WORKER_POLL_SECONDS",
    )
    bm25_index_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        validation_alias="BM25_INDEX_BATCH_SIZE",
    )
    bm25_min_score: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias="BM25_MIN_SCORE",
    )
    embedding_provider: str = Field(
        default="litellm", validation_alias="EMBEDDING_PROVIDER"
    )
    embedding_model_name: str = Field(
        default="gemini/gemini-embedding-001",
        validation_alias="EMBEDDING_MODEL_NAME",
    )
    embedding_dimension: int = Field(
        default=768,
        ge=1,
        validation_alias="EMBEDDING_DIMENSION",
    )
    embedding_vector_version: int = Field(
        default=1,
        ge=1,
        validation_alias="EMBEDDING_VECTOR_VERSION",
    )
    embedding_timeout_seconds: int = Field(
        default=30,
        ge=1,
        validation_alias="EMBEDDING_TIMEOUT_SECONDS",
    )
    embedding_max_input_chars: int = Field(
        default=128000,
        ge=1000,
        validation_alias="EMBEDDING_MAX_INPUT_CHARS",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        validation_alias="EMBEDDING_BATCH_SIZE",
    )
    embedding_worker_poll_seconds: int = Field(
        default=5,
        ge=1,
        validation_alias="EMBEDDING_WORKER_POLL_SECONDS",
    )
    embedding_worker_max_chunks_per_loop: int = Field(
        default=100,
        ge=1,
        le=10000,
        validation_alias="EMBEDDING_WORKER_MAX_CHUNKS_PER_LOOP",
    )
    llm_synthesis_enabled: bool = Field(
        default=False,
        validation_alias="LLM_SYNTHESIS_ENABLED",
    )
    llm_provider: str = Field(default="litellm", validation_alias="LLM_PROVIDER")
    llm_provider_database_enabled: bool = Field(
        default=True,
        validation_alias="LLM_PROVIDER_DATABASE_ENABLED",
    )
    llm_provider_env_fallback_enabled: bool = Field(
        default=True,
        validation_alias="LLM_PROVIDER_ENV_FALLBACK_ENABLED",
    )
    llm_provider_encryption_key: str = Field(
        default="",
        validation_alias="LLM_PROVIDER_ENCRYPTION_KEY",
    )
    default_llm_model: str = Field(
        default="gemini/gemini-2.5-flash",
        validation_alias="DEFAULT_LLM_MODEL",
    )
    default_small_model: str = Field(
        default="gemini/gemini-2.5-flash",
        validation_alias="DEFAULT_SMALL_MODEL",
    )
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    litellm_base_url: str = Field(default="", validation_alias="LITELLM_BASE_URL")
    litellm_api_key: str = Field(default="", validation_alias="LITELLM_API_KEY")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        validation_alias="LLM_TEMPERATURE",
    )
    llm_max_tokens: int = Field(
        default=700,
        ge=1,
        le=8000,
        validation_alias="LLM_MAX_TOKENS",
    )
    llm_max_input_chars: int = Field(
        default=64000,
        ge=1000,
        validation_alias="LLM_MAX_INPUT_CHARS",
    )
    llm_max_output_tokens: int = Field(
        default=8000,
        ge=1,
        validation_alias="LLM_MAX_OUTPUT_TOKENS",
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias="LLM_MAX_RETRIES",
    )
    llm_retry_backoff_seconds: float = Field(
        default=0.5,
        ge=0.0,
        le=30.0,
        validation_alias="LLM_RETRY_BACKOFF_SECONDS",
    )
    llm_circuit_breaker_enabled: bool = Field(
        default=True,
        validation_alias="LLM_CIRCUIT_BREAKER_ENABLED",
    )
    llm_circuit_breaker_state_backend: Literal["memory", "redis"] = Field(
        default="memory",
        validation_alias="LLM_CIRCUIT_BREAKER_STATE_BACKEND",
    )
    llm_circuit_breaker_redis_key_prefix: str = Field(
        default="agentic-rag:llm:circuit-breaker",
        validation_alias="LLM_CIRCUIT_BREAKER_REDIS_KEY_PREFIX",
    )
    llm_circuit_breaker_failure_threshold: int = Field(
        default=3,
        ge=1,
        le=100,
        validation_alias="LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    )
    llm_circuit_breaker_cooldown_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        validation_alias="LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS",
    )
    llm_timeout_seconds: int = Field(
        default=30,
        ge=1,
        validation_alias="LLM_TIMEOUT_SECONDS",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins_csv.split(",")
            if origin.strip()
        ]

    @staticmethod
    def _split_csv(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def local_roles(self) -> list[str]:
        return self._split_csv(self.local_roles_csv)

    @property
    def local_groups(self) -> list[str]:
        return self._split_csv(self.local_groups_csv)

    @property
    def local_scopes(self) -> list[str]:
        return self._split_csv(self.local_scopes_csv)

    @property
    def auth0_identity_connections(self) -> list[str]:
        return self._split_csv(self.auth0_identity_connections_csv)

    @property
    def auth0_organization_connection_ids(self) -> list[str]:
        connection_ids = self._split_csv(self.auth0_organization_connection_ids_csv)
        if self.auth0_database_connection_id:
            connection_ids.append(self.auth0_database_connection_id.strip())
        return list(dict.fromkeys(connection_ids))

    @property
    def auth0_issuer_url(self) -> str:
        domain = self.auth0_domain.strip().rstrip("/")
        if domain.startswith(("http://", "https://")):
            return f"{domain}/"
        if not domain:
            return ""
        return f"https://{domain}/"

    @property
    def auth0_management_audience(self) -> str:
        issuer_url = self.auth0_issuer_url
        if not issuer_url:
            return ""
        return f"{issuer_url}api/v2/"

    @property
    def sync_database_url(self) -> str:
        if self.database_url.startswith("postgresql+asyncpg://"):
            return self.database_url.replace(
                "postgresql+asyncpg://",
                "postgresql+psycopg://",
                1,
            )
        if self.database_url.startswith("sqlite+aiosqlite://"):
            return self.database_url.replace("sqlite+aiosqlite://", "sqlite://", 1)
        return self.database_url

    @property
    def is_sqlite_database(self) -> bool:
        return self.database_url.startswith(("sqlite://", "sqlite+aiosqlite://"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
