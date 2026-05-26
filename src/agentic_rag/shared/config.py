from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    auth_provider: str = Field(default="local", validation_alias="AUTH_PROVIDER")
    app_name: str = Field(default="Agentic RAG", validation_alias="APP_NAME")
    app_version: str = Field(default="0.1.0", validation_alias="APP_VERSION")
    allowed_origins_csv: str = Field(default="*", validation_alias="ALLOWED_ORIGINS")
    oidc_issuer_url: str = Field(default="", validation_alias="OIDC_ISSUER_URL")
    oidc_audience: str = Field(default="", validation_alias="OIDC_AUDIENCE")
    oidc_jwks_url: str = Field(default="", validation_alias="OIDC_JWKS_URL")
    local_auth_token: str = Field(default="local-dev-token", validation_alias="LOCAL_AUTH_TOKEN")
    local_user_id: str = Field(default="local-user", validation_alias="LOCAL_USER_ID")
    local_tenant_id: str = Field(default="local-tenant", validation_alias="LOCAL_TENANT_ID")
    local_workspace_id: str | None = Field(default=None, validation_alias="LOCAL_WORKSPACE_ID")
    local_roles_csv: str = Field(default="admin,user", validation_alias="LOCAL_ROLES")
    local_groups_csv: str = Field(default="", validation_alias="LOCAL_GROUPS")
    local_scopes_csv: str = Field(
        default="documents:read,documents:write,documents:delete,query:run,ingestion:write",
        validation_alias="LOCAL_SCOPES",
    )
    local_acl_version: int = Field(default=1, ge=1, validation_alias="LOCAL_ACL_VERSION")
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
    s3_endpoint_url: str = Field(default="http://localhost:9000", validation_alias="S3_ENDPOINT_URL")
    s3_access_key_id: str = Field(default="agentic_rag", validation_alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str = Field(
        default="agentic_rag_password",
        validation_alias="S3_SECRET_ACCESS_KEY",
    )
    s3_bucket_name: str = Field(default="agentic-rag", validation_alias="S3_BUCKET_NAME")
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
    opensearch_request_timeout_seconds: int = Field(
        default=10,
        ge=1,
        validation_alias="OPENSEARCH_REQUEST_TIMEOUT_SECONDS",
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
    llm_synthesis_enabled: bool = Field(
        default=False,
        validation_alias="LLM_SYNTHESIS_ENABLED",
    )
    llm_provider: str = Field(default="litellm", validation_alias="LLM_PROVIDER")
    default_llm_model: str = Field(
        default="ollama/llama3.1",
        validation_alias="DEFAULT_LLM_MODEL",
    )
    default_small_model: str = Field(
        default="ollama/llama3.1",
        validation_alias="DEFAULT_SMALL_MODEL",
    )
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    litellm_base_url: str = Field(default="", validation_alias="LITELLM_BASE_URL")
    litellm_api_key: str = Field(default="", validation_alias="LITELLM_API_KEY")
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
