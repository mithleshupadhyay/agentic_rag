import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

import boto3  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from agentic_rag.shared.config import settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectStoreConfig:
    bucket_name: str
    region_name: str
    endpoint_url: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None


class ObjectStoreClient:
    def __init__(
        self,
        config: Optional[ObjectStoreConfig] = None,
        client: Optional[Any] = None,
    ):
        self.config = config or ObjectStoreConfig(
            bucket_name=settings.s3_bucket_name,
            region_name=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url or None,
            access_key_id=settings.s3_access_key_id or None,
            secret_access_key=settings.s3_secret_access_key or None,
        )
        self.client = client or boto3.client(
            "s3",
            region_name=self.config.region_name,
            endpoint_url=self.config.endpoint_url,
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
        )

    def build_object_key(
        self,
        tenant_id: str,
        workspace_id: Optional[str],
        document_id: UUID,
        file_name: str,
    ) -> str:
        safe_file_name = file_name.replace("\\", "/").split("/")[-1].strip()
        if not safe_file_name:
            safe_file_name = "document.bin"

        workspace = workspace_id or "default"
        return (
            f"tenants/{tenant_id}/workspaces/{workspace}/"
            f"documents/{document_id}/raw/{safe_file_name}"
        )

    def put_bytes(
        self,
        object_key: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        logger.info(
            f"[ObjectStore] Uploading object bucket={self.config.bucket_name}, "
            f"key={object_key}, bytes={len(data)}"
        )

        try:
            extra_args: dict[str, Any] = {
                "Bucket": self.config.bucket_name,
                "Key": object_key,
                "Body": data,
            }
            if content_type:
                extra_args["ContentType"] = content_type
            if metadata:
                extra_args["Metadata"] = metadata

            response = self.client.put_object(**extra_args)
            content_hash = hashlib.sha256(data).hexdigest()

            logger.info(
                f"[ObjectStore] Uploaded object bucket={self.config.bucket_name}, "
                f"key={object_key}, sha256={content_hash}"
            )
            return {
                "bucket": self.config.bucket_name,
                "object_key": object_key,
                "byte_size": len(data),
                "content_hash": content_hash,
                "etag": response.get("ETag"),
            }

        except ClientError as e:
            logger.exception(f"[ObjectStore] Failed to upload object {object_key}: {e}")
            raise

    def get_bytes(self, object_key: str) -> bytes:
        logger.info(
            f"[ObjectStore] Reading object bucket={self.config.bucket_name}, "
            f"key={object_key}"
        )

        try:
            response = self.client.get_object(
                Bucket=self.config.bucket_name,
                Key=object_key,
            )
            data = response["Body"].read()

            logger.info(
                f"[ObjectStore] Read object bucket={self.config.bucket_name}, "
                f"key={object_key}, bytes={len(data)}"
            )
            return data

        except ClientError as e:
            logger.exception(f"[ObjectStore] Failed to read object {object_key}: {e}")
            raise

    def delete_object(self, object_key: str) -> None:
        logger.info(
            f"[ObjectStore] Deleting object bucket={self.config.bucket_name}, "
            f"key={object_key}"
        )

        try:
            self.client.delete_object(
                Bucket=self.config.bucket_name,
                Key=object_key,
            )
            logger.info(
                f"[ObjectStore] Deleted object bucket={self.config.bucket_name}, "
                f"key={object_key}"
            )

        except ClientError as e:
            logger.exception(f"[ObjectStore] Failed to delete object {object_key}: {e}")
            raise
