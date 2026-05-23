from io import BytesIO
from uuid import UUID

from agentic_rag.storage.object_store import ObjectStoreClient, ObjectStoreConfig


class FakeS3Client:
    def __init__(self):
        self.objects = {}
        self.deleted = []

    def put_object(self, **kwargs):
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]
        self.objects[(bucket, key)] = {
            "body": kwargs["Body"],
            "content_type": kwargs.get("ContentType"),
            "metadata": kwargs.get("Metadata"),
        }
        return {"ETag": '"etag-1"'}

    def get_object(self, Bucket, Key):
        stored = self.objects[(Bucket, Key)]
        return {"Body": BytesIO(stored["body"])}

    def delete_object(self, Bucket, Key):
        self.deleted.append((Bucket, Key))
        self.objects.pop((Bucket, Key), None)
        return {}


def build_client() -> tuple[ObjectStoreClient, FakeS3Client]:
    fake_client = FakeS3Client()
    object_store = ObjectStoreClient(
        config=ObjectStoreConfig(
            bucket_name="agentic-rag-test",
            region_name="us-east-1",
            endpoint_url="http://localhost:9000",
            access_key_id="test",
            secret_access_key="test",
        ),
        client=fake_client,
    )
    return object_store, fake_client


def test_build_object_key_uses_tenant_workspace_document_and_safe_file_name() -> None:
    object_store, _ = build_client()
    document_id = UUID("11111111-1111-1111-1111-111111111111")

    object_key = object_store.build_object_key(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        document_id=document_id,
        file_name="../reports/security policy.pdf",
    )

    assert object_key == (
        "tenants/tenant-a/workspaces/workspace-a/"
        "documents/11111111-1111-1111-1111-111111111111/raw/security policy.pdf"
    )


def test_build_object_key_uses_default_workspace_and_file_name() -> None:
    object_store, _ = build_client()
    document_id = UUID("11111111-1111-1111-1111-111111111111")

    object_key = object_store.build_object_key(
        tenant_id="tenant-a",
        workspace_id=None,
        document_id=document_id,
        file_name="   ",
    )

    assert object_key == (
        "tenants/tenant-a/workspaces/default/"
        "documents/11111111-1111-1111-1111-111111111111/raw/document.bin"
    )


def test_put_get_and_delete_bytes() -> None:
    object_store, fake_client = build_client()
    data = b"raw document bytes"

    result = object_store.put_bytes(
        object_key="tenants/tenant-a/documents/doc-1/raw/file.txt",
        data=data,
        content_type="text/plain",
        metadata={"tenant_id": "tenant-a"},
    )
    stored_data = object_store.get_bytes(result["object_key"])
    object_store.delete_object(result["object_key"])

    assert result["bucket"] == "agentic-rag-test"
    assert result["byte_size"] == len(data)
    assert result["content_hash"] == (
        "e22d390698653b7caba80c8f22ffcd8f5a6f33ba2d5338da0664c13f3d1728d7"
    )
    assert result["etag"] == '"etag-1"'
    assert stored_data == data
    assert fake_client.deleted == [
        ("agentic-rag-test", "tenants/tenant-a/documents/doc-1/raw/file.txt")
    ]
