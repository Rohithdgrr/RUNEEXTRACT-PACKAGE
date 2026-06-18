"""Cloud storage connectors — unified interface for S3, GCS, Azure Blob."""

import io
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import BinaryIO, List, Optional

from runeextract.exceptions import FileTooLargeError

logger = logging.getLogger(__name__)

_MAX_READ_SIZE = 500 * 1024 * 1024  # 500 MB max read from cloud storage


@dataclass
class StorageObject:
    key: str
    size: int
    last_modified: Optional[str] = None
    etag: Optional[str] = None


class StorageConnector(ABC):
    """Abstract base for cloud storage connectors."""

    def __init__(self, max_read_size: int = _MAX_READ_SIZE):
        self.max_read_size = max_read_size

    @abstractmethod
    def list(self, prefix: str = "", recursive: bool = True) -> List[StorageObject]:
        ...

    @abstractmethod
    def read(self, key: str) -> bytes:
        ...

    @abstractmethod
    def write(self, key: str, data: bytes) -> str:
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...


class S3Connector(StorageConnector):
    """Amazon S3 connector."""

    def __init__(self, bucket: str, region: str = "us-east-1", max_read_size: int = _MAX_READ_SIZE, **kwargs):
        super().__init__(max_read_size=max_read_size)
        self.bucket = bucket
        self.region = region
        self._kwargs = kwargs
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("S3 connector requires 'boto3'. Install: pip install boto3")
            self._client = boto3.client("s3", region_name=self.region, **self._kwargs)
        return self._client

    def list(self, prefix: str = "", recursive: bool = True) -> List[StorageObject]:
        client = self._get_client()
        delimiter = "" if recursive else "/"
        objects = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter=delimiter):
            for obj in page.get("Contents", []):
                objects.append(StorageObject(
                    key=obj["Key"],
                    size=obj["Size"],
                    last_modified=str(obj.get("LastModified", "")),
                    etag=obj.get("ETag", ""),
                ))
        return objects

    def read(self, key: str) -> bytes:
        client = self._get_client()
        resp = client.get_object(Bucket=self.bucket, Key=key)
        data = resp["Body"].read()
        if len(data) > self.max_read_size:
            raise FileTooLargeError(key, len(data), self.max_read_size)
        return data

    def write(self, key: str, data: bytes) -> str:
        client = self._get_client()
        resp = client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return resp.get("ETag", "")

    def delete(self, key: str) -> bool:
        client = self._get_client()
        client.delete_object(Bucket=self.bucket, Key=key)
        return True

    def exists(self, key: str) -> bool:
        client = self._get_client()
        try:
            client.head_object(Bucket=self.bucket, Key=key)
            return True
        except client.exceptions.ClientError:
            return False


class GCSConnector(StorageConnector):
    """Google Cloud Storage connector."""

    def __init__(self, bucket: str, project: Optional[str] = None, max_read_size: int = _MAX_READ_SIZE):
        super().__init__(max_read_size=max_read_size)
        self.bucket = bucket
        self.project = project
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import storage
            except ImportError:
                raise ImportError("GCS connector requires 'google-cloud-storage'. Install: pip install google-cloud-storage")
            self._client = storage.Client(project=self.project)
        return self._client

    def _get_bucket(self):
        return self._get_client().bucket(self.bucket)

    def list(self, prefix: str = "", recursive: bool = True) -> List[StorageObject]:
        bucket = self._get_bucket()
        blobs = bucket.list_blobs(prefix=prefix, delimiter=None if recursive else "/")
        return [
            StorageObject(key=b.name, size=b.size, last_modified=str(b.updated or ""), etag=b.etag or "")
            for b in blobs
        ]

    def read(self, key: str) -> bytes:
        bucket = self._get_bucket()
        blob = bucket.blob(key)
        data = blob.download_as_bytes()
        if len(data) > self.max_read_size:
            raise FileTooLargeError(key, len(data), self.max_read_size)
        return data

    def write(self, key: str, data: bytes) -> str:
        bucket = self._get_bucket()
        blob = bucket.blob(key)
        blob.upload_from_string(data)
        return blob.etag or ""

    def delete(self, key: str) -> bool:
        bucket = self._get_bucket()
        blob = bucket.blob(key)
        blob.delete()
        return True

    def exists(self, key: str) -> bool:
        bucket = self._get_bucket()
        blob = bucket.blob(key)
        return blob.exists()


class AzureConnector(StorageConnector):
    """Azure Blob Storage connector."""

    def __init__(self, container: str, connection_string: Optional[str] = None, max_read_size: int = _MAX_READ_SIZE):
        super().__init__(max_read_size=max_read_size)
        self.container = container
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from azure.storage.blob import BlobServiceClient
            except ImportError:
                raise ImportError("Azure connector requires 'azure-storage-blob'. Install: pip install azure-storage-blob")
            service = BlobServiceClient.from_connection_string(self.connection_string)
            self._client = service.get_container_client(self.container)
        return self._client

    def list(self, prefix: str = "", recursive: bool = True) -> List[StorageObject]:
        client = self._get_client()
        blobs = client.list_blobs(name_starts_with=prefix)
        return [
            StorageObject(key=b.name, size=b.size, last_modified=str(b.last_modified or ""), etag=b.etag or "")
            for b in blobs
        ]

    def read(self, key: str) -> bytes:
        client = self._get_client()
        blob = client.get_blob_client(key)
        data = blob.download_blob().readall()
        if len(data) > self.max_read_size:
            raise FileTooLargeError(key, len(data), self.max_read_size)
        return data

    def write(self, key: str, data: bytes) -> str:
        client = self._get_client()
        blob = client.get_blob_client(key)
        blob.upload_blob(data, overwrite=True)
        return ""

    def delete(self, key: str) -> bool:
        client = self._get_client()
        blob = client.get_blob_client(key)
        blob.delete_blob()
        return True

    def exists(self, key: str) -> bool:
        client = self._get_client()
        try:
            blob = client.get_blob_client(key)
            blob.get_blob_properties()
            return True
        except Exception as exc:
            logger.debug(f"Azure blob exists error: {exc}")
            return False


def get_storage_connector(provider: str, **kwargs) -> StorageConnector:
    """Get a storage connector by provider name.

    Args:
        provider: "s3", "gcs", or "azure"
        **kwargs: Provider-specific arguments (bucket, region, etc.)

    Returns:
        StorageConnector instance
    """
    providers = {
        "s3": S3Connector,
        "gcs": GCSConnector,
        "azure": AzureConnector,
    }
    if provider.lower() not in providers:
        raise ValueError(f"Unknown provider '{provider}'. Options: {', '.join(providers)}")
    return providers[provider.lower()](**kwargs)
