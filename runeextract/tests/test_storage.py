"""Tests for cloud storage connectors."""

import pytest

from runeextract.storage.connectors import (
    S3Connector, GCSConnector, AzureConnector,
    get_storage_connector, StorageObject,
)


class TestStorageObject:
    def test_create(self):
        obj = StorageObject(key="test.txt", size=100)
        assert obj.key == "test.txt"
        assert obj.size == 100
        assert obj.last_modified is None
        assert obj.etag is None


class TestGetConnector:
    def test_s3(self):
        conn = get_storage_connector("s3", bucket="test")
        assert isinstance(conn, S3Connector)

    def test_gcs(self):
        conn = get_storage_connector("gcs", bucket="test")
        assert isinstance(conn, GCSConnector)

    def test_azure(self):
        conn = get_storage_connector("azure", container="test")
        assert isinstance(conn, AzureConnector)

    def test_invalid_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_storage_connector("invalid")


class TestS3Connector:
    def test_init(self):
        conn = S3Connector(bucket="my-bucket", region="eu-west-1")
        assert conn.bucket == "my-bucket"
        assert conn.region == "eu-west-1"

    def test_exists_without_boto(self):
        conn = S3Connector("test")
        with pytest.raises(ImportError, match="boto3"):
            conn.exists("x")


class TestGCSConnector:
    def test_init(self):
        conn = GCSConnector(bucket="my-bucket", project="my-project")
        assert conn.bucket == "my-bucket"
        assert conn.project == "my-project"

    def test_read_without_gcs(self):
        conn = GCSConnector("test")
        with pytest.raises(ImportError, match="google-cloud-storage"):
            conn.read("x")


class TestAzureConnector:
    def test_init(self):
        conn = AzureConnector(container="my-container", connection_string="conn_str")
        assert conn.container == "my-container"
        assert conn.connection_string == "conn_str"

    def test_list_without_azure(self):
        conn = AzureConnector("test")
        with pytest.raises(ImportError, match="azure-storage-blob"):
            conn.list("")
