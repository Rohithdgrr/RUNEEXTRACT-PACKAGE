"""Multi-tenant isolation — per-tenant vector collections and caches."""

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TenantStore:
    def __init__(self, base_collection: str = "documents"):
        self.base_collection = base_collection
        self._tenants: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def collection_name(self, tenant_id: str) -> str:
        return f"{self.base_collection}__{tenant_id}"

    def register(self, tenant_id: str, rag_instance: Any):
        with self._lock:
            original_collection = getattr(rag_instance, "collection_name", self.base_collection)
            rag_instance.collection_name = self.collection_name(tenant_id)
            self._tenants[tenant_id] = rag_instance
            logger.info(
                "Registered tenant '%s' with collection '%s'",
                tenant_id, rag_instance.collection_name,
            )

    def get(self, tenant_id: str):
        return self._tenants.get(tenant_id)

    def list_tenants(self) -> List[str]:
        return list(self._tenants.keys())

    def remove(self, tenant_id: str):
        self._tenants.pop(tenant_id, None)


class MultiTenantRAG:
    def __init__(self, rag_factory: Any):
        self._factory = rag_factory
        self._store = TenantStore()
        self._locks: Dict[str, threading.Lock] = {}

    def get_or_create(self, tenant_id: str, **kwargs) -> Any:
        existing = self._store.get(tenant_id)
        if existing is not None:
            return existing
        rag = self._factory(**kwargs)
        self._store.register(tenant_id, rag)
        self._locks[tenant_id] = threading.Lock()
        return rag

    def query(self, tenant_id: str, question: str, **kwargs):
        rag = self.get_or_create(tenant_id)
        with self._locks.get(tenant_id, threading.Lock()):
            return rag.query(question, **kwargs)

    def ingest(self, tenant_id: str, source: str, **kwargs):
        rag = self.get_or_create(tenant_id)
        return rag.ingest(source, **kwargs)

    def list_tenants(self) -> List[str]:
        return self._store.list_tenants()
