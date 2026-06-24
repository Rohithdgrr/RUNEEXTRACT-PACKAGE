"""
Feature 13: RAG-as-a-Service API

Production-ready REST API for RAG with auto-scaling.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# API Models (Pydantic)
try:
    from pydantic import BaseModel, Field
    
    class QueryRequest(BaseModel):
        """Query request model."""
        question: str = Field(..., description="User question")
        top_k: int = Field(5, description="Number of chunks to retrieve")
        return_citations: bool = Field(True, description="Include citations")
        user_id: Optional[str] = Field(None, description="User identifier")
        metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    
    class IngestRequest(BaseModel):
        """Ingestion request model."""
        source: str = Field(..., description="File path or URL")
        chunking: str = Field("auto", description="Chunking strategy")
        metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    
    class HealthResponse(BaseModel):
        """Health check response."""
        status: str = Field(..., description="Service status")
        version: str = Field(..., description="API version")
        rag_ready: bool = Field(..., description="RAG pipeline ready")
        uptime_seconds: float = Field(..., description="Uptime in seconds")
    
    class QueryResponse(BaseModel):
        """Query response model."""
        answer: str
        confidence: float
        citations: List[Dict[str, Any]]
        latency_ms: float
        cost: float
    
except ImportError:
    logger.warning("pydantic not installed, API models not available")
    BaseModel = object
    QueryRequest = IngestRequest = HealthResponse = QueryResponse = None


class RAGAPIServer:
    """REST API server for RuneExtract RAG.
    
    Features:
    - RESTful endpoints (/query, /ingest, /health, /metrics)
    - Streaming support (/query/stream)
    - API key authentication
    - Rate limiting
    - CORS support
    - OpenAPI documentation
    
    Usage::
    
        from fastapi import FastAPI
        from runeextract import auto_rag
        
        rag = auto_rag("./docs/")
        api = RAGAPIServer(rag)
        app = api.create_app()
        
        # Run with: uvicorn main:app --reload
    """
    
    def __init__(
        self,
        rag: Any,
        api_keys: Optional[List[str]] = None,
        rate_limit: int = 100,  # requests per minute
        enable_cors: bool = True
    ):
        """Initialize RAG API server.
        
        Args:
            rag: AutoRAG instance
            api_keys: List of valid API keys (optional)
            rate_limit: Max requests per minute per key
            enable_cors: Enable CORS for web clients
        """
        self.rag = rag
        self.api_keys = set(api_keys) if api_keys else None
        self.rate_limit = rate_limit
        self.enable_cors = enable_cors
        
        # Track requests for rate limiting
        self._request_counts: Dict[str, List[float]] = {}
        
        # Server metrics
        self._start_time = time.time()
        self._total_requests = 0
        self._total_errors = 0
        
        logger.info(
            f"RAG API Server initialized "
            f"(auth={bool(api_keys)}, rate_limit={rate_limit}/min)"
        )
    
    def create_app(self):
        """Create FastAPI application.
        
        Returns:
            FastAPI app instance
        """
        try:
            from fastapi import FastAPI, HTTPException, Header, Depends
            from fastapi.responses import StreamingResponse
            from fastapi.middleware.cors import CORSMiddleware
        except ImportError:
            raise ImportError(
                "FastAPI not installed. Install with: pip install fastapi uvicorn"
            )
        
        app = FastAPI(
            title="RuneExtract RAG API",
            description="Production-ready RAG API with auto-scaling",
            version="0.7.0"
        )
        
        # CORS middleware
        if self.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Authentication dependency
        async def verify_api_key(x_api_key: Optional[str] = Header(None)):
            if self.api_keys is not None:
                if not x_api_key or x_api_key not in self.api_keys:
                    raise HTTPException(status_code=401, detail="Invalid API key")
            
            # Rate limiting
            if x_api_key:
                self._check_rate_limit(x_api_key)
            
            return x_api_key
        
        # Endpoints
        @app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                version="0.7.0",
                rag_ready=self.rag is not None,
                uptime_seconds=time.time() - self._start_time
            )
        
        @app.post("/query", response_model=QueryResponse)
        async def query(
            request: QueryRequest,
            api_key: str = Depends(verify_api_key)
        ):
            """Query RAG endpoint."""
            try:
                self._total_requests += 1
                
                result = self.rag.query(
                    question=request.question,
                    top_k=request.top_k,
                    return_citations=request.return_citations,
                    metadata_filter=request.metadata_filter
                )
                
                return QueryResponse(
                    answer=result.answer,
                    confidence=result.confidence,
                    citations=[
                        {
                            "id": c.id,
                            "text": c.text,
                            "source": c.source,
                            "page": c.page,
                            "score": c.score
                        }
                        for c in result.citations
                    ],
                    latency_ms=result.latency_ms,
                    cost=result.cost
                )
            except Exception as e:
                self._total_errors += 1
                logger.error(f"Query failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/query/stream")
        async def query_stream(
            request: QueryRequest,
            api_key: str = Depends(verify_api_key)
        ):
            """Streaming query endpoint."""
            try:
                self._total_requests += 1
                
                def event_generator():
                    for event in self.rag.query_stream(
                        question=request.question,
                        top_k=request.top_k
                    ):
                        import json
                        yield f"data: {json.dumps(event.to_dict())}\n\n"
                
                return StreamingResponse(
                    event_generator(),
                    media_type="text/event-stream"
                )
            except Exception as e:
                self._total_errors += 1
                logger.error(f"Stream query failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/ingest")
        async def ingest(
            request: IngestRequest,
            api_key: str = Depends(verify_api_key)
        ):
            """Ingest document endpoint."""
            try:
                docs = self.rag.ingest(
                    source=request.source,
                    chunking=request.chunking,
                    metadata=request.metadata
                )
                
                return {
                    "status": "success",
                    "documents_ingested": len(docs),
                    "sources": [d.source_path for d in docs]
                }
            except Exception as e:
                self._total_errors += 1
                logger.error(f"Ingestion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/metrics")
        async def metrics(api_key: str = Depends(verify_api_key)):
            """Get server metrics."""
            return {
                "uptime_seconds": time.time() - self._start_time,
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "error_rate": self._total_errors / self._total_requests if self._total_requests > 0 else 0,
                "rag_metrics": {
                    "documents": len(self.rag._documents),
                    "total_cost": self.rag._total_cost if hasattr(self.rag, '_total_cost') else 0
                }
            }
        
        return app
    
    def _check_rate_limit(self, api_key: str) -> None:
        """Check if API key exceeded rate limit.
        
        Args:
            api_key: API key to check
        
        Raises:
            HTTPException: If rate limit exceeded
        """
        from fastapi import HTTPException
        
        now = time.time()
        window = 60.0  # 1 minute window
        
        # Initialize request list for key
        if api_key not in self._request_counts:
            self._request_counts[api_key] = []
        
        # Remove old requests outside window
        self._request_counts[api_key] = [
            t for t in self._request_counts[api_key]
            if now - t < window
        ]
        
        # Check limit
        if len(self._request_counts[api_key]) >= self.rate_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.rate_limit} requests per minute"
            )
        
        # Record request
        self._request_counts[api_key].append(now)


class RAGClient:
    """Python client SDK for RAG API.
    
    Usage::
    
        client = RAGClient(
            base_url="http://localhost:8000",
            api_key="your-api-key"
        )
        
        # Query
        result = client.query("What is ML?")
        print(result["answer"])
        
        # Stream
        for event in client.query_stream("Explain RAG"):
            print(event["text"], end="", flush=True)
        
        # Ingest
        client.ingest("document.pdf")
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """Initialize RAG client.
        
        Args:
            base_url: API server URL
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        logger.info(f"RAG Client initialized: {base_url}")
    
    def _headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        return_citations: bool = True
    ) -> Dict[str, Any]:
        """Query RAG.
        
        Args:
            question: User question
            top_k: Number of chunks
            return_citations: Include citations
        
        Returns:
            Dict with answer, confidence, citations
        """
        import requests
        
        response = requests.post(
            f"{self.base_url}/query",
            json={
                "question": question,
                "top_k": top_k,
                "return_citations": return_citations
            },
            headers=self._headers(),
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def query_stream(self, question: str, top_k: int = 5):
        """Stream query results.
        
        Args:
            question: User question
            top_k: Number of chunks
        
        Yields:
            Stream events
        """
        import requests
        import json
        
        response = requests.post(
            f"{self.base_url}/query/stream",
            json={"question": question, "top_k": top_k},
            headers=self._headers(),
            stream=True,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data = json.loads(line_str[6:])
                    yield data
    
    def ingest(
        self,
        source: str,
        chunking: str = "auto",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest document.
        
        Args:
            source: File path or URL
            chunking: Chunking strategy
            metadata: Document metadata
        
        Returns:
            Ingestion result
        """
        import requests
        
        response = requests.post(
            f"{self.base_url}/ingest",
            json={
                "source": source,
                "chunking": chunking,
                "metadata": metadata
            },
            headers=self._headers(),
            timeout=self.timeout * 3  # Longer timeout for ingestion
        )
        
        response.raise_for_status()
        return response.json()
    
    def health(self) -> Dict[str, Any]:
        """Check server health.
        
        Returns:
            Health status
        """
        import requests
        
        response = requests.get(
            f"{self.base_url}/health",
            timeout=5
        )
        
        response.raise_for_status()
        return response.json()
    
    def metrics(self) -> Dict[str, Any]:
        """Get server metrics.
        
        Returns:
            Server metrics
        """
        import requests
        
        response = requests.get(
            f"{self.base_url}/metrics",
            headers=self._headers(),
            timeout=5
        )
        
        response.raise_for_status()
        return response.json()
