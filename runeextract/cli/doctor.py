"""
``runeextract doctor`` — diagnostic CLI for RAG index health.

Usage::

    runeextract doctor ./chroma_db
    runeextract doctor ./chroma_db --samples 100
    runeextract doctor ./chroma_db --fix
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Diagnostic:
    name: str
    status: str  # "pass", "warn", "fail"
    message: str
    detail: str = ""


@dataclass
class DoctorReport:
    index_path: str
    total_chunks: int = 0
    total_documents: int = 0
    diagnostics: List[Diagnostic] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    scan_time_ms: float = 0.0

    def print(self):
        lines = []
        lines.append("")
        lines.append("\033[1m🔍  RuneExtract Doctor — Diagnostic Report\033[0m")
        lines.append("\033[2m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
        lines.append(f"  Index:    {self.index_path}")
        lines.append(f"  Scanned:  \033[1m{self.total_chunks:,}\033[0m chunks across \033[1m{self.total_documents:,}\033[0m documents")
        lines.append(f"  Duration: {self.scan_time_ms:.0f} ms")
        lines.append("")
        for d in self.diagnostics:
            icon = {"pass": "\033[32m✅\033[0m", "warn": "\033[33m⚠️ \033[0m", "fail": "\033[31m❌\033[0m"}.get(d.status, "  ")
            lines.append(f"  {icon}  \033[1m{d.name}\033[0m")
            lines.append(f"       {d.message}")
            if d.detail:
                lines.append(f"       \033[2m{d.detail}\033[0m")
            lines.append("")
        if self.recommendations:
            lines.append("  \033[1m💡  Recommendations\033[0m")
            for r in self.recommendations:
                lines.append(f"     • {r}")
            lines.append("")
        print("\n".join(lines))


def _check_chroma(index_path: str, samples: int, fix: bool) -> DoctorReport:
    import chromadb
    from chromadb.config import Settings

    t0 = time.time()
    report = DoctorReport(index_path=index_path)

    client = chromadb.PersistentClient(
        path=index_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collections = client.list_collections()
    if not collections:
        report.diagnostics.append(Diagnostic("Index health", "fail", "No collections found in database"))
        report.scan_time_ms = (time.time() - t0) * 1000
        return report

    collection = collections[0]
    name = collection.name

    all_data = collection.get(include=["documents", "metadatas", "embeddings"])
    docs = all_data.get("documents", []) or []
    metas = all_data.get("metadatas", []) or []
    embeds = all_data.get("embeddings", []) or []

    report.total_chunks = len(docs)
    sources = set()
    for m in metas:
        src = (m or {}).get("source", "")
        if src:
            sources.add(src)
    report.total_documents = len(sources)

    # --- Index health ---
    if report.total_chunks == 0:
        report.diagnostics.append(Diagnostic("Index health", "fail", "Index is empty (0 chunks)"))
    else:
        embed_dim = len(embeds[0]) if embeds else 0
        report.diagnostics.append(Diagnostic(
            "Index health", "pass",
            f"OK ({report.total_chunks:,} chunks, {report.total_documents:,} docs, {embed_dim}-dim embeddings)",
        ))

    # --- Chunk quality ---
    tiny = 0
    huge = 0
    missing_meta = 0
    empty_chunks = 0
    for i, doc_text in enumerate(docs):
        if not doc_text or not doc_text.strip():
            empty_chunks += 1
        token_est = len((doc_text or "").split())
        if token_est < 10:
            tiny += 1
        elif token_est > 3000:
            huge += 1
        meta = metas[i] if i < len(metas) else {}
        if not meta or not meta.get("source"):
            missing_meta += 1

    quality_issues = []
    if empty_chunks:
        quality_issues.append(f"{empty_chunks} empty")
    if tiny:
        quality_issues.append(f"{tiny} tiny (<10 tokens)")
    if huge:
        quality_issues.append(f"{huge} oversized (>3000 tokens)")
    if missing_meta:
        quality_issues.append(f"{missing_meta} missing source metadata")

    if quality_issues:
        report.diagnostics.append(Diagnostic(
            "Chunk quality", "warn",
            f"{', '.join(quality_issues)} — may degrade retrieval",
        ))
        if tiny:
            report.recommendations.append(
                "Run: runeextract doctor --fix to drop tiny chunks"
            )
    else:
        report.diagnostics.append(Diagnostic("Chunk quality", "pass", "All chunks look healthy"))

    # --- Orphan chunks ---
    orphan_count = 0
    for m in metas:
        src = (m or {}).get("source", "")
        if src and not os.path.exists(src):
            orphan_count += 1
    if orphan_count:
        report.diagnostics.append(Diagnostic(
            "Orphan chunks", "warn",
            f"{orphan_count} chunks reference source files that no longer exist",
        ))
        report.recommendations.append(
            f"Run: runeextract doctor --fix to remove {orphan_count} orphan chunks"
        )
    else:
        report.diagnostics.append(Diagnostic("Orphan chunks", "pass", "None found"))

    # --- Stale documents ---
    stale = []
    for src in sources:
        if os.path.exists(src):
            mtime = os.path.getmtime(src)
            try:
                index_time = _get_index_time(metas, src)
                if index_time and mtime > index_time:
                    stale.append(src)
            except Exception:
                pass
    if stale:
        report.diagnostics.append(Diagnostic(
            "Stale documents", "warn",
            f"{len(stale)} source file(s) modified after last index",
        ))
        for s in stale[:5]:
            report.recommendations.append(f"  Re-index: {os.path.basename(s)}")
        if len(stale) > 5:
            report.recommendations.append(f"  ... and {len(stale) - 5} more")
    else:
        report.diagnostics.append(Diagnostic("Stale documents", "pass", "All sources up to date"))

    # --- Missing metadata fields ---
    required = {"source", "source_type", "document_id", "chunk_id"}
    missing_fields: Dict[str, int] = {}
    for m in metas:
        m = m or {}
        for field in required:
            if not m.get(field):
                missing_fields[field] = missing_fields.get(field, 0) + 1
    if missing_fields:
        details = ", ".join(f"{k} ({v}/{report.total_chunks})" for k, v in missing_fields.items())
        report.diagnostics.append(Diagnostic("Metadata completeness", "warn", f"Missing fields: {details}"))
    else:
        report.diagnostics.append(Diagnostic("Metadata completeness", "pass", "All required fields present"))

    # --- Auto-fix ---
    if fix:
        _run_fixes(report, collection, docs, metas)

    report.scan_time_ms = (time.time() - t0) * 1000
    return report


def _check_faiss(index_path: str, samples: int, fix: bool) -> DoctorReport:
    t0 = time.time()
    report = DoctorReport(index_path=index_path)
    report.diagnostics.append(Diagnostic(
        "Index health", "pass",
        "FAISS index detected — limited introspection available",
    ))
    report.scan_time_ms = (time.time() - t0) * 1000
    return report


def _get_index_time(metas: List[Dict], source: str) -> Optional[float]:
    for m in metas:
        if (m or {}).get("source") == source:
            ts = (m or {}).get("indexed_at")
            if ts:
                try:
                    return float(ts)
                except (ValueError, TypeError):
                    return None
    return None


def _run_fixes(report: DoctorReport, collection, docs: List[str], metas: List[Dict]):
    fixed = 0
    to_delete = []

    # Remove empty chunks
    for i, doc_text in enumerate(docs):
        if not doc_text or not doc_text.strip():
            meta = metas[i] if i < len(metas) else {}
            cid = meta.get("chunk_id", "")
            if cid:
                to_delete.append(cid)

    # Remove tiny chunks
    for i, doc_text in enumerate(docs):
        if doc_text and len(doc_text.split()) < 10:
            meta = metas[i] if i < len(metas) else {}
            cid = meta.get("chunk_id", "")
            if cid:
                to_delete.append(cid)

    # Remove orphan chunks
    for i, m in enumerate(metas):
        m = m or {}
        src = m.get("source", "")
        if src and not os.path.exists(src):
            cid = m.get("chunk_id", "")
            if cid:
                to_delete.append(cid)

    if to_delete:
        unique = list(set(to_delete))
        collection.delete(ids=unique)
        fixed = len(unique)
        report.diagnostics.append(Diagnostic(
            "Auto-fix", "pass" if fixed else "warn",
            f"Removed {fixed} problematic chunk(s)",
        ))
    else:
        report.diagnostics.append(Diagnostic("Auto-fix", "pass", "No fixes needed"))


def run_doctor(index_path: str, samples: int = 100, fix: bool = False) -> DoctorReport:
    """Run diagnostics on a RAG index and return a report.

    Args:
        index_path: Path to the vector store directory (ChromaDB) or FAISS index prefix.
        samples: Number of chunks to sample for analysis (default 100).
        fix: If True, auto-fix issues found (dry-run by default).

    Returns:
        A DoctorReport with findings and recommendations.
    """
    if not os.path.exists(index_path):
        print(f"Error: path not found: {index_path}", file=sys.stderr)
        sys.exit(1)

    meta_file = os.path.join(index_path, "chroma.sqlite3")
    if os.path.exists(meta_file) or os.path.isdir(os.path.join(index_path, "chroma.sqlite3")):
        return _check_chroma(index_path, samples, fix)

    index_file = index_path + ".index"
    if os.path.exists(index_file):
        return _check_faiss(index_path, samples, fix)

    if os.path.isdir(index_path):
        for entry in os.listdir(index_path):
            full = os.path.join(index_path, entry)
            if os.path.isdir(full):
                sqlite = os.path.join(full, "chroma.sqlite3")
                if os.path.exists(sqlite):
                    return _check_chroma(full, samples, fix)
        # Maybe it's a FAISS directory
        for entry in os.listdir(index_path):
            if entry.endswith(".index"):
                prefix = os.path.join(index_path, entry.replace(".index", ""))
                return _check_faiss(prefix, samples, fix)

    print(f"Error: no recognized index at {index_path}", file=sys.stderr)
    print("Supported: ChromaDB directory or FAISS index (.index + .meta.json)", file=sys.stderr)
    sys.exit(1)


def main(args: Optional[List[str]] = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="runeextract doctor",
        description="Diagnose and repair a RAG index",
    )
    parser.add_argument("index_path", help="Path to the vector store directory (ChromaDB) or FAISS index prefix")
    parser.add_argument("--samples", type=int, default=100, help="Number of chunks to sample (default: 100)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues found")
    parsed = parser.parse_args(args)
    report = run_doctor(parsed.index_path, samples=parsed.samples, fix=parsed.fix)
    report.print()
    if any(d.status == "fail" for d in report.diagnostics):
        sys.exit(2)


if __name__ == "__main__":
    main()
