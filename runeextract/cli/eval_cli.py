"""
``runeextract eval`` — evaluate RAG pipeline quality with regression gating.

Usage::

    runeextract eval --index ./chroma_db
    runeextract eval --index ./chroma_db --dataset ./golden_qa.jsonl
    runeextract eval --index ./chroma_db --judge gpt-4o-mini --report report.html
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


BASELINE_FILE = ".runeextract_baseline.json"


@dataclass
class EvalReport:
    index_path: str
    dataset_path: str
    n_questions: int = 0
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regressions: List[Dict[str, Any]] = field(default_factory=list)
    improvements: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0

    def print(self):
        lines = []
        lines.append("")
        lines.append("\033[1m📊  RuneExtract Eval Report\033[0m")
        lines.append("\033[2m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
        lines.append(f"  Index:     {self.index_path}")
        lines.append(f"  Dataset:   {os.path.basename(self.dataset_path)} ({self.n_questions} questions)")
        lines.append(f"  Duration:  {self.duration_ms:.0f} ms")
        lines.append("")
        for metric_name, agg in self.metrics.items():
            label = metric_name.replace("_", " ").title()
            lines.append(f"  \033[1m{label}\033[0m")
            lines.append(f"    Mean: {agg.get('mean', 0.0):.3f}   "
                         f"Std: {agg.get('std', 0.0):.3f}   "
                         f"Min: {agg.get('min', 0.0):.3f}   "
                         f"Max: {agg.get('max', 0.0):.3f}")
            lines.append("")
        if self.regressions:
            lines.append(f"\033[31m🔴  {len(self.regressions)} Regression(s) detected\033[0m")
            for r in self.regressions:
                lines.append(f"     • {r['metric']}: {r['previous']:.3f} → {r['current']:.3f} "
                             f"(Δ {r['delta']:+.3f})")
                if r.get("example"):
                    lines.append(f"       e.g. \"{r['example']}\"")
            lines.append("")
        if self.improvements:
            lines.append(f"\033[32m🟢  {len(self.improvements)} Improvement(s)\033[0m")
            for r in self.improvements:
                lines.append(f"     • {r['metric']}: {r['previous']:.3f} → {r['current']:.3f} "
                             f"(Δ {r['delta']:+.3f})")
            lines.append("")
        print("\n".join(lines))

    def to_html(self, path: str):
        rows = ""
        for metric_name, agg in self.metrics.items():
            label = metric_name.replace("_", " ").title()
            rows += f"""
            <tr>
                <td>{label}</td>
                <td>{agg.get('mean', 0.0):.3f}</td>
                <td>{agg.get('std', 0.0):.3f}</td>
                <td>{agg.get('min', 0.0):.3f}</td>
                <td>{agg.get('max', 0.0):.3f}</td>
                <td>{agg.get('count', 0)}</td>
            </tr>"""
        reg_rows = ""
        for r in self.regressions:
            cls = "regression" if r["delta"] < 0 else "improvement"
            reg_rows += f"""
            <tr class="{cls}">
                <td>{r['metric']}</td>
                <td>{r['previous']:.3f}</td>
                <td>{r['current']:.3f}</td>
                <td>{r['delta']:+.3f}</td>
            </tr>"""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RuneExtract Eval Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 2rem; background: #f9f9f9; color: #222; }}
h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
.meta {{ color: #666; font-size: 0.9rem; margin-bottom: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; background: #fff; border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #f0f0f0; font-weight: 500; }}
tr.regression {{ background: #fff0f0; }}
tr.improvement {{ background: #f0fff0; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }}
.badge.red {{ background: #fee; color: #c33; }}
.badge.green {{ background: #efe; color: #3a3; }}
</style>
</head>
<body>
<h1>📊 RuneExtract Evaluation Report</h1>
<div class="meta">
    <p>Index: {self.index_path}</p>
    <p>Dataset: {os.path.basename(self.dataset_path)} ({self.n_questions} questions)</p>
    <p>Duration: {self.duration_ms:.0f} ms</p>
</div>
<h2>Metrics</h2>
<table>
    <thead><tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Count</th></tr></thead>
    <tbody>{rows}</tbody>
</table>
{"<h2>Regressions / Improvements</h2><table><thead><tr><th>Metric</th><th>Previous</th><th>Current</th><th>Delta</th></tr></thead><tbody>" + reg_rows + "</tbody></table>" if reg_rows else ""}
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)


def _load_or_generate_dataset(index_path: str, dataset_path: Optional[str],
                               samples: int) -> tuple:
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            if dataset_path.endswith(".jsonl"):
                test_set = [json.loads(line) for line in f if line.strip()]
            else:
                test_set = json.load(f)
        return test_set, dataset_path

    from runeextract.rag.evaluate import RAGEvaluator
    from runeextract.retriever import ChromaRetriever
    from runeextract import extract

    retriever = ChromaRetriever(persist_directory=index_path)
    sources = retriever.list_sources()
    documents = []
    for src in sources:
        if os.path.exists(src):
            try:
                documents.append(extract(src))
            except Exception as exc:
                print(f"  Warning: could not extract {src}: {exc}", file=sys.stderr)

    if not documents:
        print("Error: no documents could be loaded to generate test set", file=sys.stderr)
        sys.exit(1)

    print(f"  Generating test set from {len(documents)} documents...", file=sys.stderr)
    evaluator = RAGEvaluator()
    from runeextract.processors.ai import AIProcessor
    try:
        ai = AIProcessor()
        evaluator.llm_complete = ai.complete
    except Exception:
        print("  Warning: no LLM available — test set will be empty", file=sys.stderr)

    test_set = evaluator.generate_test_set(documents, num_questions=samples)
    ds_path = dataset_path or os.path.join(os.path.dirname(index_path) or ".", "_generated_test_set.json")
    with open(ds_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2)
    print(f"  Generated {len(test_set)} Q&A pairs → {ds_path}", file=sys.stderr)
    return test_set, ds_path


def _load_baseline() -> Optional[Dict]:
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r") as f:
            return json.load(f)
    return None


def _save_baseline(metrics: Dict):
    with open(BASELINE_FILE, "w") as f:
        json.dump(metrics, f, indent=2)


def run_eval(index_path: str, dataset_path: Optional[str] = None,
             judge: Optional[str] = None, samples: int = 50,
             report_path: Optional[str] = None,
             update_baseline: bool = False) -> EvalReport:
    """Evaluate a RAG index and compare against a stored baseline.

    Args:
        index_path: Path to the ChromaDB persist directory.
        dataset_path: Path to a JSON/JSONL file of Q&A pairs. If None, auto-generates.
        judge: LLM model to use as judge (e.g. "gpt-4o-mini").
        samples: Number of test questions to use (default 50).
        report_path: If set, writes an HTML report to this path.
        update_baseline: If True, updates the stored baseline after evaluation.

    Returns:
        An EvalReport with metrics and regression information.
    """
    t0 = time.time()

    from runeextract.rag.evaluate import RAGEvaluator
    from runeextract.rag.auto_pipeline import auto_rag

    test_set, ds_path = _load_or_generate_dataset(index_path, dataset_path, samples)

    if not test_set:
        print("Error: empty test set — cannot evaluate", file=sys.stderr)
        sys.exit(1)

    rag = auto_rag(index_path, vector_store="chromadb", persist_directory=index_path)

    def query_fn(question, top_k=5, return_citations=True):
        return rag.query(question, top_k=top_k, cite=return_citations)

    evaluator = RAGEvaluator(query_fn=query_fn)
    if judge:
        from runeextract.processors.ai import AIProcessor
        try:
            ai = AIProcessor(model=judge)
            evaluator.llm_complete = ai.complete
        except Exception as exc:
            print(f"  Warning: judge AI failed: {exc}", file=sys.stderr)

    metrics = evaluator.evaluate(test_set)

    # Build report
    report = EvalReport(
        index_path=index_path,
        dataset_path=ds_path,
        n_questions=len(test_set),
        metrics=metrics,
        duration_ms=(time.time() - t0) * 1000,
    )

    # Compare against baseline
    baseline = _load_baseline()
    if baseline:
        for metric_name, agg in metrics.items():
            prev_mean = baseline.get(metric_name, {}).get("mean")
            curr_mean = agg.get("mean", 0.0)
            if prev_mean is not None:
                delta = curr_mean - prev_mean
                entry = {
                    "metric": metric_name.replace("_", " ").title(),
                    "previous": prev_mean,
                    "current": curr_mean,
                    "delta": delta,
                }
                if delta < -0.02:
                    report.regressions.append(entry)
                elif delta > 0.02:
                    report.improvements.append(entry)

    if update_baseline:
        _save_baseline(metrics)
        print(f"  Baseline updated → {BASELINE_FILE}", file=sys.stderr)

    if report_path:
        report.to_html(report_path)
        print(f"  Report written → {report_path}", file=sys.stderr)

    return report


def main(args: Optional[List[str]] = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="runeextract eval",
        description="Evaluate RAG pipeline quality with regression gating",
    )
    parser.add_argument("--index", required=True, help="Path to the ChromaDB persist directory")
    parser.add_argument("--dataset", help="Path to Q&A dataset (JSON/JSONL). Auto-generates if not set")
    parser.add_argument("--judge", help="LLM judge model (e.g. gpt-4o-mini)")
    parser.add_argument("--samples", type=int, default=50, help="Number of test questions (default: 50)")
    parser.add_argument("--report", help="Write HTML report to this path")
    parser.add_argument("--update-baseline", action="store_true", help="Update stored baseline after eval")
    parsed = parser.parse_args(args)

    report = run_eval(
        index_path=parsed.index,
        dataset_path=parsed.dataset,
        judge=parsed.judge,
        samples=parsed.samples,
        report_path=parsed.report,
        update_baseline=parsed.update_baseline,
    )
    report.print()

    if report.regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
