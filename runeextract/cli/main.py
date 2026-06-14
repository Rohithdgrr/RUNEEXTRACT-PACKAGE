"""
Command-line interface for RuneExtract.
"""

import argparse
import sys
import json
import os
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runeextract",
        description="One extraction API for every document type - PDF, DOCX, PPTX, XLSX, HTML, Markdown.",
    )
    parser.add_argument("input", nargs="*", help="File(s) or URL(s) to extract")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for images and scanned documents")
    parser.add_argument("--ocr-lang", type=str, default="en", help="OCR language(s), comma-separated (default: en)")
    parser.add_argument("--no-tables", action="store_false", dest="tables", help="Skip table extraction")
    parser.add_argument("--no-images", action="store_false", dest="images", help="Skip image extraction")
    parser.add_argument("--no-metadata", action="store_false", dest="metadata", help="Skip metadata extraction")
    parser.add_argument("--chunking", choices=["by_page", "by_heading", "semantic", "fixed_size", "by_token"], help="Chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Target chunk size in characters (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Character overlap between chunks (default: 100)")
    parser.add_argument("--chunk-tokens", action="store_true", help="Use token-based chunking (requires tiktoken)")
    parser.add_argument("--no-cache", action="store_true", help="Disable extraction cache")
    parser.add_argument("--format", "-f", choices=["text", "json", "pretty", "markdown"], default="text", help="Output format (default: text)")
    parser.add_argument("--output-dir", "-o", type=str, help="Directory to write output files (one per input)")
    parser.add_argument("--tree", action="store_true", help="Show document structure tree")
    parser.add_argument("--watch", type=str, metavar="DIR", help="Watch a directory for new files and extract them")
    parser.add_argument("--youtube-format", choices=["transcript", "metadata", "chapters", "full"], default="full",
                        help="YouTube output format (default: full)")
    parser.add_argument("--ai-summarize", action="store_true", help="Run AI summary after extraction")
    parser.add_argument("--version", "-v", action="store_true", help="Show version and exit")
    return parser


def _show_tree(doc, prefix: str = "") -> str:
    lines = []
    lines.append(f"{prefix}[FILE] {doc.source_path or 'document'} ({doc.source_type})")
    lines.append(f"{prefix}  ├── text: {len(doc.text)} chars")
    lines.append(f"{prefix}  ├── tables: {len(doc.tables)}")
    for i, t in enumerate(doc.tables):
        lines.append(f"{prefix}  │   └── table[{i}]: {len(t.columns)} cols x {len(t.rows)} rows")
    lines.append(f"{prefix}  ├── images: {len(doc.images)}")
    if doc._chunks:
        lines.append(f"{prefix}  └── chunks: {len(doc._chunks)}")
    return "\n".join(lines)


def _watch_directory(watch_dir: str, kwargs: dict) -> None:
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("--watch requires 'watchdog'. Install with: pip install watchdog", file=sys.stderr)
        sys.exit(1)

    processed = set()
    from runeextract import extract

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            if event.src_path in processed:
                return
            processed.add(event.src_path)
            print(f"\n[watch] New file: {event.src_path}")
            try:
                doc = extract(event.src_path, **kwargs)
                print(f"[watch] Extracted: {len(doc.text)} chars, {len(doc.tables)} tables")
            except Exception as e:
                print(f"[watch] Error: {e}")

    observer = Observer()
    observer.schedule(Handler(), watch_dir, recursive=False)
    observer.start()
    print(f"Watching {watch_dir} for new files... (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def main() -> None:
    parser = build_parser()
    if '--version' in sys.argv or '-v' in sys.argv:
        from runeextract import __version__
        print(f"runeextract {__version__}")
        sys.exit(0)

    args = parser.parse_args()

    # Handle --watch mode
    if args.watch:
        kwargs = {"ocr": args.ocr, "tables": args.tables, "images": args.images, "metadata": args.metadata, "use_cache": not args.no_cache}
        if args.chunk_tokens:
            kwargs["chunking_strategy"] = "by_token"
        if args.chunking:
            kwargs["chunking_strategy"] = args.chunking
            kwargs["chunk_size"] = args.chunk_size
            kwargs["chunk_overlap"] = args.chunk_overlap
        _watch_directory(args.watch, kwargs)
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    from runeextract import extract, extract_many

    kwargs = {
        "ocr": args.ocr, "tables": args.tables, "images": args.images, "metadata": args.metadata,
        "ocr_languages": [l.strip() for l in args.ocr_lang.split(",") if l.strip()],
        "use_cache": not args.no_cache,
    }
    if args.chunk_tokens:
        kwargs["chunking_strategy"] = "by_token"
    if args.chunking:
        kwargs["chunking_strategy"] = args.chunking
        kwargs["chunk_size"] = args.chunk_size
        kwargs["chunk_overlap"] = args.chunk_overlap

    if args.youtube_format != "full":
        kwargs["youtube_format"] = args.youtube_format

    if len(args.input) == 1:
        documents = [extract(args.input[0], **kwargs)]
    else:
        documents = extract_many(args.input, **kwargs)

    # Run AI summary if requested
    if args.ai_summarize:
        for i, doc in enumerate(documents):
            try:
                s = doc.summary()
                print(f"\n=== AI Summary [{i + 1}] ===")
                print(s)
            except Exception as exc:
                print(f"\nAI summary failed for doc {i + 1}: {exc}")

    # --output-dir writes each document to a separate file
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for doc in documents:
            ext = os.path.splitext(os.path.basename(doc.source_path or "output"))[0]
            if args.format == "json":
                out_path = os.path.join(args.output_dir, f"{ext}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(doc.to_dict(), f, indent=2, default=str)
            elif args.format == "markdown":
                out_path = os.path.join(args.output_dir, f"{ext}.md")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(doc.to_markdown())
            else:
                out_path = os.path.join(args.output_dir, f"{ext}.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(doc.text)
            print(f"Written: {out_path}")
        return

    for i, doc in enumerate(documents):
        if args.tree:
            print(_show_tree(doc))
            continue

        if args.format == "json":
            print(json.dumps(doc.to_dict(), indent=2, default=str))
        elif args.format == "markdown":
            print(doc.to_markdown())
        elif args.format == "pretty":
            print(f"=== Document {i + 1} ===")
            print(f"Source: {doc.source_path} ({doc.source_type})")
            print(f"Text length: {len(doc.text)} chars")
            print(f"Tables: {len(doc.tables)}")
            print(f"Images: {len(doc.images)}")
            if doc.metadata:
                print(f"Metadata: {json.dumps(doc.metadata, indent=2, default=str)}")
            if doc._chunks:
                print(f"Chunks: {len(doc._chunks)}")
            print()
            print(doc.text[:2000])
            if len(doc.text) > 2000:
                print(f"... ({len(doc.text) - 2000} more characters)")
            print()
        else:
            if i > 0:
                print()
                print("=" * 40)
                print()
            print(doc.text)


if __name__ == "__main__":
    main()
