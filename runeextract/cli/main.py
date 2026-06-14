"""
Command-line interface for RuneExtract.
"""

import argparse
import sys
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runeextract",
        description="One extraction API for every document type - PDF, DOCX, PPTX, XLSX, HTML, Markdown.",
    )
    parser.add_argument("input", nargs="*", help="File(s) or URL(s) to extract")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for images and scanned documents")
    parser.add_argument("--no-tables", action="store_false", dest="tables", help="Skip table extraction")
    parser.add_argument("--no-images", action="store_false", dest="images", help="Skip image extraction")
    parser.add_argument("--no-metadata", action="store_false", dest="metadata", help="Skip metadata extraction")
    parser.add_argument("--chunking", choices=["by_page", "by_heading", "semantic", "fixed_size"], help="Chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Target chunk size in characters (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Character overlap between chunks (default: 100)")
    parser.add_argument("--format", "-f", choices=["text", "json", "pretty"], default="text", help="Output format (default: text)")
    parser.add_argument("--version", "-v", action="store_true", help="Show version and exit")
    return parser


def main() -> None:
    parser = build_parser()
    if '--version' in sys.argv or '-v' in sys.argv:
        from runeextract import __version__
        print(f"runeextract {__version__}")
        sys.exit(0)
    args = parser.parse_args()
    if not args.input:
        parser.print_help()
        sys.exit(1)

    from runeextract import extract, extract_many

    kwargs = {"ocr": args.ocr, "tables": args.tables, "images": args.images, "metadata": args.metadata}
    if args.chunking:
        kwargs["chunking_strategy"] = args.chunking
        kwargs["chunk_size"] = args.chunk_size
        kwargs["chunk_overlap"] = args.chunk_overlap

    if len(args.input) == 1:
        documents = [extract(args.input[0], **kwargs)]
    else:
        documents = extract_many(args.input, **kwargs)

    for i, doc in enumerate(documents):
        if args.format == "json":
            print(json.dumps(doc.to_dict(), indent=2))
        elif args.format == "pretty":
            print(f"=== Document {i + 1} ===")
            print(f"Source: {doc.source_path} ({doc.source_type})")
            print(f"Text length: {len(doc.text)} chars")
            print(f"Tables: {len(doc.tables)}")
            print(f"Images: {len(doc.images)}")
            if doc.metadata:
                print(f"Metadata: {json.dumps(doc.metadata, indent=2)}")
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