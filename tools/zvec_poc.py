#!/usr/bin/env python3
"""
ChromaDB POC: Semantic Search for YTV2 Summaries

This script demonstrates embedding generation and semantic search
using ChromaDB (embedded vector database) for the YTV2 summary collection.

Note: Originally planned for Zvec, but Zvec doesn't support Intel macOS.
ChromaDB provides similar capabilities and works on all platforms.

Prerequisites:
  pip install chromadb

Usage:
  python tools/zvec_poc.py --help
  python tools/zvec_poc.py index     # Index all summaries
  python tools/zvec_poc.py search "videos about space exploration"
  python tools/zvec_poc.py similar <content_id>  # Find similar content
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try imports with helpful error messages
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ERROR: chromadb not installed")
    print("  pip install chromadb")
    sys.exit(1)


# Configuration
REPORTS_DIR = Path(__file__).parent.parent / "data" / "reports"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chromadb"
COLLECTION_NAME = "ytv2_summaries"


def load_reports(limit: Optional[int] = None) -> list[dict]:
    """Load JSON reports from the reports directory."""
    reports = []
    json_files = sorted(REPORTS_DIR.glob("*.json"))

    if limit:
        json_files = json_files[:limit]

    for fp in json_files:
        try:
            with open(fp) as f:
                report = json.load(f)
                report["_filepath"] = str(fp)
                reports.append(report)
        except Exception as e:
            print(f"Warning: Failed to load {fp}: {e}")

    print(f"Loaded {len(reports)} reports from {REPORTS_DIR}")
    return reports


def extract_text_for_embedding(report: dict) -> str:
    """Extract the text to embed from a report."""
    parts = []

    # Title is important
    if title := report.get("title"):
        parts.append(f"Title: {title}")

    # Channel/source
    if channel := report.get("channel_name"):
        parts.append(f"Channel: {channel}")

    # Categories from analysis
    if analysis := report.get("analysis"):
        if categories := analysis.get("category"):
            parts.append(f"Categories: {', '.join(categories)}")

    # Summary text (most important for semantic search)
    if summaries := report.get("summary_variants"):
        for summary in summaries:
            if text := summary.get("summary_text"):
                parts.append(text[:2000])  # Limit length
                break  # Use first summary
    elif summary_text := report.get("summary_text"):
        parts.append(summary_text[:2000])

    return "\n\n".join(parts)


def get_chroma_client():
    """Get or create ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client


def get_or_create_collection(client):
    """Get or create the collection."""
    # Use default embedding function (all-MiniLM-L6-v2 via ONNX)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "YTV2 content summaries for semantic search"}
    )
    return collection


def index_reports(reports: list[dict], collection):
    """Index reports in ChromaDB."""
    print(f"\nIndexing {len(reports)} reports...")

    # Prepare data for batch insert
    ids = []
    documents = []
    metadatas = []

    for i, report in enumerate(reports):
        text = extract_text_for_embedding(report)
        content_id = report.get("id", f"doc_{i}")

        ids.append(content_id)
        documents.append(text)
        metadatas.append({
            "title": report.get("title", "")[:500],
            "source": report.get("content_source", ""),
            "channel": report.get("channel_name", ""),
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(reports)}")

    # Upsert all documents
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Indexed {len(ids)} documents (upsert)")

    # Print stats
    count = collection.count()
    print(f"Collection now contains {count} documents")


def semantic_search(query: str, collection, topk: int = 10):
    """Search for similar content using semantic similarity."""
    print(f"\nSearching for: '{query}'")

    # Search using ChromaDB's query (automatically generates embedding)
    results = collection.query(
        query_texts=[query],
        n_results=topk
    )

    print(f"\nTop {len(results['ids'][0])} results:\n")
    for i, (doc_id, distance, metadata, doc) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0],
        results['documents'][0]
    ), 1):
        # Convert distance to similarity score (ChromaDB uses L2 distance by default)
        # Lower distance = more similar
        score = 1 / (1 + distance)
        print(f"{i}. [{score:.4f}] {metadata.get('title', 'N/A')}")
        print(f"   Source: {metadata.get('source', 'N/A')} | Channel: {metadata.get('channel', 'N/A')}")
        print(f"   ID: {doc_id}")
        # Show snippet of document
        snippet = doc[:200] + "..." if len(doc) > 200 else doc
        print(f"   Snippet: {snippet}")
        print()

    return results


def find_similar(content_id: str, reports: list[dict], collection, topk: int = 5):
    """Find content similar to a specific report."""
    # Find the report
    report = None
    for r in reports:
        if r.get("id") == content_id:
            report = r
            break

    if not report:
        print(f"Report not found: {content_id}")
        return []

    print(f"\nFinding content similar to: {report.get('title')}")

    # Get the document text and search
    text = extract_text_for_embedding(report)

    # Search (will include the document itself)
    results = collection.query(
        query_texts=[text],
        n_results=topk + 1  # Get one extra to exclude self
    )

    print(f"\nSimilar content:\n")
    count = 0
    for doc_id, distance, metadata in zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0]
    ):
        if doc_id == content_id:
            continue  # Skip the query document itself

        count += 1
        if count > topk:
            break

        score = 1 / (1 + distance)
        print(f"{count}. [{score:.4f}] {metadata.get('title', 'N/A')}")
        print(f"   Source: {metadata.get('source', 'N/A')} | ID: {doc_id}")
        print()

    return results


def interactive_mode(collection, reports):
    """Interactive search mode."""
    print("\n" + "=" * 60)
    print("YTV2 Semantic Search (Interactive Mode)")
    print("Type your query and press Enter. 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Query> ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break

            semantic_search(query, collection)

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(
        description="ChromaDB POC: Semantic Search for YTV2 Summaries"
    )
    parser.add_argument(
        "command",
        choices=["index", "search", "similar", "interactive"],
        help="Command to run"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query or content ID for 'similar' command"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of reports to index (default: all)"
    )
    parser.add_argument(
        "--topk", "-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete existing collection and reindex from scratch"
    )

    args = parser.parse_args()

    # Get ChromaDB client and collection
    client = get_chroma_client()

    # Handle reindex
    if args.reindex:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection for reindex...")
        except Exception:
            pass

    collection = get_or_create_collection(client)

    # Load reports
    reports = load_reports(args.limit)

    if args.command == "index":
        index_reports(reports, collection)

    elif args.command == "search":
        if not args.query:
            print("ERROR: search requires a query argument")
            sys.exit(1)
        semantic_search(args.query, collection, args.topk)

    elif args.command == "similar":
        if not args.query:
            print("ERROR: similar requires a content_id argument")
            sys.exit(1)
        find_similar(args.query, reports, collection, args.topk)

    elif args.command == "interactive":
        # Check if indexed
        if collection.count() == 0:
            print("Collection is empty. Indexing first...")
            index_reports(reports, collection)
        interactive_mode(collection, reports)


if __name__ == "__main__":
    main()
