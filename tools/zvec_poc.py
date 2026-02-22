#!/usr/bin/env python3
"""
ChromaDB POC: Semantic Search for YTV2 Summaries

This script demonstrates embedding generation and semantic search
using ChromaDB (embedded vector database) for the YTV2 summary collection.

Note: Originally planned for Zvec, but Zvec doesn't support Intel macOS.
ChromaDB provides similar capabilities and works on all platforms.

Prerequisites:
  pip install chromadb psycopg2-binary

Usage:
  python tools/zvec_poc.py --help
  python tools/zvec_poc.py index                     # Index from JSON files
  python tools/zvec_poc.py index --source postgres   # Index from PostgreSQL
  python tools/zvec_poc.py search "space exploration"
  python tools/zvec_poc.py similar <content_id>
  python tools/zvec_poc.py interactive
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

# PostgreSQL config
PG_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "ytv2",
    "user": "ytv2",
    "password": "pass"
}


def get_postgres_connection():
    """Get PostgreSQL connection."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed")
        print("  pip install psycopg2-binary")
        sys.exit(1)

    return psycopg2.connect(**PG_CONFIG)


def load_reports_from_postgres(limit: Optional[int] = None) -> list[dict]:
    """Load summaries directly from PostgreSQL database."""
    conn = get_postgres_connection()
    cur = conn.cursor()

    # Query content with summaries - use 'comprehensive' or 'bullet-points' variant
    # The content_summaries table has columns: video_id, variant, text (not summary_text)
    query = """
        SELECT DISTINCT ON (c.video_id)
            c.id,
            c.video_id,
            c.title,
            c.channel_name,
            c.analysis_json,
            cs.text as summary_text,
            cs.variant
        FROM content c
        JOIN content_summaries cs ON c.video_id = cs.video_id
        WHERE cs.text IS NOT NULL AND length(cs.text) > 0
          AND cs.variant IN ('comprehensive', 'bullet-points', 'key-insights')
        ORDER BY c.video_id,
          CASE cs.variant
            WHEN 'comprehensive' THEN 1
            WHEN 'bullet-points' THEN 2
            WHEN 'key-insights' THEN 3
          END
    """
    if limit:
        # Need subquery for limit with DISTINCT ON
        query = f"""
            SELECT * FROM (
                {query}
            ) subq LIMIT {limit}
        """

    cur.execute(query)
    rows = cur.fetchall()

    reports = []
    for row in rows:
        content_id, video_id, title, channel_name, analysis_json, summary_text, variant = row

        # Parse categories from analysis_json
        categories = []
        if analysis_json:
            try:
                if isinstance(analysis_json, str):
                    analysis = json.loads(analysis_json)
                else:
                    analysis = analysis_json
                categories = analysis.get("category", [])
            except:
                pass

        # Determine source from ID prefix
        source = "unknown"
        if content_id:
            if content_id.startswith("yt:"):
                source = "youtube"
            elif content_id.startswith("web:"):
                source = "web"
            elif content_id.startswith("reddit:"):
                source = "reddit"

        reports.append({
            "id": content_id or video_id,
            "title": title or "",
            "channel_name": channel_name or "",
            "analysis": {"category": categories} if categories else {},
            "summary_text": summary_text,
            "content_source": source,
        })

    cur.close()
    conn.close()

    print(f"Loaded {len(reports)} summaries from PostgreSQL")
    return reports


def load_reports_from_json(limit: Optional[int] = None) -> list[dict]:
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


def load_reports(limit: Optional[int] = None, source: str = "json") -> list[dict]:
    """Load reports from specified source."""
    if source == "postgres":
        return load_reports_from_postgres(limit)
    else:
        return load_reports_from_json(limit)


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
    if not reports:
        print("No reports to index")
        return

    print(f"\nIndexing {len(reports)} reports...")

    # Prepare data for batch insert
    ids = []
    documents = []
    metadatas = []

    for i, report in enumerate(reports):
        text = extract_text_for_embedding(report)
        if not text.strip():
            continue  # Skip empty documents

        content_id = report.get("id", f"doc_{i}")

        ids.append(content_id)
        documents.append(text)
        metadatas.append({
            "title": report.get("title", "")[:500],
            "source": report.get("content_source", ""),
            "channel": report.get("channel_name", ""),
        })

        if (i + 1) % 25 == 0:
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

    if not results['ids'][0]:
        print("No results found")
        return results

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


def find_similar_by_id(content_id: str, collection, topk: int = 5):
    """Find content similar to a specific document by ID."""
    # Get the document from ChromaDB
    try:
        doc = collection.get(ids=[content_id], include=["documents", "metadatas"])
        if not doc['ids']:
            print(f"Document not found in index: {content_id}")
            return []
        text = doc['documents'][0]
        title = doc['metadatas'][0].get('title', 'N/A')
    except Exception as e:
        print(f"Error fetching document: {e}")
        return []

    print(f"\nFinding content similar to: {title}")

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


def show_stats(collection):
    """Show collection statistics."""
    count = collection.count()
    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Documents: {count}")

    # Check PostgreSQL for comparison
    try:
        conn = get_postgres_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(DISTINCT video_id)
            FROM content_summaries
            WHERE text IS NOT NULL AND length(text) > 0
              AND variant IN ('comprehensive', 'bullet-points', 'key-insights')
        """)
        pg_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        print(f"PostgreSQL summaries (indexable): {pg_count}")
        if count < pg_count:
            print(f"  (!) {pg_count - count} summaries not yet indexed")
    except Exception as e:
        print(f"PostgreSQL: Unable to connect ({e})")


def interactive_mode(collection):
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
        choices=["index", "search", "similar", "interactive", "stats"],
        help="Command to run"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query or content ID for 'similar' command"
    )
    parser.add_argument(
        "--source", "-s",
        choices=["json", "postgres"],
        default="json",
        help="Data source: 'json' (files) or 'postgres' (database)"
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

    if args.command == "stats":
        show_stats(collection)
        return

    if args.command == "index":
        reports = load_reports(args.limit, args.source)
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
        find_similar_by_id(args.query, collection, args.topk)

    elif args.command == "interactive":
        # Check if indexed
        if collection.count() == 0:
            print("Collection is empty. Indexing from PostgreSQL first...")
            reports = load_reports(None, "postgres")
            index_reports(reports, collection)
        interactive_mode(collection)


if __name__ == "__main__":
    main()
