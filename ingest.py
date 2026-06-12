import os
import re
import sys
import time
import hashlib
import argparse
import requests
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# GitLab handbook details
GITLAB_REPO_API = "https://gitlab.com/api/v4/projects/gitlab-com%2Fcontent-sites%2Fhandbook/repository"
HANDBOOK_DIRS = [
    "content/handbook/about",
    "content/handbook/acquisitions",
    "content/handbook/alliances",
    "content/handbook/board-meetings",
    "content/handbook/business-technology",
    "content/handbook/ceo",
    "content/handbook/communication",
    "content/handbook/company",
    "content/handbook/customer-experience",
    "content/handbook/customer-success",
    "content/handbook/eba",
    "content/handbook/engineering",
    "content/handbook/enterprise-data",
    "content/handbook/entity",
    "content/handbook/eta",
    "content/handbook/finance",
    "content/handbook/hiring",
    "content/handbook/it",
    "content/handbook/job-description-library",
    "content/handbook/labor-and-employment-notices",
    "content/handbook/leadership",
    "content/handbook/legal",
    "content/handbook/marketing",
    "content/handbook/people-group",
    "content/handbook/people-policies",
    "content/handbook/product",
    "content/handbook/product-development",
    "content/handbook/resellers",
    "content/handbook/sales",
    "content/handbook/security",
    "content/handbook/solutions-architects",
    "content/handbook/support",
    "content/handbook/teamops",
    "content/handbook/tools-and-tips",
    "content/handbook/total-rewards",
    "content/handbook/upstream-studios",
    "content/handbook/values",
]


def map_path_to_url(path: str) -> str:
    clean_path = path.replace("content/", "")
    clean_path = re.sub(r"(?:_)?index\.md$", "", clean_path)
    clean_path = re.sub(r"\.md$", "", clean_path)
    clean_path = clean_path.rstrip("/")
    return f"https://handbook.gitlab.com/{clean_path}/"


def clean_markdown(content: str) -> str:
    cleaned = re.sub(r"^---[\s\S]*?---", "", content)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def get_file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def merge_paragraphs(paragraphs: list[str], min_chars: int = 500, max_chars: int = 1500) -> list[str]:
    if not paragraphs:
        return []

    chunks = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            if len(current) >= min_chars:
                chunks.append(current.strip())
                current = para
            else:
                current += "\n\n" + para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        # Too small? merge with the previous chunk instead
        if len(current) < min_chars and chunks:
            chunks[-1] += "\n\n" + current.strip()
        else:
            chunks.append(current.strip())

    return chunks


def fetch_markdown_files_from_repo(
    dirs: list[str], branch: str = "main", token: Optional[str] = None
) -> dict[str, str]:
    files = {}
    headers = {}
    if token:
        headers["PRIVATE-TOKEN"] = token

    for dir_path in dirs:
        print(f"  Fetching directory: {dir_path}")
        try:
            url = f"{GITLAB_REPO_API}/tree"
            params = {"path": dir_path, "ref": branch, "recursive": "true", "per_page": 100}
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            entries = resp.json()

            for entry in entries:
                if entry["type"] != "blob" or not entry["path"].endswith(".md"):
                    continue

                file_url = f"{GITLAB_REPO_API}/files/{requests.utils.quote(entry['path'], safe='')}"
                file_resp = requests.get(
                    file_url, headers=headers, params={"ref": branch}, timeout=30
                )
                file_resp.raise_for_status()
                file_data = file_resp.json()

                import base64
                content = base64.b64decode(file_data["content"]).decode("utf-8")
                files[entry["path"]] = content
                print(f"    Fetched: {entry['path']}")

        except requests.RequestException as e:
            print(f"  Warning: Failed to fetch {dir_path}: {e}")
            continue

    return files


def read_markdown_files_from_local(
    local_dir: str, dirs: list[str]
) -> dict[str, str]:
    """Read markdown files from a local handbook clone. Returns {relative_path: content}.

    `dirs` are repo-relative paths like "content/handbook/values".
    `local_dir` is the path to the cloned repo root (e.g. "handbook").
    The returned keys match the API format so map_path_to_url() still works.
    """
    base = Path(local_dir)
    files = {}

    for dir_path in dirs:
        local_path = base / dir_path
        if not local_path.is_dir():
            print(f"  Warning: Directory not found: {local_path}")
            continue

        print(f"  Reading directory: {dir_path}")
        for md_file in sorted(local_path.rglob("*.md")):
            # Store path relative to repo root (matches API format)
            rel_path = str(md_file.relative_to(base))
            try:
                content = md_file.read_text(encoding="utf-8")
                files[rel_path] = content
                print(f"    Read: {rel_path}")
            except Exception as e:
                print(f"    Warning: Failed to read {rel_path}: {e}")

    return files


def main():
    parser = argparse.ArgumentParser(description="GitLab Handbook Ingestion Pipeline (vecs)")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=HANDBOOK_DIRS,
        help="Directories to fetch from the handbook repo",
    )
    parser.add_argument(
        "--local-dir",
        default="handbook",
        help="Path to local handbook repo clone (default: handbook)",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Fetch from GitLab API instead of reading from local clone",
    )
    parser.add_argument("--branch", default="main", help="Git branch to fetch from (API mode)")
    parser.add_argument(
        "--gitlab-token",
        default=os.environ.get("GITLAB_TOKEN"),
        help="GitLab API token (optional, for rate limits)",
    )
    parser.add_argument(
        "--db-connection",
        default=os.environ.get("SUPABASE_DB_CONNECTION"),
        help="PostgreSQL connection string for Supabase (required for vecs)",
    )
    parser.add_argument(
        "--min-chunk",
        type=int,
        default=500,
        help="Minimum chunk size in characters (default: 500)",
    )
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=1500,
        help="Maximum chunk size in characters (default: 1500)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing embeddings before ingesting",
    )
    args = parser.parse_args()

    if not args.db_connection:
        print("Error: --db-connection or SUPABASE_DB_CONNECTION env var required")
        print("Format: postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres")
        sys.exit(1)

    print("=" * 60)
    print("GitLab Handbook Ingestion Pipeline (vecs)")
    print("=" * 60)

    # Fetch files
    if args.use_api:
        print("\n[1/3] Fetching markdown files from GitLab repository (API mode)...")
        files = fetch_markdown_files_from_repo(
            dirs=args.dirs, branch=args.branch, token=args.gitlab_token
        )
    else:
        local_path = Path(args.local_dir)
        if not local_path.is_dir():
            print(f"Error: Local directory '{args.local_dir}' not found.")
            print("Clone the repo first or use --use-api to fetch from GitLab.")
            sys.exit(1)
        print(f"\n[1/3] Reading markdown files from local clone: {args.local_dir}/...")
        files = read_markdown_files_from_local(
            local_dir=args.local_dir, dirs=args.dirs
        )
    print(f"  Found {len(files)} markdown files")

    if not files:
        print("  No files found. Exiting.")
        sys.exit(0)

    # Initialize vecs + embedding model
    print("\n[2/4] Setting up vecs collection...")
    import vecs
    import numpy as np
    from sentence_transformers import SentenceTransformer

    vx = vecs.create_client(args.db_connection)
    collection = vx.get_or_create_collection(
        name="handbook_embeddings",
        dimension=384,
    )

    if args.clear:
        print("\n  Clearing existing embeddings...")
        vx.delete_collection("handbook_embeddings")
        collection = vx.get_or_create_collection(
            name="handbook_embeddings",
            dimension=384,
        )
        print("  Done.")

    print(f"\n[3/4] Chunking files (min={args.min_chunk}, max={args.max_chunk} chars)...")
    embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Collect all chunks and their metadata
    all_chunks = []       # flat list of chunk texts
    chunk_meta = []       # parallel list of (doc_id, metadata)
    stats = {"files": 0, "skipped": 0}

    for i, (file_path, content) in enumerate(files.items(), 1):
        clean_text = clean_markdown(content)
        if len(clean_text) < 100:
            stats["skipped"] += 1
            continue

        paragraphs = [p.strip() for p in clean_text.split("\n\n") if len(p.strip()) > 20]
        chunks = merge_paragraphs(paragraphs, min_chars=args.min_chunk, max_chars=args.max_chunk)
        if not chunks:
            stats["skipped"] += 1
            continue

        url = map_path_to_url(file_path)
        title = file_path.replace(".md", "").replace("_", " ").title()

        for idx, chunk in enumerate(chunks):
            doc_id = f"{file_path}::{idx}"
            metadata = {
                "title": title,
                "source_path": file_path,
                "url": url,
                "chunk_index": idx,
                "content": chunk,
                "file_path": file_path,
            }
            all_chunks.append(chunk)
            chunk_meta.append((doc_id, metadata))

        stats["files"] += 1

    print(f"  {stats['files']} files → {len(all_chunks)} chunks ({stats['skipped']} files skipped)")

    # Embed using batches for faster processing
    print(f"\n  Embedding {len(all_chunks)} chunks...")
    EMBED_BATCH = 512
    all_embeddings = []
    t0 = time.time()

    for offset in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[offset : offset + EMBED_BATCH]
        print(f"    Batch {offset//EMBED_BATCH + 1} ({len(batch)} chunks)...", end="", flush=True)
        batch_t0 = time.time()
        batch_embs = embedding_model.encode(batch, batch_size=64, show_progress_bar=False)
        all_embeddings.extend(batch_embs)
        batch_time = time.time() - batch_t0
        elapsed = time.time() - t0
        done = min(offset + EMBED_BATCH, len(all_chunks))
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(all_chunks) - done) / rate if rate > 0 else 0
        print(f" {batch_time:.1f}s ({done}/{len(all_chunks)}, ~{eta:.0f}s remaining)")

    # Make records for upsert (doc_id, embedding, metadata)
    all_records = [
        (doc_id, np.array(emb), meta)
        for (doc_id, meta), emb in zip(chunk_meta, all_embeddings)
    ]
    print(f"  Done. {len(all_records)} records ready.")

    # Upsert all records in batches
    print("\n[4/4] Upserting into Supabase...")
    BATCH_SIZE = 500
    upserted = 0
    errors = 0
    t0 = time.time()

    for batch_start in range(0, len(all_records), BATCH_SIZE):
        batch = all_records[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(all_records) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"    Batch {batch_num}/{total_batches} ({len(batch)} records)...", end="", flush=True)
        batch_t0 = time.time()
        try:
            collection.upsert(batch)
            upserted += len(batch)
            batch_time = time.time() - batch_t0
            elapsed = time.time() - t0
            done = min(batch_start + BATCH_SIZE, len(all_records))
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(all_records) - done) / rate if rate > 0 else 0
            print(f" {batch_time:.1f}s ({done}/{len(all_records)}, ~{eta:.0f}s remaining)")
        except Exception as e:
            errors += 1
            print(f" ERROR: {e}")

    stats["ingested"] = upserted
    stats["errors"] = errors

    # INdexing
    print("\n  Creating vector index...")
    import psycopg2
    idx_conn = psycopg2.connect(args.db_connection)
    idx_conn.autocommit = True
    idx_cur = idx_conn.cursor()
    idx_cur.execute("SET statement_timeout = '600000'")  # 10 min
    idx_cur.execute("""
        CREATE INDEX IF NOT EXISTS ix_vector_cosine_ops_hnsw_m16_efc64
        ON vecs."handbook_embeddings"
        USING hnsw (vec vector_cosine_ops) WITH (m=16, ef_construction=64)
    """)
    idx_cur.close()
    idx_conn.close()
    print("  Done.")

    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print(f"  Files processed: {stats['files']}")
    print(f"  Chunks upserted: {stats['ingested']}")
    print(f"  Files skipped:   {stats['skipped']}")
    print(f"  Batch errors:    {stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
