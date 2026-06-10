"""
GitLab Handbook Ingestion Pipeline

Fetches markdown files from the GitLab handbook repository via the GitLab API,
generates embeddings, and stores them in Supabase with incremental sync support.
"""

import os
import re
import sys
import time
import hashlib
import argparse
import requests
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client
# from google.genai import Client as GenAIClient  # Gemini embeddings (commented out in favor of local Ollama)

load_dotenv()


# GitLab handbook repo details
GITLAB_REPO_API = "https://gitlab.com/api/v4/projects/gitlab-com%2Fcontent-sites%2Fhandbook/repository"
HANDBOOK_DIRS = [
    "content/handbook/values",
    "content/handbook/company",
    "content/handbook/communication",
    "content/handbook/product",
    "content/handbook/hiring",
    "content/handbook/engineering",
    "content/handbook/marketing",
    "content/handbook/leadership",
    "content/handbook/people-group",
]


def map_path_to_url(path: str) -> str:
    """Convert a file path to the corresponding handbook.gitlab.com URL."""
    clean_path = path.replace("content/", "")
    # Remove _index.md or index.md (directory index files)
    clean_path = re.sub(r"(?:_)?index\.md$", "", clean_path)
    # Remove .md extension from regular files
    clean_path = re.sub(r"\.md$", "", clean_path)
    clean_path = clean_path.rstrip("/")
    return f"https://handbook.gitlab.com/{clean_path}/"


def clean_markdown(content: str) -> str:
    """Strip Hugo frontmatter and clean markdown content."""
    cleaned = re.sub(r"^---[\s\S]*?---", "", content)
    # Remove HTML tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # Remove excessive whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def get_file_hash(text: str) -> str:
    """Generate SHA256 hash for content deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_markdown_files_from_repo(
    dirs: list[str], branch: str = "main", token: Optional[str] = None
) -> dict[str, str]:
    """Fetch markdown files from GitLab repo via API. Returns {relative_path: content}."""
    files = {}
    headers = {}
    if token:
        headers["PRIVATE-TOKEN"] = token

    for dir_path in dirs:
        print(f"  Fetching directory: {dir_path}")
        try:
            # List files in directory recursively
            url = f"{GITLAB_REPO_API}/tree"
            params = {"path": dir_path, "ref": branch, "recursive": "true", "per_page": 100}
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            entries = resp.json()

            for entry in entries:
                if entry["type"] != "blob" or not entry["path"].endswith(".md"):
                    continue

                # Fetch file content
                file_url = f"{GITLAB_REPO_API}/files/{requests.utils.quote(entry['path'], safe='')}"
                file_resp = requests.get(
                    file_url, headers=headers, params={"ref": branch}, timeout=30
                )
                file_resp.raise_for_status()
                file_data = file_resp.json()

                # Decode base64 content
                import base64

                content = base64.b64decode(file_data["content"]).decode("utf-8")
                files[entry["path"]] = content
                print(f"    Fetched: {entry['path']}")

        except requests.RequestException as e:
            print(f"  Warning: Failed to fetch {dir_path}: {e}")
            continue

    return files


class IncrementalSyncManager:
    """Manages incremental sync of handbook files to Supabase."""

    def __init__(self, supabase_url: str, supabase_key: str, ollama_url: str = "http://localhost:11434", ollama_model: str = "embeddinggemma"):
        self.supabase: Client = create_client(
            supabase_url=supabase_url, supabase_key=supabase_key
        )
        # self.genai_client = GenAIClient(api_key=gemini_api_key)  # Gemini embeddings
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.sync_run_id = str(int(time.time()))

    def get_embedding(self, text: str, max_retries: int = 3) -> list[float]:
        """Generate embedding for a single text with retry."""
        results = self.get_embeddings_batch([text], max_retries=max_retries)
        return results[0]

    def get_embeddings_batch(self, texts: list[str], max_retries: int = 3, batch_size: int = 100) -> list[list[float]]:
        """Generate embeddings for multiple texts using Ollama local model."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    resp = requests.post(
                        f"{self.ollama_url}/api/embed",
                        json={"model": self.ollama_model, "input": batch},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    all_embeddings.extend(data["embeddings"])
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = 5 * (attempt + 1)
                        print(f"    Ollama embed failed (attempt {attempt+1}): {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    raise
            else:
                raise RuntimeError(f"Failed to generate embeddings after {max_retries} retries")
        return all_embeddings

    # --- Gemini embeddings (commented out) ---
    # def get_embeddings_batch(self, texts: list[str], max_retries: int = 3, batch_size: int = 100) -> list[list[float]]:
    #     """Generate embeddings for multiple texts, batching in groups of 100 (Gemini API limit)."""
    #     import re
    #
    #     all_embeddings = []
    #     for i in range(0, len(texts), batch_size):
    #         batch = texts[i : i + batch_size]
    #         for attempt in range(max_retries):
    #             try:
    #                 result = self.genai_client.models.embed_content(
    #                     model="gemini-embedding-001", contents=batch
    #                 )
    #                 all_embeddings.extend([emb.values for emb in result.embeddings])
    #                 break
    #             except Exception as e:
    #                 error_str = str(e)
    #                 if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
    #                     delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_str)
    #                     delay = float(delay_match.group(1)) if delay_match else 30
    #                     delay = min(delay * (attempt + 1), 120)
    #                     print(f"    Rate limited. Waiting {delay:.0f}s before retry...")
    #                     time.sleep(delay)
    #                     continue
    #                 raise
    #         else:
    #             raise RuntimeError(f"Failed to generate embeddings after {max_retries} retries")
    #     return all_embeddings

    def sync_file(self, file_rel_path: str, content: str) -> str:
        """Sync a single file to Supabase. Returns 'SKIPPED', 'UPDATED', or 'ERROR'."""
        clean_text = clean_markdown(content)
        if len(clean_text) < 50:
            return "SKIPPED"

        current_hash = get_file_hash(clean_text)

        # Check if file has changed
        db_res = (
            self.supabase.table("documents")
            .select("file_hash")
            .eq("file_path", file_rel_path)
            .execute()
        )

        if db_res.data and db_res.data[0]["file_hash"] == current_hash:
            # File unchanged, just update sync_run_id
            self.supabase.table("documents").update(
                {"last_sync_run_id": self.sync_run_id}
            ).eq("file_path", file_rel_path).execute()
            self.supabase.table("data_embeddings").update(
                {"sync_run_id": self.sync_run_id}
            ).eq("file_path", file_rel_path).execute()
            return "SKIPPED"

        # File changed or new: delete old embeddings and re-index
        self.supabase.table("data_embeddings").delete().eq(
            "file_path", file_rel_path
        ).execute()
        self.supabase.table("documents").delete().eq(
            "file_path", file_rel_path
        ).execute()

        # Insert document record
        self.supabase.table("documents").insert(
            {
                "file_path": file_rel_path,
                "file_hash": current_hash,
                "last_sync_run_id": self.sync_run_id,
            }
        ).execute()

        # Chunk and embed
        chunks = [c.strip() for c in clean_text.split("\n\n") if len(c.strip()) > 30]
        url = map_path_to_url(file_rel_path)

        if not chunks:
            return "SKIPPED"

        # Batch embed all chunks in one API call (Gemini supports up to 2048)
        try:
            embeddings = self.get_embeddings_batch(chunks)
        except Exception as e:
            print(f"    Warning: Failed to embed chunks of {file_rel_path}: {e}")
            return "ERROR"

        # Insert all chunks with their embeddings
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = {
                "title": file_rel_path.replace(".md", "").replace("_", " ").title(),
                "source_path": file_rel_path,
                "url": url,
                "chunk_index": idx,
            }
            self.supabase.table("data_embeddings").insert(
                {
                    "content": chunk,
                    "metadata": metadata,
                    "embedding": embedding,
                    "file_path": file_rel_path,
                    "sync_run_id": self.sync_run_id,
                }
            ).execute()

        return "UPDATED"

    def cleanup_orphans(self):
        """Remove documents and embeddings from previous runs that weren't updated."""
        self.supabase.table("data_embeddings").delete().neq(
            "sync_run_id", self.sync_run_id
        ).execute()
        self.supabase.table("documents").delete().neq(
            "last_sync_run_id", self.sync_run_id
        ).execute()


def main():
    parser = argparse.ArgumentParser(description="GitLab Handbook Ingestion Pipeline")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=HANDBOOK_DIRS,
        help="Directories to fetch from the handbook repo",
    )
    parser.add_argument("--branch", default="main", help="Git branch to fetch from")
    parser.add_argument(
        "--gitlab-token",
        default=os.environ.get("GITLAB_TOKEN"),
        help="GitLab API token (optional, for rate limits)",
    )
    parser.add_argument(
        "--supabase-url",
        default=os.environ.get("SUPABASE_URL"),
        help="Supabase project URL",
    )
    parser.add_argument(
        "--supabase-key",
        default=os.environ.get("SUPABASE_API_KEY"),
        help="Supabase API key",
    )
    # parser.add_argument(
    #     "--gemini-api-key",
    #     default=os.environ.get("GEMINI_API_KEY"),
    #     help="Google Gemini API key",
    # )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama server URL for local embeddings",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_EMBEDDING_MODEL", "embeddinggemma"),
        help="Ollama embedding model name",
    )
    args = parser.parse_args()

    # Validate required args
    if not args.supabase_url:
        print("Error: --supabase-url or SUPABASE_URL env var required")
        sys.exit(1)
    if not args.supabase_key:
        print("Error: --supabase-key or SUPABASE_API_KEY env var required")
        sys.exit(1)
    # if not args.gemini_api_key:
    #     print("Error: --gemini-api-key or GEMINI_API_KEY env var required")
    #     sys.exit(1)

    print("=" * 60)
    print("GitLab Handbook Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Fetch files
    print("\n[1/3] Fetching markdown files from GitLab repository...")
    files = fetch_markdown_files_from_repo(
        dirs=args.dirs, branch=args.branch, token=args.gitlab_token
    )
    print(f"  Found {len(files)} markdown files")

    if not files:
        print("  No files found. Exiting.")
        sys.exit(0)

    # Step 2: Sync to Supabase
    print("\n[2/3] Syncing files to Supabase...")
    manager = IncrementalSyncManager(
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
    )

    stats = {"skipped": 0, "updated": 0, "errors": 0}
    total = len(files)
    for i, (file_path, content) in enumerate(files.items(), 1):
        try:
            print(f"  [{i}/{total}] Syncing: {file_path}")
            result = manager.sync_file(file_path, content)
            if result == "SKIPPED":
                stats["skipped"] += 1
            elif result == "UPDATED":
                stats["updated"] += 1
            else:
                stats["errors"] += 1
        except Exception as e:
            print(f"  Error syncing {file_path}: {e}")
            stats["errors"] += 1

    # Step 3: Cleanup orphans
    print("\n[3/3] Cleaning up orphaned entries...")
    try:
        manager.cleanup_orphans()
        print("  Orphan cleanup complete")
    except Exception as e:
        print(f"  Warning: Orphan cleanup failed: {e}")

    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print(f"  Updated: {stats['updated']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors:  {stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
