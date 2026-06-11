import logging
import sqlite3
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticCache:
    """Semantic cache with in-memory numpy matrix for fast lookups.

    Supports two backends:
      - SQLite (db_path) — for local development and tests
      - Supabase/Postgres (db_connection) — for production
    """

    def __init__(self, db_path: str = "", db_connection: str = "", dimension: int = 384):
        self.dimension = dimension
        self._use_postgres = bool(db_connection)
        self.db_path = db_path or "cache.db"
        self._db_connection = db_connection
        self._conn = None  # psycopg2 connection (postgres only)

        if self._use_postgres:
            self._connect_postgres()
        self._init_table()
        self._load_into_memory()

    # ── Postgres connection management ──

    def _connect_postgres(self):
        """Create a fresh psycopg2 connection."""
        import psycopg2
        if self._conn and not self._conn.closed:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = psycopg2.connect(self._db_connection)
        logger.info("Cache: connected to Supabase")

    def _ensure_conn(self):
        """Reconnect if the connection is stale or closed (postgres only)."""
        if not self._use_postgres:
            return
        import psycopg2
        if self._conn is None or self._conn.closed:
            self._connect_postgres()
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            logger.warning("Cache: reconnecting to Supabase")
            self._connect_postgres()

    # ── Table/schema init ──

    def _init_table(self):
        if self._use_postgres:
            with self._conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_cache (
                        id BIGSERIAL PRIMARY KEY,
                        query TEXT NOT NULL,
                        embedding BYTEA NOT NULL,
                        response TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                self._conn.commit()
            logger.info("Cache: Postgres table ready")
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        response TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()

    def _load_into_memory(self):
        """Load all cached embeddings into a numpy matrix for fast lookup."""
        if self._use_postgres:
            self._ensure_conn()
            with self._conn.cursor() as cur:
                cur.execute("SELECT embedding, response FROM semantic_cache ORDER BY id")
                rows = cur.fetchall()
            # psycopg2 returns memoryview for BYTEA
            rows = [(bytes(emb), resp) for emb, resp in rows]
        else:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT embedding, response FROM cache")
                rows = cursor.fetchall()

        if not rows:
            self._matrix = np.empty((0, self.dimension), dtype=np.float32)
            self._responses = []
            return

        embeddings = []
        self._responses = []
        for emb_bytes, response in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            if emb.shape[0] == self.dimension:
                embeddings.append(emb)
                self._responses.append(response)

        if embeddings:
            self._matrix = np.stack(embeddings).astype(np.float32)
        else:
            self._matrix = np.empty((0, self.dimension), dtype=np.float32)

        logger.info(f"Cache: loaded {len(self._responses)} entries into memory")

    # ── Public API ──

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def store(self, query: str, embedding: list, response: str):
        """Store in DB and append to in-memory matrix."""
        emb_array = np.array(embedding, dtype=np.float32)
        emb_bytes = emb_array.tobytes()

        if self._use_postgres:
            import psycopg2
            self._ensure_conn()
            with self._conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO semantic_cache (query, embedding, response) VALUES (%s, %s, %s)",
                    (query, psycopg2.Binary(emb_bytes), response),
                )
                self._conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO cache (query, embedding, response) VALUES (?, ?, ?)",
                    (query, emb_bytes, response),
                )
                conn.commit()

        # Append to in-memory structures
        if self._matrix.shape[0] == 0:
            self._matrix = emb_array.reshape(1, -1)
        else:
            self._matrix = np.vstack([self._matrix, emb_array.reshape(1, -1)])
        self._responses.append(response)

    def lookup(self, query_embedding: list, threshold: float = 0.95) -> Optional[str]:
        """Fast lookup using matrix-vector dot product."""
        if self._matrix.shape[0] == 0:
            return None

        q_emb = np.array(query_embedding, dtype=np.float32)
        norm_q = np.linalg.norm(q_emb)
        if norm_q == 0:
            return None

        dots = self._matrix @ q_emb
        norms = np.linalg.norm(self._matrix, axis=1)
        similarities = dots / (norms * norm_q + 1e-10)

        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        if best_sim >= threshold:
            return self._responses[best_idx]
        return None

    def get_stats(self) -> dict:
        return {"total_entries": self._matrix.shape[0]}

    def clear_cache(self):
        if self._use_postgres:
            self._ensure_conn()
            with self._conn.cursor() as cur:
                cur.execute("DELETE FROM semantic_cache")
                self._conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()
        self._matrix = np.empty((0, self.dimension), dtype=np.float32)
        self._responses = []
