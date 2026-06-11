import sqlite3
import numpy as np
from typing import Optional


class SemanticCache:
    """Semantic cache with in-memory numpy matrix for fast lookups."""

    def __init__(self, db_path: str = "cache.db", dimension: int = 384):
        self.db_path = db_path
        self.dimension = dimension
        self._init_db()
        self._matrix: Optional[np.ndarray] = None
        self._responses: list[str] = []
        self._load_into_memory()

    def _init_db(self):
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

    def store(self, query: str, embedding: list, response: str):
        """Store in SQLite and append to in-memory matrix."""
        emb_array = np.array(embedding, dtype=np.float32)
        emb_bytes = emb_array.tobytes()

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

        # Matrix dot product: compute cosine similarity against all cached embeddings at once
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        self._matrix = np.empty((0, self.dimension), dtype=np.float32)
        self._responses = []
