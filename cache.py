import sqlite3
import numpy as np
from typing import Optional


class SemanticCache:
    """Semantic cache using SQLite and cosine similarity on embeddings."""

    def __init__(self, db_path: str = "cache.db"):
        self.db_path = db_path
        self._init_db()

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

    def store(self, query: str, embedding: list, response: str):
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO cache (query, embedding, response) VALUES (?, ?, ?)",
                (query, emb_bytes, response),
            )
            conn.commit()

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot_prod = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_prod / (norm_a * norm_b)

    def lookup(self, query_embedding: list, threshold: float = 0.95) -> Optional[str]:
        q_emb = np.array(query_embedding, dtype=np.float32)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT query, embedding, response FROM cache")
            rows = cursor.fetchall()

            best_match = None
            best_sim = 0.0
            for _, emb_bytes, response in rows:
                db_emb = np.frombuffer(emb_bytes, dtype=np.float32)
                similarity = self.cosine_similarity(q_emb, db_emb)
                if similarity >= threshold and similarity > best_sim:
                    best_sim = similarity
                    best_match = response

        return best_match

    def get_stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]
        return {"total_entries": count}

    def clear_cache(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
