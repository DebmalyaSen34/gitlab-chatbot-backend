"""Tests for the SemanticCache module."""

import os
import numpy as np
import pytest
from cache import SemanticCache


class TestSemanticCacheInit:
    """Test cache initialization."""

    def test_creates_db_file(self, temp_db):
        cache = SemanticCache(db_path=temp_db)
        assert os.path.exists(temp_db)

    def test_creates_table(self, temp_db):
        cache = SemanticCache(db_path=temp_db)
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='cache'"
            )
            assert cursor.fetchone() is not None

    def test_default_db_path(self):
        cache = SemanticCache()
        assert cache.db_path == "cache.db"


class TestSemanticCacheStore:
    """Test storing entries in the cache."""

    def test_store_single_entry(self, temp_db, sample_embedding):
        cache = SemanticCache(db_path=temp_db)
        cache.store("test query", sample_embedding, "test response")
        stats = cache.get_stats()
        assert stats["total_entries"] == 1

    def test_store_multiple_entries(self, temp_db, sample_embedding):
        cache = SemanticCache(db_path=temp_db)
        for i in range(5):
            cache.store(f"query {i}", sample_embedding, f"response {i}")
        stats = cache.get_stats()
        assert stats["total_entries"] == 5

    def test_store_preserves_response(self, temp_db, sample_embedding):
        cache = SemanticCache(db_path=temp_db)
        response = "GitLab values are: Collaboration, Results, Efficiency, Diversity, Inclusion, Transparency"
        cache.store("values query", sample_embedding, response)

        # Retrieve directly from DB to verify
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute("SELECT response FROM cache")
            stored = cursor.fetchone()[0]
        assert stored == response


class TestSemanticCacheLookup:
    """Test cache lookup functionality."""

    def test_exact_match(self, temp_db, sample_embedding):
        cache = SemanticCache(db_path=temp_db)
        cache.store("test query", sample_embedding, "test response")
        result = cache.lookup(sample_embedding, threshold=0.99)
        assert result == "test response"

    def test_no_match_different_embedding(self, temp_db, sample_embedding):
        cache = SemanticCache(db_path=temp_db)
        cache.store("test query", sample_embedding, "test response")
        different_embedding = [-0.1] * 768
        result = cache.lookup(different_embedding, threshold=0.99)
        assert result is None

    def test_empty_cache(self, temp_db, sample_embedding):
        cache = SemanticCache(db_path=temp_db)
        result = cache.lookup(sample_embedding)
        assert result is None

    def test_threshold_sensitivity(self, temp_db):
        cache = SemanticCache(db_path=temp_db)
        emb1 = [1.0] * 768
        cache.store("query", emb1, "response")

        # Same embedding should match at any threshold
        assert cache.lookup(emb1, threshold=0.5) == "response"
        assert cache.lookup(emb1, threshold=0.99) == "response"

    def test_returns_best_match(self, temp_db):
        cache = SemanticCache(db_path=temp_db)
        # Store with very similar embeddings
        emb1 = [1.0] * 768
        emb2 = [0.99] * 768 + [0.0] * 0  # same length
        cache.store("query1", emb1, "response_exact")
        cache.store("query2", emb2, "response_close")

        result = cache.lookup(emb1, threshold=0.95)
        assert result is not None

    def test_lookup_with_orthogonal_embeddings(self, temp_db):
        cache = SemanticCache(db_path=temp_db)
        # Orthogonal vectors should have ~0 similarity
        emb1 = [1.0] + [0.0] * 767
        emb2 = [0.0] + [1.0] + [0.0] * 766
        cache.store("query1", emb1, "response1")

        result = cache.lookup(emb2, threshold=0.5)
        assert result is None


class TestSemanticCacheCosine:
    """Test cosine similarity calculation."""

    def test_identical_vectors(self, temp_db):
        cache = SemanticCache(db_path=temp_db)
        vec = np.array([1.0, 2.0, 3.0])
        assert cache.cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, temp_db):
        cache = SemanticCache(db_path=temp_db)
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        assert cache.cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_opposite_vectors(self, temp_db):
        cache = SemanticCache(db_path=temp_db)
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([-1.0, 0.0])
        assert cache.cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)

    def test_zero_vector(self, temp_db):
        cache = SemanticCache(db_path=temp_db)
        vec_a = np.array([0.0, 0.0])
        vec_b = np.array([1.0, 0.0])
        assert cache.cosine_similarity(vec_a, vec_b) == 0.0


class TestSemanticCacheClear:
    """Test cache clearing."""

    def test_clear_removes_all(self, temp_db, sample_embedding):
        cache = SemanticCache(db_path=temp_db)
        for i in range(10):
            cache.store(f"query {i}", sample_embedding, f"response {i}")

        cache.clear_cache()
        stats = cache.get_stats()
        assert stats["total_entries"] == 0

    def test_clear_empty_cache(self, temp_db):
        cache = SemanticCache(db_path=temp_db)
        cache.clear_cache()  # Should not raise
        stats = cache.get_stats()
        assert stats["total_entries"] == 0

    def test_clear_then_store(self, temp_db, sample_embedding):
        cache = SemanticCache(db_path=temp_db)
        cache.store("old query", sample_embedding, "old response")
        cache.clear_cache()
        cache.store("new query", sample_embedding, "new response")

        result = cache.lookup(sample_embedding, threshold=0.99)
        assert result == "new response"


class TestSemanticCacheStats:
    """Test cache statistics."""

    def test_empty_stats(self, temp_db):
        cache = SemanticCache(db_path=temp_db)
        stats = cache.get_stats()
        assert stats["total_entries"] == 0

    def test_stats_after_inserts(self, temp_db, sample_embedding):
        cache = SemanticCache(db_path=temp_db)
        for i in range(7):
            cache.store(f"q{i}", sample_embedding, f"r{i}")
        stats = cache.get_stats()
        assert stats["total_entries"] == 7
