"""Tests for the RAG Controller module."""

import os
import pytest
from unittest.mock import patch, MagicMock
from rag import RAGController


# Apply patches that persist across all tests in this module
_patcher_supabase = patch("rag.create_client")
_patcher_openai_client = patch("rag.OpenAIClient")

mock_supabase_cls = _patcher_supabase.start()
mock_openai_client_cls = _patcher_openai_client.start()

# Configure default mock LLM client
mock_llm_client = MagicMock()
mock_openai_client_cls.return_value = mock_llm_client


def _mock_llm_response(text: str):
    """Configure mock LLM client to return a specific text."""
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_llm_client.chat.completions.create.return_value = mock_resp


def create_rag(**kwargs):
    """Create a RAGController with mocked dependencies."""
    defaults = {
        "api_key": "test-key",
        "supabase_url": "https://test.supabase.co",
        "supabase_key": "test-key",
    }
    defaults.update(kwargs)
    mock_supabase_cls.reset_mock()
    mock_openai_client_cls.reset_mock()
    _mock_llm_response("Test response.")
    mock_llm_client.chat.completions.create.side_effect = None
    return RAGController(**defaults)


def teardown_module():
    _patcher_supabase.stop()
    _patcher_openai_client.stop()


class TestRAGControllerInit:
    """Test RAGController initialization."""

    def test_init_with_supabase(self):
        rag = create_rag()
        assert rag.api_key == "test-key"
        mock_supabase_cls.assert_called_once()

    def test_init_requires_credentials(self):
        with pytest.raises(ValueError, match="supabase_url and supabase_key required"):
            RAGController(api_key="test-key")


class TestGetQueryEmbedding:
    """Test query embedding generation."""

    @patch("rag._get_embed_model")
    def test_returns_list(self, mock_get_model):
        rag = create_rag()
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([MagicMock(tolist=lambda: [0.1] * 384)])
        mock_get_model.return_value = mock_model

        embedding = rag.get_query_embedding("test query")
        assert len(embedding) == 384


class TestBuildNodesFromResults:
    """Test converting search results to LlamaIndex nodes."""

    def test_builds_nodes(self):
        rag = create_rag()
        results = [
            {
                "content": "Test content",
                "metadata": {"title": "Test", "url": "https://example.com"},
                "similarity": 0.85,
            }
        ]
        nodes = rag.build_nodes_from_results(results)
        assert len(nodes) == 1
        assert nodes[0].node.get_content() == "Test content"
        assert nodes[0].score == 0.85

    def test_handles_string_metadata(self):
        rag = create_rag()
        results = [
            {
                "content": "Test",
                "metadata": '{"title": "Test"}',
                "similarity": 0.5,
            }
        ]
        nodes = rag.build_nodes_from_results(results)
        assert len(nodes) == 1

    def test_empty_results(self):
        rag = create_rag()
        nodes = rag.build_nodes_from_results([])
        assert nodes == []

    def test_handles_missing_fields(self):
        rag = create_rag()
        results = [{"content": "Test"}]
        nodes = rag.build_nodes_from_results(results)
        assert len(nodes) == 1
        assert nodes[0].score == 0.0

    def test_handles_invalid_json_metadata(self):
        rag = create_rag()
        results = [
            {
                "content": "Test",
                "metadata": "not valid json {{{",
                "similarity": 0.5,
            }
        ]
        nodes = rag.build_nodes_from_results(results)
        assert len(nodes) == 1


class TestVectorSearch:
    """Test vector search via vecs."""

    @patch("rag.get_vecs_collection")
    def test_uses_vecs_query(self, mock_get_collection):
        rag = create_rag()
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection

        # vecs returns (id, metadata, distance) tuples
        mock_collection.query.return_value = [
            ("doc::0", {"content": "test", "title": "Test"}, 0.1)
        ]

        results = rag.vector_search([0.1] * 384, top_k=10)
        assert len(results) == 1
        mock_collection.query.assert_called_once()

    @patch("rag.get_vecs_collection")
    def test_returns_empty_on_no_data(self, mock_get_collection):
        rag = create_rag()
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        mock_collection.query.return_value = []

        results = rag.vector_search([0.1] * 384)
        assert results == []


class TestSelectTopNodes:
    """Test top-N selection by vector similarity score."""

    def test_selects_top_by_score(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        rag = create_rag()
        nodes = [
            NodeWithScore(node=TextNode(text="low"), score=0.1),
            NodeWithScore(node=TextNode(text="high"), score=0.9),
            NodeWithScore(node=TextNode(text="mid"), score=0.5),
        ]
        result = rag.select_top_nodes(nodes, top_n=2)
        assert len(result) == 2
        assert result[0].node.get_content() == "high"
        assert result[1].node.get_content() == "mid"

    def test_empty_nodes(self):
        rag = create_rag()
        result = rag.select_top_nodes([], top_n=5)
        assert result == []

    def test_returns_all_when_fewer_than_top_n(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        rag = create_rag()
        nodes = [
            NodeWithScore(node=TextNode(text="only one"), score=0.8),
        ]
        result = rag.select_top_nodes(nodes, top_n=5)
        assert len(result) == 1


class TestQuery:
    """Test the full RAG query pipeline."""

    @patch("rag.get_vecs_collection")
    @patch("rag._get_embed_model")
    def test_query_returns_expected_keys(self, mock_get_model, mock_get_collection):
        rag = create_rag()

        # Mock embedding model
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([MagicMock(tolist=lambda: [0.1] * 384)])
        mock_get_model.return_value = mock_model

        # Mock vecs collection
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        mock_collection.query.return_value = [
            ("doc::0", {"content": "GitLab values collaboration", "title": "Values", "url": "https://handbook.gitlab.com/values/"}, 0.1)
        ]

        result = rag.query("What are GitLab's values?")

        assert "response" in result
        assert "retrieved_chunks" in result
        assert "latency" in result
        assert "time_to_first_token" in result
        assert "num_chunks_retrieved" in result
        assert "num_chunks_reranked" in result

    @patch("rag.get_vecs_collection")
    @patch("rag._get_embed_model")
    def test_query_with_empty_search(self, mock_get_model, mock_get_collection):
        """Test query when no results found."""
        rag = create_rag()

        # Mock embedding model
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([MagicMock(tolist=lambda: [0.1] * 384)])
        mock_get_model.return_value = mock_model

        # Mock empty vecs search
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        mock_collection.query.return_value = []

        result = rag.query("What is quantum computing?")

        assert "response" in result
        assert result["num_chunks_retrieved"] == 0

    @patch("rag.get_vecs_collection")
    def test_query_uses_precomputed_embedding(self, mock_get_collection):
        """Test query does not call embed if pre-computed embedding is provided."""
        rag = create_rag()

        # Mock vecs collection
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        mock_collection.query.return_value = []

        # Call query with pre-computed embedding
        result = rag.query("test query", query_embedding=[0.1] * 384)
        assert result["num_chunks_retrieved"] == 0
