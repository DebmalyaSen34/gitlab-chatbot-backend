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
        with pytest.raises(ValueError, match="Either supabase_url"):
            RAGController(api_key="test-key")

    def test_init_with_postgres_string(self):
        rag = create_rag(
            supabase_url="",
            supabase_key="",
            postgres_connection_string="postgresql://user:pass@host:5432/db",
        )
        assert rag._postgres_connection_string == "postgresql://user:pass@host:5432/db"


class TestGetQueryEmbedding:
    """Test query embedding generation."""

    def test_returns_list(self):
        rag = create_rag()
        import requests as req
        with patch.object(req, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"embeddings": [[0.1] * 768]}
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            embedding = rag.get_query_embedding("test query")
            assert len(embedding) == 768
            mock_post.assert_called_once()


class TestBuildNodesFromResults:
    """Test converting Supabase results to LlamaIndex nodes."""

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
    """Test vector search via Supabase."""

    def test_uses_supabase_rpc(self):
        rag = create_rag()
        mock_supabase = MagicMock()
        rag.supabase = mock_supabase

        mock_result = MagicMock()
        mock_result.data = [{"content": "test", "metadata": {}, "similarity": 0.9}]
        mock_supabase.rpc.return_value.execute.return_value = mock_result

        results = rag.vector_search([0.1] * 768, top_k=10)
        assert len(results) == 1
        mock_supabase.rpc.assert_called_once_with(
            "match_data_embeddings",
            {"query_embedding": [0.1] * 768, "match_count": 10},
        )

    def test_returns_empty_on_no_data(self):
        rag = create_rag()
        mock_supabase = MagicMock()
        rag.supabase = mock_supabase

        mock_result = MagicMock()
        mock_result.data = None
        mock_supabase.rpc.return_value.execute.return_value = mock_result

        results = rag.vector_search([0.1] * 768)
        assert results == []


class TestRerankNodes:
    """Test node reranking."""

    def test_rerank_scores_with_llm(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        rag = create_rag()
        # LLM returns relevance scores
        _mock_llm_response("8")

        nodes = [
            NodeWithScore(node=TextNode(text="highly relevant"), score=0.5),
            NodeWithScore(node=TextNode(text="less relevant"), score=0.9),
        ]
        result = rag.rerank_nodes(nodes, "query", top_n=1)
        assert len(result) == 1

    def test_rerank_falls_back_on_error(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        rag = create_rag()
        rag.llm_client.chat.completions.create.side_effect = Exception("API Error")

        nodes = [
            NodeWithScore(node=TextNode(text="high"), score=0.9),
            NodeWithScore(node=TextNode(text="low"), score=0.1),
        ]
        result = rag.rerank_nodes(nodes, "query", top_n=1)
        assert len(result) == 1
        assert result[0].node.get_content() == "high"

    def test_rerank_empty_nodes(self):
        rag = create_rag()
        result = rag.rerank_nodes([], "query")
        assert result == []


class TestQuery:
    """Test the full RAG query pipeline."""

    @patch("rag.requests.post")
    def test_query_returns_expected_keys(self, mock_requests_post):
        from llama_index.core.schema import NodeWithScore, TextNode

        rag = create_rag()

        # Mock embedding (Ollama)
        mock_emb_resp = MagicMock()
        mock_emb_resp.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_emb_resp.raise_for_status = MagicMock()
        mock_requests_post.return_value = mock_emb_resp

        # Mock Supabase search
        mock_supabase = MagicMock()
        rag.supabase = mock_supabase
        mock_search_result = MagicMock()
        mock_search_result.data = [
            {
                "content": "GitLab values collaboration",
                "metadata": {"title": "Values", "url": "https://handbook.gitlab.com/values/"},
                "similarity": 0.9,
            }
        ]
        mock_supabase.rpc.return_value.execute.return_value = mock_search_result

        # Mock LLM: reranking score + final response
        mock_choice1 = MagicMock()
        mock_choice1.message.content = "9"  # reranking score
        mock_choice2 = MagicMock()
        mock_choice2.message.content = "GitLab values collaboration."  # final response
        mock_llm_client.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_choice1]),  # rerank call
            MagicMock(choices=[mock_choice2]),   # generation call
        ]

        result = rag.query("What are GitLab's values?")

        assert "response" in result
        assert "retrieved_chunks" in result
        assert "latency" in result
        assert "time_to_first_token" in result
        assert "num_chunks_retrieved" in result
        assert "num_chunks_reranked" in result
        assert result["num_chunks_reranked"] == 1
        assert result["response"] == "GitLab values collaboration."

    @patch("rag.requests.post")
    def test_query_with_empty_search(self, mock_requests_post):
        """Test query when no results found."""
        rag = create_rag()

        # Mock embedding (Ollama)
        mock_emb_resp = MagicMock()
        mock_emb_resp.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_emb_resp.raise_for_status = MagicMock()
        mock_requests_post.return_value = mock_emb_resp

        # Mock empty Supabase search
        mock_supabase = MagicMock()
        rag.supabase = mock_supabase
        mock_search_result = MagicMock()
        mock_search_result.data = []
        mock_supabase.rpc.return_value.execute.return_value = mock_search_result

        result = rag.query("What is quantum computing?")

        assert "response" in result
        assert result["num_chunks_retrieved"] == 0
