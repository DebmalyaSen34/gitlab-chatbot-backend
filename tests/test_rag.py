"""Tests for the RAG Controller module."""

import os
import pytest
from unittest.mock import patch, MagicMock
from rag import RAGController


# Apply patches that persist across all tests in this module
_patcher_supabase = patch("rag.create_client")
_patcher_genai = patch("rag.GenAIClient")
_patcher_gemini = patch("rag.Gemini")
_patcher_settings = patch("rag.Settings")

mock_supabase_cls = _patcher_supabase.start()
mock_genai_cls = _patcher_genai.start()
mock_gemini_cls = _patcher_gemini.start()
mock_settings = _patcher_settings.start()

# Configure default mocks
mock_llm = MagicMock()
mock_llm.complete.return_value = "Test response."
mock_gemini_cls.return_value = mock_llm
mock_settings.llm = mock_llm


def create_rag(**kwargs):
    """Create a RAGController with mocked dependencies."""
    defaults = {
        "api_key": "test-key",
        "supabase_url": "https://test.supabase.co",
        "supabase_key": "test-key",
    }
    defaults.update(kwargs)
    mock_supabase_cls.reset_mock()
    mock_genai_cls.reset_mock()
    return RAGController(**defaults)


def teardown_module():
    _patcher_supabase.stop()
    _patcher_genai.stop()
    _patcher_gemini.stop()
    _patcher_settings.stop()


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
        mock_result = MagicMock()
        mock_result.embeddings = [MagicMock(values=[0.1] * 768)]
        rag.genai_client.models.embed_content.return_value = mock_result

        embedding = rag.get_query_embedding("test query")
        assert len(embedding) == 768
        rag.genai_client.models.embed_content.assert_called_once()


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

    @patch("rag.LLMRerank")
    def test_rerank_calls_llm_rerank(self, mock_reranker_class):
        from llama_index.core.schema import NodeWithScore, TextNode

        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker
        mock_reranker.postprocess_nodes.return_value = [
            NodeWithScore(node=TextNode(text="reranked"), score=0.95)
        ]

        rag = create_rag()
        nodes = [NodeWithScore(node=TextNode(text="test"), score=0.5)]
        result = rag.rerank_nodes(nodes, "query", top_n=1)
        assert len(result) == 1
        mock_reranker.postprocess_nodes.assert_called_once()

    @patch("rag.LLMRerank")
    def test_rerank_falls_back_on_error(self, mock_reranker_class):
        from llama_index.core.schema import NodeWithScore, TextNode

        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker
        mock_reranker.postprocess_nodes.side_effect = Exception("Rerank failed")

        rag = create_rag()
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

    @patch("rag.LLMRerank")
    def test_query_returns_expected_keys(self, mock_reranker_class):
        from llama_index.core.schema import NodeWithScore, TextNode

        rag = create_rag()

        # Mock embedding
        mock_emb_result = MagicMock()
        mock_emb_result.embeddings = [MagicMock(values=[0.1] * 768)]
        rag.genai_client.models.embed_content.return_value = mock_emb_result

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

        # Mock reranker
        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker
        mock_reranker.postprocess_nodes.return_value = [
            NodeWithScore(
                node=TextNode(
                    text="GitLab values collaboration",
                    metadata={"title": "Values", "url": "https://handbook.gitlab.com/values/"},
                ),
                score=0.9,
            )
        ]

        # Mock LLM response
        mock_llm.complete.return_value = "GitLab values collaboration."

        result = rag.query("What are GitLab's values?")

        assert "response" in result
        assert "retrieved_chunks" in result
        assert "latency" in result
        assert "time_to_first_token" in result
        assert "num_chunks_retrieved" in result
        assert "num_chunks_reranked" in result
        assert result["num_chunks_reranked"] == 1

    @patch("rag.LLMRerank")
    def test_query_with_empty_search(self, mock_reranker_class):
        """Test query when no results found."""
        rag = create_rag()

        # Mock embedding
        mock_emb_result = MagicMock()
        mock_emb_result.embeddings = [MagicMock(values=[0.1] * 768)]
        rag.genai_client.models.embed_content.return_value = mock_emb_result

        # Mock empty Supabase search
        mock_supabase = MagicMock()
        rag.supabase = mock_supabase
        mock_search_result = MagicMock()
        mock_search_result.data = []
        mock_supabase.rpc.return_value.execute.return_value = mock_search_result

        # Mock LLM response
        mock_llm.complete.return_value = "I cannot find this in the GitLab handbook."

        result = rag.query("What is quantum computing?")

        assert "response" in result
        assert result["num_chunks_retrieved"] == 0
