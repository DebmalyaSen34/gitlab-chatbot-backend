"""Tests for the Ingestion module."""

import pytest
from unittest.mock import patch, MagicMock
from ingest import (
    map_path_to_url,
    clean_markdown,
    get_file_hash,
    IncrementalSyncManager,
)


class TestMapPathToUrl:
    """Test file path to URL mapping."""

    def test_index_md(self):
        path = "content/handbook/values/_index.md"
        expected = "https://handbook.gitlab.com/handbook/values/"
        assert map_path_to_url(path) == expected

    def test_index_md_no_underscore(self):
        path = "content/handbook/values/index.md"
        expected = "https://handbook.gitlab.com/handbook/values/"
        assert map_path_to_url(path) == expected

    def test_regular_md_file(self):
        path = "content/handbook/culture/remote-work.md"
        expected = "https://handbook.gitlab.com/handbook/culture/remote-work/"
        assert map_path_to_url(path) == expected

    def test_nested_path(self):
        path = "content/handbook/engineering/development/process/_index.md"
        expected = "https://handbook.gitlab.com/handbook/engineering/development/process/"
        assert map_path_to_url(path) == expected

    def test_simple_path(self):
        path = "content/handbook/hiring/_index.md"
        expected = "https://handbook.gitlab.com/handbook/hiring/"
        assert map_path_to_url(path) == expected


class TestCleanMarkdown:
    """Test markdown cleaning."""

    def test_strips_frontmatter(self):
        raw = "---\ntitle: Values\n---\n# Value\nThis is content."
        cleaned = clean_markdown(raw)
        assert "title: Values" not in cleaned
        assert "This is content." in cleaned

    def test_strips_frontmatter_with_metadata(self):
        raw = "---\ntitle: Test\ndescription: A test\nweight: 1\n---\nActual content here."
        cleaned = clean_markdown(raw)
        assert "title: Test" not in cleaned
        assert "description: A test" not in cleaned
        assert "Actual content here." in cleaned

    def test_no_frontmatter(self):
        raw = "# Just a heading\nSome content."
        cleaned = clean_markdown(raw)
        assert "# Just a heading" in cleaned
        assert "Some content." in cleaned

    def test_strips_html_tags(self):
        raw = "# Title\n<p>Some HTML content</p>\nMore text."
        cleaned = clean_markdown(raw)
        assert "<p>" not in cleaned
        assert "</p>" not in cleaned
        assert "Some HTML content" in cleaned

    def test_collapses_excessive_newlines(self):
        raw = "# Title\n\n\n\n\nContent here."
        cleaned = clean_markdown(raw)
        assert "\n\n\n" not in cleaned

    def test_strips_whitespace(self):
        raw = "   \n\n  Content  \n\n   "
        cleaned = clean_markdown(raw)
        assert cleaned == "Content"

    def test_empty_content(self):
        raw = "---\ntitle: empty\n---\n"
        cleaned = clean_markdown(raw)
        assert cleaned == ""


class TestGetFileHash:
    """Test file hashing."""

    def test_deterministic(self):
        hash1 = get_file_hash("Hello GitLab")
        hash2 = get_file_hash("Hello GitLab")
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        hash1 = get_file_hash("Hello")
        hash2 = get_file_hash("World")
        assert hash1 != hash2

    def test_returns_string(self):
        result = get_file_hash("test")
        assert isinstance(result, str)

    def test_sha256_length(self):
        result = get_file_hash("test")
        assert len(result) == 64  # SHA256 hex digest length

    def test_empty_string(self):
        result = get_file_hash("")
        assert len(result) == 64


class TestIncrementalSyncManager:
    """Test the IncrementalSyncManager class."""

    @patch("ingest.create_client")
    def test_init(self, mock_supabase):
        manager = IncrementalSyncManager(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        assert manager.sync_run_id is not None
        mock_supabase.assert_called_once()

    @patch("ingest.create_client")
    def test_sync_run_id_is_timestamp(self, mock_supabase):
        manager = IncrementalSyncManager(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        # Should be a numeric timestamp string
        assert manager.sync_run_id.isdigit()

    @patch("ingest.requests.post")
    @patch("ingest.create_client")
    def test_get_embedding_calls_ollama(self, mock_supabase, mock_requests_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_resp.raise_for_status = MagicMock()
        mock_requests_post.return_value = mock_resp

        manager = IncrementalSyncManager(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        embedding = manager.get_embedding("test text")
        mock_requests_post.assert_called_once()
        assert len(embedding) == 768

    @patch("ingest.create_client")
    def test_sync_file_skips_short_content(self, mock_supabase):
        mock_client = MagicMock()
        mock_supabase.return_value = mock_client

        manager = IncrementalSyncManager(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        # Very short content should be skipped
        result = manager.sync_file("test.md", "short")
        assert result == "SKIPPED"

    @patch("ingest.create_client")
    def test_sync_file_skips_unchanged_hash(self, mock_supabase_class):
        mock_supabase = MagicMock()
        mock_supabase_class.return_value = mock_supabase

        # Mock that the file already exists with same hash
        content = "This is a long enough content for testing purposes. " * 5
        current_hash = get_file_hash(clean_markdown(content))
        mock_supabase.table().select().eq().execute.return_value = MagicMock(
            data=[{"file_hash": current_hash}]
        )

        manager = IncrementalSyncManager(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        result = manager.sync_file("test.md", content)
        assert result == "SKIPPED"


class TestFetchMarkdownFiles:
    """Test GitLab API fetching."""

    @patch("ingest.requests.get")
    def test_fetch_returns_dict(self, mock_get):
        from ingest import fetch_markdown_files_from_repo

        # Mock the tree listing response
        mock_tree_resp = MagicMock()
        mock_tree_resp.json.return_value = [
            {"type": "blob", "path": "content/handbook/values/_index.md"},
        ]
        mock_tree_resp.raise_for_status = MagicMock()

        # Mock the file content response
        import base64

        content = base64.b64encode(b"# Test content").decode()
        mock_file_resp = MagicMock()
        mock_file_resp.json.return_value = {"content": content}
        mock_file_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_tree_resp, mock_file_resp]

        result = fetch_markdown_files_from_repo(["content/handbook/values"])
        assert isinstance(result, dict)
        assert "content/handbook/values/_index.md" in result

    @patch("ingest.requests.get")
    def test_fetch_handles_api_error(self, mock_get):
        from ingest import fetch_markdown_files_from_repo
        import requests

        mock_get.side_effect = requests.RequestException("API Error")
        result = fetch_markdown_files_from_repo(["content/handbook/values"])
        assert result == {}
