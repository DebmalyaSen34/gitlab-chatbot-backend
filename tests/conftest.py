"""Shared test fixtures for the GitLab Chatbot test suite."""

import os
import pytest
import tempfile


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database path for testing."""
    return str(tmp_path / "test_cache.db")


@pytest.fixture
def sample_embedding():
    """Return a sample 768-dimensional embedding vector."""
    return [0.1] * 768


@pytest.fixture
def sample_embedding_alt():
    """Return an alternative sample embedding vector."""
    return [0.2] * 768


@pytest.fixture
def sample_markdown():
    """Return sample GitLab handbook markdown content."""
    return """---
title: GitLab Values
description: Our core company values
---

# GitLab Values

GitLab has six core values that guide how we work together.

## Collaboration

Collaboration is about helping each other and working together as a team.
We believe that the best results come from working together effectively.

## Results

We focus on results and outcomes rather than processes.
We measure success by what we achieve, not by how busy we are.

## Efficiency

We strive to be efficient in everything we do.
We automate where possible and eliminate unnecessary work.

## Diversity

We embrace diversity in all its forms.
Different perspectives lead to better solutions.

## Inclusion

We create an inclusive environment where everyone feels welcome.
We actively work to remove barriers and biases.

## Transparency

We default to transparency in all our communications.
We share information openly and honestly.
"""


@pytest.fixture
def sample_chunks():
    """Return sample text chunks for testing."""
    return [
        "GitLab values collaboration. We help each other and work together.",
        "Results-oriented approach. We focus on outcomes over processes.",
        "Efficiency matters. We automate and eliminate unnecessary work.",
        "Diversity and inclusion are core to who we are.",
        "Transparency is our default. We share openly.",
    ]


@pytest.fixture
def sample_metadata():
    """Return sample metadata dict for a chunk."""
    return {
        "title": "GitLab Values",
        "source_path": "content/handbook/values/_index.md",
        "url": "https://handbook.gitlab.com/handbook/values/",
        "chunk_index": 0,
    }
