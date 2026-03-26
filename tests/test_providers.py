"""Tests for providers: base abstractions, stubs, registry, language detector."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from agent.providers.base import TaskProvider, CodeProvider, DocProvider
from agent.providers.task.stub import StubTaskProvider
from agent.providers.code.stub import StubCodeProvider
from agent.providers.doc.stub import StubDocProvider
from agent.providers.registry import build_providers
from agent.providers.code.language_detector import build_language_map
from agent.errors import PermanentError


class TestStubProviders:
    @pytest.mark.asyncio
    async def test_stub_task_provider(self):
        provider = StubTaskProvider()
        task = await provider.get_task("TEST-1")
        assert task["id"] == "TEST-1"
        assert "title" in task
        comments = await provider.get_comments("TEST-1")
        assert isinstance(comments, list)

    @pytest.mark.asyncio
    async def test_stub_code_provider(self):
        provider = StubCodeProvider()
        results = await provider.search_code("query")
        assert results == []
        content = await provider.get_file("path.py")
        assert content == ""

    @pytest.mark.asyncio
    async def test_stub_doc_provider(self):
        provider = StubDocProvider()
        results = await provider.search_docs("query")
        assert results == []
        page = await provider.get_page("page-1")
        assert "title" in page


class TestProviderRegistry:
    def test_build_providers_with_stubs(self):
        from agent.config import Config
        config = Config(task_provider="stub", code_provider="stub", doc_provider="stub")
        task, code, doc = build_providers(config)
        assert isinstance(task, StubTaskProvider)
        assert isinstance(code, StubCodeProvider)
        assert isinstance(doc, StubDocProvider)

    def test_build_providers_unknown_raises(self):
        from agent.config import Config
        config = Config(task_provider="nonexistent", code_provider="stub", doc_provider="stub")
        with pytest.raises(PermanentError, match="Unknown task provider"):
            build_providers(config)


class TestLanguageDetector:
    def test_detects_python(self):
        files = ["main.py", "utils.py", "test.py"]
        result = build_language_map(files)
        assert result.primary_language == "python"
        assert result.languages["python"] == 3

    def test_detects_mixed(self):
        files = ["app.ts", "index.tsx", "server.py"]
        result = build_language_map(files)
        assert "typescript" in result.languages
        assert "python" in result.languages

    def test_handles_empty(self):
        result = build_language_map([])
        assert result.primary_language == "unknown"

    def test_ignores_unknown_extensions(self):
        files = ["Makefile", "Dockerfile", "data.csv"]
        result = build_language_map(files)
        assert result.primary_language == "unknown"
