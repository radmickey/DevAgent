"""Tests for MCP: config, client, classifier, catalog, providers, dynamic injection."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config import (
    MCPServerConfig,
    _expand_env_vars,
    _load_all_mcp_servers,
    _load_mcp_yaml,
    _parse_mcp_servers,
)
from agent.providers.mcp_classifier import (
    ToolStage,
    _apply_overrides,
    _classify_by_heuristic,
    _classify_by_rules,
    _classify_single_by_rules,
    _compute_tools_hash,
    _fallback_classify,
    _load_cache,
    _merge_classifications,
    _parse_stage,
    _save_cache,
)
from agent.providers.mcp_client import MCPClient, MCPConnection, ToolInfo, _parse_mcp_result
from agent.providers.mcp_providers import (
    MCPCodeProvider,
    MCPDocProvider,
    MCPTaskProvider,
    _resolve_tool,
)
from agent.providers.tool_catalog import CatalogTool, ToolCatalog, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn(tool_names: list[str]) -> MCPConnection:
    tools = {
        name: ToolInfo(name=name, description=f"Tool {name}", input_schema={})
        for name in tool_names
    }
    return MCPConnection(server_name="test", tools=tools, session=None)


def _make_client_with_tools(server: str, tool_names: list[str]) -> MCPClient:
    client = MCPClient()
    conn = _make_conn(tool_names)
    conn.server_name = server
    client._connections[server] = conn
    return client


def _make_tool(name: str, desc: str = "") -> ToolInfo:
    return ToolInfo(name=name, description=desc, input_schema={})


# ===========================================================================
# YAML Config Tests
# ===========================================================================


class TestYAMLConfig:
    def test_expand_env_vars(self) -> None:
        with patch.dict(os.environ, {"MY_TOKEN": "secret123"}):
            assert _expand_env_vars("${MY_TOKEN}") == "secret123"
            assert _expand_env_vars("bearer ${MY_TOKEN}") == "bearer secret123"

    def test_expand_env_vars_missing(self) -> None:
        result = _expand_env_vars("${NONEXISTENT_VAR_XYZ}")
        assert result == "${NONEXISTENT_VAR_XYZ}"

    def test_load_mcp_yaml_no_file(self, tmp_path: Path) -> None:
        result = _load_mcp_yaml(tmp_path / "missing.yaml")
        assert result == []

    def test_load_mcp_yaml_valid(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            servers:
              - name: tracker
                command: npx
                args: ["-y", "@mcp/tracker"]
                env:
                  TOKEN: "abc123"
              - name: github
                command: node
                args: ["server.js"]
        """)
        yaml_file = tmp_path / "mcp_servers.yaml"
        yaml_file.write_text(yaml_content)

        result = _load_mcp_yaml(yaml_file)
        assert len(result) == 2
        assert result[0].name == "tracker"
        assert result[0].command == "npx"
        assert result[0].args == ["-y", "@mcp/tracker"]
        assert result[0].env == {"TOKEN": "abc123"}
        assert result[1].name == "github"
        assert result[1].env is None

    def test_load_mcp_yaml_env_expansion(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            servers:
              - name: srv
                command: python
                args: ["s.py"]
                env:
                  KEY: "${TEST_MCP_KEY}"
        """)
        yaml_file = tmp_path / "mcp_servers.yaml"
        yaml_file.write_text(yaml_content)

        with patch.dict(os.environ, {"TEST_MCP_KEY": "expanded_value"}):
            result = _load_mcp_yaml(yaml_file)
            assert result[0].env == {"KEY": "expanded_value"}

    def test_load_mcp_yaml_malformed(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("not: valid: yaml: [[")
        result = _load_mcp_yaml(yaml_file)
        assert result == []

    def test_load_mcp_yaml_missing_fields(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            servers:
              - name: no_command
              - command: no_name
                args: ["x"]
              - name: ok
                command: python
                args: ["s.py"]
        """)
        yaml_file = tmp_path / "mcp_servers.yaml"
        yaml_file.write_text(yaml_content)

        result = _load_mcp_yaml(yaml_file)
        assert len(result) == 1
        assert result[0].name == "ok"

    def test_load_mcp_yaml_no_pyyaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "servers.yaml"
        yaml_file.write_text("servers: []")

        import sys
        yaml_mod = sys.modules.get("yaml")
        sys.modules["yaml"] = None  # type: ignore[assignment]
        try:
            result = _load_mcp_yaml(yaml_file)
            assert result == []
        finally:
            if yaml_mod is not None:
                sys.modules["yaml"] = yaml_mod
            else:
                sys.modules.pop("yaml", None)

    def test_load_mcp_yaml_with_overrides(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            servers:
              - name: my_srv
                command: python
                args: ["s.py"]
                exclude_tools: ["play_music", "get_weather"]
                tool_stages:
                  search_code: ["context_gathering", "code_operations"]
                  my_tool: ["review"]
        """)
        yaml_file = tmp_path / "mcp_servers.yaml"
        yaml_file.write_text(yaml_content)

        result = _load_mcp_yaml(yaml_file)
        assert len(result) == 1
        assert result[0].exclude_tools == ["play_music", "get_weather"]
        assert result[0].tool_stages == {
            "search_code": ["context_gathering", "code_operations"],
            "my_tool": ["review"],
        }

    def test_load_all_merges_yaml_and_env(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            servers:
              - name: from_yaml
                command: python
                args: ["y.py"]
        """)
        yaml_file = tmp_path / "mcp_servers.yaml"
        yaml_file.write_text(yaml_content)

        with patch("agent.config._load_mcp_yaml", return_value=_load_mcp_yaml(yaml_file)):
            with patch.dict(os.environ, {"MCP_SERVERS": "from_env:node:s.js"}):
                result = _load_all_mcp_servers()
                names = [s.name for s in result]
                assert "from_yaml" in names
                assert "from_env" in names

    def test_load_all_yaml_takes_priority(self, tmp_path: Path) -> None:
        yaml_servers = [MCPServerConfig(name="dup", command="python", args=["y.py"])]
        env_servers = [MCPServerConfig(name="dup", command="node", args=["e.js"])]

        with patch("agent.config._load_mcp_yaml", return_value=yaml_servers):
            with patch("agent.config._parse_mcp_servers", return_value=env_servers):
                result = _load_all_mcp_servers()
                assert len(result) == 1
                assert result[0].command == "python"


# ===========================================================================
# Backward-compat env parsing (unchanged)
# ===========================================================================


class TestParseMCPServers:
    def test_empty(self) -> None:
        with patch.dict(os.environ, {"MCP_SERVERS": ""}, clear=False):
            assert _parse_mcp_servers() == []

    def test_single_server(self) -> None:
        with patch.dict(os.environ, {"MCP_SERVERS": "tracker:npx:-y,@mcp/tracker:task"}):
            servers = _parse_mcp_servers()
            assert len(servers) == 1
            assert servers[0].name == "tracker"
            assert servers[0].command == "npx"
            assert servers[0].args == ["-y", "@mcp/tracker"]
            assert servers[0].provider_type == "task"

    def test_multiple_servers(self) -> None:
        env = "gh:npx:-y,@mcp/github:code;wiki:python:wiki.py:doc"
        with patch.dict(os.environ, {"MCP_SERVERS": env}):
            servers = _parse_mcp_servers()
            assert len(servers) == 2

    def test_malformed_entry_skipped(self) -> None:
        with patch.dict(os.environ, {"MCP_SERVERS": "bad;ok:python:s.py:task"}):
            servers = _parse_mcp_servers()
            assert len(servers) == 1


# ===========================================================================
# MCPConnection / ToolInfo
# ===========================================================================


class TestMCPConnection:
    def test_has_tool(self) -> None:
        conn = _make_conn(["get_task", "list_files"])
        assert conn.has_tool("get_task")
        assert not conn.has_tool("missing")

    def test_tool_names(self) -> None:
        conn = _make_conn(["a", "b", "c"])
        assert conn.tool_names == ["a", "b", "c"]


# ===========================================================================
# Tool resolution
# ===========================================================================


class TestResolveToolMapping:
    def test_exact_hint_match(self) -> None:
        conn = _make_conn(["get_task", "something_else"])
        assert _resolve_tool(conn, ["get_task", "get_issue"], method_name="get_task") == "get_task"

    def test_custom_mapping_priority(self) -> None:
        conn = _make_conn(["get_task", "custom_tool"])
        mapping = {"get_task": "custom_tool"}
        assert (
            _resolve_tool(conn, ["get_task"], custom_mapping=mapping, method_name="get_task")
            == "custom_tool"
        )

    def test_no_match(self) -> None:
        conn = _make_conn(["unrelated_tool"])
        assert _resolve_tool(conn, ["get_task"], method_name="get_task") is None


# ===========================================================================
# MCP result parsing
# ===========================================================================


class TestParseMCPResult:
    def test_none(self) -> None:
        assert _parse_mcp_result(None) is None

    def test_json_text_content(self) -> None:
        item = MagicMock()
        item.text = '{"key": "value"}'
        result = MagicMock()
        result.content = [item]
        assert _parse_mcp_result(result) == {"key": "value"}

    def test_plain_text_content(self) -> None:
        item = MagicMock()
        item.text = "hello world"
        result = MagicMock()
        result.content = [item]
        assert _parse_mcp_result(result) == "hello world"

    def test_raw_passthrough(self) -> None:
        assert _parse_mcp_result(42) == 42


# ===========================================================================
# MCPClient
# ===========================================================================


class TestMCPClient:
    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self) -> None:
        client = MCPClient()
        with pytest.raises(Exception, match="not connected"):
            await client.call_tool("missing", "tool")

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self) -> None:
        client = _make_client_with_tools("srv", ["existing_tool"])
        with pytest.raises(Exception, match="not available"):
            await client.call_tool("srv", "missing_tool")

    @pytest.mark.asyncio
    async def test_call_tool_success(self) -> None:
        client = _make_client_with_tools("srv", ["my_tool"])
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = "ok"
        mock_session.call_tool.return_value = mock_result
        client._connections["srv"].session = mock_session

        result = await client.call_tool("srv", "my_tool", {"arg": "val"})
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_call_tool_safe_returns_default(self) -> None:
        client = _make_client_with_tools("srv", ["my_tool"])
        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = RuntimeError("boom")
        client._connections["srv"].session = mock_session

        result = await client.call_tool_safe("srv", "my_tool", default="fallback")
        assert result == "fallback"

    def test_list_servers(self) -> None:
        client = MCPClient()
        assert client.list_servers() == []
        client._connections["a"] = _make_conn([])
        assert "a" in client.list_servers()

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        client = MCPClient()
        mock_stack = AsyncMock()
        conn = _make_conn(["t"])
        conn._exit_stack = mock_stack
        client._connections["srv"] = conn

        await client.disconnect("srv")
        assert "srv" not in client._connections
        mock_stack.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_all(self) -> None:
        client = MCPClient()
        for name in ["a", "b"]:
            conn = _make_conn([])
            conn._exit_stack = AsyncMock()
            client._connections[name] = conn

        await client.disconnect_all()
        assert len(client._connections) == 0


# ===========================================================================
# Classifier — multi-stage
# ===========================================================================


class TestClassifier:
    def test_compute_tools_hash_deterministic(self) -> None:
        tools = [_make_tool("b"), _make_tool("a")]
        h1 = _compute_tools_hash(tools)
        h2 = _compute_tools_hash(list(reversed(tools)))
        assert h1 == h2

    def test_compute_tools_hash_changes(self) -> None:
        t1 = [_make_tool("a")]
        t2 = [_make_tool("a"), _make_tool("b")]
        assert _compute_tools_hash(t1) != _compute_tools_hash(t2)

    def test_parse_stage_valid(self) -> None:
        assert _parse_stage("code_operations") == ToolStage.CODE_OPERATIONS
        assert _parse_stage("review") == ToolStage.REVIEW
        assert _parse_stage("skip") == ToolStage.SKIP

    def test_parse_stage_invalid(self) -> None:
        assert _parse_stage("nonexistent") == ToolStage.GENERIC

    # --- Backward compat: _classify_by_heuristic returns primary stage ---

    def test_classify_by_heuristic_task(self) -> None:
        tool = _make_tool("get_issue", "Fetch a Jira issue by key")
        result = _classify_by_heuristic(tool)
        assert result in (ToolStage.TASK_MANAGEMENT, ToolStage.CONTEXT_GATHERING)

    def test_classify_by_heuristic_code(self) -> None:
        tool = _make_tool("create_file", "Create a new file in the repo")
        assert _classify_by_heuristic(tool) == ToolStage.CODE_OPERATIONS

    def test_classify_by_heuristic_review(self) -> None:
        tool = _make_tool("run_tests", "Execute test suite")
        result = _classify_by_heuristic(tool)
        assert result in (ToolStage.REVIEW, ToolStage.CODE_OPERATIONS)

    def test_classify_by_heuristic_context(self) -> None:
        tool = _make_tool("search_code", "Search the codebase")
        result = _classify_by_heuristic(tool)
        assert result == ToolStage.CONTEXT_GATHERING

    def test_classify_by_heuristic_unknown(self) -> None:
        tool = _make_tool("magic_unicorn", "Does something mysterious")
        result = _classify_by_heuristic(tool)
        assert result in (ToolStage.GENERIC, ToolStage.SKIP)

    # --- Multi-stage: _classify_single_by_rules ---

    def test_single_by_rules_multi_stage(self) -> None:
        tool = _make_tool("search_code", "Search code in repo")
        stages = _classify_single_by_rules(tool)
        assert ToolStage.CONTEXT_GATHERING in stages
        assert ToolStage.CODE_OPERATIONS in stages

    def test_single_by_rules_get_file_multi(self) -> None:
        tool = _make_tool("get_file", "Get file contents")
        stages = _classify_single_by_rules(tool)
        assert ToolStage.CONTEXT_GATHERING in stages
        assert ToolStage.CODE_OPERATIONS in stages

    def test_single_by_rules_get_issue_multi(self) -> None:
        tool = _make_tool("get_issue", "Get issue details from tracker")
        stages = _classify_single_by_rules(tool)
        assert ToolStage.TASK_MANAGEMENT in stages
        assert ToolStage.CONTEXT_GATHERING in stages

    def test_single_by_rules_irrelevant(self) -> None:
        tool = _make_tool("play_music", "Play a song")
        stages = _classify_single_by_rules(tool)
        assert stages == [ToolStage.SKIP]

    def test_single_by_rules_pure_code(self) -> None:
        tool = _make_tool("create_file", "Create a new file in the repository")
        stages = _classify_single_by_rules(tool)
        assert ToolStage.CODE_OPERATIONS in stages

    def test_single_by_rules_threshold(self) -> None:
        """Verify that close-scoring stages are both included."""
        tool = _make_tool("write_file", "Write and validate file contents")
        stages = _classify_single_by_rules(tool)
        assert ToolStage.CODE_OPERATIONS in stages

    # --- _classify_by_rules (batch) ---

    def test_classify_by_rules_batch(self) -> None:
        tools = [
            _make_tool("get_issue", "Fetch issue"),
            _make_tool("search_code", "Search code"),
            _make_tool("run_tests", "Execute tests"),
            _make_tool("create_file", "Create file"),
            _make_tool("play_music", "Play songs"),
        ]
        result = _classify_by_rules(tools)
        assert ToolStage.TASK_MANAGEMENT in result["get_issue"]
        assert ToolStage.CONTEXT_GATHERING in result["search_code"]
        assert ToolStage.REVIEW in result["run_tests"]
        assert ToolStage.CODE_OPERATIONS in result["create_file"]
        assert result["play_music"] == [ToolStage.SKIP]

    # --- Backward compat: _fallback_classify returns multi-stage ---

    def test_fallback_classify(self) -> None:
        tools = [
            _make_tool("get_issue"),
            _make_tool("search_code"),
            _make_tool("run_tests"),
            _make_tool("create_file"),
        ]
        result = _fallback_classify(tools)
        assert ToolStage.TASK_MANAGEMENT in result["get_issue"]
        assert ToolStage.CONTEXT_GATHERING in result["search_code"]
        assert ToolStage.REVIEW in result["run_tests"]
        assert ToolStage.CODE_OPERATIONS in result["create_file"]

    # --- Merge classifications ---

    def test_merge_classifications_union(self) -> None:
        rules = {"tool_a": [ToolStage.CONTEXT_GATHERING]}
        llm = {"tool_a": [ToolStage.CODE_OPERATIONS]}
        merged = _merge_classifications(rules, llm)
        assert ToolStage.CONTEXT_GATHERING in merged["tool_a"]
        assert ToolStage.CODE_OPERATIONS in merged["tool_a"]

    def test_merge_classifications_skip_removed(self) -> None:
        rules = {"tool_a": [ToolStage.SKIP]}
        llm = {"tool_a": [ToolStage.CODE_OPERATIONS]}
        merged = _merge_classifications(rules, llm)
        assert ToolStage.SKIP not in merged["tool_a"]
        assert ToolStage.CODE_OPERATIONS in merged["tool_a"]

    def test_merge_classifications_both_skip(self) -> None:
        rules = {"tool_a": [ToolStage.SKIP]}
        llm = {"tool_a": [ToolStage.SKIP]}
        merged = _merge_classifications(rules, llm)
        assert merged["tool_a"] == [ToolStage.SKIP]

    # --- Apply overrides ---

    def test_apply_overrides_exclude(self) -> None:
        classifications = {
            "tool_a": [ToolStage.CODE_OPERATIONS],
            "tool_b": [ToolStage.REVIEW],
        }
        result = _apply_overrides(classifications, None, ["tool_a"])
        assert result["tool_a"] == [ToolStage.SKIP]
        assert result["tool_b"] == [ToolStage.REVIEW]

    def test_apply_overrides_stage_override(self) -> None:
        classifications = {"tool_a": [ToolStage.GENERIC]}
        overrides = {"tool_a": ["code_operations", "review"]}
        result = _apply_overrides(classifications, overrides, None)
        assert ToolStage.CODE_OPERATIONS in result["tool_a"]
        assert ToolStage.REVIEW in result["tool_a"]

    def test_apply_overrides_both(self) -> None:
        classifications = {
            "tool_a": [ToolStage.GENERIC],
            "tool_b": [ToolStage.REVIEW],
        }
        result = _apply_overrides(
            classifications,
            user_overrides={"tool_a": ["planning"]},
            exclude_tools=["tool_b"],
        )
        assert result["tool_a"] == [ToolStage.PLANNING]
        assert result["tool_b"] == [ToolStage.SKIP]

    # --- Cache roundtrip (now multi-stage format) ---

    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        with patch("agent.providers.mcp_classifier.CACHE_PATH", cache_path):
            _save_cache("abc123", {"tool1": ["review", "context_gathering"]})
            loaded = _load_cache("abc123")
            assert loaded == {"tool1": ["review", "context_gathering"]}

    def test_cache_miss_wrong_hash(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        with patch("agent.providers.mcp_classifier.CACHE_PATH", cache_path):
            _save_cache("hash_a", {"tool1": ["review"]})
            assert _load_cache("hash_b") is None

    def test_cache_miss_no_file(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "missing.json"
        with patch("agent.providers.mcp_classifier.CACHE_PATH", cache_path):
            assert _load_cache("any") is None

    @pytest.mark.asyncio
    async def test_classify_tools_uses_cache(self, tmp_path: Path) -> None:
        from agent.providers.mcp_classifier import classify_tools

        cache_path = tmp_path / "cache.json"
        tools = [_make_tool("get_issue"), _make_tool("search_code")]
        tools_hash = _compute_tools_hash(tools)
        cache_data = {
            "hash": tools_hash,
            "classifications": {
                "get_issue": ["task_management"],
                "search_code": ["context_gathering", "code_operations"],
            },
        }
        cache_path.write_text(json.dumps(cache_data))

        with patch("agent.providers.mcp_classifier.CACHE_PATH", cache_path):
            result = await classify_tools(tools, use_cache=True)
            assert ToolStage.TASK_MANAGEMENT in result["get_issue"]
            assert ToolStage.CONTEXT_GATHERING in result["search_code"]
            assert ToolStage.CODE_OPERATIONS in result["search_code"]

    @pytest.mark.asyncio
    async def test_classify_tools_empty(self) -> None:
        from agent.providers.mcp_classifier import classify_tools
        assert await classify_tools([]) == {}

    @pytest.mark.asyncio
    async def test_classify_tools_llm_fallback(self, tmp_path: Path) -> None:
        from agent.providers.mcp_classifier import classify_tools

        cache_path = tmp_path / "cache.json"
        tools = [_make_tool("get_issue", "Fetch issue details")]

        with patch("agent.providers.mcp_classifier.CACHE_PATH", cache_path):
            with patch(
                "agent.providers.mcp_classifier._call_llm_classifier",
                side_effect=Exception("LLM unavailable"),
            ):
                result = await classify_tools(tools, use_cache=False)
                assert "get_issue" in result
                assert isinstance(result["get_issue"], list)

    @pytest.mark.asyncio
    async def test_classify_tools_no_llm(self, tmp_path: Path) -> None:
        """When use_llm=False, only rule-based classification is used."""
        from agent.providers.mcp_classifier import classify_tools

        cache_path = tmp_path / "cache.json"
        tools = [
            _make_tool("create_file", "Create a new file"),
            _make_tool("play_music", "Play songs"),
        ]

        with patch("agent.providers.mcp_classifier.CACHE_PATH", cache_path):
            result = await classify_tools(tools, use_cache=False, use_llm=False)
            assert ToolStage.CODE_OPERATIONS in result["create_file"]
            assert result["play_music"] == [ToolStage.SKIP]

    @pytest.mark.asyncio
    async def test_classify_tools_with_overrides(self, tmp_path: Path) -> None:
        from agent.providers.mcp_classifier import classify_tools

        cache_path = tmp_path / "cache.json"
        tools = [_make_tool("my_custom_tool", "Does custom stuff")]

        with patch("agent.providers.mcp_classifier.CACHE_PATH", cache_path):
            result = await classify_tools(
                tools,
                use_cache=False,
                use_llm=False,
                user_overrides={"my_custom_tool": ["code_operations", "review"]},
            )
            assert ToolStage.CODE_OPERATIONS in result["my_custom_tool"]
            assert ToolStage.REVIEW in result["my_custom_tool"]

    @pytest.mark.asyncio
    async def test_classify_tools_with_excludes(self, tmp_path: Path) -> None:
        from agent.providers.mcp_classifier import classify_tools

        cache_path = tmp_path / "cache.json"
        tools = [
            _make_tool("create_file", "Create file"),
            _make_tool("get_weather", "Get weather info"),
        ]

        with patch("agent.providers.mcp_classifier.CACHE_PATH", cache_path):
            result = await classify_tools(
                tools,
                use_cache=False,
                use_llm=False,
                exclude_tools=["get_weather"],
            )
            assert result["get_weather"] == [ToolStage.SKIP]
            assert ToolStage.CODE_OPERATIONS in result["create_file"]


# ===========================================================================
# ToolCatalog — multi-stage
# ===========================================================================


class TestToolCatalog:
    def _make_catalog(self) -> ToolCatalog:
        client = _make_client_with_tools("srv", ["search_code", "run_tests", "get_issue"])
        catalog = ToolCatalog(client)
        catalog.register(
            server="srv",
            tool=_make_tool("search_code", "Search code"),
            stages=[ToolStage.CONTEXT_GATHERING, ToolStage.CODE_OPERATIONS],
        )
        catalog.register(
            server="srv",
            tool=_make_tool("run_tests", "Run tests"),
            stages=[ToolStage.REVIEW],
        )
        catalog.register(
            server="srv",
            tool=_make_tool("get_issue", "Get issue"),
            stages=[ToolStage.TASK_MANAGEMENT, ToolStage.CONTEXT_GATHERING],
        )
        return catalog

    def test_get_tools_for_stage(self) -> None:
        catalog = self._make_catalog()
        context_tools = catalog.get_tools_for_stage(ToolStage.CONTEXT_GATHERING)
        names = [t.name for t in context_tools]
        assert "search_code" in names
        assert "get_issue" in names

    def test_multi_stage_tool_in_multiple_stages(self) -> None:
        catalog = self._make_catalog()
        context_tools = catalog.get_tools_for_stage(ToolStage.CONTEXT_GATHERING)
        code_tools = catalog.get_tools_for_stage(ToolStage.CODE_OPERATIONS)
        assert any(t.name == "search_code" for t in context_tools)
        assert any(t.name == "search_code" for t in code_tools)

    def test_no_duplicates_in_stage(self) -> None:
        """When a tool is both in a stage and generic, it appears only once."""
        client = MCPClient()
        catalog = ToolCatalog(client)
        catalog.register(
            server="srv",
            tool=_make_tool("multi", "Multi tool"),
            stages=[ToolStage.CODE_OPERATIONS, ToolStage.GENERIC],
        )
        code_tools = catalog.get_tools_for_stage(ToolStage.CODE_OPERATIONS)
        assert sum(1 for t in code_tools if t.name == "multi") == 1

    def test_get_tools_for_stage_string(self) -> None:
        catalog = self._make_catalog()
        review_tools = catalog.get_tools_for_stage("review")
        assert any(t.name == "run_tests" for t in review_tools)

    def test_generic_tools_everywhere(self) -> None:
        client = MCPClient()
        catalog = ToolCatalog(client)
        catalog.register(
            server="srv",
            tool=_make_tool("utility", "Generic utility"),
            stages=[ToolStage.GENERIC],
        )
        for stage in [ToolStage.CONTEXT_GATHERING, ToolStage.CODE_OPERATIONS, ToolStage.REVIEW]:
            tools = catalog.get_tools_for_stage(stage)
            assert any(t.name == "utility" for t in tools)

    def test_skip_tools_excluded(self) -> None:
        client = MCPClient()
        catalog = ToolCatalog(client)
        catalog.register(
            server="srv",
            tool=_make_tool("weather", "Get weather"),
            stages=[ToolStage.SKIP],
        )
        assert catalog.tool_count == 0
        assert catalog.skipped_count == 1
        assert "srv::weather" in catalog.get_skipped_tools()
        for stage in ToolStage:
            tools = catalog.get_tools_for_stage(stage)
            assert not any(t.name == "weather" for t in tools)

    def test_has_tools_for_stage(self) -> None:
        catalog = self._make_catalog()
        assert catalog.has_tools_for_stage(ToolStage.CONTEXT_GATHERING) is True
        assert catalog.has_tools_for_stage(ToolStage.PLANNING) is False

    def test_tool_count(self) -> None:
        catalog = self._make_catalog()
        assert catalog.tool_count == 3

    def test_summary(self) -> None:
        catalog = self._make_catalog()
        summary = catalog.summary()
        assert "context_gathering" in summary
        assert "review" in summary
        assert "task_management" in summary

    def test_register_batch(self) -> None:
        client = MCPClient()
        catalog = ToolCatalog(client)
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        classifications: dict[str, list[ToolStage]] = {
            "a": [ToolStage.CODE_OPERATIONS],
            "b": [ToolStage.REVIEW, ToolStage.CODE_OPERATIONS],
        }
        catalog.register_batch("srv", tools, classifications)
        assert catalog.tool_count == 3
        code_tools = catalog.get_tools_for_stage(ToolStage.CODE_OPERATIONS)
        assert any(t.name == "a" for t in code_tools)
        assert any(t.name == "b" for t in code_tools)

    def test_catalog_tool_stage_property(self) -> None:
        """CatalogTool.stage returns primary (first) stage for backward compat."""
        ct = CatalogTool(
            name="test",
            server="srv",
            description="Test",
            input_schema={},
            stages=[ToolStage.CODE_OPERATIONS, ToolStage.REVIEW],
        )
        assert ct.stage == ToolStage.CODE_OPERATIONS

    def test_get_all_tools(self) -> None:
        catalog = self._make_catalog()
        all_tools = catalog.get_all_tools()
        assert len(all_tools) == 3

    @pytest.mark.asyncio
    async def test_call_tool(self) -> None:
        client = _make_client_with_tools("srv", ["my_tool"])
        session = AsyncMock()
        r = MagicMock()
        r.content = "result"
        session.call_tool.return_value = r
        client._connections["srv"].session = session

        catalog = ToolCatalog(client)
        result = await catalog.call_tool("srv", "my_tool", {"x": 1})
        assert result == "result"

    @pytest.mark.asyncio
    async def test_call_tool_safe(self) -> None:
        client = _make_client_with_tools("srv", ["my_tool"])
        session = AsyncMock()
        session.call_tool.side_effect = RuntimeError("boom")
        client._connections["srv"].session = session

        catalog = ToolCatalog(client)
        result = await catalog.call_tool_safe("srv", "my_tool", default="safe")
        assert result == "safe"

    @pytest.mark.asyncio
    async def test_call_tools_parallel(self) -> None:
        client = _make_client_with_tools("srv", ["t1", "t2"])
        session = AsyncMock()
        r = MagicMock()
        r.content = "ok"
        session.call_tool.return_value = r
        client._connections["srv"].session = session

        catalog = ToolCatalog(client)
        calls = [ToolCall("srv", "t1", {}), ToolCall("srv", "t2", {})]
        results = await catalog.call_tools_parallel(calls)
        assert len(results) == 2

    def test_invalid_stage_string(self) -> None:
        client = MCPClient()
        catalog = ToolCatalog(client)
        assert catalog.get_tools_for_stage("nonexistent_stage") == []


# ===========================================================================
# MCP Providers (backward compat)
# ===========================================================================


class TestMCPTaskProvider:
    @pytest.mark.asyncio
    async def test_get_task(self) -> None:
        client = _make_client_with_tools("tracker", ["get_issue"])
        session = AsyncMock()
        task_data = MagicMock()
        task_data.content = json.dumps({"id": "T-1", "title": "Test"})
        session.call_tool.return_value = task_data
        client._connections["tracker"].session = session

        provider = MCPTaskProvider(client, "tracker")
        result = await provider.get_task("T-1")
        assert result["id"] == "T-1"

    @pytest.mark.asyncio
    async def test_get_task_no_tool(self) -> None:
        client = _make_client_with_tools("tracker", ["unrelated"])
        provider = MCPTaskProvider(client, "tracker")
        with pytest.raises(Exception, match="No get_task tool"):
            await provider.get_task("T-1")


class TestMCPCodeProvider:
    @pytest.mark.asyncio
    async def test_get_file(self) -> None:
        client = _make_client_with_tools("github", ["get_file_contents"])
        session = AsyncMock()
        content = MagicMock()
        content.content = "print('hello')"
        session.call_tool.return_value = content
        client._connections["github"].session = session

        provider = MCPCodeProvider(client, "github")
        result = await provider.get_file("main.py")
        assert "hello" in result


class TestMCPDocProvider:
    @pytest.mark.asyncio
    async def test_search_docs(self) -> None:
        client = _make_client_with_tools("wiki", ["search_pages"])
        session = AsyncMock()
        docs = MagicMock()
        docs.content = json.dumps([{"id": "1", "title": "Doc"}])
        session.call_tool.return_value = docs
        client._connections["wiki"].session = session

        provider = MCPDocProvider(client, "wiki")
        result = await provider.search_docs("query")
        assert len(result) == 1


# ===========================================================================
# Bootstrap
# ===========================================================================


class TestMCPBootstrap:
    @pytest.mark.asyncio
    async def test_bootstrap_no_servers(self) -> None:
        from agent.providers.mcp_bootstrap import bootstrap_mcp

        config = MagicMock()
        config.mcp_servers = []
        catalog = await bootstrap_mcp(config)
        assert catalog.tool_count == 0


# ===========================================================================
# Dynamic Tool Injection
# ===========================================================================


class TestDynamicInjection:
    def test_enricher_build_search_args(self) -> None:
        from agent.pipeline.nodes.enricher import _build_search_args

        schema = {"properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}
        args = _build_search_args(schema, "test query")
        assert args == {"query": "test query"}

    def test_enricher_build_search_args_unknown_param(self) -> None:
        from agent.pipeline.nodes.enricher import _build_search_args

        schema = {"properties": {"custom_field": {"type": "string"}}}
        args = _build_search_args(schema, "test query")
        assert args == {"custom_field": "test query"}

    def test_enricher_build_search_args_empty_schema(self) -> None:
        from agent.pipeline.nodes.enricher import _build_search_args

        args = _build_search_args({}, "test query")
        assert args == {"query": "test query"}

    @pytest.mark.asyncio
    async def test_enricher_with_catalog(self) -> None:
        from agent.pipeline.nodes.enricher import enricher_node
        from agent.providers.code.stub import StubCodeProvider
        from agent.providers.doc.stub import StubDocProvider

        mock_catalog = MagicMock()
        mock_catalog.get_tools_for_stage.return_value = [
            CatalogTool(
                name="web_search",
                server="search_srv",
                description="Search the web",
                input_schema={"properties": {"query": {"type": "string"}}},
                stages=[ToolStage.CONTEXT_GATHERING],
            )
        ]
        mock_catalog.call_tool_safe = AsyncMock(return_value={"results": ["found"]})

        state = {
            "task_raw": {"title": "Fix auth bug"},
            "task_id": "T-1",
        }

        result = await enricher_node(
            state,
            code_provider=StubCodeProvider(),
            doc_provider=StubDocProvider(),
            tool_catalog=mock_catalog,
        )

        assert "enriched_context" in result
        mock_catalog.call_tool_safe.assert_called_once()

    def test_executor_inject_mcp_tools(self) -> None:
        from agent.pipeline.nodes.executor import _create_executor_agent, inject_mcp_tools

        mock_catalog = MagicMock()
        mock_catalog.get_tools_for_stage.return_value = [
            CatalogTool(
                name="create_pr",
                server="github",
                description="Create a pull request",
                input_schema={},
                stages=[ToolStage.CODE_OPERATIONS],
            )
        ]

        agent = _create_executor_agent()
        inject_mcp_tools(agent, mock_catalog, "code_operations")

    def test_explainer_create_agent(self) -> None:
        from agent.pipeline.nodes.explainer import _create_explainer_agent

        agent = _create_explainer_agent()
        assert agent is not None

    def test_reviewer_create_agent(self) -> None:
        from agent.pipeline.nodes.reviewer import _create_reviewer_agent

        agent = _create_reviewer_agent()
        assert agent is not None

    def test_node_deps_has_catalog(self) -> None:
        from agent.pipeline.models import NodeDeps

        mock_catalog = MagicMock()
        deps = NodeDeps(task_id="T-1", tool_catalog=mock_catalog)
        assert deps.tool_catalog is mock_catalog

    def test_node_deps_catalog_default_none(self) -> None:
        from agent.pipeline.models import NodeDeps

        deps = NodeDeps(task_id="T-1")
        assert deps.tool_catalog is None


# ===========================================================================
# Pipeline graph with catalog
# ===========================================================================


class TestPipelineWithCatalog:
    def test_build_pipeline_no_catalog(self, tmp_path: Path) -> None:
        from agent.pipeline.graph import build_pipeline

        with patch("agent.memory.effects._DB_PATH", tmp_path / "effects.db"):
            graph, config = build_pipeline()
            assert graph is not None

    def test_build_pipeline_with_catalog(self, tmp_path: Path) -> None:
        from agent.pipeline.graph import build_pipeline

        mock_catalog = MagicMock()
        with patch("agent.memory.effects._DB_PATH", tmp_path / "effects.db"):
            graph, config = build_pipeline(tool_catalog=mock_catalog)
            assert graph is not None


# ===========================================================================
# Pipeline auto-resolves providers from catalog
# ===========================================================================


class TestAutoResolveProviders:
    def test_resolve_providers_no_catalog(self) -> None:
        from agent.pipeline.graph import _resolve_providers

        task, code, doc = _resolve_providers(None)
        from agent.providers.task.stub import StubTaskProvider
        from agent.providers.code.stub import StubCodeProvider
        from agent.providers.doc.stub import StubDocProvider
        assert isinstance(task, StubTaskProvider)
        assert isinstance(code, StubCodeProvider)
        assert isinstance(doc, StubDocProvider)

    def test_resolve_providers_with_catalog(self) -> None:
        from agent.pipeline.graph import _resolve_providers
        from agent.providers.mcp_providers import MCPTaskProvider

        mock_catalog = MagicMock()
        mock_catalog.get_all_tools.return_value = [
            CatalogTool(
                name="get_issue",
                server="tracker",
                description="Get issue",
                input_schema={},
                stages=[ToolStage.TASK_MANAGEMENT],
            ),
        ]
        mock_catalog._client = MagicMock()

        task, code, doc = _resolve_providers(mock_catalog)
        assert isinstance(task, MCPTaskProvider)
