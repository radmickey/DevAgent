"""DevAgent configuration: env loading, LLM setup, timeouts, runtime settings.

Priority chain for LLM flags:
  1. Runtime override (set via CLI/Web API → persisted in settings.yaml)
  2. Environment variable (e.g. USE_LLM_ENRICHERS=true)
  3. Default value (false for all LLM flags)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv

load_dotenv()

log = structlog.get_logger()


def _get_devagent_home() -> Path:
    p = Path(os.getenv("DEVAGENT_HOME", str(Path.home() / ".devagent")))
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return p


DEVAGENT_HOME = _get_devagent_home()
SETTINGS_PATH = DEVAGENT_HOME / "settings.yaml"

TIMEOUTS: dict[str, int] = {
    "mcp_read": 30,
    "mcp_search": 20,
    "mcp_write": 45,
    "llm_fast": 30,
    "llm_strong": 120,
}

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")

# All known boolean settings with their env var name and default value
_BOOL_SETTINGS: dict[str, tuple[str, bool]] = {
    "use_external_llm":    ("USE_EXTERNAL_LLM", True),
    "use_llm_enrichers":   ("USE_LLM_ENRICHERS", False),
    "use_llm_doc_writer":  ("USE_LLM_DOC_WRITER", False),
    "use_llm_classifier":  ("USE_LLM_CLASSIFIER", False),
    "use_llm_meta_agent":  ("USE_LLM_META_AGENT", False),
}

_STR_SETTINGS: dict[str, tuple[str, str]] = {
    "llm_strong_model": ("LLM_STRONG_MODEL", "claude-sonnet-4-20250514"),
    "llm_fast_model":   ("LLM_FAST_MODEL", "claude-haiku-4-20250414"),
    "env":              ("DEVAGENT_ENV", "dev"),
}


# ---------------------------------------------------------------------------
# Settings YAML: persistent runtime overrides
# ---------------------------------------------------------------------------

def _load_settings_yaml(path: Path | None = None) -> dict[str, Any]:
    """Load settings from ~/.devagent/settings.yaml."""
    if path is None:
        path = SETTINGS_PATH
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        result: dict[str, Any] = raw.get("settings", raw)
        return result
    except Exception:
        return {}


def _save_settings_yaml(settings: dict[str, Any], path: Path | None = None) -> None:
    """Save settings to ~/.devagent/settings.yaml."""
    if path is None:
        path = SETTINGS_PATH
    try:
        import yaml  # type: ignore[import-untyped]
        existing: dict[str, Any] = {}
        if path.exists():
            existing = yaml.safe_load(path.read_text()) or {}
        existing["settings"] = settings
        path.write_text(yaml.dump(existing, default_flow_style=False, sort_keys=True))
    except ImportError:
        log.warning("settings_save_failed", reason="pyyaml not installed")
    except Exception as exc:
        log.warning("settings_save_failed", error=str(exc))


def get_all_settings() -> dict[str, Any]:
    """Get all resolved settings with their current values and sources.

    Returns dict[name, {value, source, default, env_var}].
    """
    yaml_overrides = _load_settings_yaml()
    result: dict[str, Any] = {}

    for name, (env_var, bool_default) in _BOOL_SETTINGS.items():
        bool_val, source = _resolve_bool(name, env_var, bool_default, yaml_overrides)
        result[name] = {
            "value": bool_val,
            "source": source,
            "default": bool_default,
            "env_var": env_var,
            "type": "bool",
        }

    for name, (env_var, str_default) in _STR_SETTINGS.items():
        str_val, source = _resolve_str(name, env_var, str_default, yaml_overrides)
        result[name] = {
            "value": str_val,
            "source": source,
            "default": str_default,
            "env_var": env_var,
            "type": "str",
        }

    return result


def set_setting(name: str, value: Any) -> dict[str, Any]:
    """Set a runtime setting. Persists to settings.yaml.

    Returns the updated setting info.
    """
    all_known = {**_BOOL_SETTINGS, **_STR_SETTINGS}
    if name not in all_known:
        raise KeyError(
            f"Unknown setting '{name}'. Available: {sorted(all_known.keys())}"
        )

    if name in _BOOL_SETTINGS:
        if isinstance(value, str):
            value = value.lower() in ("true", "1", "yes", "on")
        value = bool(value)

    yaml_settings = _load_settings_yaml()
    yaml_settings[name] = value
    _save_settings_yaml(yaml_settings)
    log.info("setting_updated", name=name, value=value)

    _invalidate_config_cache()

    return {"name": name, "value": value, "source": "settings.yaml"}


def _resolve_bool(
    name: str, env_var: str, default: bool, yaml_overrides: dict[str, Any],
) -> tuple[bool, str]:
    """Resolve a boolean setting: YAML > env > default."""
    if name in yaml_overrides:
        raw = yaml_overrides[name]
        if isinstance(raw, bool):
            return raw, "settings.yaml"
        if isinstance(raw, str):
            return raw.lower() in ("true", "1", "yes", "on"), "settings.yaml"

    env_val = os.getenv(env_var)
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes", "on"), f"env:{env_var}"

    return default, "default"


def _resolve_str(
    name: str, env_var: str, default: str, yaml_overrides: dict[str, Any],
) -> tuple[str, str]:
    """Resolve a string setting: YAML > env > default."""
    if name in yaml_overrides and isinstance(yaml_overrides[name], str):
        return yaml_overrides[name], "settings.yaml"

    env_val = os.getenv(env_var)
    if env_val is not None:
        return env_val, f"env:{env_var}"

    return default, "default"


# ---------------------------------------------------------------------------
# MCP Server Config
# ---------------------------------------------------------------------------

@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server connection."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    provider_type: str = ""
    tool_mapping: dict[str, str] | None = None
    exclude_tools: list[str] | None = None
    tool_stages: dict[str, list[str]] | None = None


def _expand_env_vars(value: str) -> str:
    """Expand ${VAR} references in a string from os.environ."""
    def _replacer(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), m.group(0))
    return _ENV_VAR_RE.sub(_replacer, value)


def _load_mcp_yaml(path: Path | None = None) -> list[MCPServerConfig]:
    """Load MCP server configs from a YAML file.

    Default path: ~/.devagent/mcp_servers.yaml
    Supports ${VAR} expansion in env values.
    """
    if path is None:
        path = _get_devagent_home() / "mcp_servers.yaml"

    if not path.exists():
        return []

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return []

    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return []

    servers: list[MCPServerConfig] = []
    for entry in raw.get("servers", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        command = entry.get("command", "")
        args = entry.get("args", [])
        if not name or not command:
            continue

        env_raw: dict[str, str] | None = entry.get("env")
        env_expanded: dict[str, str] | None = None
        if isinstance(env_raw, dict):
            env_expanded = {k: _expand_env_vars(str(v)) for k, v in env_raw.items()}

        exclude_tools: list[str] | None = None
        raw_exclude = entry.get("exclude_tools")
        if isinstance(raw_exclude, list):
            exclude_tools = [str(t) for t in raw_exclude]

        tool_stages: dict[str, list[str]] | None = None
        raw_stages = entry.get("tool_stages")
        if isinstance(raw_stages, dict):
            tool_stages = {
                str(k): ([str(s) for s in v] if isinstance(v, list) else [str(v)])
                for k, v in raw_stages.items()
            }

        servers.append(
            MCPServerConfig(
                name=str(name),
                command=str(command),
                args=[str(a) for a in args],
                env=env_expanded,
                exclude_tools=exclude_tools,
                tool_stages=tool_stages,
            )
        )

    return servers


def _parse_mcp_servers() -> list[MCPServerConfig]:
    """Parse MCP_SERVERS from env (backward compat).

    Format: name:command:arg1,arg2:provider_type
    Multiple servers separated by ';'

    Examples:
      MCP_SERVERS=tracker:npx:-y,@mcp/tracker:task;github:npx:-y,@mcp/github:code
      MCP_SERVERS=my-server:python:server.py:task
    """
    raw = os.getenv("MCP_SERVERS", "")
    if not raw.strip():
        return []

    servers: list[MCPServerConfig] = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) < 3:
            continue
        name = parts[0].strip()
        command = parts[1].strip()
        args = [a.strip() for a in parts[2].split(",") if a.strip()]
        provider_type = parts[3].strip() if len(parts) > 3 else ""
        servers.append(
            MCPServerConfig(
                name=name,
                command=command,
                args=args,
                provider_type=provider_type,
            )
        )
    return servers


def _load_all_mcp_servers() -> list[MCPServerConfig]:
    """Load MCP servers: YAML first, then env fallback. Merged by name."""
    yaml_servers = _load_mcp_yaml()
    env_servers = _parse_mcp_servers()

    seen: dict[str, MCPServerConfig] = {}
    for s in yaml_servers:
        seen[s.name] = s
    for s in env_servers:
        if s.name not in seen:
            seen[s.name] = s

    return list(seen.values())


# ---------------------------------------------------------------------------
# Config dataclass (frozen, resolved from settings chain)
# ---------------------------------------------------------------------------

def _make_bool_resolver(name: str) -> Any:
    """Create a factory for a boolean config field using the settings chain."""
    env_var, default = _BOOL_SETTINGS[name]
    def _factory() -> bool:
        yaml_overrides = _load_settings_yaml()
        val, _ = _resolve_bool(name, env_var, default, yaml_overrides)
        return val
    return _factory


def _make_str_resolver(name: str) -> Any:
    """Create a factory for a string config field using the settings chain."""
    env_var, default = _STR_SETTINGS[name]
    def _factory() -> str:
        yaml_overrides = _load_settings_yaml()
        val, _ = _resolve_str(name, env_var, default, yaml_overrides)
        return val
    return _factory


@dataclass(frozen=True)
class Config:
    llm_strong_model: str = field(default_factory=_make_str_resolver("llm_strong_model"))
    llm_fast_model: str = field(default_factory=_make_str_resolver("llm_fast_model"))
    use_external_llm: bool = field(default_factory=_make_bool_resolver("use_external_llm"))
    use_llm_enrichers: bool = field(default_factory=_make_bool_resolver("use_llm_enrichers"))
    use_llm_doc_writer: bool = field(default_factory=_make_bool_resolver("use_llm_doc_writer"))
    use_llm_classifier: bool = field(default_factory=_make_bool_resolver("use_llm_classifier"))
    use_llm_meta_agent: bool = field(default_factory=_make_bool_resolver("use_llm_meta_agent"))
    env: str = field(default_factory=_make_str_resolver("env"))
    home: Path = field(default_factory=_get_devagent_home)
    mcp_servers: list[MCPServerConfig] = field(default_factory=_load_all_mcp_servers)


_config_cache: Config | None = None


def get_config() -> Config:
    """Get the current Config (cached). Call _invalidate_config_cache() after settings change."""
    global _config_cache
    if _config_cache is None:
        _config_cache = Config()
    return _config_cache


def _invalidate_config_cache() -> None:
    """Reset the config cache so next get_config() re-reads settings."""
    global _config_cache
    _config_cache = None
