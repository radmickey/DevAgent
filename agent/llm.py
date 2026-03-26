"""LLM model registry: resolves models for pipeline nodes.

Priority chain for model resolution:
  1. node_models[node_name] → explicit per-node alias from settings.yaml
  2. node_models[default_strong/default_fast] → tier-level override
  3. Config.llm_strong_model / Config.llm_fast_model → env vars
  4. Hardcoded defaults (claude-sonnet / claude-haiku)

Custom models are defined in ~/.devagent/settings.yaml under `models:` section.
Each model specifies provider, model name, base_url, and api_key.
"""

from __future__ import annotations

from typing import Any

import structlog

from agent.config import ModelConfig, get_config

log = structlog.get_logger()

_NODE_MODEL_MAP: dict[str, str] = {
    "input_router": "fast",
    "reader": "fast",
    "enricher": "fast",
    "ranker": "fast",
    "code_search": "fast",
    "doc_search": "fast",
    "task_search": "fast",
    "diff_agent": "fast",
    "explainer": "strong",
    "executor": "strong",
    "reviewer": "strong",
    "doc_writer": "fast",
    "pattern_extractor": "strong",
    "meta_agent": "strong",
}


class ModelRegistry:
    """Creates and caches pydantic-ai model instances from ModelConfig."""

    def __init__(self, models: list[ModelConfig] | None = None) -> None:
        self._configs: dict[str, ModelConfig] = {}
        self._cache: dict[str, Any] = {}
        if models:
            for m in models:
                self._configs[m.name] = m

    def get(self, alias: str) -> Any:
        """Get or create a pydantic-ai model instance by alias.

        Returns an object accepted by pydantic-ai Agent.run(model=...).
        """
        if alias in self._cache:
            return self._cache[alias]

        cfg = self._configs.get(alias)
        if cfg is None:
            return alias

        model = _create_model(cfg)
        self._cache[alias] = model
        log.info(
            "model_created",
            alias=alias,
            provider=cfg.provider,
            model=cfg.model,
            base_url=cfg.base_url or "default",
        )
        return model

    def has(self, alias: str) -> bool:
        return alias in self._configs

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered model configs (without api_key)."""
        return [
            {
                "name": c.name,
                "provider": c.provider,
                "model": c.model,
                "base_url": c.base_url or "default",
            }
            for c in self._configs.values()
        ]


def _create_model(cfg: ModelConfig) -> Any:
    """Create a pydantic-ai model from ModelConfig using the appropriate provider."""
    provider_type = cfg.provider.lower()

    if provider_type in ("openai", "openai-compatible"):
        return _create_openai_model(cfg)
    elif provider_type == "anthropic":
        return _create_anthropic_model(cfg)
    elif provider_type == "ollama":
        return _create_ollama_model(cfg)
    elif provider_type == "deepseek":
        return _create_deepseek_model(cfg)
    else:
        return _create_openai_model(cfg)


def _create_openai_model(cfg: ModelConfig) -> Any:
    """Create OpenAI or OpenAI-compatible model."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    kwargs: dict[str, Any] = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url

    provider = OpenAIProvider(**kwargs)
    return OpenAIChatModel(cfg.model, provider=provider)


def _create_anthropic_model(cfg: ModelConfig) -> Any:
    """Create Anthropic model."""
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    kwargs: dict[str, Any] = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key

    provider = AnthropicProvider(**kwargs)
    return AnthropicModel(cfg.model, provider=provider)


def _create_ollama_model(cfg: ModelConfig) -> Any:
    """Create Ollama model."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider

    kwargs: dict[str, Any] = {}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url

    provider = OllamaProvider(**kwargs)
    return OpenAIChatModel(cfg.model, provider=provider)


def _create_deepseek_model(cfg: ModelConfig) -> Any:
    """Create DeepSeek model."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider

    kwargs: dict[str, Any] = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key

    provider = DeepSeekProvider(**kwargs)
    return OpenAIChatModel(cfg.model, provider=provider)


# ---------------------------------------------------------------------------
# Singleton registry + resolution
# ---------------------------------------------------------------------------

_registry: ModelRegistry | None = None


def _get_registry() -> ModelRegistry:
    """Get or create the global ModelRegistry from Config."""
    global _registry
    if _registry is None:
        config = get_config()
        _registry = ModelRegistry(config.models)
    return _registry


def invalidate_registry() -> None:
    """Reset the registry (called when config changes)."""
    global _registry
    _registry = None


def get_model_for_node(node: str) -> Any:
    """Return the model for a given pipeline node.

    Resolution priority:
      1. node_models[node] → explicit alias
      2. node_models[default_strong / default_fast] → tier override
      3. Config.llm_strong_model / llm_fast_model → env vars
      4. Hardcoded defaults
    """
    config = get_config()
    registry = _get_registry()
    tier = _NODE_MODEL_MAP.get(node, "fast")
    node_models = config.node_models

    if node in node_models:
        alias = node_models[node]
        model = registry.get(alias)
        log.debug("llm_model_selected", node=node, source="node_models", alias=alias)
        return model

    tier_key = f"default_{tier}"
    if tier_key in node_models:
        alias = node_models[tier_key]
        model = registry.get(alias)
        log.debug("llm_model_selected", node=node, source=tier_key, alias=alias)
        return model

    if tier == "strong":
        model_name = config.llm_strong_model
    else:
        model_name = config.llm_fast_model

    if registry.has(model_name):
        return registry.get(model_name)

    log.debug("llm_model_selected", node=node, tier=tier, model=model_name)
    return model_name


def resolve_model_name_for_node(node: str) -> str:
    """Return the resolved model name/alias for a node WITHOUT instantiating.

    Used for display/listing purposes only.
    """
    config = get_config()
    tier = _NODE_MODEL_MAP.get(node, "fast")
    node_models = config.node_models

    if node in node_models:
        return node_models[node]

    tier_key = f"default_{tier}"
    if tier_key in node_models:
        return node_models[tier_key]

    if tier == "strong":
        return config.llm_strong_model
    return config.llm_fast_model


def get_fast_model() -> Any:
    """Get the fast (cheap) model."""
    return get_model_for_node("enricher")


def get_strong_model() -> Any:
    """Get the strong (capable) model."""
    return get_model_for_node("executor")
