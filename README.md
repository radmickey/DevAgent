# DevAgent

AI-powered developer task automation agent. Reads tasks from your tracker, gathers context, generates plans, writes code, runs reviews — all in a structured LangGraph pipeline.

## Features

- **LangGraph pipeline**: input_router → reader → enricher → ranker → explainer → HITL → executor → reviewer → doc_writer
- **Dual LLM strategy**: fast model for routine tasks, strong model for planning/execution/review
- **Pydantic AI agents**: type-safe agents with `output_type` and `RunContext[NodeDeps]`
- **ChromaDB vector memory**: context retrieval with relevance threshold and language filter
- **Side effect tracking**: every file write and command is recorded for rollback
- **Self-evolution L1**: Meta-Agent analyzes outcomes and improves prompts with versioning
- **Observability**: LangSmith + Pydantic Logfire tracing (optional)
- **Security**: automatic sanitization of secrets before sending to external LLMs

## Quick Start

### Prerequisites

- Python 3.11+
- `uv` (recommended) or `pip`

### Installation

```bash
# With uv (recommended)
uv tool install .

# With pip
pip install -e .

# With dev dependencies
pip install -e ".[dev]"

# With observability
pip install -e ".[observability]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

Key environment variables:

| Variable | Description | Default |
|---|---|---|
| `TASK_PROVIDER` | Task tracker: `jira`, `yandex_tracker`, `linear` | `yandex_tracker` |
| `CODE_PROVIDER` | Code source: `github`, `arcanum`, `gitlab` | `github` |
| `DOC_PROVIDER` | Docs: `notion`, `confluence`, `yandex_wiki` | `notion` |
| `LLM_STRONG_MODEL` | Model for planning/execution/review | `claude-sonnet-4-20250514` |
| `LLM_FAST_MODEL` | Model for enrichment/sub-agents | `claude-haiku-4-20250414` |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `USE_EXTERNAL_LLM` | Enable sanitization for external LLMs | `true` |
| `LANGSMITH_API_KEY` | LangSmith tracing (optional) | — |
| `LOGFIRE_TOKEN` | Pydantic Logfire tracing (optional) | — |

### Usage

```bash
# Run on a task
devagent run PROJ-123

# Dry run (plan only, no execution)
devagent run PROJ-123 --dry-run

# Run on free text
devagent run "Add rate limiting to the API"

# Rollback side effects
devagent rollback PROJ-123

# Show cost summary
devagent status
```

## Architecture

```
agent/
├── main.py                 # Entry point, lockfile, signal handlers
├── config.py               # Environment, LLM config, timeouts
├── errors.py               # TransientError / PermanentError / DegradedError
├── llm.py                  # Dual LLM: fast vs strong model routing
├── logging.py              # structlog: dev (pretty) / prod (JSON)
├── observability.py        # LangSmith + Logfire tracing
├── interface/
│   ├── cli.py              # Typer + Rich CLI
│   ├── renderer.py         # Rich rendering for plans/reviews
│   └── hitl_page.html      # HITL approval page
├── memory/
│   ├── vector.py           # ChromaDB vector memory + lang filter
│   ├── session.py          # LangGraph SqliteSaver checkpoint
│   ├── longterm.py         # Patterns + prompt versioning (SQLite)
│   ├── effects.py          # Side effect tracker + rollback
│   ├── budget.py           # Token budget (tiktoken)
│   ├── cost.py             # LLM cost tracker (SQLite)
│   └── agent_docs.py       # JSON Lines documentation
├── pipeline/
│   ├── graph.py            # LangGraph state machine
│   ├── state.py            # PipelineState TypedDict
│   ├── models.py           # Pydantic v2 models + NodeDeps
│   ├── meta_agent.py       # Self-Evolution L1
│   ├── prompts/            # Prompt templates per node
│   └── nodes/
│       ├── input_router.py # Task ID vs free text routing
│       ├── reader.py       # Task data fetching
│       ├── enricher.py     # Parallel context gathering
│       ├── ranker.py       # ChromaDB threshold ranking
│       ├── explainer.py    # Plan generation (Pydantic AI)
│       ├── executor.py     # Code execution (Pydantic AI)
│       ├── reviewer.py     # Tests → lint → LLM review
│       └── doc_writer.py   # JSON Lines + ChromaDB docs
├── providers/
│   ├── base.py             # TaskProvider, CodeProvider, DocProvider ABCs
│   ├── registry.py         # Provider factory
│   ├── task/               # Tracker integrations
│   ├── code/               # Repository integrations
│   └── doc/                # Documentation integrations
├── security/
│   └── sanitizer.py        # Secret redaction for external LLMs
├── skills/                 # Pydantic AI sub-agents
└── registry/               # Skill and tool registries
```

## Adding a Provider

1. Create a class implementing the appropriate ABC from `agent/providers/base.py`:

```python
from agent.providers.base import TaskProvider

class MyTrackerProvider(TaskProvider):
    async def get_task(self, task_id: str) -> dict:
        ...

    async def get_comments(self, task_id: str) -> list[dict]:
        ...

    async def update_status(self, task_id: str, status: str) -> None:
        ...
```

2. Register it in `agent/providers/registry.py`:

```python
from agent.providers.task.my_tracker import MyTrackerProvider

register_task_provider("my_tracker", MyTrackerProvider)
```

3. Set in `.env`:

```
TASK_PROVIDER=my_tracker
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Type checking
mypy agent/ --ignore-missing-imports

# Linting
ruff check agent/ tests/

# Benchmark
python tests/bench.py --tasks 5
```

## License

MIT
