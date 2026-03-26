"""Web interface: FastAPI + SSE for real-time pipeline monitoring.

Provides:
- POST /api/tasks — submit a task
- GET /api/tasks/{id} — get task status
- GET /api/tasks/{id}/stream — SSE stream of pipeline events
- GET /api/status — agent status + cost summary
- GET / — simple dashboard
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import structlog

log = structlog.get_logger()

_event_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}


def create_app():  # type: ignore[no-untyped-def]
    """Create the FastAPI application. Import is deferred to avoid hard dependency."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, StreamingResponse
        from pydantic import BaseModel
    except ImportError:
        raise RuntimeError(
            "FastAPI not installed. Install with: pip install 'devagent[web]' "
            "or pip install fastapi uvicorn sse-starlette"
        )

    app = FastAPI(title="DevAgent", version="0.2.0")

    class TaskRequest(BaseModel):
        task_id: str
        dry_run: bool = False

    class TaskResponse(BaseModel):
        task_id: str
        status: str
        message: str

    @app.post("/api/tasks", response_model=TaskResponse)
    async def submit_task(request: TaskRequest) -> TaskResponse:
        """Submit a task for processing."""
        emit_event(request.task_id, "submitted", {"dry_run": request.dry_run})
        return TaskResponse(
            task_id=request.task_id,
            status="queued",
            message=f"Task {request.task_id} queued for processing",
        )

    @app.get("/api/tasks/{task_id}")
    async def get_task_status(task_id: str) -> dict[str, Any]:
        """Get current status of a task."""
        return {
            "task_id": task_id,
            "has_stream": task_id in _event_queues,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/tasks/{task_id}/stream")
    async def stream_events(task_id: str) -> StreamingResponse:
        """SSE stream of pipeline events for a task."""
        if task_id not in _event_queues:
            _event_queues[task_id] = asyncio.Queue()

        async def event_generator() -> AsyncGenerator[str, None]:
            queue = _event_queues[task_id]
            yield f"data: {json.dumps({'event': 'connected', 'task_id': task_id})}\n\n"
            try:
                while True:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("event") == "completed":
                        break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/status")
    async def agent_status() -> dict[str, Any]:
        """Get agent status and cost summary."""
        try:
            from agent.memory.cost import CostTracker
            ct = CostTracker()
            total = ct.get_total_cost()
            by_model = ct.get_summary_by_model()
            ct.close()
        except Exception:
            total = 0.0
            by_model = []

        return {
            "status": "running",
            "total_cost_usd": total,
            "cost_by_model": by_model,
            "active_tasks": list(_event_queues.keys()),
        }

    # --- Settings API ---

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        """Get all settings with current values and sources."""
        from agent.config import get_all_settings
        return {"settings": get_all_settings()}

    @app.get("/api/settings/{name}")
    async def get_setting(name: str) -> dict[str, Any]:
        """Get a specific setting."""
        from agent.config import get_all_settings
        settings = get_all_settings()
        if name not in settings:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Unknown setting: {name}")
        return {"name": name, **settings[name]}

    class SettingUpdate(BaseModel):
        value: Any

    @app.put("/api/settings/{name}")
    async def update_setting(name: str, body: SettingUpdate) -> dict[str, Any]:
        """Update a setting. Persists to ~/.devagent/settings.yaml."""
        from agent.config import set_setting
        try:
            result = set_setting(name, body.value)
            return result
        except KeyError as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/settings/{name}")
    async def reset_setting(name: str) -> dict[str, Any]:
        """Reset a setting to its default (remove YAML override)."""
        from agent.config import (
            _invalidate_config_cache,
            _load_settings_yaml,
            _save_settings_yaml,
            get_all_settings,
        )
        yaml_settings = _load_settings_yaml()
        if name in yaml_settings:
            del yaml_settings[name]
            _save_settings_yaml(yaml_settings)
            _invalidate_config_cache()

        settings = get_all_settings()
        if name not in settings:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Unknown setting: {name}")
        return {"name": name, "reset": True, **settings[name]}

    # --- Models API ---

    @app.get("/api/models")
    async def list_models() -> dict[str, Any]:
        """List configured LLM models."""
        from agent.config import get_models
        models = get_models()
        return {"models": [
            {"name": m.name, "provider": m.provider, "model": m.model,
             "base_url": m.base_url or "default"}
            for m in models
        ]}

    class ModelCreate(BaseModel):
        name: str
        provider: str = "openai-compatible"
        model: str
        base_url: str | None = None
        api_key: str | None = None

    @app.post("/api/models")
    async def create_model(body: ModelCreate) -> dict[str, Any]:
        """Add or update a custom LLM model."""
        from agent.config import ModelConfig, save_model
        m = save_model(ModelConfig(
            name=body.name,
            provider=body.provider,
            model=body.model,
            base_url=body.base_url or None,
            api_key=body.api_key or None,
        ))
        return {"name": m.name, "provider": m.provider, "model": m.model,
                "base_url": m.base_url or "default"}

    @app.put("/api/models/{name}")
    async def update_model(name: str, body: ModelCreate) -> dict[str, Any]:
        """Update an existing model (name in URL is the target)."""
        from agent.config import ModelConfig, save_model
        m = save_model(ModelConfig(
            name=name,
            provider=body.provider,
            model=body.model,
            base_url=body.base_url or None,
            api_key=body.api_key or None,
        ))
        return {"name": m.name, "provider": m.provider, "model": m.model,
                "base_url": m.base_url or "default"}

    @app.delete("/api/models/{name}")
    async def remove_model(name: str) -> dict[str, Any]:
        """Remove a custom LLM model."""
        from agent.config import delete_model
        found = delete_model(name)
        if not found:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        return {"name": name, "deleted": True}

    @app.get("/api/node-models")
    async def list_node_models() -> dict[str, Any]:
        """Get node→model mapping with resolved values and available choices."""
        from agent.config import get_config, get_models, get_node_models
        from agent.llm import _NODE_MODEL_MAP, resolve_model_name_for_node
        cfg = get_config()
        node_models = get_node_models()
        resolved: dict[str, Any] = {}
        for node in sorted(_NODE_MODEL_MAP):
            tier = _NODE_MODEL_MAP[node]
            model_name = resolve_model_name_for_node(node)
            resolved[node] = {
                "tier": tier,
                "resolved": model_name,
                "override": node_models.get(node, None),
            }
        custom_aliases = [m.name for m in get_models()]
        builtin = [
            {"value": cfg.llm_strong_model, "label": f"{cfg.llm_strong_model} (strong)"},
            {"value": cfg.llm_fast_model, "label": f"{cfg.llm_fast_model} (fast)"},
        ]
        return {
            "node_models": resolved,
            "overrides": node_models,
            "choices": {
                "builtin": builtin,
                "custom": custom_aliases,
            },
        }

    class NodeModelUpdate(BaseModel):
        model: str

    @app.put("/api/node-models/{node}")
    async def update_node_model(node: str, body: NodeModelUpdate) -> dict[str, Any]:
        """Assign a model to a pipeline node."""
        from agent.config import set_node_model
        result = set_node_model(node, body.model)
        return result

    @app.delete("/api/node-models/{node}")
    async def reset_node_model(node: str) -> dict[str, Any]:
        """Remove a per-node model override (revert to default)."""
        from agent.config import delete_node_model
        found = delete_node_model(node)
        return {"node": node, "reset": True, "had_override": found}

    # --- MCP Servers API ---

    @app.get("/api/mcp-servers")
    async def list_mcp_servers() -> dict[str, Any]:
        """List configured MCP servers."""
        from agent.config import get_mcp_servers_raw
        return {"servers": get_mcp_servers_raw()}

    class MCPServerCreate(BaseModel):
        name: str
        command: str
        args: list[str] = []
        env: dict[str, str] | None = None
        exclude_tools: list[str] | None = None
        tool_stages: dict[str, list[str]] | None = None

    @app.post("/api/mcp-servers")
    async def create_mcp_server(body: MCPServerCreate) -> dict[str, Any]:
        """Add or update an MCP server."""
        from agent.config import save_mcp_server
        data: dict[str, Any] = {
            "name": body.name, "command": body.command, "args": body.args,
        }
        if body.env:
            data["env"] = body.env
        if body.exclude_tools:
            data["exclude_tools"] = body.exclude_tools
        if body.tool_stages:
            data["tool_stages"] = body.tool_stages
        return save_mcp_server(data)

    @app.put("/api/mcp-servers/{name}")
    async def update_mcp_server(name: str, body: MCPServerCreate) -> dict[str, Any]:
        """Update an existing MCP server."""
        from agent.config import save_mcp_server
        data: dict[str, Any] = {
            "name": name, "command": body.command, "args": body.args,
        }
        if body.env:
            data["env"] = body.env
        if body.exclude_tools:
            data["exclude_tools"] = body.exclude_tools
        if body.tool_stages:
            data["tool_stages"] = body.tool_stages
        return save_mcp_server(data)

    @app.delete("/api/mcp-servers/{name}")
    async def remove_mcp_server(name: str) -> dict[str, Any]:
        """Remove an MCP server."""
        from agent.config import delete_mcp_server
        found = delete_mcp_server(name)
        if not found:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"MCP server '{name}' not found")
        return {"name": name, "deleted": True}

    # --- Prompts API ---

    @app.get("/api/prompts")
    async def list_prompts() -> dict[str, Any]:
        """List all prompt names with previews."""
        from agent.pipeline.prompts import get_all_prompts
        return {"prompts": get_all_prompts()}

    @app.get("/api/prompts/{name}")
    async def get_prompt_detail(name: str) -> dict[str, Any]:
        """Get full prompt text."""
        from agent.pipeline.prompts import PROMPTS, _load_prompt_overrides
        if name not in PROMPTS:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found")
        overrides = _load_prompt_overrides()
        return {
            "name": name,
            "text": PROMPTS[name],
            "is_overridden": name in overrides,
        }

    class PromptUpdate(BaseModel):
        text: str

    @app.put("/api/prompts/{name}")
    async def update_prompt(name: str, body: PromptUpdate) -> dict[str, Any]:
        """Save a prompt override."""
        from agent.pipeline.prompts import PROMPTS, set_prompt_override
        if name not in PROMPTS:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found")
        set_prompt_override(name, body.text)
        return {"name": name, "updated": True}

    @app.delete("/api/prompts/{name}")
    async def reset_prompt_endpoint(name: str) -> dict[str, Any]:
        """Reset a prompt to its default."""
        from agent.pipeline.prompts import PROMPTS, reset_prompt
        if name not in PROMPTS:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found")
        reset_prompt(name)
        return {"name": name, "reset": True}

    # --- Tool Catalog API ---

    @app.get("/api/tool-catalog")
    async def get_tool_catalog() -> dict[str, Any]:
        """Get classified tools grouped by stage."""
        from agent.config import DEVAGENT_HOME
        cache_path = DEVAGENT_HOME / "tool_classifications.json"
        if not cache_path.exists():
            return {"stages": {}, "note": "No tool classifications cached yet. Run a pipeline first."}
        try:
            raw: dict[str, Any] = json.loads(cache_path.read_text())
            stages: dict[str, list[dict[str, Any]]] = {}
            for tool_name, info in raw.items():
                tool_stages = info.get("stages", [])
                server = info.get("server", "unknown")
                for stage in tool_stages:
                    if stage not in stages:
                        stages[stage] = []
                    stages[stage].append({"name": tool_name, "server": server})
            return {"stages": stages}
        except Exception:
            return {"stages": {}, "note": "Failed to parse tool classifications cache."}

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        """Monitoring dashboard — full management."""
        return _DASHBOARD_HTML

    return app


def emit_event(task_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
    """Emit an SSE event for a task."""
    queue = _event_queues.get(task_id)
    if queue is None:
        return

    event = {
        "event": event_type,
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(data or {}),
    }
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        log.warning("event_queue_full", task_id=task_id)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>DevAgent Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }
        h1 { font-size: 1.5rem; margin-bottom: 1rem; color: #38bdf8; }
        h2 { font-size: 1.1rem; margin-bottom: 0.5rem; }
        .card { background: #1e293b; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
        .status { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; }
        .status.running { background: #22c55e; color: #000; }
        #events { max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.85rem; }
        .event { padding: 4px 0; border-bottom: 1px solid #334155; }
        input, button { padding: 8px 16px; border-radius: 4px; border: 1px solid #475569; background: #1e293b; color: #e2e8f0; }
        button { cursor: pointer; background: #3b82f6; border: none; }
        button:hover { background: #2563eb; }
        .setting-row { display: flex; align-items: center; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #334155; }
        .setting-name { font-family: monospace; color: #38bdf8; }
        .setting-source { font-size: 0.75rem; color: #64748b; margin-left: 8px; }
        .toggle { position: relative; width: 44px; height: 24px; cursor: pointer; }
        .toggle input { opacity: 0; width: 0; height: 0; }
        .toggle .slider { position: absolute; inset: 0; background: #475569; border-radius: 12px; transition: 0.2s; }
        .toggle .slider::before { content: ''; position: absolute; width: 18px; height: 18px; left: 3px; top: 3px; background: #e2e8f0; border-radius: 50%; transition: 0.2s; }
        .toggle input:checked + .slider { background: #22c55e; }
        .toggle input:checked + .slider::before { transform: translateX(20px); }
        .toast { position: fixed; bottom: 1rem; right: 1rem; background: #22c55e; color: #000; padding: 8px 16px; border-radius: 4px; font-size: 0.85rem; display: none; }
        .model-form { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; padding-top: 12px; border-top: 1px solid #334155; }
        .model-form input, .model-form select { padding: 6px 10px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #e2e8f0; font-size: 0.85rem; }
        .model-form .full-width { grid-column: 1 / -1; }
        .btn-sm { padding: 4px 10px; font-size: 0.8rem; border-radius: 4px; border: none; cursor: pointer; }
        .btn-danger { background: #ef4444; color: #fff; }
        .btn-danger:hover { background: #dc2626; }
        .btn-edit { background: #475569; color: #e2e8f0; }
        .btn-edit:hover { background: #64748b; }
        .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: none; align-items: center; justify-content: center; z-index: 100; }
        .modal { background: #1e293b; border-radius: 8px; padding: 1.5rem; width: 480px; max-width: 90vw; }
        .modal h3 { color: #38bdf8; margin-bottom: 1rem; }
        .modal .form-group { margin-bottom: 0.75rem; }
        .modal .form-group label { display: block; font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; }
        .modal .form-group input, .modal .form-group select, .modal .form-group textarea { width: 100%; padding: 8px 10px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #e2e8f0; font-size: 0.9rem; }
        .modal .form-group textarea { font-family: monospace; resize: vertical; min-height: 120px; }
        .modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 1rem; }
        .modal-wide { width: 720px; max-width: 95vw; }
        .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.7rem; margin-left: 6px; }
        .badge-override { background: #f59e0b; color: #000; }
        .stage-group { margin-bottom: 0.75rem; }
        .stage-title { font-size: 0.85rem; color: #f59e0b; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
        .tool-chip { display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 4px; background: #334155; font-family: monospace; font-size: 0.8rem; }
        .tool-server { color: #64748b; font-size: 0.7rem; }
        .note { font-size: 0.75rem; color: #64748b; margin-top: 8px; }
    </style>
</head>
<body>
    <h1>DevAgent Dashboard</h1>
    <div class="card">
        <span class="status running">Running</span>
        <p style="margin-top: 0.5rem">Submit tasks and monitor pipeline execution in real-time.</p>
    </div>
    <div class="card">
        <input id="taskId" placeholder="Task ID (e.g. PROJ-123)" style="width: 300px">
        <button onclick="submitTask()">Submit</button>
    </div>
    <div class="card">
        <h2>Settings</h2>
        <div id="settings">Loading...</div>
    </div>
    <div class="card">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem">
            <h2>Models</h2>
            <button onclick="openModelModal()" class="btn-sm" style="background:#3b82f6;color:#fff">+ Add Model</button>
        </div>
        <div id="models">Loading...</div>
    </div>
    <div id="modelModal" class="modal-overlay" onclick="if(event.target===this)closeModelModal()">
        <div class="modal">
            <h3 id="modalTitle">Add Model</h3>
            <input type="hidden" id="editingModelName" value="">
            <div class="form-group">
                <label>Name (alias)</label>
                <input id="mName" placeholder="my-gpt4o">
            </div>
            <div class="form-group">
                <label>Provider</label>
                <select id="mProvider">
                    <option value="openai">OpenAI</option>
                    <option value="openai-compatible">OpenAI-compatible</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="ollama">Ollama</option>
                    <option value="deepseek">DeepSeek</option>
                </select>
            </div>
            <div class="form-group">
                <label>Model</label>
                <input id="mModel" placeholder="gpt-4o">
            </div>
            <div class="form-group">
                <label>Base URL (optional)</label>
                <input id="mBaseUrl" placeholder="https://api.openai.com/v1">
            </div>
            <div class="form-group">
                <label>API Key (optional, supports ${VAR})</label>
                <input id="mApiKey" type="password" placeholder="${OPENAI_API_KEY} or sk-...">
            </div>
            <div class="modal-actions">
                <button onclick="closeModelModal()" class="btn-sm btn-edit">Cancel</button>
                <button onclick="saveModel()" class="btn-sm" style="background:#22c55e;color:#000">Save</button>
            </div>
        </div>
    </div>
    <div class="card">
        <h2>Node Model Mapping</h2>
        <div id="nodeModels">Loading...</div>
    </div>
    <div class="card">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem">
            <h2>MCP Servers</h2>
            <button onclick="openMcpModal()" class="btn-sm" style="background:#3b82f6;color:#fff">+ Add MCP Server</button>
        </div>
        <div id="mcpServers">Loading...</div>
        <p class="note">Changes apply on next pipeline restart.</p>
    </div>
    <div id="mcpModal" class="modal-overlay" onclick="if(event.target===this)closeMcpModal()">
        <div class="modal">
            <h3 id="mcpModalTitle">Add MCP Server</h3>
            <input type="hidden" id="editingMcpName" value="">
            <div class="form-group">
                <label>Name</label>
                <input id="mcpName" placeholder="my-mcp-server">
            </div>
            <div class="form-group">
                <label>Command</label>
                <input id="mcpCommand" placeholder="npx or python">
            </div>
            <div class="form-group">
                <label>Args (comma-separated)</label>
                <input id="mcpArgs" placeholder="-y, @mcp/server">
            </div>
            <div class="form-group">
                <label>Env (one KEY=VALUE per line, supports ${VAR})</label>
                <textarea id="mcpEnv" rows="3" placeholder="API_KEY=${MY_KEY}&#10;HOST=localhost"></textarea>
            </div>
            <div class="modal-actions">
                <button onclick="closeMcpModal()" class="btn-sm btn-edit">Cancel</button>
                <button onclick="saveMcp()" class="btn-sm" style="background:#22c55e;color:#000">Save</button>
            </div>
        </div>
    </div>
    <div class="card">
        <h2>Prompts</h2>
        <div id="prompts">Loading...</div>
    </div>
    <div id="promptModal" class="modal-overlay" onclick="if(event.target===this)closePromptModal()">
        <div class="modal modal-wide">
            <h3 id="promptModalTitle">Edit Prompt</h3>
            <input type="hidden" id="editingPromptName" value="">
            <div class="form-group">
                <textarea id="promptText" rows="20"></textarea>
            </div>
            <div class="modal-actions">
                <button onclick="resetPrompt()" class="btn-sm btn-danger" id="resetPromptBtn" style="margin-right:auto;display:none">Reset to Default</button>
                <button onclick="closePromptModal()" class="btn-sm btn-edit">Cancel</button>
                <button onclick="savePrompt()" class="btn-sm" style="background:#22c55e;color:#000">Save</button>
            </div>
        </div>
    </div>
    <div class="card">
        <h2>Tool Classifications</h2>
        <div id="toolCatalog">Loading...</div>
    </div>
    <div class="card">
        <h2>Events</h2>
        <div id="events"></div>
    </div>
    <div id="toast" class="toast"></div>
    <script>
        async function submitTask() {
            const taskId = document.getElementById('taskId').value;
            if (!taskId) return;
            const res = await fetch('/api/tasks', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({task_id: taskId})
            });
            const data = await res.json();
            addEvent(JSON.stringify(data));
            const es = new EventSource('/api/tasks/' + taskId + '/stream');
            es.onmessage = e => addEvent(e.data);
        }
        function addEvent(text) {
            const div = document.getElementById('events');
            const el = document.createElement('div');
            el.className = 'event';
            el.textContent = new Date().toISOString().slice(11,19) + ' ' + text;
            div.prepend(el);
        }
        function showToast(msg) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.style.display = 'block';
            setTimeout(() => t.style.display = 'none', 2000);
        }
        async function loadSettings() {
            const res = await fetch('/api/settings');
            const data = await res.json();
            const container = document.getElementById('settings');
            container.innerHTML = '';
            const names = Object.keys(data.settings).sort();
            for (const name of names) {
                const info = data.settings[name];
                const row = document.createElement('div');
                row.className = 'setting-row';
                if (info.type === 'bool') {
                    row.innerHTML = `
                        <div>
                            <span class="setting-name">${name}</span>
                            <span class="setting-source">${info.source}</span>
                        </div>
                        <label class="toggle">
                            <input type="checkbox" ${info.value ? 'checked' : ''} onchange="toggleSetting('${name}', this.checked)">
                            <span class="slider"></span>
                        </label>`;
                } else {
                    row.innerHTML = `
                        <div>
                            <span class="setting-name">${name}</span>
                            <span class="setting-source">${info.source}</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:6px">
                            <input id="str_${name}" value="${info.value}" style="font-family:monospace;padding:4px 8px;border-radius:4px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;font-size:0.85rem;width:280px">
                            <button class="btn-sm" style="background:#3b82f6;color:#fff" onclick="saveSetting('${name}')">Save</button>
                        </div>`;
                }
                container.appendChild(row);
            }
        }
        async function toggleSetting(name, value) {
            const res = await fetch('/api/settings/' + name, {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({value: value})
            });
            if (res.ok) {
                showToast(name + ' = ' + value);
                loadSettings();
            }
        }
        async function saveSetting(name) {
            const input = document.getElementById('str_' + name);
            if (!input) return;
            const value = input.value.trim();
            const res = await fetch('/api/settings/' + name, {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({value: value})
            });
            if (res.ok) {
                showToast(name + ' = ' + value);
                loadSettings();
                loadNodeModels();
            }
        }
        async function loadModels() {
            const res = await fetch('/api/models');
            const data = await res.json();
            const container = document.getElementById('models');
            if (!data.models.length) {
                container.innerHTML = '<span style="color:#64748b">No models yet. Click "+ Add Model" to configure one.</span>';
                return;
            }
            container.innerHTML = '';
            for (const m of data.models) {
                const row = document.createElement('div');
                row.className = 'setting-row';
                row.innerHTML = `
                    <div>
                        <span class="setting-name">${m.name}</span>
                        <span class="setting-source">${m.provider}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:8px">
                        <span style="font-family:monospace;font-size:0.85rem">${m.model} <span style="color:#64748b">(${m.base_url})</span></span>
                        <button class="btn-sm btn-edit" onclick="editModel('${m.name}','${m.provider}','${m.model}','${m.base_url}')">Edit</button>
                        <button class="btn-sm btn-danger" onclick="deleteModel('${m.name}')">Delete</button>
                    </div>`;
                container.appendChild(row);
            }
        }
        function openModelModal(name, provider, model, baseUrl) {
            const editing = name || '';
            document.getElementById('editingModelName').value = editing;
            document.getElementById('modalTitle').textContent = editing ? 'Edit Model' : 'Add Model';
            document.getElementById('mName').value = name || '';
            document.getElementById('mName').disabled = !!editing;
            document.getElementById('mProvider').value = provider || 'openai-compatible';
            document.getElementById('mModel').value = model || '';
            document.getElementById('mBaseUrl').value = (baseUrl && baseUrl !== 'default') ? baseUrl : '';
            document.getElementById('mApiKey').value = '';
            document.getElementById('modelModal').style.display = 'flex';
        }
        function closeModelModal() {
            document.getElementById('modelModal').style.display = 'none';
        }
        function editModel(name, provider, model, baseUrl) {
            openModelModal(name, provider, model, baseUrl);
        }
        async function saveModel() {
            const editing = document.getElementById('editingModelName').value;
            const name = document.getElementById('mName').value.trim();
            const provider = document.getElementById('mProvider').value;
            const model = document.getElementById('mModel').value.trim();
            const baseUrl = document.getElementById('mBaseUrl').value.trim();
            const apiKey = document.getElementById('mApiKey').value.trim();
            if (!name || !model) { showToast('Name and Model are required'); return; }
            const payload = {name, provider, model, base_url: baseUrl || null, api_key: apiKey || null};
            const url = editing ? '/api/models/' + encodeURIComponent(editing) : '/api/models';
            const method = editing ? 'PUT' : 'POST';
            const res = await fetch(url, {method, headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)});
            if (res.ok) {
                showToast((editing ? 'Updated ' : 'Added ') + name);
                closeModelModal();
                loadModels();
                loadNodeModels();
            } else {
                const err = await res.json();
                showToast('Error: ' + (err.detail || 'save failed'));
            }
        }
        async function deleteModel(name) {
            if (!confirm('Delete model "' + name + '"? Node mappings using it will also be removed.')) return;
            const res = await fetch('/api/models/' + encodeURIComponent(name), {method: 'DELETE'});
            if (res.ok) {
                showToast('Deleted ' + name);
                loadModels();
                loadNodeModels();
            }
        }
        async function loadNodeModels() {
            const res = await fetch('/api/node-models');
            const data = await res.json();
            const container = document.getElementById('nodeModels');
            container.innerHTML = '';
            const builtins = data.choices.builtin;
            const customs = data.choices.custom;
            for (const [node, info] of Object.entries(data.node_models).sort()) {
                const row = document.createElement('div');
                row.className = 'setting-row';
                let opts = `<option value="__default__" ${!info.override ? 'selected' : ''}>(default: ${info.tier})</option>`;
                if (builtins.length) {
                    opts += '<optgroup label="Built-in">';
                    for (const b of builtins) {
                        opts += `<option value="${b.value}" ${info.override === b.value ? 'selected' : ''}>${b.label}</option>`;
                    }
                    opts += '</optgroup>';
                }
                if (customs.length) {
                    opts += '<optgroup label="Custom">';
                    for (const c of customs) {
                        opts += `<option value="${c}" ${info.override === c ? 'selected' : ''}>${c}</option>`;
                    }
                    opts += '</optgroup>';
                }
                opts += `<option value="__custom__">Other...</option>`;
                const overrideMatch = info.override && ![...builtins.map(b=>b.value), ...customs].includes(info.override);
                if (overrideMatch) {
                    opts += `<option value="${info.override}" selected>${info.override}</option>`;
                }
                row.innerHTML = `
                    <div>
                        <span class="setting-name">${node}</span>
                        <span class="setting-source">${info.tier}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:8px">
                        <span style="font-family:monospace;font-size:0.8rem;color:#94a3b8">${info.resolved}</span>
                        <select onchange="setNodeModel('${node}', this.value)" style="padding:4px 8px;border-radius:4px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;font-size:0.8rem">${opts}</select>
                    </div>`;
                container.appendChild(row);
            }
        }
        async function setNodeModel(node, value) {
            if (value === '__default__') {
                const res = await fetch('/api/node-models/' + node, {method: 'DELETE'});
                if (res.ok) { showToast(node + ' → default'); loadNodeModels(); }
                return;
            }
            if (value === '__custom__') {
                const custom = prompt('Enter model name (e.g. gpt-4o, claude-sonnet-4-20250514):');
                if (!custom) { loadNodeModels(); return; }
                value = custom.trim();
            }
            const res = await fetch('/api/node-models/' + node, {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model: value})
            });
            if (res.ok) {
                showToast(node + ' → ' + value);
                loadNodeModels();
            }
        }
        async function loadMcpServers() {
            const res = await fetch('/api/mcp-servers');
            const data = await res.json();
            const container = document.getElementById('mcpServers');
            if (!data.servers.length) {
                container.innerHTML = '<span style="color:#64748b">No MCP servers configured.</span>';
                return;
            }
            container.innerHTML = '';
            for (const s of data.servers) {
                const row = document.createElement('div');
                row.className = 'setting-row';
                const args = (s.args || []).join(', ');
                const envCount = s.env ? Object.keys(s.env).length : 0;
                const envBadge = envCount ? `<span class="badge" style="background:#334155;color:#94a3b8">${envCount} env</span>` : '';
                row.innerHTML = `
                    <div>
                        <span class="setting-name">${s.name}</span>${envBadge}
                    </div>
                    <div style="display:flex;align-items:center;gap:8px">
                        <span style="font-family:monospace;font-size:0.8rem;color:#94a3b8">${s.command} ${args}</span>
                        <button class="btn-sm btn-edit" onclick='editMcp(${JSON.stringify(s)})'>Edit</button>
                        <button class="btn-sm btn-danger" onclick="deleteMcp('${s.name}')">Delete</button>
                    </div>`;
                container.appendChild(row);
            }
        }
        function openMcpModal(s) {
            const editing = s ? s.name : '';
            document.getElementById('editingMcpName').value = editing;
            document.getElementById('mcpModalTitle').textContent = editing ? 'Edit MCP Server' : 'Add MCP Server';
            document.getElementById('mcpName').value = s ? s.name : '';
            document.getElementById('mcpName').disabled = !!editing;
            document.getElementById('mcpCommand').value = s ? s.command : '';
            document.getElementById('mcpArgs').value = s ? (s.args || []).join(', ') : '';
            document.getElementById('mcpEnv').value = s && s.env ? Object.entries(s.env).map(([k,v]) => k+'='+v).join('\\n') : '';
            document.getElementById('mcpModal').style.display = 'flex';
        }
        function closeMcpModal() { document.getElementById('mcpModal').style.display = 'none'; }
        function editMcp(s) { openMcpModal(s); }
        async function saveMcp() {
            const editing = document.getElementById('editingMcpName').value;
            const name = document.getElementById('mcpName').value.trim();
            const command = document.getElementById('mcpCommand').value.trim();
            const argsStr = document.getElementById('mcpArgs').value.trim();
            const envStr = document.getElementById('mcpEnv').value.trim();
            if (!name || !command) { showToast('Name and Command are required'); return; }
            const args = argsStr ? argsStr.split(',').map(a => a.trim()).filter(Boolean) : [];
            let env = null;
            if (envStr) {
                env = {};
                for (const line of envStr.split('\\n')) {
                    const eq = line.indexOf('=');
                    if (eq > 0) env[line.slice(0,eq).trim()] = line.slice(eq+1).trim();
                }
            }
            const payload = {name, command, args, env};
            const url = editing ? '/api/mcp-servers/' + encodeURIComponent(editing) : '/api/mcp-servers';
            const method = editing ? 'PUT' : 'POST';
            const res = await fetch(url, {method, headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
            if (res.ok) { showToast((editing?'Updated ':'Added ')+name); closeMcpModal(); loadMcpServers(); }
            else { const err = await res.json(); showToast('Error: '+(err.detail||'failed')); }
        }
        async function deleteMcp(name) {
            if (!confirm('Delete MCP server "'+name+'"?')) return;
            const res = await fetch('/api/mcp-servers/'+encodeURIComponent(name), {method:'DELETE'});
            if (res.ok) { showToast('Deleted '+name); loadMcpServers(); }
        }
        async function loadPrompts() {
            const res = await fetch('/api/prompts');
            const data = await res.json();
            const container = document.getElementById('prompts');
            container.innerHTML = '';
            for (const [name, info] of Object.entries(data.prompts).sort()) {
                const row = document.createElement('div');
                row.className = 'setting-row';
                row.style.cursor = 'pointer';
                const badge = info.is_overridden ? '<span class="badge badge-override">overridden</span>' : '';
                row.innerHTML = `
                    <div onclick="openPromptEditor('${name}')">
                        <span class="setting-name">${name}</span>${badge}
                        <span class="setting-source">${info.length} chars</span>
                    </div>
                    <span style="font-size:0.8rem;color:#94a3b8;max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:inline-block" onclick="openPromptEditor('${name}')">${info.preview}</span>`;
                container.appendChild(row);
            }
        }
        async function openPromptEditor(name) {
            const res = await fetch('/api/prompts/'+encodeURIComponent(name));
            if (!res.ok) return;
            const data = await res.json();
            document.getElementById('editingPromptName').value = name;
            document.getElementById('promptModalTitle').textContent = 'Edit: ' + name;
            document.getElementById('promptText').value = data.text;
            document.getElementById('resetPromptBtn').style.display = data.is_overridden ? 'block' : 'none';
            document.getElementById('promptModal').style.display = 'flex';
        }
        function closePromptModal() { document.getElementById('promptModal').style.display = 'none'; }
        async function savePrompt() {
            const name = document.getElementById('editingPromptName').value;
            const text = document.getElementById('promptText').value;
            const res = await fetch('/api/prompts/'+encodeURIComponent(name), {
                method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})
            });
            if (res.ok) { showToast('Saved '+name); closePromptModal(); loadPrompts(); }
        }
        async function resetPrompt() {
            const name = document.getElementById('editingPromptName').value;
            if (!confirm('Reset "'+name+'" to default?')) return;
            const res = await fetch('/api/prompts/'+encodeURIComponent(name), {method:'DELETE'});
            if (res.ok) { showToast('Reset '+name); closePromptModal(); loadPrompts(); }
        }
        async function loadToolCatalog() {
            const res = await fetch('/api/tool-catalog');
            const data = await res.json();
            const container = document.getElementById('toolCatalog');
            const stages = data.stages || {};
            const stageNames = Object.keys(stages).sort();
            if (!stageNames.length) {
                container.innerHTML = '<span style="color:#64748b">' + (data.note || 'No tools classified yet.') + '</span>';
                return;
            }
            container.innerHTML = '';
            for (const stage of stageNames) {
                const group = document.createElement('div');
                group.className = 'stage-group';
                const tools = stages[stage];
                group.innerHTML = `<div class="stage-title">${stage} (${tools.length})</div>` +
                    tools.map(t => `<span class="tool-chip">${t.name} <span class="tool-server">${t.server}</span></span>`).join('');
                container.appendChild(group);
            }
        }
        loadSettings();
        loadModels();
        loadNodeModels();
        loadMcpServers();
        loadPrompts();
        loadToolCatalog();
    </script>
</body>
</html>"""
