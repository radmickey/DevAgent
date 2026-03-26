"""Web interface: FastAPI + SSE for real-time pipeline monitoring.

Provides:
- POST /api/tasks — submit a task
- GET /api/tasks/{id} — get task status
- GET /api/tasks/{id}/stream — SSE stream of pipeline events
- GET /api/status — agent status + cost summary
- GET / — simple dashboard
"""

from __future__ import annotations

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

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        """Monitoring dashboard with settings management."""
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
                        <span style="font-family:monospace">${info.value}</span>`;
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
        loadSettings();
    </script>
</body>
</html>"""
