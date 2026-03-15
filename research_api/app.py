"""Standalone Flask app for the portable research service."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).with_name(".env"))

from research_service import get_research_capabilities, run_research, serialize_research_run


def _api_prefix() -> str:
    prefix = str(os.environ.get("RESEARCH_API_PREFIX", "/api") or "/api").strip()
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    return prefix.rstrip("/") or "/api"


def _configure_logging() -> None:
    level_name = str(os.environ.get("LOG_LEVEL", "INFO") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> Flask:
    _configure_logging()
    app = Flask(__name__)
    api_prefix = _api_prefix()

    @app.get("/health")
    def health() -> Response:
        return jsonify({"ok": True, "service": "portable-research-api"})

    @app.get(f"{api_prefix}/research/status")
    def research_status() -> Response:
        return jsonify(get_research_capabilities())

    @app.post(f"{api_prefix}/research")
    def research() -> tuple[Response, int] | Response:
        data = request.get_json(silent=True) or {}
        message = str(data.get("message", "") or "").strip()
        if not message:
            return jsonify({
                "status": "error",
                "error": "Message is required",
                "response": "",
                "sources": [],
                "meta": {},
            }), 400

        run = run_research(
            message=message,
            history=data.get("history", []) or [],
            provider_mode=str(data.get("provider_mode", "auto") or "auto"),
            tool_mode=str(data.get("tool_mode", "auto") or "auto"),
            depth=str(data.get("depth", "balanced") or "balanced"),
            compare=bool(data.get("compare", False)),
            manual_tools=data.get("manual_tools", {}) or {},
        )
        return jsonify(serialize_research_run(run))

    @app.post(f"{api_prefix}/research/stream")
    def research_stream() -> tuple[Response, int] | Response:
        data = request.get_json(silent=True) or {}
        message = str(data.get("message", "") or "").strip()
        if not message:
            return jsonify({
                "status": "error",
                "error": "Message is required",
                "response": "",
                "sources": [],
                "meta": {},
            }), 400

        history = data.get("history", []) or []
        provider_mode = str(data.get("provider_mode", "auto") or "auto")
        tool_mode = str(data.get("tool_mode", "auto") or "auto")
        depth = str(data.get("depth", "balanced") or "balanced")
        compare = bool(data.get("compare", False))
        manual_tools = data.get("manual_tools", {}) or {}

        def emit_data(payload: dict[str, Any]) -> str:
            return f"data: {json.dumps(payload, default=str)}\n\n"

        def emit_done() -> str:
            return "data: [DONE]\n\n"

        def generate():
            event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

            def progress(event: dict[str, Any]) -> None:
                event_queue.put({
                    "type": "progress",
                    **event,
                })

            def worker() -> None:
                try:
                    run = run_research(
                        message=message,
                        history=history,
                        provider_mode=provider_mode,
                        tool_mode=tool_mode,
                        depth=depth,
                        compare=compare,
                        manual_tools=manual_tools,
                        progress=progress,
                    )
                    event_queue.put({
                        "type": "result",
                        "result": serialize_research_run(run),
                    })
                except Exception as exc:  # pragma: no cover
                    event_queue.put({
                        "type": "error",
                        "error": str(exc) or "Research request failed",
                    })
                finally:
                    event_queue.put(None)

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            while True:
                event = event_queue.get()
                if event is None:
                    break
                yield emit_data(event)
            yield emit_done()

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    host = str(os.environ.get("HOST", "127.0.0.1") or "127.0.0.1").strip()
    port = int(os.environ.get("PORT", "8090") or "8090")
    debug = str(os.environ.get("DEBUG", "false") or "false").strip().lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug, threaded=True)
