"""Small HTTP server for publishing live agent pose estimates."""

from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


class PoseStateStore:
    """Thread-safe container for the latest simulation pose snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "schema_version": 1,
            "updated_wall_time": None,
            "step": None,
            "sim_time": None,
            "dogs": {},
            "intruder": None,
        }

    def update(self, **fields: Any) -> None:
        """Merge top-level state fields into the current snapshot."""
        with self._lock:
            self._state.update(fields)
            self._state["updated_wall_time"] = time.time()

    def update_dog(self, dog_name: str, state: dict[str, Any]) -> None:
        """Set the latest state for one dog."""
        with self._lock:
            dogs = dict(self._state.get("dogs") or {})
            dogs[dog_name] = state
            self._state["dogs"] = dogs
            self._state["updated_wall_time"] = time.time()

    def update_intruder(self, state: dict[str, Any]) -> None:
        """Set the latest intruder state."""
        with self._lock:
            self._state["intruder"] = state
            self._state["updated_wall_time"] = time.time()

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy safe for serialization by another thread."""
        with self._lock:
            return deepcopy(self._state)


class PoseHttpServer:
    """Background HTTP JSON API for external programs."""

    def __init__(self, store: PoseStateStore, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.store = store
        self.host = host
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            return f"http://{self.host}:{self.port}"
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        """Start serving in a daemon thread."""
        if self._server is not None:
            return

        store = self.store

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                path = urlparse(self.path).path.rstrip("/") or "/"
                state = store.snapshot()

                if path == "/":
                    self._write_html(self._dashboard_html())
                elif path in {"/poses", "/poses/latest"}:
                    self._write_json(state)
                elif path == "/health":
                    self._write_json({"ok": True, "updated_wall_time": state.get("updated_wall_time")})
                elif path == "/poses/dogs":
                    self._write_json(state.get("dogs") or {})
                elif path == "/poses/intruder":
                    self._write_json(state.get("intruder") or {})
                elif path.startswith("/poses/"):
                    name = path.split("/", 2)[2]
                    if name == "intruder":
                        payload = state.get("intruder")
                    else:
                        payload = (state.get("dogs") or {}).get(name)
                    if payload is None:
                        self._write_json({"error": f"pose '{name}' not found"}, status=404)
                    else:
                        self._write_json(payload)
                else:
                    self._write_json(
                        {
                            "error": "not found",
                            "endpoints": [
                                "/",
                                "/health",
                                "/poses",
                                "/poses/latest",
                                "/poses/dogs",
                                "/poses/intruder",
                                "/poses/go2_dog_1",
                                "/poses/go2_dog_2",
                            ],
                        },
                        status=404,
                    )

            def _dashboard_html(self) -> str:
                return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multiagent Live Poses</title>
  <style>
    body { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #101418; color: #e8eef2; }
    header { padding: 18px 22px; background: #17212b; border-bottom: 1px solid #2a3a46; }
    h1 { margin: 0 0 6px; font-size: 20px; }
    .meta { color: #9fb1bd; font-size: 13px; }
    main { padding: 18px; display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    section { background: #151c23; border: 1px solid #2a3a46; border-radius: 12px; padding: 14px; }
    h2 { margin: 0 0 10px; font-size: 16px; color: #8bd3ff; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; line-height: 1.35; font-size: 12px; }
    .ok { color: #8ff0a4; }
    .bad { color: #ff9b9b; }
  </style>
</head>
<body>
  <header>
    <h1>Multiagent Live Poses</h1>
    <div class="meta">Polling <code>/poses</code> every 250 ms. JSON endpoints: <code>/poses/dogs</code>, <code>/poses/go2_dog_1</code>, <code>/poses/go2_dog_2</code>, <code>/poses/intruder</code>.</div>
    <div id="status" class="meta">Connecting...</div>
  </header>
  <main>
    <section><h2>go2_dog_1</h2><pre id="dog1">{}</pre></section>
    <section><h2>go2_dog_2</h2><pre id="dog2">{}</pre></section>
    <section><h2>intruder</h2><pre id="intruder">{}</pre></section>
    <section><h2>all</h2><pre id="all">{}</pre></section>
  </main>
  <script>
    const fmt = (obj) => JSON.stringify(obj ?? {}, null, 2);
    async function refresh() {
      try {
        const response = await fetch('/poses', { cache: 'no-store' });
        const data = await response.json();
        document.getElementById('status').innerHTML =
          `<span class="ok">live</span> step=${data.step} sim_time=${data.sim_time}`;
        const dogs = data.dogs || {};
        document.getElementById('dog1').textContent = fmt(dogs.go2_dog_1);
        document.getElementById('dog2').textContent = fmt(dogs.go2_dog_2);
        document.getElementById('intruder').textContent = fmt(data.intruder);
        document.getElementById('all').textContent = fmt(data);
      } catch (err) {
        document.getElementById('status').innerHTML = `<span class="bad">offline</span> ${err}`;
      }
    }
    refresh();
    setInterval(refresh, 250);
  </script>
</body>
</html>
"""

            def _write_html(self, html: str, status: int = 200) -> None:
                body = html.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_json(self, payload: Any, status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="pose-http-server", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background server."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
