"""HTTP transport for the perception pipeline.

The simulation process posts an `EnvironmentSensorFrame` (pickled) to
`POST /perceive` and gets back a pickled `PerceptionOutput`. This is the
formal wire boundary between the environment and perception layers.

Pickle is used for body framing: both endpoints run in the same trust
domain (same machine, same user), and `EnvironmentSensorFrame` carries
torch tensors and dataclasses that round-trip cleanly through pickle. If
you ever split the two layers across machines, swap pickle for msgpack +
explicit tensor schemas — the contract surface here is intentionally tiny
so that swap is local.
"""

from __future__ import annotations

import io
import pickle
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from environment.types import EnvironmentSensorFrame

from .pipeline import PerceptionPipeline
from .types import PerceiveRequest, PerceptionOutput


PICKLE_MIME = "application/x-python-pickle"


class PerceptionHttpServer:
    """Background HTTP server that runs `PerceptionPipeline.update` per request."""

    def __init__(
        self,
        pipeline: PerceptionPipeline,
        host: str = "127.0.0.1",
        port: int = 8767,
    ) -> None:
        self.pipeline = pipeline
        self.host = host
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        # Pipeline state is mutable across calls; serialize requests so a
        # single Python perception instance owns its state machine.
        self._lock = threading.Lock()

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

        pipeline = self.pipeline
        lock = self._lock

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path.rstrip("/") in ("", "/health"):
                    self._write_json_ok()
                else:
                    self._write_error(404, "not found")

            def do_POST(self) -> None:
                if self.path.rstrip("/") != "/perceive":
                    self._write_error(404, "unknown endpoint")
                    return

                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._write_error(400, "bad Content-Length")
                    return
                if length <= 0:
                    self._write_error(400, "empty body")
                    return

                buf = io.BytesIO()
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:
                        self._write_error(400, "short body")
                        return
                    buf.write(chunk)
                    remaining -= len(chunk)

                try:
                    payload = pickle.loads(buf.getvalue())
                except Exception as exc:  # noqa: BLE001
                    self._write_error(400, f"could not decode body: {exc}")
                    return

                if isinstance(payload, PerceiveRequest):
                    frame = payload.frame
                    update_dogs = bool(payload.update_dogs)
                    update_intruder = bool(payload.update_intruder)
                elif isinstance(payload, EnvironmentSensorFrame):
                    frame = payload
                    update_dogs = True
                    update_intruder = True
                else:
                    self._write_error(
                        400,
                        "body must be a pickled PerceiveRequest or EnvironmentSensorFrame",
                    )
                    return

                try:
                    with lock:
                        output: PerceptionOutput = pipeline.update(
                            frame,
                            update_dogs=update_dogs,
                            update_intruder=update_intruder,
                        )
                except Exception as exc:  # noqa: BLE001
                    self._write_error(500, f"pipeline update failed: {exc}")
                    return

                payload = pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)
                self.send_response(200)
                self.send_header("Content-Type", PICKLE_MIME)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _write_json_ok(self, status: int = 200) -> None:
                body = b'{"ok": true}'
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_error(self, status: int, message: str) -> None:
                body = f'{{"error": "{message}"}}'.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="perception-http-server",
            daemon=True,
        )
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
