# SPDX-License-Identifier: MIT
"""Hardened loopback-only static HTTP server for the DiffCAM 3D Toolpath & Stock Simulator.

Security posture:
- Binds strictly to ``127.0.0.1``, ``::1``, or ``localhost``; refuses ``0.0.0.0`` and non-loopback binds.
- Read-only HTTP method allow-list (``GET``, ``HEAD`` only; ``405`` on all other verbs).
- Strict ``Content-Security-Policy`` (no ``unsafe-inline`` or ``unsafe-eval``), ``X-Frame-Options: DENY``,
  ``X-Content-Type-Options: nosniff``, ``Permissions-Policy``, ``Cache-Control: no-store``, and COOP/CORP.
- All file paths resolved via ``safe_resolve_under`` with trailing ``os.sep`` boundary verification,
  strict extension allow-listing, symlink escape prevention, and per-client sliding-window rate limiting.
"""

# TODO(security): This server is intentionally scoped as a local loopback-only (127.0.0.1 / ::1)
# single-user scientific visualization tool with zero state-changing endpoints, zero file uploads,
# zero cookies/sessions, and zero database backing. If this viewer is ever deployed beyond localhost,
# terminate TLS 1.3+ at a reverse proxy and add OAuth 2.0 / OIDC authentication (with MFA and leaked-
# password checks), server-side RBAC authorization, and distributed rate limiting.

from __future__ import annotations

import argparse
import collections
import ipaddress
import re
import socket
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verification.simulator_export import safe_resolve_under

ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_SERVED_BYTES = 64 * 1024 * 1024  # 64 MiB defensive ceiling
RATE_LIMIT_MAX_REQUESTS = 240
RATE_LIMIT_WINDOW_S = 10.0

SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

EXTENSION_CONTENT_TYPES: Dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".stl": "application/octet-stream",
    ".rustcnc": "application/octet-stream",
    ".camotics": "application/json; charset=utf-8",
    ".nc": "text/plain; charset=utf-8",
}

SECURITY_HEADERS: Tuple[Tuple[str, str], ...] = (
    (
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; "
        "form-action 'none'; frame-ancestors 'none'",
    ),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()"),
    ("Referrer-Policy", "no-referrer"),
    ("Cache-Control", "no-store"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
)


def validate_loopback_host(host: str) -> str:
    """Ensure ``host`` resolves strictly to a loopback interface (never ``0.0.0.0`` or ``::``)."""
    if not isinstance(host, str):
        raise ValueError("Host must be a string.")
    cleaned = host.strip().lower()
    if cleaned in ALLOWED_LOOPBACK_HOSTS:
        return cleaned
    try:
        ip = ipaddress.ip_address(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"Refusing non-loopback bind host {host!r}; use '127.0.0.1', '::1', or 'localhost'."
        ) from exc
    if not ip.is_loopback or ip.is_unspecified:
        raise ValueError(
            f"Refusing non-loopback bind address {host!r}; servers MUST listen on 127.0.0.1 or localhost."
        )
    return cleaned


class SlidingWindowRateLimiter:
    """Thread-safe per-client sliding-window rate limiter."""

    def __init__(self, max_requests: int = RATE_LIMIT_MAX_REQUESTS, window_s: float = RATE_LIMIT_WINDOW_S) -> None:
        self._max_requests = int(max_requests)
        self._window_s = float(window_s)
        self._lock = threading.Lock()
        self._hits: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)

    def allow(self, client_key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window_s
        with self._lock:
            q = self._hits[client_key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self._max_requests:
                return False
            q.append(now)
            return True


class SimulatorHTTPServer(ThreadingHTTPServer):
    """Loopback HTTP server holding immutable root paths and rate limiter state."""

    def __init__(
        self,
        server_address: Tuple[str, int],
        viewer_root: Path,
        run_dir: Path,
        rate_limiter: Optional[SlidingWindowRateLimiter] = None,
    ) -> None:
        host, port = server_address
        validated_host = validate_loopback_host(host)
        if validated_host == "::1":
            self.address_family = socket.AF_INET6
        self.viewer_root = Path(viewer_root).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.sim_data_root = safe_resolve_under(self.run_dir, "simulator_viewer")
        self.rustcnc_root = safe_resolve_under(self.run_dir, "rustcnc_projects")
        self.camotics_root = safe_resolve_under(self.run_dir, "camotics_projects")
        self.haas_root = safe_resolve_under(self.run_dir, "haas_ngc_packets")
        self.rate_limiter = rate_limiter or SlidingWindowRateLimiter()
        super().__init__((validated_host, int(port)), SimulatorRequestHandler)


class SimulatorRequestHandler(BaseHTTPRequestHandler):
    """Read-only static handler enforcing strict CSP, allow-lists, and path containment."""

    server_version = "DiffCAMSimulator/1.0"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Emit concise sanitized method/status logs without echoing untrusted headers or query params.
        sys.stderr.write(f"[SimulatorHTTP] {self.address_string()} - {format % args}\n")

    def _send_security_headers(self) -> None:
        for k, v in SECURITY_HEADERS:
            self.send_header(k, v)

    def _send_plain_error(self, status: HTTPStatus, message: str, *, allow_header: bool = False) -> None:
        body = (message + "\n").encode("utf-8")
        self.send_response(int(status))
        self._send_security_headers()
        if allow_header:
            self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _resolve_route(self, raw_target: str) -> Tuple[Optional[Path], bool, Optional[HTTPStatus]]:
        """Map an untrusted request URI to ``(resolved_path, force_attachment, error_status)``."""
        if not raw_target or len(raw_target) > 512:
            return None, False, HTTPStatus.BAD_REQUEST
        parsed = urllib.parse.urlsplit(raw_target)
        # Reject raw or percent-encoded traversal tokens before any filesystem operation.
        unquoted = urllib.parse.unquote(parsed.path)
        if not unquoted.startswith("/") or "\x00" in unquoted or "\\" in unquoted:
            return None, False, HTTPStatus.BAD_REQUEST
        segments = [seg for seg in unquoted.split("/") if seg != ""]
        if any(seg in (".", "..") for seg in segments):
            return None, False, HTTPStatus.FORBIDDEN

        srv: SimulatorHTTPServer = self.server  # type: ignore[assignment]
        try:
            if len(segments) == 0 or segments == ["index.html"]:
                return safe_resolve_under(srv.viewer_root, "index.html"), False, None
            if segments == ["styles.css"]:
                return safe_resolve_under(srv.viewer_root, "styles.css"), False, None
            if segments == ["app.js"]:
                return safe_resolve_under(srv.viewer_root, "app.js"), False, None

            if segments == ["data", "manifest.json"]:
                return safe_resolve_under(srv.sim_data_root, "manifest.json"), False, None
            if len(segments) == 3 and segments[0] == "data" and segments[1] == "cases":
                fname = segments[2]
                if not SAFE_FILENAME_RE.fullmatch(fname) or not fname.endswith(".json"):
                    return None, False, HTTPStatus.NOT_FOUND
                return safe_resolve_under(srv.sim_data_root, f"cases/{fname}"), False, None
            if len(segments) == 3 and segments[0] == "data" and segments[1] == "meshes":
                fname = segments[2]
                if not SAFE_FILENAME_RE.fullmatch(fname) or not fname.endswith(".stl"):
                    return None, False, HTTPStatus.NOT_FOUND
                return safe_resolve_under(srv.sim_data_root, f"meshes/{fname}"), False, None

            if len(segments) == 3 and segments[0] == "downloads":
                bucket, fname = segments[1], segments[2]
                if not SAFE_FILENAME_RE.fullmatch(fname):
                    return None, False, HTTPStatus.NOT_FOUND
                if bucket == "rustcnc" and fname.endswith(".rustcnc"):
                    return safe_resolve_under(srv.rustcnc_root, fname), True, None
                if bucket == "camotics" and fname.endswith((".camotics", ".nc")):
                    return safe_resolve_under(srv.camotics_root, fname), True, None
                if bucket == "haas" and fname.endswith(".nc"):
                    return safe_resolve_under(srv.haas_root, fname), True, None
                return None, False, HTTPStatus.NOT_FOUND
        except ValueError:
            return None, False, HTTPStatus.FORBIDDEN

        return None, False, HTTPStatus.NOT_FOUND

    def _handle_read_request(self, send_body: bool) -> None:
        srv: SimulatorHTTPServer = self.server  # type: ignore[assignment]
        client_ip = str(self.client_address[0]) if self.client_address else "local"
        if not srv.rate_limiter.allow(client_ip):
            self._send_plain_error(HTTPStatus.TOO_MANY_REQUESTS, "Rate limit exceeded.")
            return

        target_path, as_attachment, err = self._resolve_route(self.path)
        if err is not None or target_path is None:
            self._send_plain_error(err or HTTPStatus.NOT_FOUND, "Resource unavailable.")
            return

        ext = target_path.suffix.lower()
        content_type = EXTENSION_CONTENT_TYPES.get(ext)
        if content_type is None or not target_path.is_file() or target_path.is_symlink():
            self._send_plain_error(HTTPStatus.NOT_FOUND, "Not found.")
            return

        try:
            stat = target_path.stat()
            if stat.st_size > MAX_SERVED_BYTES:
                self._send_plain_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Artifact exceeds size limit.")
                return
            payload = target_path.read_bytes() if send_body else b""
        except OSError:
            self._send_plain_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error.")
            return

        self.send_response(int(HTTPStatus.OK))
        self._send_security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(stat.st_size))
        if as_attachment and SAFE_FILENAME_RE.fullmatch(target_path.name):
            self.send_header("Content-Disposition", f'attachment; filename="{target_path.name}"')
        self.end_headers()
        if send_body:
            self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._handle_read_request(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle_read_request(send_body=False)

    def do_POST(self) -> None:  # noqa: N802
        self._send_plain_error(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_header=True)

    def do_PUT(self) -> None:  # noqa: N802
        self._send_plain_error(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_header=True)

    def do_DELETE(self) -> None:  # noqa: N802
        self._send_plain_error(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_header=True)

    def do_PATCH(self) -> None:  # noqa: N802
        self._send_plain_error(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_header=True)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_plain_error(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_header=True)

    def do_TRACE(self) -> None:  # noqa: N802
        self._send_plain_error(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_header=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "pub-v2-20261110",
        help="Campaign run directory containing simulator_viewer/ and rustcnc_projects/.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Loopback address to bind ('127.0.0.1', '::1', or 'localhost' ONLY).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="TCP port on loopback interface (0 for ephemeral OS port).",
    )
    args = parser.parse_args()

    host = validate_loopback_host(args.host)
    viewer_root = (PROJECT_ROOT / "viewer").resolve()
    run_dir = args.run_dir.resolve()
    manifest_file = safe_resolve_under(run_dir, "simulator_viewer/manifest.json")
    if not manifest_file.is_file():
        raise SystemExit(
            f"Missing {manifest_file}. Run `python scripts/export_simulator_bundles.py --run-dir {run_dir}` first."
        )

    httpd = SimulatorHTTPServer((host, args.port), viewer_root=viewer_root, run_dir=run_dir)
    bound_host, bound_port = httpd.server_address[:2]
    url_host = f"[{bound_host}]" if ":" in str(bound_host) else str(bound_host)
    print(
        f"DiffCAM 3D Toolpath & Resulting Stock Simulator listening on http://{url_host}:{bound_port}/ "
        f"(run={run_dir.name})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down simulator server.", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
