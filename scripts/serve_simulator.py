# SPDX-License-Identifier: MIT
"""Hardened loopback-only HTTP server for the DiffCAM 3D Toolpath, Stock & CAD Simulator.

Security posture:
- Binds strictly to ``127.0.0.1``, ``::1``, or ``localhost``; refuses ``0.0.0.0`` and non-loopback binds.
- Per-route HTTP method allow-list (``GET``, ``HEAD`` on static assets & ``/api/session``; ``POST`` only on
  ``/api/machine-part``; ``405`` with accurate ``Allow`` header on all other verbs).
- Strict ``Content-Security-Policy`` (no ``unsafe-inline`` or ``unsafe-eval``), ``X-Frame-Options: DENY``,
  ``X-Content-Type-Options: nosniff``, ``Permissions-Policy``, ``Cache-Control: no-store``, and COOP/CORP.
- Expiring ``HttpOnly; SameSite=Strict`` server-side sessions + HMAC-bound Synchronizer CSRF token
  (``X-CSRF-Token`` verified via ``hmac.compare_digest``) + strict ``Origin`` / ``Sec-Fetch-Site`` validation.
- Hardened CAD file upload (``.step``, ``.stp``, ``.stl`` allow-list, <= 10 MiB limit, magic-byte & structural
  parser verification, random UUID4 quarantine filename in ``0o700`` directory outside web roots with ``0o600`` mode).
- All served file paths resolved via ``safe_resolve_under`` with trailing ``os.sep`` boundary verification,
  symlink escape prevention, and dual sliding-window rate limiters (static read + CAD compute).
"""

# TODO(security): This server is scoped as a local loopback-only (127.0.0.1 / ::1) single-operator scientific
# CAM workstation. If ever deployed beyond localhost:
#   1. Terminate TLS 1.3+ at a reverse proxy so session cookies always enforce `__Host-` + `Secure`.
#   2. Add OAuth 2.0 / OIDC authentication with MFA, leaked-password checks, and server-side RBAC ownership.
#   3. Integrate sandboxed antivirus / CDR malware scanning on quarantined CAD uploads prior to persistence.

from __future__ import annotations

import argparse
import base64
import binascii
import collections
import hashlib
import hmac
import http.cookies
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import sys
import threading
import time
import urllib.parse
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePath
from typing import Any, Deque, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cam.feeds_and_speeds import STOCK_MATERIALS, TOOL_MATERIALS
from cam.multi_dialect_post import SUPPORTED_MULTI_DIALECTS
from cam.step_brep import is_step_payload
from cam.tamp_planner import (
    execute_tamp_pipeline,
    export_tamp_result_to_viewer_bundle,
    load_stl_mesh,
)
from cam.tool_library import DEFAULT_TOOL_LIBRARY
from verification.simulator_export import safe_resolve_under

LOGGER = logging.getLogger("diffcam.simulator_http")

ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_SERVED_BYTES = 64 * 1024 * 1024  # 64 MiB defensive static ceiling
MAX_JSON_BODY_BYTES = 15 * 1024 * 1024  # 15 MiB max JSON envelope (covers 10 MiB base64 CAD)
MAX_UPLOAD_CAD_BYTES = 10 * 1024 * 1024  # 10 MiB max decoded CAD binary/STEP payload

RATE_LIMIT_MAX_REQUESTS = 240
RATE_LIMIT_WINDOW_S = 10.0
COMPUTE_RATE_LIMIT_MAX = 15
COMPUTE_RATE_LIMIT_WINDOW_S = 60.0

SESSION_TTL_S = 1800  # 30-minute session expiration
MAX_ACTIVE_SESSIONS = 256

SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SAFE_TITLE_CHARS_RE = re.compile(r"[^A-Za-z0-9 _.#+/()-]+")

ALLOWED_CAD_EXTENSIONS = frozenset({".step", ".stp", ".stl"})
ALLOWED_UNIT_SYSTEMS = frozenset({"inches", "millimeters", "both"})
ALLOWED_SETUP_MODES = frozenset({"auto", "single_op1", "dual_op1_op2"})
ALLOWED_CRIB_MODES = frozenset({"selected_crib", "all_unit"})

EXTENSION_CONTENT_TYPES: Dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".stl": "application/octet-stream",
    ".rustcnc": "application/octet-stream",
    ".camotics": "application/json; charset=utf-8",
    ".nc": "text/plain; charset=utf-8",
    ".ngc": "text/plain; charset=utf-8",
    ".eia": "text/plain; charset=utf-8",
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

EXAMPLE_PARTS_ALLOWLIST: Dict[str, Tuple[str, str]] = {
    f"titan-m{i}": (f"Titan-M{i}.step", f"TITAN-{i}M Building Block")
    for i in range(1, 11)
}


def _resolve_session_secret() -> bytes:
    """Multi-tiered secret resolution without any hardcoded literal fallback."""
    env_val = os.getenv("DIFFCAM_SESSION_SECRET", "").strip()
    if len(env_val) >= 32:
        return env_val.encode("utf-8")
    secret_file = Path(os.getenv("DIFFCAM_SESSION_SECRET_FILE", "")).expanduser() if os.getenv("DIFFCAM_SESSION_SECRET_FILE") else None
    if secret_file and secret_file.is_file():
        raw = secret_file.read_text(encoding="utf-8").strip()
        if len(raw) >= 32:
            return raw.encode("utf-8")
    LOGGER.warning(
        "DIFFCAM_SESSION_SECRET not configured; generating ephemeral 256-bit OS CSPRNG secret (instance-isolated)."
    )
    return secrets.token_bytes(32)


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


class SessionStore:
    """Expiring server-side session & Synchronizer CSRF token manager."""

    def __init__(self, secret_key: bytes, ttl_s: float = SESSION_TTL_S) -> None:
        self._secret = bytes(secret_key)
        self._ttl_s = float(ttl_s)
        self._lock = threading.Lock()
        self._sessions: Dict[str, Tuple[str, float]] = {}  # sid -> (csrf_nonce, expires_monotonic)

    def _derive_csrf(self, sid: str, nonce: str) -> str:
        msg = f"{sid}:{nonce}".encode("ascii")
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def _prune_unlocked(self, now: float) -> None:
        expired = [k for k, (_, exp) in self._sessions.items() if exp <= now]
        for k in expired:
            self._sessions.pop(k, None)
        while len(self._sessions) >= MAX_ACTIVE_SESSIONS:
            oldest_sid = next(iter(self._sessions))
            self._sessions.pop(oldest_sid, None)

    def get_or_create(self, incoming_sid: Optional[str]) -> Tuple[str, str, bool]:
        """Return ``(sid, csrf_token, is_new_cookie)``."""
        now = time.monotonic()
        with self._lock:
            self._prune_unlocked(now)
            if incoming_sid and SAFE_ID_RE.fullmatch(incoming_sid) and incoming_sid in self._sessions:
                nonce, _ = self._sessions[incoming_sid]
                self._sessions[incoming_sid] = (nonce, now + self._ttl_s)
                return incoming_sid, self._derive_csrf(incoming_sid, nonce), False
            sid = secrets.token_hex(24)
            nonce = secrets.token_hex(24)
            self._sessions[sid] = (nonce, now + self._ttl_s)
            return sid, self._derive_csrf(sid, nonce), True

    def verify_csrf(self, incoming_sid: Optional[str], presented_token: Optional[str]) -> bool:
        if not incoming_sid or not presented_token:
            return False
        if not SAFE_ID_RE.fullmatch(incoming_sid) or len(presented_token) != 64:
            return False
        now = time.monotonic()
        with self._lock:
            self._prune_unlocked(now)
            rec = self._sessions.get(incoming_sid)
            if rec is None:
                return False
            nonce, exp = rec
            if exp <= now:
                self._sessions.pop(incoming_sid, None)
                return False
            expected = self._derive_csrf(incoming_sid, nonce)
            if not hmac.compare_digest(expected, presented_token):
                return False
            self._sessions[incoming_sid] = (nonce, now + self._ttl_s)
            return True


class SimulatorHTTPServer(ThreadingHTTPServer):
    """Loopback HTTP server holding immutable root paths, session store, and rate limiters."""

    def __init__(
        self,
        server_address: Tuple[str, int],
        viewer_root: Path,
        run_dir: Path,
        rate_limiter: Optional[SlidingWindowRateLimiter] = None,
        compute_rate_limiter: Optional[SlidingWindowRateLimiter] = None,
        example_parts_dir: Optional[Path] = None,
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
        self.tamp_packets_root = safe_resolve_under(self.run_dir, "tamp_shop_packets")

        # Isolated non-executable quarantine directory OUTSIDE viewer_root and sim_data_root
        self.quarantine_root = safe_resolve_under(self.run_dir, "uploads_quarantine")
        self.quarantine_root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.quarantine_root, 0o700)
        except OSError:
            pass

        self.example_parts_root = (
            Path(example_parts_dir).resolve()
            if example_parts_dir is not None
            else (WORKSPACE_ROOT / "example_parts").resolve()
        )
        self.rate_limiter = rate_limiter or SlidingWindowRateLimiter()
        self.compute_rate_limiter = compute_rate_limiter or SlidingWindowRateLimiter(
            max_requests=COMPUTE_RATE_LIMIT_MAX, window_s=COMPUTE_RATE_LIMIT_WINDOW_S
        )
        self.sessions = SessionStore(_resolve_session_secret(), ttl_s=SESSION_TTL_S)
        self.tool_catalog_cache: Dict[str, Any] = DEFAULT_TOOL_LIBRARY.export_viewer_catalog()
        self.valid_tool_ids = frozenset(t.tool_id for t in DEFAULT_TOOL_LIBRARY.tools)

        self.state_lock = threading.Lock()
        self.cad_registry: Dict[str, Tuple[Path, str]] = {}
        self.dynamic_cases: Dict[str, Dict[str, Any]] = {}
        self.next_program_number: int = 2401
        super().__init__((validated_host, int(port)), SimulatorRequestHandler)

    def allowed_origins(self) -> Tuple[str, ...]:
        port = int(self.server_address[1])
        return (
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
            f"http://[::1]:{port}",
            f"https://127.0.0.1:{port}",
            f"https://localhost:{port}",
            f"https://[::1]:{port}",
        )


class SimulatorRequestHandler(BaseHTTPRequestHandler):
    """Hardened HTTP handler enforcing CSP, CSRF, Origin checks, upload quarantine, and path containment."""

    server_version = "DiffCAMSimulator/2.0"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        sys.stderr.write(f"[SimulatorHTTP] {self.address_string()} - {format % args}\n")

    def _send_security_headers(self) -> None:
        for k, v in SECURITY_HEADERS:
            self.send_header(k, v)

    def _is_tls_request(self) -> bool:
        proto = (self.headers.get("X-Forwarded-Proto") or "").strip().lower()
        return proto == "https"

    def _extract_session_id(self) -> Optional[str]:
        raw_cookie = self.headers.get("Cookie")
        if not raw_cookie or len(raw_cookie) > 4096:
            return None
        jar = http.cookies.SimpleCookie()
        try:
            jar.load(raw_cookie)
        except http.cookies.CookieError:
            return None
        for cname in ("__Host-diffcam_sid", "diffcam_sid"):
            morsel = jar.get(cname)
            if morsel is not None and SAFE_ID_RE.fullmatch(morsel.value):
                return morsel.value
        return None

    def _send_session_cookie(self, sid: str) -> None:
        # Upgrade cookie to __Host- + Secure automatically over HTTPS; use HttpOnly + SameSite=Strict on HTTP loopback
        if self._is_tls_request():
            cookie_val = (
                f"__Host-diffcam_sid={sid}; Path=/; Max-Age={SESSION_TTL_S}; HttpOnly; Secure; SameSite=Strict"
            )
        else:
            cookie_val = f"diffcam_sid={sid}; Path=/; Max-Age={SESSION_TTL_S}; HttpOnly; SameSite=Strict"
        self.send_header("Set-Cookie", cookie_val)

    def _send_plain_error(
        self, status: HTTPStatus, message: str, *, allow_methods: Optional[str] = None
    ) -> None:
        body = (message + "\n").encode("utf-8")
        self.send_response(int(status))
        self._send_security_headers()
        if allow_methods:
            self.send_header("Allow", allow_methods)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(
        self, status: HTTPStatus, payload: Dict[str, Any], *, set_sid: Optional[str] = None
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self._send_security_headers()
        if set_sid is not None:
            self._send_session_cookie(set_sid)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _clean_path_segments(self, raw_target: str) -> Tuple[Optional[List[str]], Optional[HTTPStatus]]:
        if not raw_target or len(raw_target) > 512:
            return None, HTTPStatus.BAD_REQUEST
        parsed = urllib.parse.urlsplit(raw_target)
        unquoted = urllib.parse.unquote(parsed.path)
        if not unquoted.startswith("/") or "\x00" in unquoted or "\\" in unquoted:
            return None, HTTPStatus.BAD_REQUEST
        segments = [seg for seg in unquoted.split("/") if seg != ""]
        if any(seg in (".", "..") for seg in segments):
            return None, HTTPStatus.FORBIDDEN
        return segments, None

    def _resolve_route(self, raw_target: str) -> Tuple[Optional[Path], bool, Optional[HTTPStatus]]:
        """Map an untrusted static URI to ``(resolved_path, force_attachment, error_status)``."""
        segments, path_err = self._clean_path_segments(raw_target)
        if path_err is not None or segments is None:
            return None, False, path_err

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
                if bucket == "haas" and fname.endswith((".nc", ".ngc", ".eia")):
                    return safe_resolve_under(srv.haas_root, fname), True, None
                return None, False, HTTPStatus.NOT_FOUND
        except ValueError:
            return None, False, HTTPStatus.FORBIDDEN

        return None, False, HTTPStatus.NOT_FOUND

    def _handle_api_session(self) -> None:
        srv: SimulatorHTTPServer = self.server  # type: ignore[assignment]
        incoming_sid = self._extract_session_id()
        sid, csrf_tok, _ = srv.sessions.get_or_create(incoming_sid)

        ex_list: List[Dict[str, str]] = []
        if srv.example_parts_root.is_dir():
            for ex_id, (fname, lbl) in EXAMPLE_PARTS_ALLOWLIST.items():
                try:
                    p = safe_resolve_under(srv.example_parts_root, fname)
                    if p.is_file() and not p.is_symlink():
                        ex_list.append({"id": ex_id, "filename": fname, "label": f"{lbl} ({fname})"})
                except ValueError:
                    continue

        with srv.state_lock:
            dyn_cases = list(srv.dynamic_cases.values())

        self._send_json(
            HTTPStatus.OK,
            {
                "csrf_token": csrf_tok,
                "tool_catalog": srv.tool_catalog_cache,
                "example_parts": ex_list,
                "dynamic_cases": dyn_cases,
            },
            set_sid=sid,
        )

    def _verify_same_origin_post(self, srv: SimulatorHTTPServer) -> bool:
        sec_fetch = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if sec_fetch and sec_fetch not in ("same-origin", "none"):
            return False
        origin = (self.headers.get("Origin") or "").strip().rstrip("/")
        if origin:
            return origin in srv.allowed_origins()
        referer = (self.headers.get("Referer") or "").strip()
        if referer:
            return any(referer.startswith(ao + "/") or referer == ao for ao in srv.allowed_origins())
        return True

    def _quarantine_uploaded_cad(
        self, srv: SimulatorHTTPServer, raw_filename: str, cad_b64: str
    ) -> Tuple[Optional[Path], Optional[str], Optional[str], Optional[HTTPStatus], str]:
        """Validate extension, size, and magic bytes; write to 0o600 UUID file outside web roots."""
        base_name = PurePath(str(raw_filename or "").replace("\\", "/")).name
        ext = Path(base_name).suffix.lower()
        if not base_name or ext not in ALLOWED_CAD_EXTENSIONS:
            return None, None, None, HTTPStatus.BAD_REQUEST, "Only .step, .stp, and .stl CAD files are permitted."

        if not isinstance(cad_b64, str) or len(cad_b64) == 0:
            return None, None, None, HTTPStatus.BAD_REQUEST, "Missing CAD file payload."
        if len(cad_b64) > (MAX_UPLOAD_CAD_BYTES * 4 // 3) + 4096:
            return None, None, None, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "CAD upload exceeds 10 MiB limit."

        try:
            raw_bytes = base64.b64decode(cad_b64, validate=True)
        except (binascii.Error, ValueError):
            return None, None, None, HTTPStatus.BAD_REQUEST, "Malformed base64 CAD encoding."

        if len(raw_bytes) == 0 or len(raw_bytes) > MAX_UPLOAD_CAD_BYTES:
            return None, None, None, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "CAD upload must be between 1 B and 10 MiB."

        # Strict content inspection (magic header + full structural parse check)
        try:
            if ext in (".step", ".stp"):
                if not is_step_payload(raw_bytes):
                    return None, None, None, HTTPStatus.BAD_REQUEST, "File content is not a valid ISO-10303-21 STEP solid."
            else:
                mesh = load_stl_mesh(raw_bytes)
                if mesh.triangle_count < 4:
                    return None, None, None, HTTPStatus.BAD_REQUEST, "STL mesh contains fewer than 4 facets."
        except Exception:
            return None, None, None, HTTPStatus.BAD_REQUEST, "CAD payload failed structural geometry validation."

        cad_id = f"cad-{uuid.uuid4().hex[:12]}"
        q_name = f"{uuid.uuid4().hex}{ext}"
        q_path = safe_resolve_under(srv.quarantine_root, q_name)
        fd = os.open(str(q_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, raw_bytes)
        finally:
            os.close(fd)
        return q_path, base_name, cad_id, None, ""

    def _resolve_replan_or_example_cad(
        self, srv: SimulatorHTTPServer, source_mode: str, req_data: Dict[str, Any]
    ) -> Tuple[Optional[Path], str, str, Optional[HTTPStatus], str]:
        if source_mode == "example":
            ex_id = str(req_data.get("example_id") or "").strip().lower()
            spec = EXAMPLE_PARTS_ALLOWLIST.get(ex_id)
            if spec is None:
                return None, "", "", HTTPStatus.BAD_REQUEST, "Unknown example part identifier."
            fname, _ = spec
            try:
                ex_path = safe_resolve_under(srv.example_parts_root, fname)
            except ValueError:
                return None, "", "", HTTPStatus.FORBIDDEN, "Invalid example path."
            if not ex_path.is_file() or ex_path.is_symlink():
                return None, "", "", HTTPStatus.NOT_FOUND, "Requested example STEP file is unavailable."
            return ex_path, fname, ex_id, None, ""

        # source_mode == "replan_case"
        src_id = str(req_data.get("source_cad_id") or req_data.get("case_id") or "").strip()
        if not SAFE_ID_RE.fullmatch(src_id):
            return None, "", "", HTTPStatus.BAD_REQUEST, "Invalid source_cad_id for re-planning."

        with srv.state_lock:
            reg = srv.cad_registry.get(src_id)
        if reg is not None and reg[0].is_file() and not reg[0].is_symlink():
            return reg[0], reg[1], src_id, None, ""

        if src_id in EXAMPLE_PARTS_ALLOWLIST:
            fname, _ = EXAMPLE_PARTS_ALLOWLIST[src_id]
            ex_path = safe_resolve_under(srv.example_parts_root, fname)
            if ex_path.is_file() and not ex_path.is_symlink():
                return ex_path, fname, src_id, None, ""

        # Check TAMP STL input packet fallback (e.g. tamp-flanged-bracket)
        try:
            tamp_stl = safe_resolve_under(srv.tamp_packets_root, f"{src_id}_input.stl")
            if tamp_stl.is_file() and not tamp_stl.is_symlink():
                return tamp_stl, tamp_stl.name, src_id, None, ""
        except ValueError:
            pass

        return None, "", "", HTTPStatus.NOT_FOUND, "Original CAD source not found for re-planning; please upload the CAD file."

    def _handle_machine_part_post(self) -> None:
        srv: SimulatorHTTPServer = self.server  # type: ignore[assignment]
        client_ip = str(self.client_address[0]) if self.client_address else "local"

        if not self._verify_same_origin_post(srv):
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Cross-origin POST forbidden."})
            return

        sid = self._extract_session_id()
        csrf_hdr = (self.headers.get("X-CSRF-Token") or "").strip()
        if not srv.sessions.verify_csrf(sid, csrf_hdr):
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Invalid or expired session CSRF token."})
            return

        if not srv.compute_rate_limiter.allow(client_ip):
            self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "Machining rate limit exceeded; please wait a moment."})
            return

        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Expected Content-Type: application/json."})
            return

        try:
            content_len = int(self.headers.get("Content-Length") or "-1")
        except ValueError:
            content_len = -1
        if content_len <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Empty request body."})
            return
        if content_len > MAX_JSON_BODY_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Request payload exceeds 15 MiB limit."})
            return

        try:
            raw_body = self.rfile.read(content_len)
            req_data = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Malformed JSON request payload."})
            return

        if not isinstance(req_data, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Expected JSON object body."})
            return

        # Strict allow-list validation of all machining options
        source_mode = str(req_data.get("source_mode") or "upload").strip().lower()
        if source_mode not in ("upload", "example", "replan_case"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid source_mode."})
            return

        unit_sys = str(req_data.get("unit_system") or "inches").strip().lower()
        if unit_sys not in ALLOWED_UNIT_SYSTEMS:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid unit_system."})
            return

        crib_mode = str(req_data.get("crib_mode") or "all_unit").strip().lower()
        if crib_mode not in ALLOWED_CRIB_MODES:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid crib_mode."})
            return

        allowed_tool_ids: Optional[List[str]] = None
        if crib_mode == "selected_crib":
            raw_ids = req_data.get("allowed_tool_ids")
            if not isinstance(raw_ids, list) or len(raw_ids) == 0 or len(raw_ids) > 800:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Select at least one valid tool in the Tool Crib."})
                return
            validated_ids = [str(tid) for tid in raw_ids if isinstance(tid, str) and tid in srv.valid_tool_ids]
            if not validated_ids:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "None of the selected tool IDs belong to the catalog."})
                return
            allowed_tool_ids = validated_ids

        setup_mode = str(req_data.get("setup_mode") or "auto").strip().lower()
        if setup_mode not in ALLOWED_SETUP_MODES:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid setup_mode."})
            return

        stock_mat = str(req_data.get("stock_material_id") or "al_6061_t6").strip()
        if stock_mat not in STOCK_MATERIALS:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Unsupported stock_material_id."})
            return

        tool_mat = str(req_data.get("tool_material_id") or "carbide_coated").strip()
        if tool_mat not in TOOL_MATERIALS:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Unsupported tool_material_id."})
            return

        dialect = str(req_data.get("controller_dialect") or "haas_ngc").strip().lower()
        if dialect not in SUPPORTED_MULTI_DIALECTS:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Unsupported controller_dialect."})
            return

        try:
            opt_iters = max(4, min(80, int(req_data.get("opt_iters", 28))))
        except (TypeError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "opt_iters must be an integer."})
            return

        enable_probing = bool(req_data.get("enable_probing", False))
        enable_chamfer = bool(req_data.get("enable_chamfer", True))
        enable_threading = bool(req_data.get("enable_threading", True))

        if source_mode == "upload":
            cad_path, fname_hint, src_cad_id, err_st, err_msg = self._quarantine_uploaded_cad(
                srv, str(req_data.get("filename") or ""), str(req_data.get("cad_base64") or "")
            )
            if err_st is not None or cad_path is None or fname_hint is None or src_cad_id is None:
                self._send_json(err_st or HTTPStatus.BAD_REQUEST, {"error": err_msg})
                return
            with srv.state_lock:
                srv.cad_registry[src_cad_id] = (cad_path, fname_hint)
        else:
            cad_path, fname_hint, src_cad_id, err_st, err_msg = self._resolve_replan_or_example_cad(
                srv, source_mode, req_data
            )
            if err_st is not None or cad_path is None:
                self._send_json(err_st or HTTPStatus.BAD_REQUEST, {"error": err_msg})
                return

        raw_title = str(req_data.get("part_title") or "").strip()
        if not raw_title:
            raw_title = Path(fname_hint).stem
        clean_title = SAFE_TITLE_CHARS_RE.sub("", raw_title).strip()[:48] or "Custom CAD Part"

        with srv.state_lock:
            prog_num = srv.next_program_number
            srv.next_program_number += 1

        if source_mode == "replan_case" and SAFE_ID_RE.fullmatch(str(req_data.get("case_id") or "")):
            base_cid = str(req_data.get("case_id"))
            part_id = base_cid if base_cid.startswith("custom-") else f"custom-{base_cid[:24]}-{uuid.uuid4().hex[:4]}"
        else:
            slug = re.sub(r"[^a-z0-9]+", "-", clean_title.lower()).strip("-")[:20] or "part"
            part_id = f"custom-{slug}-{uuid.uuid4().hex[:6]}"

        try:
            res = execute_tamp_pipeline(
                cad_path,
                part_id=part_id,
                title=clean_title,
                program_number=prog_num,
                stock_material_id=stock_mat,
                opt_iters_per_op=opt_iters,
                filename_hint=fname_hint,
                unit_system=None if unit_sys == "both" else unit_sys,
                allowed_tool_ids=allowed_tool_ids,
                tool_material_id=tool_mat,
                setup_mode=setup_mode,
                controller_dialect=dialect,
                enable_probing=enable_probing,
                enable_chamfer=enable_chamfer,
                enable_threading=enable_threading,
            )
            _, manifest_entry = export_tamp_result_to_viewer_bundle(
                res,
                srv.run_dir,
                program_number=prog_num,
                split_name="custom_cad",
                source_cad_id=src_cad_id,
                can_replan=True,
            )
        except Exception as exc:
            LOGGER.error("Automated CAD planning failed for %s: %s", part_id, type(exc).__name__)
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": "Unable to synthesize a collision-free machining plan for the supplied CAD geometry."},
            )
            return

        with srv.state_lock:
            srv.cad_registry[part_id] = (cad_path, fname_hint)
            srv.dynamic_cases[part_id] = manifest_entry

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "case_entry": manifest_entry,
                "case_json_url": manifest_entry["case_json_url"],
                "warnings": res.warnings,
            },
        )

    def _handle_read_request(self, send_body: bool) -> None:
        srv: SimulatorHTTPServer = self.server  # type: ignore[assignment]
        client_ip = str(self.client_address[0]) if self.client_address else "local"
        if not srv.rate_limiter.allow(client_ip):
            self._send_plain_error(HTTPStatus.TOO_MANY_REQUESTS, "Rate limit exceeded.")
            return

        segments, path_err = self._clean_path_segments(self.path)
        if path_err is not None or segments is None:
            self._send_plain_error(path_err or HTTPStatus.BAD_REQUEST, "Invalid request URI.")
            return

        if segments == ["api", "session"]:
            self._handle_api_session()
            return
        if segments == ["api", "machine-part"]:
            self._send_plain_error(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_methods="POST")
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
        if target_path.name == "index.html":
            sid, _, is_new = srv.sessions.get_or_create(self._extract_session_id())
            if is_new:
                self._send_session_cookie(sid)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(stat.st_size))
        if as_attachment and SAFE_FILENAME_RE.fullmatch(target_path.name):
            self.send_header("Content-Disposition", f'attachment; filename="{target_path.name}"')
        self.end_headers()
        if send_body:
            self.wfile.write(payload)

    def _allowed_methods_for_current_path(self) -> str:
        segments, err = self._clean_path_segments(self.path)
        if err is None and segments == ["api", "machine-part"]:
            return "POST"
        return "GET, HEAD"

    def do_GET(self) -> None:  # noqa: N802
        self._handle_read_request(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle_read_request(send_body=False)

    def do_POST(self) -> None:  # noqa: N802
        segments, err = self._clean_path_segments(self.path)
        if err is not None or segments is None:
            self._send_plain_error(err or HTTPStatus.BAD_REQUEST, "Invalid request URI.")
            return
        if segments == ["api", "machine-part"]:
            self._handle_machine_part_post()
            return
        self._send_plain_error(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_methods="GET, HEAD")

    def do_PUT(self) -> None:  # noqa: N802
        self._send_plain_error(
            HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_methods=self._allowed_methods_for_current_path()
        )

    def do_DELETE(self) -> None:  # noqa: N802
        self._send_plain_error(
            HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_methods=self._allowed_methods_for_current_path()
        )

    def do_PATCH(self) -> None:  # noqa: N802
        self._send_plain_error(
            HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_methods=self._allowed_methods_for_current_path()
        )

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_plain_error(
            HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_methods=self._allowed_methods_for_current_path()
        )

    def do_TRACE(self) -> None:  # noqa: N802
        self._send_plain_error(
            HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed.", allow_methods=self._allowed_methods_for_current_path()
        )


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
        f"DiffCAM 3D Toolpath, Stock & Arbitrary CAD Simulator listening on http://{url_host}:{bound_port}/ "
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
