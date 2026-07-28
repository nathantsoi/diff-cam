#!/usr/bin/env python3
"""Serve the train_csg results dashboard over HTTPS with a self-signed cert.

Why HTTPS: WebGPU is gated to "secure contexts" (HTTPS, localhost, or file://).
Serving over plain http://<lan-host>:<port> leaves `navigator.gpu` undefined in
Chrome, so the WebGPU voxel viewer falls back to SVG. A self-signed cert makes
any LAN origin a secure context so WebGPU activates; Chrome will warn about the
untrusted cert on first visit — click "Advanced → Proceed" to continue.

The cert is generated once into <web_dir>/.cert/ and reused on subsequent runs.

Usage:
    uv run python scripts/serve_web_https.py [--port 8443] [--host 0.0.0.0] \
        [--root .] [--web autoresearch/tasks/train_csg/web]

Defaults: host 0.0.0.0 (LAN-reachable), port 8443, repo root as the serve root
so runs/... download links in data.json resolve.

Open in Chrome (accept the cert warning):
    https://robolidar:8443/autoresearch/tasks/train_csg/web/index.html
    https://128.83.141.126:8443/autoresearch/tasks/train_csg/web/index.html
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parent.parents]:
        if (p / ".git").exists():
            return p
    return Path.cwd()


def _json_sanitize(obj):
    """Recursively replace non-finite floats (inf / -inf / NaN) with None.

    Defense-in-depth at the API egress: Python's ``json`` would otherwise emit
    ``Infinity`` / ``-Infinity`` / ``NaN`` (invalid JSON), which the browser's
    ``JSON.parse`` rejects, silently emptying the dashboard. ``build_results_web``
    sanitizes at the source (``load_run``), but this guarantees every response —
    including a cached payload built before that fix or any future non-finite
    value — is strict JSON. ``null`` renders as "—" in the page.
    """
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
    return obj


def ensure_cert(cert_dir: Path, host: str) -> tuple[Path, Path]:
    """Generate a self-signed cert+key via openssl if not already present."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert = cert_dir / "cert.pem"
    key = cert_dir / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    subj = f"/CN={host}"
    # SANs so the cert is valid for localhost, the LAN hostname, and the LAN IP.
    import ipaddress
    dns_names = ["localhost"]
    ip_addrs = ["127.0.0.1"]
    try:
        ipaddress.ip_address(host)
        if host not in ip_addrs:
            ip_addrs.append(host)
    except ValueError:
        if host not in dns_names:
            dns_names.append(host)
    for h in ("robolidar",):
        if h not in dns_names:
            dns_names.append(h)
    for ip in ("128.83.141.126",):
        if ip not in ip_addrs:
            ip_addrs.append(ip)
    san_lines = [f"DNS.{i+1}={n}" for i, n in enumerate(dns_names)]
    san_lines += [f"IP.{i+1}={a}" for i, a in enumerate(ip_addrs)]
    tmp_conf = cert_dir / "openssl.cnf"
    tmp_conf.write_text(
        "[req]\n"
        "distinguished_name=req\n"
        "x509_extensions=v3_ca\n"
        "prompt=no\n"
        "[v3_ca]\n"
        "subjectAltName=@alt_names\n"
        "[alt_names]\n"
        + "\n".join(san_lines) + "\n"
    )
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(cert),
        "-days", "3650", "-subj", subj,
        "-config", str(tmp_conf),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        sys.exit("error: openssl not found; install it to generate the self-signed cert.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: openssl failed:\n{e.stderr.decode(errors='replace')}")
    finally:
        try:
            tmp_conf.unlink()
        except OSError:
            pass
    return cert, key


# ---------------------------------------------------------------------------
# On-demand run-video generation (called from the dashboard's "Generate video"
# button). Renders a saved run's carve to an mp4 via the Taichi CSG simulator
# (scripts/render_run_video.py), then serves it as a static file under runs/.
# ---------------------------------------------------------------------------
def _pick_free_gpu() -> str | None:
    """Return the index of the GPU with the most free memory, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return None
    best_idx, best_free = None, -1
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            idx, free = parts[0], int(parts[1])
        except ValueError:
            continue
        if free > best_free:
            best_idx, best_free = idx, free
    return best_idx


def _safe_run_path(root: Path, run_rel: str) -> Path | None:
    """Resolve a `runs/<name>` path under root, rejecting traversal escapes."""
    if not run_rel:
        return None
    root_runs = (root / "runs").resolve()
    candidate = (root / run_rel).resolve()
    try:
        candidate.relative_to(root_runs)
    except ValueError:
        return None
    return candidate if candidate.is_dir() else None


def _resolve_run_by_name(root: Path, name: str) -> Path | None:
    """Resolve a bare run basename (e.g. ``CamEnvDiff-v0__train_csg__1__1783725757990``)
    to its ``runs/<batch>/<name>`` dir, searching under ``runs/`` recursively.

    Lets the dashboard be direct-linked with just the run name — no need to know
    which batch subdir it lives under. Run names are unique across batches, so a
    name maps to at most one dir; if several somehow match, the newest (by mtime)
    wins. Returns None if nothing matches.
    """
    root_runs = (root / "runs").resolve()
    if not root_runs.is_dir():
        return None
    hits = [p for p in root_runs.rglob(name) if p.is_dir() and p.name == name]
    if not hits:
        return None
    if len(hits) > 1:
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def _resolve_run_arg(root: Path, run: str) -> Path | None:
    """Resolve a `run` query value to a runs/<name> dir under root.

    Accepts:
      - the sentinel ``latest`` → newest viewable run dir;
      - an explicit ``runs/<batch>/<name>`` path;
      - a bare run basename (``<name>``) → resolved by searching under ``runs/``.

    The bare-name form lets runs be direct-linked without knowing their batch
    subdir, e.g. ``?run=CamEnvDiff-v0__train_csg__1__1783725757990``.
    """
    if not run:
        return None
    if run == "latest":
        # Import here so the server starts even if numpy is absent; list_runs
        # reads runs/ on demand and reuses build_results_web's per-run loader.
        import sys
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from build_results_web import list_runs
        runs = list_runs()
        if not runs:
            return None
        return _safe_run_path(root, runs[0]["run_dir"])
    # Explicit runs/<...> path (with traversal-escape guard).
    resolved = _safe_run_path(root, run)
    if resolved is not None:
        return resolved
    # Bare run basename: search runs/ for a matching dir.
    return _resolve_run_by_name(root, run)


# ---------------------------------------------------------------------------
# Human feedback store (star ratings + free-text notes per run).
#
# The dashboard lets a user rate each run 1-7 stars and attach a note. The
# store is a single JSON file under the task dir, keyed by run basename (run
# names are unique across batches). train_csg.py reads the same file at startup
# so human feedback flows into future runs (logged +, opt-in, warm-started).
# ---------------------------------------------------------------------------
def feedback_path(root: Path) -> Path:
    """Path to the shared run_feedback.json under the train_csg task dir."""
    return root / "autoresearch" / "tasks" / "train_csg" / "run_feedback.json"


def load_feedback(root: Path) -> dict:
    """Read the feedback store; returns {} if missing/corrupt (never raises)."""
    p = feedback_path(root)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text() or "{}")
    except (OSError, ValueError):
        return {}


def save_feedback(root: Path, data: dict) -> None:
    """Atomically write the feedback store (temp file + replace)."""
    p = feedback_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, p)


def _run_key_from_rel(run_rel: str) -> str:
    """Normalize a runs/<batch>/<name> path (or bare <name>) to its basename.

    The basename is the unique key into the feedback store; the batch subdir is
    not part of the identity so a run keeps its rating regardless of how it was
    addressed.
    """
    return run_rel.rstrip("/").rsplit("/", 1)[-1]


# Sentinel for "field not provided" (distinct from None, which means "clear").
_UNSET = object()

# Serializes read-modify-write on the feedback store. The server is a
# ThreadingTCPServer, so concurrent POSTs (e.g. rating several runs in quick
# succession, or a star click landing during a note save) run in separate
# threads. Without a lock, two set_feedback calls each load the file, each add
# one entry, and each save -- the second save silently clobbers the first, so
# one of the ratings is LOST ("not all persisted"). Holding this lock across
# the whole load->modify->save makes each update atomic.
_FEEDBACK_LOCK = threading.Lock()


def set_feedback(root: Path, run: str, stars=_UNSET, feedback=_UNSET) -> dict:
    """Set/clear one run's feedback entry. Returns the stored entry.

    `stars` is an integer 1-7 (or None to clear); `feedback` is a free-text
    string (or "" to clear). Either may be omitted (_UNSET) to leave that field
    unchanged. An entry left with no stars and empty text is removed so the
    store stays clean.

    The full load->modify->save is held under _FEEDBACK_LOCK so concurrent
    ratings don't clobber each other (lost-update fix).
    """
    with _FEEDBACK_LOCK:
        data = load_feedback(root)
        key = _run_key_from_rel(run)
        entry = data.get(key, {})
        if stars is not _UNSET:
            if stars is None:
                entry["stars"] = None
            else:
                try:
                    s = int(stars)
                except (TypeError, ValueError):
                    s = None
                # Accept only the documented 1-7 ratings; anything else -> null.
                entry["stars"] = s if (s is not None and 1 <= s <= 7) else None
        if feedback is not _UNSET:
            entry["feedback"] = str(feedback).strip()
        entry["ts"] = time.time()
        if not entry.get("stars") and not entry.get("feedback"):
            data.pop(key, None)
        else:
            data[key] = entry
        save_feedback(root, data)
        return data.get(key, {})


# ---------------------------------------------------------------------------
# Pairwise comparison store (A/B trajectory preferences).
#
# The autoresearch agent enqueues pairs of runs it wants a human to compare;
# compare.html fetches the pending pairs, renders both trajectories side by
# side, and the user picks A / B / tie. Answers persist here so they can flow
# back into future runs (train_csg.py reads the same file at startup, mirroring
# the star-rating feedback path).
#
# Schema: a list of pair objects
#   {"id": "p_0001", "run_a": "<basename>", "run_b": "<basename>",
#    "prompt": "...", "dimension": "w_air_time", "magnitude_a": "1e-3",
#    "magnitude_b": "1e-2", "scenario": "sphere s1",
#    "ts": <epoch>, "answer": "a"|"b"|"tie"|null, "answer_ts": <epoch>|null,
#    "note": ""}
# dimension/magnitude_a/magnitude_b/scenario are optional (added for the
# preference-based-objective-learning loop); old pairs omit them. run_a/run_b
# are stored as basenames (the unique key, same convention as the star-rating
# store) so a pair survives regardless of batch folder moves.
# ---------------------------------------------------------------------------
def pairwise_path(root: Path) -> Path:
    """Path to pairwise.json under the train_csg task dir."""
    return root / "autoresearch" / "tasks" / "train_csg" / "pairwise.json"


def load_pairs(root: Path) -> list:
    """Read the pairwise store; returns [] if missing/corrupt (never raises)."""
    p = pairwise_path(root)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text() or "[]")
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def save_pairs(root: Path, data: list) -> None:
    """Atomically write the pairwise store (temp file + replace)."""
    p = pairwise_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, p)


def _new_pair_id(pairs: list) -> str:
    """Next unused p_NNNN id."""
    used = {p.get("id") for p in pairs}
    n = 1
    while f"p_{n:04d}" in used:
        n += 1
    return f"p_{n:04d}"


# Serializes read-modify-write on the pairwise store (same lost-update risk as
# _FEEDBACK_LOCK: the threading server can POST two pair answers concurrently
# and the second save would clobber the first).
_PAIRS_LOCK = threading.Lock()


def add_pair(
    root: Path,
    run_a: str,
    run_b: str,
    prompt: str = "",
    dimension: str = "",
    magnitude_a: str = "",
    magnitude_b: str = "",
    scenario: str = "",
) -> dict:
    """Append a new unanswered pair; returns the stored pair object.

    `dimension` is the single objective knob the pair varies (e.g. `w_air_time`);
    `magnitude_a` / `magnitude_b` are the two values of that knob; `scenario`
    is a short label for the fixed config (shape/seed/iters). All optional and
    backward compatible — old callers and old pairs omit them.
    """
    with _PAIRS_LOCK:
        data = load_pairs(root)
        pair = {
            "id": _new_pair_id(data),
            "run_a": _run_key_from_rel(run_a),
            "run_b": _run_key_from_rel(run_b),
            "prompt": (prompt or "").strip(),
            "dimension": (dimension or "").strip(),
            "magnitude_a": str(magnitude_a or "").strip(),
            "magnitude_b": str(magnitude_b or "").strip(),
            "scenario": (scenario or "").strip(),
            "ts": time.time(),
            "answer": None,
            "answer_ts": None,
            "note": "",
        }
        data.append(pair)
        save_pairs(root, data)
        return pair


def record_pair_answer(root: Path, pair_id: str, answer: str, note: str = "") -> dict | None:
    """Record a human answer for one pair. Returns the updated pair or None."""
    if answer not in ("a", "b", "tie"):
        return None
    with _PAIRS_LOCK:
        data = load_pairs(root)
        for p in data:
            if p.get("id") == pair_id:
                p["answer"] = answer
                p["answer_ts"] = time.time()
                p["note"] = str(note).strip() if note is not None else ""
                save_pairs(root, data)
                return p
        return None


def update_pair_note(root: Path, pair_id: str, note: str) -> dict | None:
    """Update only the free-text note on an already-answered pair.

    Leaves the recorded answer (and answer_ts) untouched so the learned
    preference is not disturbed; lets a user refine the rationale later.
    """
    with _PAIRS_LOCK:
        data = load_pairs(root)
        for p in data:
            if p.get("id") == pair_id:
                p["note"] = str(note).strip() if note is not None else ""
                save_pairs(root, data)
                return p
        return None


def generate_run_video(root: Path, run_rel: str, force: bool = False) -> dict:
    """Ensure runs/<run>/videos/run.mp4 exists; generate it if missing.

    Returns {"ok": True, "path": <repo-relative mp4>} or {"ok": False, "error": ...}.
    """
    run_dir = _safe_run_path(root, run_rel)
    if run_dir is None:
        return {"ok": False, "error": f"invalid or unknown run: {run_rel}"}

    mp4 = run_dir / "videos" / "run.mp4"
    if mp4.exists() and not force:
        return {"ok": True, "path": os.path.relpath(mp4, root)}

    script = root / "scripts" / "render_run_video.py"
    if not script.exists():
        return {"ok": False, "error": "render_run_video.py not found in scripts/"}

    env = dict(os.environ)
    gpu = _pick_free_gpu()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    cmd = [sys.executable, str(script), "--run", run_rel, "--mode", "both"]
    print(f"[video] generating {run_rel} (gpu={gpu}) ...")
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), env=env, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "render timed out (>10 min)"}
    if proc.returncode != 0 or not mp4.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        return {"ok": False, "error": "render failed", "detail": "\n".join(tail)}
    print(f"[video] done: {mp4}")
    return {"ok": True, "path": os.path.relpath(mp4, root)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--root", default=None, help="serve root (default: repo root)")
    ap.add_argument("--web", default="autoresearch/tasks/train_csg/web", help="web dir (for cert storage)")
    args = ap.parse_args()

    repo = find_repo_root()
    root = Path(args.root).resolve() if args.root else repo
    web_dir = (repo / args.web).resolve()
    if not web_dir.is_dir():
        fallback = (repo / "autoresearch" / "tasks" / "train_csg" / "web").resolve()
        if fallback.is_dir():
            web_dir = fallback
        else:
            sys.exit(f"error: web dir not found: {web_dir}")

    cert, key = ensure_cert(web_dir / ".cert", args.host)

    os.chdir(root)

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from build_results_web import IncrementalResultsBuilder

    builder = IncrementalResultsBuilder(generate_gcode=True, verbose=True)
    builder.get_payload()

    def background_builder_loop():
        while True:
            time.sleep(3)
            try:
                builder.get_payload()
            except Exception as e:
                print(f"[builder sync error] {e}")

    threading.Thread(target=background_builder_loop, daemon=True).start()

    # Dev server: send no-store so browsers never serve a cached JS module (the
    # dynamic import("./voxel.js") otherwise stays stale across edits, making code
    # changes appear to do nothing).
    class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            path = urllib.parse.urlparse(self.path).path
            if not path.startswith("/runs/"):
                self.send_header("Cache-Control", "no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
            super().end_headers()

        def _json(self, obj, status=200):
            body = json.dumps(_json_sanitize(obj)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path in ("/web/data.json", "/data.json", "/__api/data.json") or parsed.path.endswith("/data.json"):
                data = builder.get_payload()
                if data is not None:
                    return self._json(data)
            # On-demand video generation: GET /__api/video?run=runs/<name>[&force=1]
            if parsed.path == "/__api/video" or parsed.path.endswith("/__api/video"):
                qs = urllib.parse.parse_qs(parsed.query)
                run = (qs.get("run", [""])[0] or "").strip()
                force = "1" in qs.get("force", [])
                if not run:
                    return self._json({"ok": False, "error": "missing run param"}, 400)
                return self._json(generate_run_video(root, run, force=force))
            # List all viewable run dirs (newest first) for the dashboard's
            # arbitrary-run picker. Accepts ?batch=old|current|all to filter by
            # batch folder; no param returns every runs/<name>.
            if parsed.path == "/__api/runs" or parsed.path.endswith("/__api/runs"):
                import sys
                scripts_dir = str(Path(__file__).resolve().parent)
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)
                from build_results_web import list_runs
                qs = urllib.parse.parse_qs(parsed.query)
                batch = (qs.get("batch", [""])[0] or "").strip() or None
                return self._json({"runs": list_runs(batch=batch)})
            # Discover experiment batch directories under runs/ — auto-populates
            # the dashboard's batch selector. New branches added to runs/ show up
            # here without code changes.
            if parsed.path == "/__api/batches" or parsed.path.endswith("/__api/batches"):
                import sys
                scripts_dir = str(Path(__file__).resolve().parent)
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)
                from build_results_web import discover_batches
                return self._json({"batches": discover_batches()})
            # Fetch one arbitrary run's full record (args/metrics/trajectory/stl/
            # gcode/tool_geom). `run=latest` resolves to the newest run dir, so a
            # fresh train_csg run can be inspected without knowing its name.
            if parsed.path == "/__api/run" or parsed.path.endswith("/__api/run"):
                qs = urllib.parse.parse_qs(parsed.query)
                run = (qs.get("run", [""])[0] or "").strip()
                run_dir = _resolve_run_arg(root, run)
                if run_dir is None:
                    return self._json({"ok": False, "error": f"invalid or unknown run: {run}"}, 404)
                import sys
                scripts_dir = str(Path(__file__).resolve().parent)
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)
                from build_results_web import run_record
                rec = run_record(run_dir)
                if rec is None:
                    return self._json({"ok": False, "error": f"no viewable artifacts in {run}"}, 404)
                return self._json(rec)
            # All human feedback (star ratings + notes), keyed by run basename.
            # The dashboard fetches this once at load and merges it into the run
            # rows; train_csg.py reads the same file directly to feed ratings
            # into future runs.
            if parsed.path == "/__api/feedback" or parsed.path.endswith("/__api/feedback"):
                return self._json({"feedback": load_feedback(root)})
            # Pairwise comparison pairs (agent-queued A/B trajectory
            # comparisons + recorded human answers). ?status=pending returns
            # only unanswered pairs; otherwise the full list (newest-aware
            # order: as written).
            if parsed.path == "/__api/pairs" or parsed.path.endswith("/__api/pairs"):
                pairs = load_pairs(root)
                qs = urllib.parse.parse_qs(parsed.query)
                status = (qs.get("status", [""])[0] or "").strip()
                if status == "pending":
                    pairs = [p for p in pairs if not p.get("answer")]
                elif status == "answered":
                    pairs = [p for p in pairs if p.get("answer")]
                return self._json({"pairs": pairs})
            # Preference digest: answered pairs aggregated by dimension (the
            # single objective knob a pair varies). Same view the agent gets via
            # scripts/pref_digest.py; the compare.html digest panel fetches this.
            if parsed.path == "/__api/pref-digest" or parsed.path.endswith("/__api/pref-digest"):
                import sys
                scripts_dir = str(Path(__file__).resolve().parent)
                if scripts_dir not in sys.path:
                    sys.path.insert(0, scripts_dir)
                from pref_lib import digest, pending, summary_counts
                pairs = load_pairs(root)
                return self._json({
                    "by_dimension": digest(pairs),
                    "pending": pending(pairs),
                    "counts": summary_counts(pairs),
                })
            return super().do_GET()

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            # Save one run's star rating / feedback note. Body is JSON:
            # {"run": "runs/<batch>/<name>" | "<name>", "stars": 1-7|null,
            #  "feedback": "..."}. Returns the stored entry.
            if parsed.path == "/__api/feedback" or parsed.path.endswith("/__api/feedback"):
                try:
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    body = json.loads(raw.decode() or "{}")
                except (ValueError, OSError):
                    return self._json({"ok": False, "error": "invalid JSON body"}, 400)
                run = (body.get("run") or "").strip()
                if not run:
                    return self._json({"ok": False, "error": "missing run param"}, 400)
                # Only override a field when its key is present in the body — a
                # present null clears it, an absent key leaves it unchanged (so a
                # star click doesn't wipe the note, and a note save doesn't touch
                # the stars).
                entry = set_feedback(
                    root, run,
                    stars=body["stars"] if "stars" in body else _UNSET,
                    feedback=body["feedback"] if "feedback" in body else _UNSET,
                )
                return self._json({"ok": True, "entry": entry})
            # Pairwise comparison actions. Body branches on intent:
            #  - add a pair:        {"run_a": "...", "run_b": "...", "prompt": "...",
            #                        "dimension": "...", "magnitude_a": "...",
            #                        "magnitude_b": "...", "scenario": "..."}
            #                       (dimension/magnitude_*/scenario optional)
            #  - record an answer:  {"id": "p_0001", "answer": "a"|"b"|"tie", "note": "..."}
            if parsed.path == "/__api/pairs" or parsed.path.endswith("/__api/pairs"):
                try:
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    body = json.loads(raw.decode() or "{}")
                except (ValueError, OSError):
                    return self._json({"ok": False, "error": "invalid JSON body"}, 400)
                if body.get("update_note") and body.get("id"):
                    pair = update_pair_note(
                        root, str(body["id"]).strip(), body.get("note", ""))
                    if pair is None:
                        return self._json({"ok": False, "error": "unknown pair id"}, 400)
                    return self._json({"ok": True, "pair": pair})
                if "answer" in body and body.get("id"):
                    pair = record_pair_answer(
                        root, str(body["id"]).strip(), str(body["answer"]).strip(),
                        body.get("note", ""))
                    if pair is None:
                        return self._json({"ok": False, "error": "invalid answer or unknown pair id"}, 400)
                    return self._json({"ok": True, "pair": pair})
                if body.get("run_a") and body.get("run_b"):
                    pair = add_pair(
                        root, str(body["run_a"]), str(body["run_b"]), body.get("prompt", ""),
                        dimension=body.get("dimension", ""),
                        magnitude_a=body.get("magnitude_a", ""),
                        magnitude_b=body.get("magnitude_b", ""),
                        scenario=body.get("scenario", ""),
                    )
                    return self._json({"ok": True, "pair": pair})
                return self._json({"ok": False, "error": "provide {run_a,run_b} to add a pair or {id,answer} to record an answer"}, 400)
            return self._json({"ok": False, "error": "unknown POST endpoint"}, 404)

    handler = NoCacheHandler

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    httpd = Server((args.host, args.port), handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    rel = os.path.relpath(str(web_dir), str(root))
    urls = [
        f"https://{h}:{args.port}/{rel}/index.html"
        for h in (args.host, "localhost", "127.0.0.1")
        if h not in ("0.0.0.0",)
    ]
    if args.host == "0.0.0.0":
        urls = [
            f"https://localhost:{args.port}/{rel}/index.html",
            f"https://127.0.0.1:{args.port}/{rel}/index.html",
            f"https://robolidar:{args.port}/{rel}/index.html",
            f"https://128.83.141.126:{args.port}/{rel}/index.html",
        ]
    print(f"serving {root} over HTTPS on {args.host}:{args.port}")
    print(f"cert: {cert}")
    print("open in Chrome (accept the self-signed cert warning):")
    for u in urls:
        print(f"  {u}")
    print("\nCtrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
