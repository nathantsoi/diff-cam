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
from contextlib import contextmanager
import fcntl
import http.server
import json
import os
import socketserver
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
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
_PAIRWISE_PATH_OVERRIDE: Path | None = None


def pairwise_path(root: Path) -> Path:
    """Path to pairwise.json under the train_csg task dir."""
    if _PAIRWISE_PATH_OVERRIDE is not None:
        return _PAIRWISE_PATH_OVERRIDE
    return root / "autoresearch" / "tasks" / "train_csg" / "pairwise.json"


def seed_isolated_smoke_pairs(root: Path, destination: Path, count: int = 2) -> list:
    """Copy pending pair descriptions into a disposable, non-research store."""
    source = root / "autoresearch" / "tasks" / "train_csg" / "pairwise.json"
    try:
        records = json.loads(source.read_text() or "[]")
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read source pair store for smoke mode: {source}") from exc
    pending = [
        record
        for record in records
        if isinstance(record, dict) and not record.get("answer")
    ]
    if len(pending) < count:
        raise ValueError(
            f"smoke mode requires {count} pending pairs; found {len(pending)}"
        )
    seeded = []
    for index, source_pair in enumerate(pending[:count], 1):
        pair = dict(source_pair)
        for key in list(pair):
            if key.startswith("direct_feedback_"):
                pair.pop(key)
        pair.update(
            {
                "id": f"smoke_{index:04d}",
                "smoke_source_pair_id": source_pair.get("id"),
                "experimental_evidence_eligible": False,
                "answer": None,
                "answer_ts": None,
                "note": "",
                "first_view_ts": None,
                "display_snapshot": None,
            }
        )
        seeded.append(pair)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(seeded, indent=2, sort_keys=True))
    return seeded


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


@contextmanager
def _pair_store_lock(root: Path):
    """Serialize pair mutations across the web server and CLI processes."""
    lock_path = pairwise_path(root).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def add_pair(
    root: Path,
    run_a: str,
    run_b: str,
    prompt: str = "",
    dimension: str = "",
    magnitude_a: str = "",
    magnitude_b: str = "",
    scenario: str = "",
    explanation_a: str = "",
    explanation_b: str = "",
) -> dict:
    """Append a new unanswered pair; returns the stored pair object.

    `dimension` is the single objective knob the pair varies (e.g. `w_air_time`);
    `magnitude_a` / `magnitude_b` are the two values of that knob; `scenario`
    is a short label for the fixed config (shape/seed/iters). All optional and
    backward compatible — old callers and old pairs omit them.
    """
    with _PAIRS_LOCK, _pair_store_lock(root):
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
            "display_order": ["a", "b"],
            "explanation_a": str(explanation_a or "").strip(),
            "explanation_b": str(explanation_b or "").strip(),
            "ts": time.time(),
            "first_view_ts": None,
            "display_snapshot": None,
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
    with _PAIRS_LOCK, _pair_store_lock(root):
        data = load_pairs(root)
        for p in data:
            if p.get("id") == pair_id:
                # A human observation is immutable.  In particular, a retried
                # browser POST must not overwrite it or trigger the agent twice.
                if p.get("answer") in ("a", "b", "tie"):
                    return p
                p["answer"] = answer
                p["answer_ts"] = time.time()
                p["note"] = str(note).strip() if note is not None else ""
                p["direct_feedback_status"] = "queued" if p["note"] else "needs_critique"
                save_pairs(root, data)
                return p
        return None


def _displayed_run_summary(root: Path, run_name: str) -> dict:
    """Capture exactly the factual run fields rendered by compare.html."""
    run_dir = _resolve_run_by_name(root, run_name)
    if run_dir is None:
        return {"run": run_name, "error": "run not found"}
    try:
        args = json.loads((run_dir / "args.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
    except (OSError, ValueError):
        return {"run": run_name, "error": "args/metrics unavailable"}
    total = metrics.get("total_time")
    air = metrics.get("air_time")
    air_pct = 100.0 * air / total if isinstance(air, (int, float)) and isinstance(total, (int, float)) and total > 0 else None
    return {
        "run": run_name,
        "shape": args.get("target_shape"),
        "iters": args.get("iters"),
        "hard_dice": metrics.get("hard_dice"),
        "air_percent": air_pct,
    }


def mark_pair_viewed(root: Path, pair_id: str) -> dict | None:
    """Record first-view time and the exact comparison framing once."""
    with _PAIRS_LOCK, _pair_store_lock(root):
        data = load_pairs(root)
        for p in data:
            if p.get("id") != pair_id:
                continue
            if p.get("first_view_ts") is None:
                p["first_view_ts"] = time.time()
                p.setdefault("display_order", ["a", "b"])
                p["display_snapshot"] = {
                    "prompt": p.get("prompt", ""),
                    "dimension": p.get("dimension", ""),
                    "magnitude_a": p.get("magnitude_a", ""),
                    "magnitude_b": p.get("magnitude_b", ""),
                    "explanation_a": p.get("explanation_a", ""),
                    "explanation_b": p.get("explanation_b", ""),
                    "a": _displayed_run_summary(root, p.get("run_a", "")),
                    "b": _displayed_run_summary(root, p.get("run_b", "")),
                }
                save_pairs(root, data)
            return p
    return None


def _set_direct_feedback_status(root: Path, pair_id: str, expected: str, status: str, **extra) -> bool:
    with _PAIRS_LOCK, _pair_store_lock(root):
        data = load_pairs(root)
        for p in data:
            if p.get("id") == pair_id and p.get("direct_feedback_status") == expected:
                p["direct_feedback_status"] = status
                p.update(extra)
                save_pairs(root, data)
                return True
    return False


def trigger_direct_feedback(root: Path, pair_id: str) -> bool:
    """Claim one answer and run the bounded local-Qwen agent asynchronously."""
    if not _set_direct_feedback_status(root, pair_id, "queued", "agent_running"):
        return False

    def worker():
        script = root / "scripts" / "direct_feedback_agent.py"
        cmd = [sys.executable, str(script), "--pair-id", pair_id]
        try:
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=660)
        except subprocess.TimeoutExpired:
            _set_direct_feedback_status(
                root, pair_id, "agent_running", "agent_failed",
                direct_feedback_error="local objective agent timed out after 660 seconds",
            )
            return
        if proc.returncode == 0:
            try:
                result = json.loads(proc.stdout)
            except ValueError:
                result = {}
            iteration_id = result.get("iteration_id")
            if not iteration_id:
                _set_direct_feedback_status(
                    root, pair_id, "agent_running", "agent_failed",
                    direct_feedback_error="agent returned no iteration id",
                )
                return
            planner = root / "scripts" / "run_direct_feedback_variants.py"
            planned = subprocess.run(
                [sys.executable, str(planner), "--iteration-id", iteration_id],
                cwd=root, capture_output=True, text=True, timeout=60,
            )
            if planned.returncode == 0:
                _set_direct_feedback_status(
                    root, pair_id, "agent_running", "launch_planned",
                    direct_feedback_iteration=iteration_id,
                )
            else:
                detail = (planned.stderr or planned.stdout or "planning failed").strip().splitlines()[-1]
                _set_direct_feedback_status(
                    root, pair_id, "agent_running", "planning_failed",
                    direct_feedback_iteration=iteration_id,
                    direct_feedback_error=detail[:500],
                )
        else:
            detail = (proc.stderr or proc.stdout or "agent failed").strip().splitlines()[-1]
            _set_direct_feedback_status(
                root, pair_id, "agent_running", "agent_failed",
                direct_feedback_error=detail[:500],
            )

    threading.Thread(target=worker, daemon=True, name=f"direct-feedback-{pair_id}").start()
    return True


def trigger_direct_feedback_for_answer(
    root: Path, pair_id: str, *, smoke_mode: bool
) -> tuple[bool, dict | None]:
    """Trigger normal processing or explicitly suppress it for isolated smoke."""
    if smoke_mode:
        _set_direct_feedback_status(root, pair_id, "queued", "smoke_disabled")
        pair = next((p for p in load_pairs(root) if p.get("id") == pair_id), None)
        return False, pair
    triggered = trigger_direct_feedback(root, pair_id)
    pair = next((p for p in load_pairs(root) if p.get("id") == pair_id), None)
    return triggered, pair


# ---------------------------------------------------------------------------
# Text-only human-belief change collection.  This remains separate from the
# direct objective-editing agent above: it observes adjacent critique text,
# launches the frozen classifier, and stores an independent training label.
# ---------------------------------------------------------------------------
_HUMAN_STATE_WORKER_LOCK = threading.Lock()
_HUMAN_STATE_ACTIVE_WORKERS: set[str] = set()


def _human_state_module():
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import human_state_lib
    return human_state_lib


def trigger_human_state_classifier(
    root: Path,
    observation_id: str,
    *,
    task_dir: Path | None = None,
) -> bool:
    """Launch at most one classifier worker per observation in this server."""
    with _HUMAN_STATE_WORKER_LOCK:
        if observation_id in _HUMAN_STATE_ACTIVE_WORKERS:
            return False
        _HUMAN_STATE_ACTIVE_WORKERS.add(observation_id)

    task_dir = task_dir or root / "autoresearch" / "tasks" / "train_csg"

    def record_unexpected_failure(error_type: str, detail: str) -> None:
        module = _human_state_module()
        store = module.HumanStateStore(task_dir)
        audit = store.get_observation(observation_id)
        if audit["prediction"] is None and audit["prediction_failure"] is None:
            store.record_prediction_failure(
                observation_id,
                error_type=error_type,
                detail=detail[:1000],
            )

    def worker():
        script = root / "scripts" / "human_state_classifier.py"
        command = [
            sys.executable,
            str(script),
            "--task-dir",
            str(task_dir),
            "--observation-id",
            observation_id,
        ]
        try:
            proc = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=660,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "classifier failed").strip()
                record_unexpected_failure("worker_failed", detail)
        except subprocess.TimeoutExpired:
            record_unexpected_failure(
                "worker_timeout", "classifier worker timed out after 660 seconds"
            )
        except Exception as exc:
            record_unexpected_failure("worker_failed", str(exc))
        finally:
            with _HUMAN_STATE_WORKER_LOCK:
                _HUMAN_STATE_ACTIVE_WORKERS.discard(observation_id)

    threading.Thread(
        target=worker,
        daemon=True,
        name=f"human-state-{observation_id}",
    ).start()
    return True


class HumanStateController:
    """Server-side Phase 3 orchestration with prediction-hidden status views."""

    MODES = frozenset({"off", "training", "deployment"})
    QUESTION = (
        "Compared with your previous response, did what you considered important "
        "in judging these machining results change?"
    )
    CHANGE_QUESTION = "What changed in what you considered important?"

    def __init__(
        self,
        *,
        root: Path,
        mode: str,
        participant_id: str | None,
        git_revision: str,
        session_id: str | None = None,
        store=None,
        classifier_trigger=None,
        smoke_mode: bool = False,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"invalid human-state mode: {mode!r}")
        module = _human_state_module()
        if mode != "off":
            if not isinstance(participant_id, str) or not module.IDENTIFIER_RE.fullmatch(
                participant_id
            ):
                raise ValueError(
                    "non-off human-state mode requires a pseudonymous participant ID"
                )
        self.root = root
        self.mode = mode
        self.participant_id = participant_id if mode != "off" else None
        self.git_revision = git_revision
        self.smoke_mode = bool(smoke_mode)
        self.store = store or module.HumanStateStore(
            root / "autoresearch" / "tasks" / "train_csg"
        )
        self._classifier_trigger = classifier_trigger or (
            lambda observation_id: trigger_human_state_classifier(
                root, observation_id, task_dir=self.store.task_dir
            )
        )
        self._session_lock = threading.Lock()
        self._session_id = session_id or uuid.uuid4().hex

    @property
    def session_id(self) -> str:
        with self._session_lock:
            return self._session_id

    def config(self) -> dict:
        return {
            "mode": self.mode,
            "smoke_mode": self.smoke_mode,
            "participant_id": self.participant_id,
            "session_id": self.session_id if self.mode != "off" else None,
            "question": self.QUESTION if self.mode == "training" else None,
            "change_question": self.CHANGE_QUESTION if self.mode == "training" else None,
            "labels": (
                ["change", "no_change", "not_sure", "skip"]
                if self.mode == "training"
                else []
            ),
        }

    def new_session(self) -> dict:
        if self.mode == "off":
            raise ValueError("human-state collection is off")
        with self._session_lock:
            self._session_id = uuid.uuid4().hex
        return self.config()

    def record_pair(self, pair: dict) -> dict:
        if self.mode == "off":
            return {"mode": "off", "status": "off"}
        result = self.store.record_critique(
            participant_id=self.participant_id,
            session_id=self.session_id,
            step_id=str(pair.get("id", "")),
            text=pair.get("note", ""),
            git_revision=self.git_revision,
        )
        transition = result["transition_event"]
        if transition is None:
            return {
                "mode": self.mode,
                "status": "insufficient_history",
                "observation_id": None,
                "classifier_triggered": False,
            }
        observation_id = transition["observation_id"]
        triggered = self._classifier_trigger(observation_id)
        status = self.observation_status(observation_id)
        status["classifier_triggered"] = triggered
        return status

    def observation_status(self, observation_id: str) -> dict:
        audit = self.store.get_observation(observation_id)
        if audit["prediction"] is not None:
            prediction_status = "completed"
        elif audit["prediction_failure"] is not None:
            prediction_status = "failed"
        else:
            prediction_status = "pending"
        label_status = "saved" if audit["human_label"] is not None else "pending"
        # Deliberately omit prediction content.  The training UI sees status,
        # never Qwen's decision or delta text before providing its own label.
        return {
            "mode": self.mode,
            "status": "transition_created",
            "observation_id": observation_id,
            "prediction_status": prediction_status,
            "human_label_status": label_status,
        }

    def record_label(
        self,
        observation_id: str,
        *,
        label: str,
        change_text: str | None,
    ) -> dict:
        if self.mode != "training":
            raise ValueError("human-state labels are accepted only in training mode")
        self.store.record_human_label(
            observation_id,
            label=label,
            change_text=change_text,
        )
        return self.observation_status(observation_id)


def _human_state_endpoint(path: str, suffix: str) -> bool:
    endpoint = f"/__api/human-state/{suffix}"
    return path == endpoint or path.endswith(endpoint)


def human_state_get_response(
    controller: HumanStateController,
    path: str,
    query: dict[str, list[str]],
) -> tuple[dict, int] | None:
    """Map a human-state GET to its exact JSON response, or decline it."""
    if _human_state_endpoint(path, "config"):
        return controller.config(), 200
    if not _human_state_endpoint(path, "status"):
        return None
    observation_id = (query.get("id", [""])[0] or "").strip()
    if not observation_id:
        return {"ok": False, "error": "missing observation id"}, 400
    try:
        return {"ok": True, **controller.observation_status(observation_id)}, 200
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 404


def human_state_post_response(
    controller: HumanStateController,
    path: str,
    body: dict,
) -> tuple[dict, int] | None:
    """Map a validated JSON-object POST to its exact JSON response."""
    if _human_state_endpoint(path, "label"):
        try:
            observation_id = str(body.get("observation_id", "")).strip()
            label = str(body.get("label", "")).strip()
            if not observation_id or not label:
                raise ValueError("observation_id and label are required")
            result = controller.record_label(
                observation_id,
                label=label,
                change_text=body.get("change_text"),
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}, 400
        except Exception as exc:
            return {"ok": False, "error": str(exc)}, 409
        return {"ok": True, **result}, 200
    if _human_state_endpoint(path, "session"):
        try:
            if body.get("action") != "new":
                raise ValueError("action must be 'new'")
            config = controller.new_session()
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}, 400
        return {"ok": True, **config}, 200
    return None


def update_pair_note(root: Path, pair_id: str, note: str) -> dict | None:
    """Update only the free-text note on an already-answered pair.

    Leaves the recorded answer (and answer_ts) untouched so the learned
    preference is not disturbed; lets a user refine the rationale later.
    """
    with _PAIRS_LOCK, _pair_store_lock(root):
        data = load_pairs(root)
        for p in data:
            if p.get("id") == pair_id:
                # New direct-feedback observations become immutable at answer
                # time. Historical rows without a direct-feedback status retain
                # the old note-edit behavior for backward compatibility.
                if p.get("answer") and p.get("direct_feedback_status"):
                    return None
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
    global _PAIRWISE_PATH_OVERRIDE

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

    smoke_mode = os.environ.get("DIFFCAM_HUMAN_STATE_SMOKE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    human_state_mode = os.environ.get("DIFFCAM_HUMAN_STATE_MODE", "off").strip().lower()
    participant_id = os.environ.get("DIFFCAM_HUMAN_STATE_PARTICIPANT_ID")
    participant_id = participant_id.strip() if participant_id else None
    smoke_workspace = None
    human_state_store = None
    if smoke_mode:
        human_state_mode = "training"
        participant_id = participant_id or "isolated-smoke"
        smoke_workspace = tempfile.TemporaryDirectory(
            prefix="diffcam-human-state-smoke-"
        )
        smoke_task_dir = Path(smoke_workspace.name) / "train_csg"
        smoke_pair_path = smoke_task_dir / "pairwise.json"
        try:
            seed_isolated_smoke_pairs(root, smoke_pair_path)
        except ValueError as exc:
            smoke_workspace.cleanup()
            sys.exit(f"error: {exc}")
        _PAIRWISE_PATH_OVERRIDE = smoke_pair_path
        module = _human_state_module()
        human_state_store = module.HumanStateStore(smoke_task_dir)
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        git_revision = "unknown"
    try:
        human_state = HumanStateController(
            root=root,
            mode=human_state_mode,
            participant_id=participant_id,
            git_revision=git_revision,
            store=human_state_store,
            smoke_mode=smoke_mode,
        )
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    cert, key = ensure_cert(web_dir / ".cert", args.host)

    os.chdir(root)

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from build_results_web import IncrementalResultsBuilder, build_data_payload

    if smoke_mode:
        # IncrementalResultsBuilder always refreshes web/data.json.  Use an
        # in-memory immutable payload so isolated smoke mode performs no writes
        # to real dashboard or run artifacts.
        smoke_payload = build_data_payload(generate_gcode=False, verbose=True)

        class ReadOnlySmokeBuilder:
            def get_payload(self, force=False):
                return smoke_payload

        builder = ReadOnlySmokeBuilder()
    else:
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

        def _json_body(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            value = json.loads(raw.decode() or "{}")
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            human_state_response = human_state_get_response(
                human_state, parsed.path, urllib.parse.parse_qs(parsed.query)
            )
            if human_state_response is not None:
                payload, status = human_state_response
                return self._json(payload, status)
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
            if _human_state_endpoint(parsed.path, "label") or _human_state_endpoint(
                parsed.path, "session"
            ):
                try:
                    body = self._json_body()
                except (ValueError, OSError):
                    return self._json({"ok": False, "error": "invalid JSON body"}, 400)
                payload, status = human_state_post_response(
                    human_state, parsed.path, body
                )
                return self._json(payload, status)
            # Save one run's star rating / feedback note. Body is JSON:
            # {"run": "runs/<batch>/<name>" | "<name>", "stars": 1-7|null,
            #  "feedback": "..."}. Returns the stored entry.
            if parsed.path == "/__api/feedback" or parsed.path.endswith("/__api/feedback"):
                try:
                    body = self._json_body()
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
                    body = self._json_body()
                except (ValueError, OSError):
                    return self._json({"ok": False, "error": "invalid JSON body"}, 400)
                if body.get("update_note") and body.get("id"):
                    pair = update_pair_note(
                        root, str(body["id"]).strip(), body.get("note", ""))
                    if pair is None:
                        return self._json({"ok": False, "error": "unknown pair id"}, 400)
                    return self._json({"ok": True, "pair": pair})
                if body.get("viewed") and body.get("id"):
                    pair = mark_pair_viewed(root, str(body["id"]).strip())
                    if pair is None:
                        return self._json({"ok": False, "error": "unknown pair id"}, 400)
                    return self._json({"ok": True, "pair": pair})
                if "answer" in body and body.get("id"):
                    pair = record_pair_answer(
                        root, str(body["id"]).strip(), str(body["answer"]).strip(),
                        body.get("note", ""))
                    if pair is None:
                        return self._json({"ok": False, "error": "invalid answer or unknown pair id"}, 400)
                    triggered, refreshed_pair = trigger_direct_feedback_for_answer(
                        root, pair["id"], smoke_mode=smoke_mode
                    )
                    pair = refreshed_pair or pair
                    try:
                        human_state_result = human_state.record_pair(pair)
                    except Exception as exc:
                        # The immutable pair answer is already safely stored.
                        # Surface collection failure without pretending the
                        # preference save failed or losing the critique.
                        human_state_result = {
                            "mode": human_state.mode,
                            "status": "error",
                            "error": str(exc),
                        }
                    return self._json({
                        "ok": True,
                        "pair": pair,
                        "agent_triggered": triggered,
                        "smoke_mode": smoke_mode,
                        "human_state": human_state_result,
                    })
                if body.get("run_a") and body.get("run_b"):
                    pair = add_pair(
                        root, str(body["run_a"]), str(body["run_b"]), body.get("prompt", ""),
                        dimension=body.get("dimension", ""),
                        magnitude_a=body.get("magnitude_a", ""),
                        magnitude_b=body.get("magnitude_b", ""),
                        scenario=body.get("scenario", ""),
                        explanation_a=body.get("explanation_a", ""),
                        explanation_b=body.get("explanation_b", ""),
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
    # Do not perform the TLS handshake in ThreadingTCPServer.get_request().
    # Browsers commonly open speculative/preconnect sockets and leave them
    # idle; a handshake on the main accept thread lets one such socket stall
    # every subsequent dashboard asset request.  With deferred handshakes,
    # the first read in each request-handler thread performs its own handshake.
    httpd.socket = ctx.wrap_socket(
        httpd.socket, server_side=True, do_handshake_on_connect=False
    )

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
    print(
        "human-state collection: "
        f"{human_state.mode}"
        + (
            f" (participant={human_state.participant_id}, session={human_state.session_id})"
            if human_state.mode != "off"
            else ""
        )
    )
    if smoke_mode:
        print(f"ISOLATED SMOKE MODE: disposable data only in {smoke_workspace.name}")
        print("ISOLATED SMOKE MODE: direct-feedback agent and GPU launch are disabled")
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
        if smoke_workspace is not None:
            _PAIRWISE_PATH_OVERRIDE = None
            smoke_workspace.cleanup()
            print("isolated smoke data deleted")


if __name__ == "__main__":
    main()
