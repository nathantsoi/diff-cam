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


def _resolve_run_arg(root: Path, run: str) -> Path | None:
    """Resolve a `run` query value to a runs/<name> dir under root.

    Accepts either an explicit ``runs/<name>`` path or the sentinel ``latest``,
    which resolves to the newest viewable run dir (so the dashboard can offer a
    one-click "view the last train_csg run" without knowing its name).
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
    return _safe_run_path(root, run)


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

    cmd = [sys.executable, str(script), "--run", run_rel]
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
            body = json.dumps(obj).encode()
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
            return super().do_GET()

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
