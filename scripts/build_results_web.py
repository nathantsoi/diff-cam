"""Build the interactive results visualization data for the train_csg autoresearch.

Joins ``results.tsv`` (the experiment log) to the per-run artifacts under
``runs/<run>/`` (``args.json``, ``metrics.json``, ``trajectory.npy``,
``trajectory_deltas.npy``, ``meshes/*.stl``), batch-generates missing Haas
G-code in-process via ``cam.save_gcode``, and emits a single ``data.json`` the
D3 page consumes.

Output: ``autoresearch/tasks/train_csg/web/data.json``

Each experiment record carries:
  - the results.tsv metadata (commit, dice, status, description, command, shape)
  - the matched run dir (relative to repo root) or null
  - the (T,3) tool trajectory + commanded (pre-clip) path (for the 3D plot)
  - relative paths to the STL meshes and G-code (for download links)

Run from the repo root:

    uv run python scripts/build_results_web.py
"""
import csv
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # so `import cam` works when run as a script
RESULTS = os.path.join(REPO, "results.tsv")
RUNS = os.path.join(REPO, "runs")
WEB = os.path.join(REPO, "autoresearch", "tasks", "train_csg", "web")
OUT = os.path.join(WEB, "data.json")

DICE_TOL = 0.05  # results.tsv dice must be within this of a run's metrics dice


def parse_cmd(cmd):
    """Pull shape / iters / seed out of a results.tsv command string."""
    shape = re.search(r"--target-shape\s+(\S+)", cmd)
    iters = re.search(r"--iters\s+(\d+)", cmd)
    seed = re.search(r"--seed\s+(\d+)", cmd)
    return {
        "shape": shape.group(1) if shape else None,
        "iters": int(iters.group(1)) if iters else None,
        "seed": int(seed.group(1)) if seed else None,
    }


# 1 inch in mm. Matches cam.units.IN_TO_MM / inch_to_mm without importing cam
# (which pulls scipy, not installed in the lightweight web-build env).
IN_TO_MM = 25.4


def tool_geom_from_args(args):
    """Tool + holder geometry in stock-normalized [0,1]^3 units, matching the
    Taichi CSGSimulatorDelta (simulator/csg_simulator.py) that renders each run's
    run.mp4 via record_video.

    The sim measures SDFs in voxel space (radius_mm / voxel_mm) over a grid of
    Nx = stock_mm / voxel_mm cells, so a normalized radius = radius_mm / stock_mm
    (the voxel size cancels). Defaults are the sim's own defaults: tool_radius_mm
    3.175, tool_height_mm 25.0, 2.5"-diameter holder, 10"-Z work volume, 1" stock.
    """
    sin = args.get("stock_size_in") if args else None
    sin = list(sin) if sin else [1.0, 1.0, 1.0]
    lx = (sin[0] if len(sin) > 0 and sin[0] else 1.0) * IN_TO_MM
    lz = (sin[2] if len(sin) > 2 and sin[2] else (sin[0] if sin else 1.0)) * IN_TO_MM
    win = None
    if args:
        win = args.get("workspace_in") or args.get("work_volume_in")
    win = list(win) if win else [16.0, 12.0, 10.0]
    wz = (win[2] if len(win) > 2 and win[2] else 10.0) * IN_TO_MM
    a = args or {}
    return {
        "toolRadius": float(a.get("tool_radius_mm", 3.175)) / lx,
        "toolHeight": float(a.get("tool_height_mm", 25.0)) / lz,
        "holderRadius": (IN_TO_MM * 2.5 / 2.0) / lx,   # 2.5"-diameter spindle
        "holderHeight": wz / lz,                       # machine Z travel
    }


def load_run(run_dir):
    """Read one run dir's args/metrics/trajectory. Returns a record or None."""
    name = os.path.basename(run_dir)
    ap = os.path.join(run_dir, "args.json")
    mp = os.path.join(run_dir, "metrics.json")
    tp = os.path.join(run_dir, "trajectory.npy")
    if not (os.path.exists(ap) and os.path.exists(mp) and os.path.exists(tp)):
        return None
    try:
        with open(ap) as f:
            args = json.load(f)
        with open(mp) as f:
            metrics = json.load(f)
    except (OSError, ValueError):
        return None
    if "dice" not in metrics:
        return None
    return {
        "run_dir": os.path.relpath(run_dir, REPO),
        "name": name,
        "shape": args.get("target_shape"),
        "iters": int(args["iters"]) if args.get("iters") is not None else None,
        "seed": int(args["seed"]) if args.get("seed") is not None else None,
        "dice": float(metrics["dice"]),
        "metrics": metrics,
        "args": args,
        "mtime": os.path.getmtime(tp),
    }


def discover_batches():
    """Discover experiment batch directories under runs/.

    Returns a list of dicts ``{name, count}`` for every direct child directory
    of ``runs/`` that contains at least one viewable run. Files (like
    ``latest_metrics.json``) and empty dirs are skipped. New branches / batches
    added to ``runs/`` show up automatically — no code changes needed.
    """
    if not os.path.isdir(RUNS):
        return []
    out = []
    for name in sorted(os.listdir(RUNS)):
        full = os.path.join(RUNS, name)
        if not os.path.isdir(full):
            continue
        # Count viewable runs inside this container without building the full list.
        count = 0

        def _count(base):
            nonlocal count
            if not os.path.isdir(base):
                return
            for n in os.listdir(base):
                p = os.path.join(base, n)
                if os.path.isfile(p):
                    continue
                rec = load_run(p)
                if rec is None:
                    _count(p)
                else:
                    count += 1

        _count(full)
        if count > 0:
            out.append({"name": name, "count": count})
    return out


def list_runs(batch=None):
    """Flat list of all viewable run dirs under runs/, newest first.

    A run is viewable when it has args.json + metrics.json + trajectory.npy (the
    same gate ``load_run`` applies). Used by the dashboard's arbitrary-run picker
    and the ``?run=latest`` URL, so any train_csg run -- not just autoresearch
    experiments matched in results.tsv -- can be inspected in the browser.

    When ``batch`` matches a discovered subdirectory of ``runs/``, only runs
    under that container are returned. Pass ``None`` or ``"all"`` to recurse
    every child (the legacy behavior). New batch directories appear here
    automatically — no code changes needed.
    """
    out = []

    def walk(base):
        if not os.path.isdir(base):
            return
        for name in sorted(os.listdir(base)):
            full = os.path.join(base, name)
            if os.path.isfile(full):
                continue
            rec = load_run(full)
            if rec is None:
                # Not a run dir — recurse deeper (e.g. branch-*/ containers).
                walk(full)
                continue
            out.append({
                "run_dir": rec["run_dir"],
                "name": rec["name"],
                "shape": rec["shape"],
                "iters": rec["iters"],
                "seed": rec["seed"],
                "dice": rec["dice"],
                "mtime": rec["mtime"],
            })

    if batch and batch != "all":
        walk(os.path.join(RUNS, batch))
    else:
        walk(RUNS)

    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def run_record(run_dir, generate_gcode=True):
    """Full experiment-shaped record for one arbitrary run dir, or None.

    Mirrors the per-experiment record ``main()`` emits into data.json (same keys:
    run_dir, metrics, stl, gcode, trajectory, tool_geom, ...), so the dashboard's
    existing detail renderer can consume it directly. ``idx``/``commit``/``command``
    are nulled since an arbitrary run has no results.tsv row. Generates Haas G-code
    on demand (unless ``generate_gcode=False``) so the download link works.
    """
    rec = load_run(run_dir)
    if rec is None:
        return None

    repro_cmd = ""
    repro_path = os.path.join(run_dir, "reproduce_command.sh")
    if os.path.exists(repro_path):
        try:
            with open(repro_path) as f:
                repro_cmd = f.read().strip()
        except OSError:
            pass

    return {
        "idx": None,
        "commit": "",
        "dice": rec["dice"],
        "memory_gb": 0.0,
        "status": "arbitrary",
        "description": rec["name"],
        "command": repro_cmd,
        "shape": rec["shape"],
        "iters": rec["iters"],
        "seed": rec["seed"],
        "run_dir": rec["run_dir"],
        "name": rec["name"],
        "metrics": rec["metrics"],
        "args": rec["args"],
        "stl": stl_paths(rec),
        "gcode": ensure_gcode(rec, generate=generate_gcode),
        "trajectory": trajectory_json(rec),
        "tool_geom": tool_geom_from_args(rec["args"]),
        "mtime": rec["mtime"],
    }


def build_run_index():
    """Index runs by (shape, iters, seed) -> list of run records."""
    index = {}

    def walk(base):
        if not os.path.isdir(base):
            return
        for name in sorted(os.listdir(base)):
            full = os.path.join(base, name)
            if os.path.isfile(full):
                continue
            rec = load_run(full)
            if rec is None:
                # Not a run dir — recurse deeper (e.g. batch-*/ containers).
                walk(full)
                continue
            for key in (
                (rec["shape"], rec["iters"], rec["seed"]),
                (rec["shape"], rec["iters"], None),  # seedless fallback
            ):
                index.setdefault(key, []).append(rec)

    walk(RUNS)
    return index, len(index)


def match_row(row_dice, shape, iters, seed, index):
    """Find the run dir whose dice best matches the results.tsv row."""
    if shape is None or iters is None:
        return None
    cands = index.get((shape, iters, seed)) or index.get((shape, iters, None)) or []
    if not cands:
        return None
    best = min(cands, key=lambda r: abs(r["dice"] - row_dice))
    if abs(best["dice"] - row_dice) > DICE_TOL:
        return None
    return best


# --- G-code generation (mirrors scripts/export_gcode.py's MachineConfig build) ---
def make_machine_config(args):
    from cam import MachineConfig
    def pick(key, default):
        v = args.get(key)
        return tuple(v) if isinstance(v, list) else (v if v is not None else default)
    return MachineConfig(
        workspace_mm=100.0,
        workspace_in=pick("workspace_in", (16.0, 12.0, 10.0)),
        stock_size_in=pick("stock_size_in", (1.0, 1.0, 1.0)),
        stock_origin_in=args.get("stock_origin_in"),
        feed=600.0,
        plunge_feed=200.0,
        spindle_rpm=5000.0,
        tool_number=1,
        program_number=1,
        units="mm",
        coolant=True,
    )


def ensure_gcode(run_rec, generate=True):
    """Write runs/<run>/gcode_haas.nc if missing. Returns relative path or None."""
    gpath = os.path.join(REPO, run_rec["run_dir"], "gcode_haas.nc")
    rel = os.path.relpath(gpath, REPO)
    if os.path.exists(gpath):
        return rel
    if not generate:
        return None
    try:
        from cam import save_gcode
        tp = os.path.join(REPO, run_rec["run_dir"], "trajectory.npy")
        positions = np.load(tp).astype(np.float64)
        cfg = make_machine_config(run_rec["args"])
        save_gcode(positions, gpath, cfg, post="haas")
        return rel
    except Exception as e:  # noqa: BLE001
        print(f"  [gcode] {run_rec['name']}: {e}", file=sys.stderr)
        return None


def trajectory_json(run_rec, decimals=4):
    """Load trajectory + commanded path, return compact JSON-ready arrays."""
    d = os.path.join(REPO, run_rec["run_dir"])
    try:
        pos = np.load(os.path.join(d, "trajectory.npy")).astype(np.float64)
    except OSError:
        return None
    traj = np.round(pos, decimals).tolist()
    cmd = None
    dp = os.path.join(d, "trajectory_deltas.npy")
    if os.path.exists(dp):
        try:
            deltas = np.load(dp).astype(np.float64)
            start = pos[0]
            commanded = start + np.cumsum(deltas, axis=0)
            commanded = np.vstack([start[None, :], commanded])
            cmd = np.round(commanded, decimals).tolist()
        except OSError:
            pass
    return {"traj": traj, "cmd": cmd}


def stl_paths(run_rec):
    d = os.path.join(REPO, run_rec["run_dir"], "meshes")
    out = {}
    if not os.path.isdir(d):
        return out
    for label, pat in (
        ("stock_initial", "stock_initial_*.stl"),
        ("stock_carved", "stock_carved_*.stl"),
        ("target", "target_*.stl"),
    ):
        import glob
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if hits:
            out[label] = os.path.relpath(hits[-1], REPO)
    return out


def main():
    if not os.path.exists(RESULTS):
        raise SystemExit(f"results.tsv not found at {RESULTS}")

    rows = []
    with open(RESULTS, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                dice = float(r["dice"])
            except (ValueError, KeyError):
                continue
            rows.append(r)

    index, n_runs = build_run_index()
    print(f"[index] {n_runs} indexed run dirs across {len(index)} keys")
    print(f"[results] {len(rows)} rows in results.tsv")

    # Pre-generate gcode for ALL indexed runs so any click-through works.
    print("[gcode] generating missing Haas G-code for indexed runs...")
    n_gen = 0
    seen = set()
    for cands in index.values():
        for rec in cands:
            if rec["run_dir"] in seen:
                continue
            seen.add(rec["run_dir"])
            if not os.path.exists(os.path.join(REPO, rec["run_dir"], "gcode_haas.nc")):
                if ensure_gcode(rec) is not None:
                    n_gen += 1
    print(f"[gcode] generated {n_gen} new G-code files")

    # Build experiment records, matching each results row to a run dir.
    experiments = []
    n_matched = 0
    for i, r in enumerate(rows):
        cmd = r.get("command", "")
        desc = r.get("description", "")
        dice = float(r["dice"])
        p = parse_cmd(cmd)
        shape = p["shape"]
        if shape is None:  # recover from description
            for s in ("sphere", "cylinder", "box", "pyramid"):
                if s in desc or s in cmd:
                    shape = s
                    break
        run = match_row(dice, shape, p["iters"], p["seed"], index)
        rec = {
            "idx": i,
            "commit": r.get("commit", ""),
            "dice": dice,
            "memory_gb": float(r["memory_gb"]) if r.get("memory_gb") else 0.0,
            "status": r.get("status", "discard"),
            "description": desc,
            "command": cmd,
            "shape": shape,
            "iters": p["iters"],
            "seed": p["seed"],
            "run_dir": run["run_dir"] if run else None,
            "metrics": run["metrics"] if run else None,
            "stl": stl_paths(run) if run else {},
            "gcode": ensure_gcode(run) if run else None,
            "trajectory": trajectory_json(run) if run else None,
            "tool_geom": tool_geom_from_args(run["args"]) if run else None,
        }
        if run:
            n_matched += 1
        experiments.append(rec)

    print(f"[match] {n_matched}/{len(rows)} rows matched to a run dir")

    os.makedirs(WEB, exist_ok=True)
    payload = {
        "experiments": experiments,
        "n_experiments": len(experiments),
        "n_matched": n_matched,
        "repo_root_note": "paths are relative to the repo root; serve from there",
    }
    with open(OUT, "w") as f:
        json.dump(payload, f)
    print(f"[write] {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
