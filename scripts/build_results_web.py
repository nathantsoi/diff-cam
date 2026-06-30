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


def build_run_index():
    """Index runs by (shape, iters, seed) -> list of run records."""
    index = {}
    if not os.path.isdir(RUNS):
        return index, 0
    for name in os.listdir(RUNS):
        full = os.path.join(RUNS, name)
        if not os.path.isdir(full) or not name.startswith("CamEnv"):
            continue
        rec = load_run(full)
        if rec is None:
            continue
        for key in (
            (rec["shape"], rec["iters"], rec["seed"]),
            (rec["shape"], rec["iters"], None),  # seedless fallback
        ):
            index.setdefault(key, []).append(rec)
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
