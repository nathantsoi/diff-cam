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
  - for ``--method sweep`` runs, the (K,3) B-spline control polygon (``ctrl``)
    so the viewer can overlay the planned spline + its control points
  - relative paths to the STL meshes and G-code (for download links)

Run from the repo root:

    uv run python scripts/build_results_web.py
"""
import csv
import json
import os
import re
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # so `import cam` works when run as a script
RUNS = os.path.join(REPO, "runs")
WEB = os.path.join(REPO, "autoresearch", "tasks", "train_csg", "web")
# The canonical experiment log for this task (the same file plot_results.py
# reads) -- NOT the legacy repo-root results.tsv, which holds an unrelated
# 14-row baseline set. Reading the task log is what makes every autoresearch
# experiment, including staged multi-trajectory runs, appear in the dashboard.
RESULTS = os.path.join(REPO, "autoresearch", "tasks", "train_csg", "results.tsv")
OUT = os.path.join(WEB, "data.json")

DICE_TOL = 0.05  # results.tsv dice must be within this of a run's metrics dice


def grid_label(npz_path):
    """Per-part shape label for a grid/STEP target, e.g. 'grid:titan'.

    All grid runs share target_shape == 'grid'; labeling by the NPZ stem keeps
    different STEP parts from colliding in the run-matching index and the
    per-shape chart."""
    return "grid:" + os.path.splitext(os.path.basename(npz_path))[0]


def parse_cmd(cmd):
    """Pull shape / iters / seed out of a results.tsv command string.

    Accepts both flag styles: run_pipeline.py commands use hyphens
    (--target-shape) while direct algorithms.train_csg commands -- the only
    way to launch grid/STEP targets -- use underscores (--target_shape).
    """
    if not isinstance(cmd, str):
        cmd = ""
    shape = re.search(r"--target[-_]shape\s+(\S+)", cmd)
    iters = re.search(r"--iters\s+(\d+)", cmd)
    seed = re.search(r"--seed\s+(\d+)", cmd)
    shape_v = shape.group(1) if shape else None
    if shape_v == "grid":
        m = re.search(r"--target[-_]sdf[-_]path\s+(\S+)", cmd)
        if m:
            shape_v = grid_label(m.group(1))
    return {
        "shape": shape_v,
        "iters": int(iters.group(1)) if iters else None,
        "seed": int(seed.group(1)) if seed else None,
    }


# 1 inch in mm. Matches cam.units.IN_TO_MM / inch_to_mm without importing cam
# (which pulls scipy, not installed in the lightweight web-build env).
IN_TO_MM = 25.4


_NPZ_STOCK_CACHE = {}


def stock_mm_from_args(args):
    """Stock box (Lx, Ly, Lz) in mm for a run, grid/STEP-aware.

    Grid targets don't pass --stock-size-in; their stock box lives in the
    target NPZ (the part bbox + padding -- the same source the simulator uses,
    see simulator/csg_simulator.py). Analytic shapes use --stock-size-in
    (default 1" cube). Falls back to the 1" cube if the NPZ is unreadable.
    """
    a = args or {}
    if a.get("target_shape") == "grid" and a.get("target_sdf_path"):
        p = a["target_sdf_path"]
        full = p if os.path.isabs(p) else os.path.join(REPO, p)
        if full not in _NPZ_STOCK_CACHE:
            try:
                with np.load(full) as z:
                    _NPZ_STOCK_CACHE[full] = [float(c) for c in z["stock_size_mm"]]
            except (OSError, KeyError, ValueError) as e:
                print(f"  [stock] can't read target NPZ {p}: {e}", file=sys.stderr)
                _NPZ_STOCK_CACHE[full] = None
        mm = _NPZ_STOCK_CACHE[full]
        if mm is not None:
            return list(mm)
    sin = a.get("stock_size_in")
    sin = list(sin) if sin else [1.0, 1.0, 1.0]
    sin = (sin + [1.0, 1.0, 1.0])[:3]
    return [(c if c else 1.0) * IN_TO_MM for c in sin]


def tool_geom_from_args(args):
    """Tool + holder geometry normalized by the LONGEST stock axis, plus the
    stock-box aspect, matching the Taichi CSGSimulatorDelta
    (simulator/csg_simulator.py) that renders each run's run.mp4.

    The viewers draw in "aspect space": each axis spans L_axis/L_max, so
    physical shapes stay undistorted on non-cubic stock (grid/STEP targets)
    and a single normalized radius (radius_mm / L_max) is valid on every
    axis -- the tool stays circular. Run coordinates are still per-axis
    normalized [0,1]; the viewers scale them by ``aspect`` before drawing.
    For the default 1" cubic stock this reduces exactly to the old
    stock-normalized values. Defaults are the sim's own defaults:
    tool_radius_mm 3.175, tool_height_mm 25.0, 2.5"-diameter holder,
    10"-Z work volume.
    """
    a = args or {}
    lx, ly, lz = stock_mm_from_args(a)
    max_l = max(lx, ly, lz)
    win = a.get("workspace_in") or a.get("work_volume_in")
    win = list(win) if win else [16.0, 12.0, 10.0]
    wz = (win[2] if len(win) > 2 and win[2] else 10.0) * IN_TO_MM
    return {
        "toolRadius": float(a.get("tool_radius_mm", 3.175)) / max_l,
        "toolHeight": float(a.get("tool_height_mm", 25.0)) / max_l,
        "holderRadius": (IN_TO_MM * 2.5 / 2.0) / max_l,   # 2.5"-diameter spindle
        "holderHeight": wz / max_l,                       # machine Z travel
        "aspect": [lx / max_l, ly / max_l, lz / max_l],
        "stockMm": [lx, ly, lz],
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
    # Headline dice, across three metric conventions that coexist in runs/:
    #   delta-era + early-sweep runs  -> "dice" already holds the hard-carve value
    #   post-traj-quality sweep runs  -> "dice" is SOFT dice, which is identically
    #                                    ~0 for the sweep method (one hard union,
    #                                    so the soft surrogate never engages) and
    #                                    "hard_dice" carries the real number.
    # Preferring hard_dice when present gives one deployable-dice column over the
    # whole history; without it every phys-plausible run reads as 0.0 and fails
    # to match its results.tsv row.
    dice = metrics.get("hard_dice")
    if dice is None:
        dice = metrics.get("dice")
    if dice is None:
        return None
    shape = args.get("target_shape")
    if shape == "grid" and args.get("target_sdf_path"):
        shape = grid_label(args["target_sdf_path"])
    return {
        "run_dir": os.path.relpath(run_dir, REPO),
        "name": name,
        "shape": shape,
        "iters": int(args["iters"]) if args.get("iters") is not None else None,
        "seed": int(args["seed"]) if args.get("seed") is not None else None,
        "dice": float(dice),
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


def parse_seed_desc(desc):
    """Recover a seed from a results.tsv description like 'cyl s4 STAGED ...'."""
    m = re.search(r"\bs(\d+)\b", desc or "")
    return int(m.group(1)) if m else None


def _stage1_seed(run_rec):
    """Seed of a staged run's stage-1 trajectory, read from the stage-1 run's
    args.json via ``init_stock_from``. The results.tsv description's 'sN' refers
    to the STAGE-1 seed (the run that got truncated), not the stage-2 run's own
    seed, so match against this."""
    args = run_rec.get("args") or {}
    init_from = args.get("init_stock_from")
    if not init_from:
        return None
    try:
        run1_dir = os.path.dirname(init_from)
        with open(os.path.join(REPO, run1_dir, "args.json")) as f:
            a1 = json.load(f)
        return int(a1.get("seed"))
    except (OSError, ValueError, TypeError):
        return None


def find_concat_run(shape, seed):
    """Find the most recent staged run dir (has trajectory_concat.npy) matching
    shape (+stage-1 seed). Staged results.tsv rows log the deployable HARD-carve
    dice (~0.72), which can't match a run's soft dice via ``match_row``; this
    links them to the stage-2 run dir that owns the concatenated trajectory.
    ``seed`` is the STAGE-1 seed (what the description's 'sN' refers to)."""
    if shape is None:
        return None
    found = []

    def walk(base):
        if not os.path.isdir(base):
            return
        for name in sorted(os.listdir(base)):
            full = os.path.join(base, name)
            if os.path.isfile(full):
                continue
            rec = load_run(full)
            if rec is None:
                walk(full)
                continue
            if (rec["shape"] == shape
                    and os.path.exists(os.path.join(full, "trajectory_concat.npy"))):
                if seed is None or _stage1_seed(rec) == seed:
                    found.append(rec)
            else:
                walk(full)

    walk(RUNS)
    if not found:
        return None
    return max(found, key=lambda r: r["mtime"])


# --- G-code generation (mirrors scripts/export_gcode.py's MachineConfig build) ---
def make_machine_config(args):
    from cam import MachineConfig
    def pick(key, default):
        v = args.get(key)
        return tuple(v) if isinstance(v, list) else (v if v is not None else default)
    # Grid/STEP runs don't pass --stock-size-in; scale G-code by the NPZ's
    # stock box instead of silently defaulting to a 1" cube.
    stock_in = pick("stock_size_in", None)
    if stock_in is None:
        stock_in = tuple(c / IN_TO_MM for c in stock_mm_from_args(args))
    return MachineConfig(
        workspace_mm=100.0,
        workspace_in=pick("workspace_in", (16.0, 12.0, 10.0)),
        stock_size_in=stock_in,
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
        d = os.path.join(REPO, run_rec["run_dir"])
        # Staged runs: the deployable path is the concatenated multi-stage
        # trajectory, not the stage-2-only trajectory.npy.
        tp = os.path.join(d, "trajectory_concat.npy")
        if not os.path.exists(tp):
            tp = os.path.join(d, "trajectory.npy")
        positions = np.load(tp).astype(np.float64)
        cfg = make_machine_config(run_rec["args"])
        save_gcode(positions, gpath, cfg, post="haas")
        return rel
    except Exception as e:  # noqa: BLE001
        print(f"  [gcode] {run_rec['name']}: {e}", file=sys.stderr)
        return None


def _staged_boundary(run_rec):
    """For a staged run, return the stage-1 boundary index t* (the index of the
    last stage-1 position within the concatenated trajectory). Read from the
    ``init_stock_from`` npz (run1/trunc_state.npz -> t_trunc). Returns None if
    the run isn't staged or the truncation point can't be recovered."""
    args = run_rec.get("args") or {}
    init_from = args.get("init_stock_from")
    if not init_from:
        return None
    try:
        z = np.load(init_from)
        return int(z["t_trunc"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def sweep_control_points(run_rec, run_dir_abs, n_samples, cmd, decimals=4):
    """B-spline control polygon (K,3) for ``--method sweep`` runs, or None.

    Prefers the ``control_points.npy`` the trainer saves alongside the
    trajectory. Older sweep runs predate that file; for those, refit the
    control points by least squares against the planned (pre-clip) path --
    ``cmd`` IS the sampled spline ``B @ P``, so the refit recovers P to float
    precision. The refit needs ``simulator.sweep.bspline_basis``, which pulls
    in taichi; if that import fails in a lightweight web-build env, skip the
    overlay for legacy runs rather than failing the build.
    """
    args = run_rec.get("args") or {}
    if args.get("method") != "sweep":
        return None
    cp = os.path.join(run_dir_abs, "control_points.npy")
    if os.path.exists(cp):
        try:
            return np.round(np.load(cp).astype(np.float64), decimals).tolist()
        except (OSError, ValueError):
            return None
    n_ctrl = args.get("n_ctrl")
    if not n_ctrl or not cmd:
        return None
    try:
        from simulator.sweep import bspline_basis
    except Exception as e:  # noqa: BLE001 — taichi may be absent here
        print(f"  [ctrl] {run_rec['name']}: no basis for refit ({e})",
              file=sys.stderr)
        return None
    # Best-checkpoint sweep deltas carry a trailing zero step, so cmd can be
    # one point longer than the T the basis was built for -- fit the first T.
    plan = np.asarray(cmd, dtype=np.float64)[:n_samples]
    B = bspline_basis(int(n_ctrl), len(plan)).astype(np.float64)
    P, *_ = np.linalg.lstsq(B, plan, rcond=None)
    return np.round(P, decimals).tolist()


def trajectory_json(run_rec, decimals=4):
    """Load trajectory + commanded path, return compact JSON-ready arrays.

    For STAGED runs (a ``trajectory_concat.npy`` exists), the deployable
    artifact is the concatenated ``stage1[:t*+1] + stage2[1:]`` path -- NOT the
    stage-2-only ``trajectory.npy``, which starts mid-cut from the saved stock
    and is misleading on its own. Return the concat as ``traj``, expose the
    stage boundary (``stage_boundary`` = t*), and keep the stage-2-only path as
    ``stage2_traj`` for reference. No ``cmd`` (commanded/pre-clip) path exists
    for a concatenated multi-stage trajectory.
    """
    d = os.path.join(REPO, run_rec["run_dir"])
    concat_path = os.path.join(d, "trajectory_concat.npy")
    if os.path.exists(concat_path):
        try:
            pos = np.load(concat_path).astype(np.float64)
        except OSError:
            return None
        out = {
            "traj": np.round(pos, decimals).tolist(),
            "cmd": None,
            "staged": True,
            "stage_boundary": _staged_boundary(run_rec),
        }
        try:
            p2 = np.load(os.path.join(d, "trajectory.npy")).astype(np.float64)
            out["stage2_traj"] = np.round(p2, decimals).tolist()
        except OSError:
            out["stage2_traj"] = None
        return out
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
    out = {"traj": traj, "cmd": cmd}
    ctrl = sweep_control_points(run_rec, d, len(pos), cmd, decimals=decimals)
    if ctrl is not None:
        out["ctrl"] = ctrl
    return out


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


def build_data_payload(generate_gcode=True, verbose=True):
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
    if verbose:
        print(f"[index] {n_runs} indexed run dirs across {len(index)} keys")
        print(f"[results] {len(rows)} rows in results.tsv")

    # Pre-generate gcode for ALL indexed runs so any click-through works.
    if verbose:
        print("[gcode] checking/generating missing Haas G-code for indexed runs...")
    n_gen = 0
    seen = set()
    for cands in index.values():
        for rec in cands:
            if rec["run_dir"] in seen:
                continue
            seen.add(rec["run_dir"])
            if not os.path.exists(os.path.join(REPO, rec["run_dir"], "gcode_haas.nc")):
                if ensure_gcode(rec, generate=generate_gcode) is not None:
                    n_gen += 1
    if verbose and n_gen > 0:
        print(f"[gcode] generated {n_gen} new G-code files")

    # Build experiment records, matching each results row to a run dir.
    experiments = []
    n_matched = 0
    for i, r in enumerate(rows):
        cmd = r.get("command", "") or ""
        desc = r.get("description", "") or ""
        dice = float(r["dice"])
        p = parse_cmd(cmd)
        shape = p["shape"]
        if shape is None:  # recover from description (desc often abbreviates
            # "cylinder" as "cyl")
            for s, alias in (("sphere","sphere"), ("cylinder","cyl"),
                             ("box","box"), ("pyramid","pyramid")):
                if s in desc or s in cmd or alias in desc or alias in cmd:
                    shape = s
                    break
        run = match_row(dice, shape, p["iters"], p["seed"], index)
        # Staged rows log the hard-carve concat dice (~0.72), which can't match a
        # run's soft dice; link them to the stage-2 run dir with the concat path.
        if run is None and ("STAGED" in desc or "concat" in desc.lower()):
            seed = p["seed"] if p["seed"] is not None else parse_seed_desc(desc)
            run = find_concat_run(shape, seed)
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
            "gcode": ensure_gcode(run, generate=generate_gcode) if run else None,
            "trajectory": trajectory_json(run) if run else None,
            "tool_geom": tool_geom_from_args(run["args"]) if run else None,
        }
        if run:
            n_matched += 1
        experiments.append(rec)

    if verbose:
        print(f"[match] {n_matched}/{len(rows)} rows matched to a run dir")

    payload = {
        "experiments": experiments,
        "n_experiments": len(experiments),
        "n_matched": n_matched,
        "repo_root_note": "paths are relative to the repo root; serve from there",
    }
    return payload


class IncrementalResultsBuilder:
    """Watches results.tsv and runs/ for new files/experiments and builds data index on demand."""

    def __init__(self, generate_gcode=True, verbose=True):
        self.generate_gcode = generate_gcode
        self.verbose = verbose
        self.lock = threading.Lock()
        self.last_results_mtime = 0.0
        self.last_runs_mtime = 0.0
        self.last_run_dirs = set()
        self.payload = None

    def has_changed(self):
        """Check if results.tsv or runs/ contents changed since last build."""
        try:
            results_mtime = os.path.getmtime(RESULTS)
        except OSError:
            results_mtime = 0.0
        if results_mtime != self.last_results_mtime:
            return True

        try:
            runs_mtime = os.path.getmtime(RUNS)
        except OSError:
            runs_mtime = 0.0
        if runs_mtime != self.last_runs_mtime:
            return True

        try:
            current_run_dirs = {
                name for name in os.listdir(RUNS)
                if os.path.isdir(os.path.join(RUNS, name))
            }
        except OSError:
            current_run_dirs = set()
        if current_run_dirs != self.last_run_dirs:
            return True

        return self.payload is None

    def get_payload(self, force=False):
        """Get the data payload, rebuilding incrementally if files changed."""
        with self.lock:
            if force or self.has_changed():
                if self.verbose and not force and self.payload is not None:
                    print("[builder] new files or results detected, updating data index...")
                try:
                    payload = build_data_payload(generate_gcode=self.generate_gcode, verbose=self.verbose)
                    os.makedirs(os.path.dirname(OUT), exist_ok=True)
                    with open(OUT, "w") as f:
                        json.dump(payload, f)
                    if self.verbose:
                        print(f"[write] {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")
                    self.payload = payload
                    try:
                        self.last_results_mtime = os.path.getmtime(RESULTS)
                    except OSError:
                        pass
                    try:
                        self.last_runs_mtime = os.path.getmtime(RUNS)
                        self.last_run_dirs = {
                            name for name in os.listdir(RUNS)
                            if os.path.isdir(os.path.join(RUNS, name))
                        }
                    except OSError:
                        pass
                except Exception as e:
                    if self.verbose:
                        print(f"[builder] error rebuilding payload: {e}")
            if self.payload is None and os.path.exists(OUT):
                try:
                    with open(OUT) as f:
                        self.payload = json.load(f)
                except Exception:
                    pass
            return self.payload


def main():
    builder = IncrementalResultsBuilder(generate_gcode=True, verbose=True)
    builder.get_payload(force=True)


if __name__ == "__main__":
    main()
