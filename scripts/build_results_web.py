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


def clean_for_json(obj):
    """Recursively replace non-finite floats (inf / -inf / NaN) with None.

    Python's ``json`` renders these as ``Infinity`` / ``-Infinity`` / ``NaN`` by
    default, which are NOT valid JSON — the browser's ``JSON.parse`` (used by
    ``d3.json`` and ``response.json()``) rejects the WHOLE response with
    "Unexpected token 'I'". That silently empties the dashboard: every fetch's
    ``.json()`` rejects into a local catch, leaving ``DATA.experiments`` and
    ``CURRENT_BATCH_RUNS`` empty. ``metrics.asd`` / ``hd95`` are ``+inf`` for any
    run whose carve never overlaps the target (no surface distance), so this
    fires routinely. ``null`` is the standards-compliant rendering the page
    already displays as "—".
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    # numpy scalars subclass float, so this covers them too.
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
    return obj


def parse_cmd(cmd):
    """Pull shape / iters / seed out of a results.tsv command string."""
    if not isinstance(cmd, str):
        cmd = ""
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
    return {
        "run_dir": os.path.relpath(run_dir, REPO),
        "name": name,
        "shape": args.get("target_shape"),
        "iters": int(args["iters"]) if args.get("iters") is not None else None,
        "seed": int(args["seed"]) if args.get("seed") is not None else None,
        "dice": float(dice),
        "hard_dice": float(metrics["hard_dice"]) if metrics.get("hard_dice") is not None else None,
        "metrics": clean_for_json(metrics),
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
                # Full metrics dict so the dashboard's per-batch table can show
                # every evaluation measure (hard_dice, air_time, break_prob, …)
                # for every run without a per-row /__api/run fetch.
                "metrics": rec["metrics"],
                "mtime": rec["mtime"],
            })

    if batch and batch != "all":
        walk(os.path.join(RUNS, batch))
    else:
        walk(RUNS)

    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def find_results_row(run_rec):
    """Reverse-lookup: which results.tsv row (if any) matches this run dir?

    ``match_row`` goes results.tsv-row -> run dir; this inverts it so a clicked
    run's detail view can show its real logged status/command/description
    instead of the generic "arbitrary" label. A row matches when it shares the
    run's (shape, iters, seed) AND its (hard) dice is within DICE_TOL of the
    run's hard_dice/dice -- the same test ``match_row`` applies. If several rows
    qualify, the closest-by-dice wins.
    """
    shape, iters, seed = run_rec["shape"], run_rec["iters"], run_rec["seed"]
    if shape is None or iters is None:
        return None
    if not os.path.exists(RESULTS):
        return None
    hd = run_rec.get("hard_dice")
    rd = run_rec.get("dice", 0.0)
    best, best_d = None, DICE_TOL
    with open(RESULTS, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                row_dice = float(r["dice"])
            except (ValueError, KeyError):
                continue
            p = parse_cmd(r.get("command", "") or "")
            if p["shape"] != shape or p["iters"] != iters:
                # description may abbreviate shape ("cyl") and omit iters; only
                # accept the mismatch if the row's command lacks the field.
                continue
            if p["seed"] is not None and seed is not None and p["seed"] != seed:
                continue
            d = abs(row_dice - rd)
            if hd is not None:
                d = min(d, abs(row_dice - hd))
            if d <= best_d:
                best_d, best = d, r
    return best


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

    # If this run dir was logged in results.tsv, surface the real row (status,
    # commit, command, description) so the detail view does not mislabel a
    # promoted run as "arbitrary". Falls back to the generic arbitrary record
    # only when no results.tsv row matches (the common case for ad-hoc runs).
    row = find_results_row(rec)
    if row is not None:
        status = row.get("status", "OK") or "OK"
        commit = row.get("commit", "") or ""
        command = row.get("command", "") or ""
        description = row.get("description", "") or rec["name"]
        try:
            row_dice = float(row["dice"])
        except (ValueError, KeyError):
            row_dice = rec["dice"]
    else:
        status, commit, command, description, row_dice = (
            "arbitrary", "", repro_cmd, rec["name"], rec["dice"],
        )

    return {
        "idx": None,
        "commit": commit,
        "dice": row_dice,
        "memory_gb": 0.0,
        "status": status,
        "description": description,
        "command": command,
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
    """Find the run dir whose dice best matches the results.tsv row.

    results.tsv's dice column is the HARD dice (the deployable metric). A run dir's
    metrics.json ``dice`` field is the SOFT dice for normal runs but 0.0 for
    best_on_hard runs (the best-checkpoint re-carve path doesn't populate soft_dice,
    so the summary falls back to 0.0). So match against whichever of the run's
    ``hard_dice`` / ``dice`` is closest to the row's (hard) dice.
    """
    if shape is None or iters is None:
        return None
    cands = index.get((shape, iters, seed)) or index.get((shape, iters, None)) or []
    if not cands:
        return None
    def dist(r):
        d = abs(r["dice"] - row_dice)
        hd = r.get("hard_dice")
        if hd is not None:
            d = min(d, abs(hd - row_dice))
        return d
    best = min(cands, key=dist)
    if dist(best) > DICE_TOL:
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
    matched_dirs = set()
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
            matched_dirs.add(run["run_dir"])
        experiments.append(rec)

    # Append every viewable run dir that results.tsv did NOT match, as a
    # lightweight "arbitrary" experiment. The autoresearch launch scripts
    # (launch_wave*.sh) run train_csg directly and never append to results.tsv,
    # so without this the dashboard's default data.json grid is frozen at the
    # last results.tsv era (e.g. 32 wave-4 rows) while hundreds of newer runs
    # sit on disk invisible until a batch is manually selected. These records
    # carry run_dir + dice + metrics (so the grid's hard-dice axis + tooltips
    # work) but NO embedded trajectory/stl/gcode; the page fetches the full
    # record via /__api/run on click (index.html ~L974), exactly as it does for
    # batch-dropdown runs. Newest first (list_runs sorts by mtime desc).
    n_arbitrary = 0
    for r in list_runs():
        if r["run_dir"] in matched_dirs:
            continue
        experiments.append({
            "idx": len(experiments),
            "commit": "",
            "dice": r["dice"],
            "memory_gb": 0.0,
            "status": "arbitrary",
            "description": r["name"],
            "command": "",
            "shape": r["shape"],
            "iters": r["iters"],
            "seed": r["seed"],
            "run_dir": r["run_dir"],
            "name": r["name"],
            "metrics": r["metrics"],
            "stl": {},
            "gcode": None,
            "trajectory": None,
            "tool_geom": None,
            "mtime": r["mtime"],
        })
        n_arbitrary += 1
    if verbose:
        print(f"[arbitrary] appended {n_arbitrary} unmatched run dirs as lightweight experiments")

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
                    # Atomic write: serialize to a temp file in the same dir, then
                    # os.replace onto data.json. A concurrent browser read of an
                    # in-place json.dump can catch a half-written file (parse fail
                    # -> the page's catch branch empties the dashboard and crashes
                    # updateStats on d3.max([])); rename is atomic on POSIX.
                    tmp = OUT + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(payload, f)
                    os.replace(tmp, OUT)
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
