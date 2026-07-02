"""One-step pipeline: train -> evaluate -> export G-code -> visualize.

Orchestrates the four diff-cam stages for the continuous (CSG / GradMill)
gradient-descent method into a single command, so a part can be taken from
"define geometry" to "machine-ready G-code + a diagnostic figure" without
manually chaining scripts:

  1. ``algorithms.train_csg``  -- optimize the toolpath; writes
     ``runs/<run>/trajectory.npy`` + ``args.json`` (and copies both to the
     repo root).
  2. ``eval.eval_csg``         -- score the carved stock vs the target (Dice /
     ASD / HD95) and, with ``--gcode``, the G-code round-trip fidelity + the
     executed-program carve.
  3. ``scripts/export_gcode``  -- export the machine G-code (Haas or RS274),
     auto-matching the run's ``args.json`` so the coordinates line up.
  4. ``scripts/visualize_trajectory`` -- render the 6-panel diagnostic figure
     (normalized + WCS frames, G-code round-trip, sim-vs-target and
     G-code-vs-sim carves, metrics).

Each stage runs as a subprocess (a fresh process per stage, so Taichi is
re-initialised cleanly) and the discovered run directory is threaded between
them. Geometry flags are forwarded consistently to every stage that needs them;
the exporter and visualizer additionally auto-read the run's ``args.json``.

Examples
--------
    # Fast end-to-end run (the proven default of 5000 iters is a ~15-min run;
    # pass --iters 50 for a quick wiring check, small part, headless, no W&B):
    uv run python scripts/run_pipeline.py --iters 50

    # A real 1" sphere at 0.5 mm voxels, Haas G-code + figure (proven operating
    # point from the autoresearch sweep: dt=0.45 unlocks tool traversal,
    # grad-clip + eval-freq + best-checkpoint saving capture the dice peak):
    uv run python scripts/run_pipeline.py --iters 5000 --max-steps 128 \
        --stock-size-in 1 1 1 --voxel-size-mm 0.5 --target-shape sphere \
        --target-radius-mm 11.43 --post haas --dt 0.45 --grad-clip 0.5 \
        --eval-freq 10

    # Fixture the stock top-centre at machine (8,6,5)" (emits G10 L2 in the Haas program):
    uv run python scripts/run_pipeline.py --stock-origin-in 8 6 5 --post haas
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Each stage is launched as `python -m ...` / `python scripts/...` from the repo
# root so that `cam_env` / `simulator` / `cam` resolve on the import path.
PYTHON = sys.executable


def _run(cmd, stage):
    """Run a stage, streaming its output, and return (returncode, stdout)."""
    print(f"\n=== [{stage}] {' '.join(shlex.quote(c) for c in cmd)} ===", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, env=env
    )
    lines = []
    if proc.stdout:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
    proc.wait()
    dt = time.time() - t0
    print(f"=== [{stage}] exit {proc.returncode} in {dt:.1f}s ===\n", flush=True)
    return proc.returncode, "".join(lines)


# Regex for train_csg's "[run] writing outputs to runs/<name>" line.
_RUN_RE = re.compile(r"writing outputs to\s+(runs/\S+)")


def _parse_run_dir(stdout):
    m = _RUN_RE.search(stdout)
    if not m:
        raise RuntimeError(
            "could not find the training run directory in train_csg output; "
            "set --run-dir explicitly or inspect runs/ for the newest run"
        )
    return m.group(1).rstrip("/")


def _as_list(v):
    """Coerce a tuple/list of numbers to a flat list of strings for CLI args."""
    return [str(x) for x in v]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # --- Stage control ---
    ap.add_argument("--stages", default="train,eval,export,viz",
                    help="comma-separated subset of stages to run "
                         "(train,eval,export,viz); useful to resume a pipeline")
    ap.add_argument("--run-dir", default=None,
                    help="existing run dir to use (skips discovering one from "
                         "training; required when 'train' is not in --stages)")
    # --- Training (forwarded to train_csg). Defaults are the proven operating
    #     point from the autoresearch sweep (514 experiments): dt=0.45 unlocks
    #     tool traversal (the real bottleneck), grad-clip + eval-freq +
    #     best-checkpoint saving capture the transient dice peak. ---
    ap.add_argument("--iters", type=int, default=5000,
                    help="Adam iterations (i5000 is the sweet spot within the "
                         "15-min budget; peaks appear later as iters grow; "
                         "i8000 gives no further gain and breaks the budget)")
    ap.add_argument("--max-steps", type=int, default=128,
                    help="trajectory length T (number of tool motions; m=128 at "
                         "dt<=0.45, m=160 at dt0.5; m>=192 NaNs)")
    ap.add_argument("--learning-rate", type=float, default=5e-3)
    ap.add_argument("--lr-decay-frac", type=float, default=0.0,
                    help="fraction of iters (at the end) over which LR decays to 0 "
                         "(dead lever on current API; best-checkpoint saving subsumes it)")
    ap.add_argument("--init-scale", type=float, default=0.05,
                    help="half-range of the uniform random init for per-step displacements")
    ap.add_argument("--init-mode", default="random", choices=("random", "raster", "raster_fine", "raster_fine_wide", "spiral", "shell", "zlayer"),
                    help="trajectory init mode")
    ap.add_argument("--w-gouge", type=float, default=4.0,
                    help="loss weight on cutting INTO the part (barrier)")
    ap.add_argument("--w-residual", type=float, default=1.0,
                    help="loss weight on leftover material outside the part (objective)")
    ap.add_argument("--grad-clip", type=float, default=0.5,
                    help="clip per-iter gradient L2 norm (0 = disabled); 0.4-0.5 "
                         "stabilizes the transient dice peak so best-checkpoint "
                         "saving captures a higher one")
    ap.add_argument("--w-air", type=float, default=0.0,
                    help="weight on the per-step air-cut penalty (0 = disabled; ~0.5-1.0)")
    ap.add_argument("--w-jerk", type=float, default=0.0,
                    help="weight on the jerk/smoothness penalty (0 = disabled; ~1e-2)")
    ap.add_argument("--w-step", type=float, default=0.0,
                    help="weight on the speed-regularity (constant-feed) penalty (0 = disabled)")
    ap.add_argument("--w-prox", type=float, default=0.0,
                    help="weight on the distance-weighted air-cut (contour-hug) penalty (0 = disabled); "
                         "charges air-cutting in proportion to squared distance from the target surface")
    ap.add_argument("--random-tool-start", action="store_true",
                    help="randomize the cutter start each fresh start (XY in the stock "
                         "footprint, Z >= stock top + --tool-start-clearance-in)")
    ap.add_argument("--tool-start-clearance-in", type=float, default=0.2,
                    help="min cutter height above the stock top (inches) for a random start")
    ap.add_argument("--tool-start-xy-margin", type=float, default=0.1,
                    help="normalized XY margin inside the stock footprint for a random start")
    ap.add_argument("--tool-start-z-jitter-in", type=float, default=0.1,
                    help="extra random height (inches) above the clearance floor for a random start")
    ap.add_argument("--restart-from-state", action="store_true",
                    help="save mid-cut simulator states and restart training from them "
                         "(robustness to initial conditions)")
    ap.add_argument("--p-restart", type=float, default=0.25,
                    help="per-iter probability of restarting from a saved state")
    ap.add_argument("--state-bank-size", type=int, default=32,
                    help="max saved simulator states (FIFO)")
    ap.add_argument("--save-state-prob", type=float, default=0.05,
                    help="per-iter probability of snapshotting a mid-cut state")
    ap.add_argument("--eval-freq", type=int, default=10,
                    help="eval (dice) cadence in iters; 0 = auto (iters//10). "
                         "Fine cadence (10) samples the transient dice peak for "
                         "best-checkpoint saving.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-save-model", action="store_true",
                    help="don't pass --save_model (trajectory won't be written to the run dir)")
    ap.add_argument("--track", action="store_true",
                    help="enable W&B tracking in training (default off for a clean one-step run)")
    # --- Geometry (forwarded consistently to every stage that needs it) ---
    ap.add_argument("--stock-size-in", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                    metavar=("X", "Y", "Z"), help="stock box in inches (the normalized cube)")
    ap.add_argument("--voxel-size-mm", type=float, default=0.5,
                    help="physical voxel edge (mm) -- the sub-mm precision knob")
    ap.add_argument("--workspace-in", type=float, nargs=3, default=(16.0, 12.0, 10.0),
                    metavar=("X", "Y", "Z"), help="machine work volume in inches")
    ap.add_argument("--stock-origin-in", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="G54 = stock top-centre in machine inches")
    ap.add_argument("--target-shape", default="sphere",
                    choices=("sphere", "cylinder", "box", "pyramid"))
    ap.add_argument("--target-radius-mm", type=float, default=11.43,
                    help="sphere/cylinder radius, or box/pyramid half-size (mm)")
    ap.add_argument("--target-height-mm", type=float, default=22.86,
                    help="cylinder/pyramid height (mm; ignored for sphere/box)")
    ap.add_argument("--dt", type=float, default=0.45,
                    help="seconds per simulator step. THE decisive lever: at low dt "
                         "(0.12/0.01) the tool is speed-limited and cannot traverse "
                         "the exterior (dice caps ~0.56); 0.45 advances ~1 voxel/step. "
                         "Sweet spot dt in [0.42, 0.5].")
    # --- G-code / viz ---
    ap.add_argument("--post", default="haas", choices=("rs274", "haas"),
                    help="post-processor for export/eval/viz")
    ap.add_argument("--units", default="mm", choices=("mm", "inch"),
                    help="G-code output units")
    ap.add_argument("--tool", type=int, default=1, help="tool number (Txx)")
    ap.add_argument("--rpm", type=float, default=5000.0, help="spindle speed")
    ap.add_argument("--feed", type=float, default=600.0, help="cutting feed, mm/min")
    ap.add_argument("--plunge-feed", type=float, default=200.0, help="Z plunge feed, mm/min")
    ap.add_argument("--no-carve", action="store_true",
                    help="skip the carved-stock panels in the visualizer (no Taichi)")
    ap.add_argument("--gcode-out", default=None,
                    help="explicit G-code output path (default runs/<run>/gcode_<post>.<ext>)")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ("train", "eval", "export", "viz")]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}; valid: train,eval,export,viz")

    run_dir = args.run_dir
    if run_dir is not None:
        run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(REPO, run_dir)
        if not os.path.isdir(run_dir):
            raise SystemExit(f"--run-dir not found: {run_dir}")

    # Common geometry args in each tool's own flag convention.
    ssi = _as_list(args.stock_size_in)
    wsi = _as_list(args.workspace_in)
    soi = _as_list(args.stock_origin_in) if args.stock_origin_in is not None else None

    artifacts = {"run_dir": run_dir}

    # ------------------------------------------------------------------ train
    if "train" in stages:
        cmd = [
            PYTHON, "-m", "algorithms.train_csg",
            "--iters", str(args.iters),
            "--max_steps", str(args.max_steps),
            "--learning_rate", str(args.learning_rate),
            "--lr_decay_frac", str(args.lr_decay_frac),
            "--init_scale", str(args.init_scale),
            "--init_mode", args.init_mode,
            "--w_gouge", str(args.w_gouge),
            "--w_residual", str(args.w_residual),
            "--grad_clip", str(args.grad_clip),
            "--w_air", str(args.w_air),
            "--w_jerk", str(args.w_jerk),
            "--w_step", str(args.w_step),
            "--w_prox", str(args.w_prox),
            "--eval_freq", str(args.eval_freq),
            "--seed", str(args.seed),
            "--stock_size_in", *ssi,
            "--voxel_size_mm", str(args.voxel_size_mm),
            "--workspace_in", *wsi,
            "--target_shape", args.target_shape,
            "--target_radius_mm", str(args.target_radius_mm),
            "--target_height_mm", str(args.target_height_mm),
            "--dt", str(args.dt),
            "--headless",
        ]
        if not args.no_save_model:
            cmd.append("--save_model")
        cmd.append("--track" if args.track else "--no-track")
        cmd.append("--eval")
        # Trajectory regularizers + robustness-to-initial-conditions options.
        if args.random_tool_start:
            cmd += ["--random_tool_start",
                    "--tool_start_clearance_in", str(args.tool_start_clearance_in),
                    "--tool_start_xy_margin", str(args.tool_start_xy_margin),
                    "--tool_start_z_jitter_in", str(args.tool_start_z_jitter_in)]
        if args.restart_from_state:
            cmd += ["--restart_from_state",
                    "--p_restart", str(args.p_restart),
                    "--state_bank_size", str(args.state_bank_size),
                    "--save_state_prob", str(args.save_state_prob)]
        rc, out = _run(cmd, "train")
        if rc != 0:
            raise SystemExit(f"training failed (exit {rc})")
        run_dir = os.path.join(REPO, _parse_run_dir(out))
        artifacts["run_dir"] = run_dir
        print(f"[pipeline] training run dir: {run_dir}")

    if run_dir is None:
        raise SystemExit("no run dir: run the 'train' stage or pass --run-dir")

    traj = os.path.join(run_dir, "trajectory.npy")
    if not os.path.exists(traj):
        raise SystemExit(
            f"trajectory not found at {traj}; train with --save-model (default) "
            f"or point --run-dir at a run that has trajectory.npy"
        )
    artifacts["trajectory"] = traj

    # ------------------------------------------------------------------- eval
    if "eval" in stages:
        cmd = [
            PYTHON, "-m", "eval.eval_csg",
            "--trajectory", traj,
            "--stock-size-in", *ssi,
            "--voxel-size-mm", str(args.voxel_size_mm),
            "--target-shape", args.target_shape,
            "--workspace-in", *wsi,
            "--gcode",
            "--post", args.post,
        ]
        if soi is not None:
            cmd += ["--stock-origin-in", *soi]
        rc, _ = _run(cmd, "eval")
        if rc != 0:
            print(f"[pipeline] WARNING: eval exited {rc}; continuing")

    # ----------------------------------------------------------------- export
    if "export" in stages:
        ext = ".nc" if args.post == "haas" else ".ngc"
        out = args.gcode_out or os.path.join(run_dir, f"gcode_{args.post}{ext}")
        cmd = [
            PYTHON, "scripts/export_gcode.py",
            "--post", args.post,
            "--trajectory", traj,
            "-o", out,
            "--tool", str(args.tool),
            "--rpm", str(args.rpm),
            "--feed", str(args.feed),
            "--plunge-feed", str(args.plunge_feed),
            "--units", args.units,
        ]
        # The exporter auto-matches runs/<run>/args.json; CLI geometry flags are
        # redundant there but harmless, and make the stage robust if args.json is
        # missing. Pass them so export works even on a hand-placed trajectory.
        cmd += ["--stock-size-in", *ssi, "--workspace-in", *wsi]
        if soi is not None:
            cmd += ["--stock-origin-in", *soi]
        rc, _ = _run(cmd, "export")
        if rc != 0:
            print(f"[pipeline] WARNING: export exited {rc}; continuing")
        else:
            artifacts["gcode"] = out

    # --------------------------------------------------------------------- viz
    if "viz" in stages:
        out = os.path.join(run_dir, f"trajectory_viz_{args.post}.png")
        cmd = [
            PYTHON, "scripts/visualize_trajectory.py",
            "--run", run_dir,
            "--post", args.post,
            "--save", out,
        ]
        if args.no_carve:
            cmd.append("--no-carve")
        rc, _ = _run(cmd, "viz")
        if rc != 0:
            print(f"[pipeline] WARNING: viz exited {rc}; continuing")
        else:
            artifacts["viz"] = out

    # --------------------------------------------------------------- summary
    print("\n================ pipeline summary ================")
    print(f"run dir     : {artifacts.get('run_dir')}")
    print(f"trajectory  : {artifacts.get('trajectory')}")
    print(f"g-code      : {artifacts.get('gcode', '(skipped/failed)')}")
    print(f"figure      : {artifacts.get('viz', '(skipped/failed)')}")
    print(f"metrics.json: {os.path.join(run_dir, 'metrics.json') if os.path.exists(os.path.join(run_dir, 'metrics.json')) else '(none)'}")
    print("==================================================")
    print("\nNext: load the .nc on the machine, or re-run a single stage, e.g.")
    print("  uv run python scripts/run_pipeline.py --stages viz --run-dir "
          f"{os.path.relpath(run_dir, REPO)}")


if __name__ == "__main__":
    main()
