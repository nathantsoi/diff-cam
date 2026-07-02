"""Staged training: train -> truncate -> finish.

Implements the two-stage scheme for parts a single trajectory cannot complete:

  1. ``algorithms.train_csg``  -- optimize a first trajectory (stage 1). The
     tool often carves the part then wanders off for the trailing steps
     ("cutting air"); the best dice is captured by best-checkpoint saving.
  2. ``algorithms.truncate_trajectory`` -- hard-carve stage 1, find the point
     ``t*`` where the tool STOPS cutting (the trailing excursion), and save the
     mid-cut stock SDF + tool position there.
  3. ``algorithms.train_csg --init-stock-from <state.npz>`` -- train a SECOND,
     fresh trajectory that starts from the saved mid-cut stock and carves the
     REMAINING material (the residual stage 1 left by wandering off). This is
     "train another trajectory to finish cutting the part".
  4. Concatenate ``stage1[:t*+1] + stage2[1:]`` and score it with
     ``eval.eval_csg`` for the deployable final dice.

Each training stage runs as a subprocess (fresh Taichi per stage). Geometry
flags are forwarded consistently. The discovered run dirs and the truncation
point are threaded between stages.

Examples
--------
    # Two-stage finish of a 1" sphere (stage 1 + stage 2, 5000 iters each):
    uv run python scripts/staged_train.py --iters 5000 --max-steps 128 \
        --target-shape sphere --target-radius-mm 11.43 --dt 0.45 --grad-clip 0.5
"""

import argparse
import os
import re
import shlex
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

_RUN_RE = re.compile(r"writing outputs to\s+(runs/\S+)")


def _run(cmd, stage):
    print(f"\n=== [{stage}] {' '.join(shlex.quote(c) for c in cmd)} ===", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, env=env,
    )
    lines = []
    if proc.stdout:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
    proc.wait()
    print(f"=== [{stage}] exit {proc.returncode} ===\n", flush=True)
    return proc.returncode, "".join(lines)


def _parse_run_dir(stdout):
    m = _RUN_RE.search(stdout)
    if not m:
        raise RuntimeError("could not find the training run directory in output")
    return m.group(1).rstrip("/")


def _as_list(v):
    return [str(x) for x in v]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # --- Training hyperparams (forwarded to BOTH stages) ---
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--init-scale", type=float, default=0.05)
    ap.add_argument("--w-gouge", type=float, default=4.0)
    ap.add_argument("--w-residual", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--w-len", type=float, default=0.03,
                    help="path-length penalty (forwarded to both stages; keeps "
                         "each trajectory from developing its own trailing drift)")
    ap.add_argument("--eval-freq", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-mode", default="raster_fine",
                    help="stage-1 init mode (stage 2 always uses 'random' so it "
                         "starts from the saved tool position, not a full raster)")
    # --- Geometry (forwarded consistently) ---
    ap.add_argument("--stock-size-in", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--voxel-size-mm", type=float, default=0.5)
    ap.add_argument("--workspace-in", type=float, nargs=3, default=(16.0, 12.0, 10.0),
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--target-shape", default="sphere",
                    choices=("sphere", "cylinder", "box", "pyramid"))
    ap.add_argument("--target-radius-mm", type=float, default=11.43)
    ap.add_argument("--target-height-mm", type=float, default=22.86)
    ap.add_argument("--dt", type=float, default=0.45)
    ap.add_argument("--post", default="haas", choices=("rs274", "haas"))
    # --- Staged control ---
    ap.add_argument("--stage1-run-dir", default=None,
                    help="existing stage-1 run dir (skips stage-1 training; "
                         "use to resume a staged run from a saved trajectory)")
    ap.add_argument("--remove-thresh", type=float, default=1e-6,
                    help="truncation removed-volume noise floor")
    ap.add_argument("--min-keep-frac", type=float, default=0.3,
                    help="min fraction of stage-1 trajectory to keep")
    args = ap.parse_args()

    ssi = _as_list(args.stock_size_in)
    wsi = _as_list(args.workspace_in)

    common_train = [
        "--iters", str(args.iters),
        "--max_steps", str(args.max_steps),
        "--learning_rate", str(args.learning_rate),
        "--init_scale", str(args.init_scale),
        "--w_gouge", str(args.w_gouge),
        "--w_residual", str(args.w_residual),
        "--grad_clip", str(args.grad_clip),
        "--w_len", str(args.w_len),
        "--eval_freq", str(args.eval_freq),
        "--seed", str(args.seed),
        "--stock_size_in", *ssi,
        "--voxel_size_mm", str(args.voxel_size_mm),
        "--workspace_in", *wsi,
        "--target_shape", args.target_shape,
        "--target_radius_mm", str(args.target_radius_mm),
        "--target_height_mm", str(args.target_height_mm),
        "--dt", str(args.dt),
        "--headless", "--save_model", "--no-track", "--eval",
    ]

    # ----------------------------------------------------------- stage 1
    if args.stage1_run_dir:
        run1 = args.stage1_run_dir
        if not os.path.isabs(run1):
            run1 = os.path.join(REPO, run1)
        print(f"[staged] reusing stage-1 run dir: {run1}")
    else:
        cmd = [PYTHON, "-m", "algorithms.train_csg",
               "--init_mode", args.init_mode, *common_train]
        rc, out = _run(cmd, "stage-1 train")
        if rc != 0:
            raise SystemExit(f"stage-1 training failed (exit {rc})")
        run1 = os.path.join(REPO, _parse_run_dir(out))
    traj1 = os.path.join(run1, "trajectory.npy")
    if not os.path.exists(traj1):
        raise SystemExit(f"stage-1 trajectory not found: {traj1}")

    # ----------------------------------------------------------- truncate
    state_npz = os.path.join(run1, "trunc_state.npz")
    cmd = [PYTHON, "-m", "algorithms.truncate_trajectory",
           "--run-dir", run1, "--out", state_npz,
           "--remove-thresh", str(args.remove_thresh),
           "--min-keep-frac", str(args.min_keep_frac)]
    rc, _ = _run(cmd, "truncate")
    if rc != 0:
        raise SystemExit(f"truncation failed (exit {rc})")
    saved = np.load(state_npz)
    t_star = int(saved["t_trunc"])
    print(f"[staged] truncation point t* = {t_star} "
          f"(kept {t_star + 1} stage-1 positions)")

    # ----------------------------------------------------------- stage 2
    cmd = [PYTHON, "-m", "algorithms.train_csg",
           "--init_mode", "random", "--init-stock-from", state_npz, *common_train]
    rc, out = _run(cmd, "stage-2 train")
    if rc != 0:
        raise SystemExit(f"stage-2 training failed (exit {rc})")
    run2 = os.path.join(REPO, _parse_run_dir(out))
    traj2 = os.path.join(run2, "trajectory.npy")

    # ----------------------------------------------------------- concat + eval
    p1 = np.load(traj1)
    p2 = np.load(traj2)
    # Stage 2's first position == saved tool_pos == stage 1's position[t*],
    # so drop it to avoid duplicating the waypoint.
    concat = np.concatenate([p1[: t_star + 1], p2[1:]], axis=0).astype(np.float32)
    concat_path = os.path.join(run2, "trajectory_concat.npy")
    np.save(concat_path, concat)
    print(f"[staged] concatenated trajectory: {concat.shape} -> {concat_path}")

    cmd = [PYTHON, "-m", "eval.eval_csg",
           "--trajectory", concat_path,
           "--stock-size-in", *ssi,
           "--voxel-size-mm", str(args.voxel_size_mm),
           "--target-shape", args.target_shape,
           "--workspace-in", *wsi,
           "--post", args.post]
    rc, _ = _run(cmd, "eval concatenated")
    if rc != 0:
        print(f"[staged] WARNING: eval exited {rc}")

    print("\n================ staged summary ================")
    print(f"stage-1 run : {run1}")
    print(f"stage-2 run : {run2}")
    print(f"truncation  : t*={t_star} (kept {t_star + 1}/{len(p1)} stage-1 positions)")
    print(f"concat traj : {concat_path}  ({len(concat)} positions)")
    m2 = os.path.join(run2, "metrics.json")
    print(f"stage-2 dice (on saved stock, soft): {m2}")
    print("================================================")


if __name__ == "__main__":
    main()
