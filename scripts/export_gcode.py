"""Export a saved trajectory to G-code with a configurable post-processor.

Loads an ``(T, 3)`` stock-normalized trajectory (e.g. ``trajectory.npy`` written
by ``algorithms/train_csg.py``) and writes a G-code program. Choose the dialect
with ``--post``:

* ``rs274`` -- generic LinuxCNC-style program (default).
* ``haas``  -- Fanuc-style program ready to load on a Haas Mini Mill.

**Matching the training run.** The stock size, work origin (G54) and machine work
volume must match what the trajectory was optimized against, or the exported
coordinates won't line up with the real part. ``train_csg.py`` writes the full run
config to ``runs/<run>/args.json`` next to ``trajectory.npy``; this script reads
that file automatically (when the trajectory lives in a run dir) and uses its
``stock_size_in`` / ``stock_origin_in`` / ``workspace_in``. Any value you pass
explicitly on the CLI overrides the run config; use ``--no-run-config`` to ignore
it entirely.

Examples
--------
    # Auto-match the run's stock/part/location from runs/<run>/args.json
    uv run python scripts/export_gcode.py --post haas \
        --trajectory runs/CamEnvDiff-v0__train_csg__1__1782599757/trajectory.npy -o part.nc

    uv run python scripts/export_gcode.py --post haas --tool 3 --rpm 6000 \
        --stock-size-in 2 2 2 --stock-origin-in 8 6 5 --program-number 1234
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cam import MachineConfig, save_gcode, POSTS


def _load_run_config(trajectory_path, explicit_path, disabled):
    """Return (config_dict, source_path) from a training run's ``args.json``.

    Looks at ``explicit_path`` if given, else ``args.json`` next to the
    trajectory (where ``train_csg.py`` writes it). Returns ({}, None) when
    disabled or not found.
    """
    if disabled:
        return {}, None
    path = explicit_path
    if path is None:
        cand = os.path.join(os.path.dirname(os.path.abspath(trajectory_path)), "args.json")
        path = cand if os.path.exists(cand) else None
    if not path or not os.path.exists(path):
        return {}, None
    try:
        with open(path) as f:
            return json.load(f), path
    except (OSError, ValueError) as e:
        print(f"[config] WARNING: could not read run config {path}: {e}")
        return {}, None


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", default=os.path.join(repo, "trajectory.npy"),
                    help="input (T,3) trajectory .npy")
    ap.add_argument("-o", "--out", default=None,
                    help="output G-code path (default derives from --post)")
    ap.add_argument("--post", default="haas", choices=sorted(POSTS),
                    help="post-processor / machine dialect")
    ap.add_argument("--run-config", default=None,
                    help="path to a training run's args.json (default: auto-detect next to --trajectory)")
    ap.add_argument("--no-run-config", action="store_true",
                    help="ignore the training run's args.json; use only CLI flags / defaults")
    # Geometry flags default to None so we can tell whether the user set them
    # (CLI overrides the run config); unset values fall back to the run config
    # then to the built-in defaults.
    ap.add_argument("--workspace-in", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="machine work volume (x y z) in inches (default: run config, else Haas Mini Mill)")
    ap.add_argument("--workspace-mm", type=float, default=None,
                    help="work-volume cube edge (mm); overrides --workspace-in with a cube")
    ap.add_argument("--stock-size-in", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="stock box (x y z) in inches -- the normalized cube (default: run config, else 1 in cube)")
    ap.add_argument("--stock-origin-in", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="work origin (G54) = stock top-centre in machine inches (default: run config)")
    ap.add_argument("--feed", type=float, default=600.0, help="cutting feed, mm/min")
    ap.add_argument("--plunge-feed", type=float, default=200.0, help="Z plunge feed, mm/min")
    ap.add_argument("--rpm", type=float, default=5000.0, help="spindle speed")
    ap.add_argument("--tool", type=int, default=1, help="tool number (Txx)")
    ap.add_argument("--program-number", type=int, default=1, help="Haas O-number")
    ap.add_argument("--units", default="mm", choices=("mm", "inch"))
    ap.add_argument("--no-coolant", action="store_true")
    args = ap.parse_args()

    positions = np.load(args.trajectory).astype(np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise SystemExit(f"{args.trajectory} must hold an (T,3) array; got {positions.shape}")

    # Read the training run's config (stock/part/location) so the exported G-code
    # matches what the trajectory was optimized against. CLI flags override it.
    run_cfg, cfg_path = _load_run_config(args.trajectory, args.run_config, args.no_run_config)
    if cfg_path:
        print(f"[config] matching training run config from {cfg_path}")
    elif not args.no_run_config:
        print("[config] no training-run args.json found next to the trajectory; "
              "using CLI flags / defaults")

    def pick(cli_val, key, default):
        """CLI value if given, else the run config's value, else the default."""
        if cli_val is not None:
            return tuple(cli_val) if isinstance(cli_val, list) else cli_val, "cli"
        v = run_cfg.get(key)
        if v is not None:
            return tuple(v) if isinstance(v, list) else v, "run"
        return default, "default"

    stock_size_in, s_src = pick(args.stock_size_in, "stock_size_in", (1.0, 1.0, 1.0))
    stock_origin_in, o_src = pick(args.stock_origin_in, "stock_origin_in", None)

    # Work volume: --workspace-mm (cube) wins, else --workspace-in, else run config, else Mini Mill.
    if args.workspace_mm is not None:
        workspace_mm, workspace_in, w_src = args.workspace_mm, None, "cli(mm)"
    else:
        wv, w_src = pick(args.workspace_in, "workspace_in", (16.0, 12.0, 10.0))
        workspace_mm, workspace_in = 100.0, wv

    print(f"[config] stock_size_in={stock_size_in} ({s_src})  "
          f"stock_origin_in={stock_origin_in} ({o_src})  "
          f"workspace_in={workspace_in} ({w_src})")
    if run_cfg.get("target_shape"):
        print(f"[config] part: target_shape={run_cfg.get('target_shape')} "
              f"radius_mm={run_cfg.get('target_radius_mm')} "
              f"height_mm={run_cfg.get('target_height_mm')} "
              f"(does not affect coordinates; shown for traceability)")

    cfg = MachineConfig(
        workspace_mm=workspace_mm,
        workspace_in=workspace_in,
        stock_size_in=stock_size_in,
        stock_origin_in=stock_origin_in,
        feed=args.feed,
        plunge_feed=args.plunge_feed,
        spindle_rpm=args.rpm,
        tool_number=args.tool,
        program_number=args.program_number,
        units=args.units,
        coolant=not args.no_coolant,
    )

    ext = ".nc" if args.post == "haas" else ".ngc"
    out = args.out or os.path.join(repo, f"trajectory_{args.post}{ext}")
    save_gcode(positions, out, cfg, post=args.post)
    print(f"Wrote {args.post} G-code for {positions.shape[0]} points -> {out}")


if __name__ == "__main__":
    main()
