"""Export a saved trajectory to G-code with a configurable post-processor.

Loads an ``(T, 3)`` unit-cube trajectory (e.g. ``trajectory.npy`` written by
``algorithms/train_csg.py``) and writes a G-code program. Choose the dialect
with ``--post``:

* ``rs274`` -- generic LinuxCNC-style program (default).
* ``haas``  -- Fanuc-style program ready to load on a Haas Mini Mill.

Examples
--------
    uv run python scripts/export_gcode.py --post haas -o part.nc
    uv run python scripts/export_gcode.py --post haas --tool 3 --rpm 6000 \
        --workspace-mm 80 --program-number 1234
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cam import MachineConfig, save_gcode, POSTS


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
    ap.add_argument("--workspace-in", type=float, nargs=3, default=[16.0, 12.0, 10.0],
                    metavar=("X", "Y", "Z"),
                    help="machine envelope (x y z) in inches (default Haas Mini Mill)")
    ap.add_argument("--workspace-mm", type=float, default=None,
                    help="cube edge length (mm); overrides --workspace-in with an isotropic cube")
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

    cfg = MachineConfig(
        workspace_mm=args.workspace_mm if args.workspace_mm else 100.0,
        workspace_in=None if args.workspace_mm else tuple(args.workspace_in),
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
