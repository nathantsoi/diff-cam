"""Export a diff-cam trajectory to G-code via a configurable post-processor.

A diff-cam trajectory is an ``(T, 3)`` array of tool-tip positions in the unit
cube ``[0, 1]^3``. Coordinates are scaled from the unit cube to millimetres via
``MachineConfig.workspace_mm`` so the parser can invert the mapping exactly.

The actual G-code dialect is chosen by ``post`` (see :mod:`cam.posts`):

* ``"rs274"`` (default) -- generic LinuxCNC-style program used for round-trip
  fidelity checks.
* ``"haas"`` -- Fanuc-style program ready to run on a Haas Mini Mill.
"""

from .config import MachineConfig
from .posts import get_post, _fmt, _axis_words  # noqa: F401  (re-exported helpers)


def trajectory_to_gcode(positions, config: MachineConfig = MachineConfig(),
                        post: str = "rs274", feed_mults=None) -> str:
    """Convert an ``(T, 3)`` unit-cube trajectory to a G-code program string
    using the named post-processor.

    ``feed_mults`` is an optional ``(T-1,)`` per-segment feed schedule (see
    ``feed_mult.npy`` from ``train_csg.py``): entry ``s`` scales the programmed
    feed of the move ``positions[s] -> positions[s+1]``. Without it the feed is
    a single modal ``F`` word, exactly as before.
    """
    return get_post(post).program(positions, config, feed_mults=feed_mults)


def save_gcode(positions, path, config: MachineConfig = MachineConfig(),
               post: str = "rs274", feed_mults=None) -> str:
    """Write the G-code for ``positions`` to ``path`` and return the program text."""
    text = trajectory_to_gcode(positions, config, post, feed_mults=feed_mults)
    with open(path, "w") as f:
        f.write(text)
    return text
