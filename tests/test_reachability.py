"""Tests for the exact vertical-tool reachability mask (utils/reachability.py).

Ground truths used:
1. A solid box has only CONVEX corners, so every waste voxel is reachable by a
   vertical tool of any radius -> dice ceiling 1.0.
2. A slot narrower than the tool diameter cut into that box is unreachable
   below the box top (the tool cannot enter it) -> those voxels, and only
   those, become unreachable.
3. Part voxels are never marked reachable; waste above the part's top plane
   always is.
"""

import numpy as np

from utils.reachability import compute_reachable_mask, dice_ceiling


def _box_sdf(shape, lo, hi):
    """Occupancy-style SDF: -1 inside the box [lo, hi), +1 outside."""
    sdf = np.ones(shape, dtype=np.float32)
    sdf[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = -1.0
    return sdf


def test_solid_box_fully_reachable():
    sdf = _box_sdf((40, 40, 30), (10, 10, 0), (30, 30, 15))
    ceiling, unreach, _, _ = dice_ceiling(sdf, r_tool_vox=5.0)
    assert unreach == 0
    assert ceiling == 1.0


def test_narrow_slot_unreachable():
    sdf = _box_sdf((40, 40, 30), (10, 10, 0), (30, 30, 15))
    # 4-voxel-wide slot through the box interior, tool radius 5 (diameter 10):
    # the tool cannot enter, so slot waste below the box top is unreachable.
    sdf[19:23, 14:26, 0:15] = 1.0
    reach = compute_reachable_mask(sdf, r_tool_vox=5.0)
    slot = np.zeros(sdf.shape, dtype=bool)
    slot[19:23, 14:26, 0:15] = True
    # Slot mouth at the top face is exposed; interior depth must be unreachable.
    assert not reach[slot & (np.arange(30)[None, None, :] < 14)].any()
    # Outside the slot, the box's surroundings stay fully reachable.
    part = sdf <= 0
    outside = ~part & ~slot
    assert reach[outside].all()


def test_part_never_reachable_and_top_air_always():
    sdf = _box_sdf((40, 40, 30), (10, 10, 0), (30, 30, 15))
    reach = compute_reachable_mask(sdf, r_tool_vox=4.0)
    part = sdf <= 0
    assert not reach[part].any()
    assert reach[:, :, 16:].all()  # everything above the part top plane
