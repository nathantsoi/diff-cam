"""DiffCAM task adapter for the direct RustCNC strategy ports.

PROVENANCE(reference): composition of the RustCNC-derived generators at commit
5fc4717a9342313c03461f80a19a79ad8996e757. No clean-room/MIT status is implied.

The individual port files own their algorithmic constants and coarse sampling.
This module only selects a strategy for a typed family and composes Pocket2D
roughing with an explicit Contour2D finish. It never unwraps the legacy raster.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping

from benchmarks import contracts as C
from cam.seeds import seed_spec_for_task

RUSTCNC_COMMIT = "5fc4717a9342313c03461f80a19a79ad8996e757"
SEED_GENERATION_VERSION = "cam-rustcnc-direct/1"
DEFAULT_MAX_SEGMENTS = 72


def _cap(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_segments must be a positive integer")
    return value


def _contour_profile(task: C.TaskSpec, spec):
    """Match contour2d_seed's explicit bounded thin-gap default profile."""
    if spec is None or isinstance(spec, Mapping):
        radius = float(task.tool.cutter_radius_mm)
        profile = {"leadInRadius": radius / 2.0, "leadOutRadius": radius / 2.0,
                   "leadInVerticalRadius": radius / 4.0, "leadOutVerticalRadius": radius / 4.0}
        if spec is not None:
            profile.update(spec)
        return profile
    return spec


def _outer_loop(task: C.TaskSpec) -> tuple[tuple[float, float], ...]:
    params = task.original_part.params
    low, high = params["pocket_min_mm"], params["pocket_max_mm"]
    xmin, ymin = float(low[0]), float(low[1])
    xmax, ymax = float(high[0]), float(high[1])
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("Pocket/Contour composition requires a nonempty pocket rectangle")
    return ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))


def _island_loop(task: C.TaskSpec, derived) -> tuple[tuple[float, float], ...] | None:
    if not derived.has_retained_island:
        return None
    params = task.original_part.params
    low, high = params["island_min_mm"], params["island_max_mm"]
    xmin, ymin = float(low[0]), float(low[1])
    xmax, ymax = float(high[0]), float(high[1])
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("retained-pad rectangle must be nonempty")
    # Clockwise retained boundary + left compensation -> rolled outside offset.
    return ((xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin))


def _provenance(raw) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {"source_provenance": str(raw)}


def pocket_with_contour_seed(task: C.TaskSpec, spec=None, *, contour_spec=None,
                             max_segments: int = DEFAULT_MAX_SEGMENTS) -> C.MotionPlan:
    """Use sourced Pocket2D roughing, replacing its deepest native wall finish.

    The lower-level Pocket rougher supplies exact segment ranges for each native
    outer/island wall finish. Only ranges at its deepest level are replaced.
    Previous elevations, any retained-pad cap phase, entry and retract remain in
    the source order. Contour2D then supplies an explicit outer closed loop and,
    when present, a closed retained-pad finish. The combined whole plan—not just
    one phase—is subject to the cap and full contract validation.
    """
    cap = _cap(max_segments)
    from cam import rustcnc_face_contour as contour
    from cam import rustcnc_pocket2d as pocket

    derived = seed_spec_for_task(task)
    profile = _contour_profile(task, contour_spec)
    roughing = pocket.emit_pocket_roughing(task, spec, max_segments=cap)
    wall = roughing.wall_finish_passes
    levels = [item.level for item in wall if isinstance(item.level, int)]
    if not levels:
        raise ValueError("Pocket2D did not identify native wall passes for Contour2D composition")
    deepest = max(levels)
    replace_ranges = [(item.first_segment, item.end_segment_exclusive, item.role)
                      for item in wall if item.level == deepest]
    if not replace_ranges:
        raise ValueError("no deepest Pocket2D wall-finish ranges to compose with Contour2D")
    native = roughing.motion.segments
    kept: list[C.Segment] = []
    removed: list[C.Segment] = []
    drop_indices = {index for first, end, _role in replace_ranges
                    for index in range(first, end)}
    for index, segment in enumerate(native):
        (removed if index in drop_indices else kept).append(segment)
    if kept and native[0] not in kept:
        raise ValueError("composition may not remove Pocket2D's initial approach")

    outer = _outer_loop(task)
    pad = _island_loop(task, derived)
    position = (kept[-1].end_mm if kept else task.initial_state.position_mm)
    combined = list(kept)
    first, last = replace_ranges[0][0], replace_ranges[-1][1]
    if any(index < first or index >= last for index in drop_indices):
        raise ValueError("replacement wall ranges are not one contiguous final finish")
    if not removed:
        raise ValueError("internal finish-range inconsistency: selected passes have no segments")
    for left, right in zip(kept, kept[1:]):
        if math.dist(left.end_mm, right.start_mm) > C.MotionPlan.CONTIGUITY_TOL_MM:
            raise ValueError("wall-finish replacement leaves disconnected roughing; a sourced bridge is required")
    added = contour.emit_contour_loop(
        task, outer, spec=profile, contour_z=float(task.original_part.params["pocket_min_mm"][2]),
        right_compensation=False, start_machine_mm=position,
        segment_id_prefix="pocket-contour-outer", max_segments=cap,
        additional_walls=(() if pad is None else (pad,)),
    )
    combined.extend(added)
    if pad is not None:
        more = contour.emit_contour_loop(
            task, pad, spec=profile, contour_z=float(task.original_part.params["pocket_min_mm"][2]),
            right_compensation=False, start_machine_mm=combined[-1].end_mm,
            segment_id_prefix="pocket-contour-pad", max_segments=cap,
            additional_walls=(outer,),
        )
        combined.extend(more)
    if len(combined) > cap:
        raise ValueError(
            f"cam.pocket2d+cam.contour2d complete composition requires {len(combined)} "
            f"segments > cap={cap}; no region, level, contour, or pad was truncated")

    provenance = _provenance(roughing.motion.provenance)
    provenance.update({
        "seed_generation_version": SEED_GENERATION_VERSION,
        "strategy_id": "cam.pocket2d+cam.contour2d",
        "composition": {
            "deepest_native_wall_level_replaced": deepest,
            "replacement_ranges": [
                {"first_segment": first, "end_segment_exclusive": end, "role": role}
                for first, end, role in replace_ranges],
            "removed_segment_ids": [segment.id for segment in removed],
            "removed_segment_count": len(removed),
            "retained_roughing_segment_count": len(kept),
            "outer_contour_segment_count": len(added),
            "pad_contour_segment_count": len(more) if pad is not None else 0,
            "complete_segment_count": len(combined),
            "max_segments": cap,
            "contour_source_commit": contour.RUSTCNC_SHA,
            "contour_profile": _provenance(profile),
            "contour_source_sha256": contour.CONTOUR_SHA256,
            "truncated_or_dropped_regions_or_levels": False,
            "note": "Named deepest native finish passes are replaced by the sourced Contour2D finish, not machined twice; no region/level is omitted.",
        },
    })
    result = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=task.initial_state,
        segments=tuple(combined),
        timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
        provenance=json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )
    result.validate_for_task(task)
    return result


def cam_seed_for_task(task: C.TaskSpec, spec=None, *,
                      max_segments: int = DEFAULT_MAX_SEGMENTS) -> C.MotionPlan:
    """Dispatch only on declared task geometry, never optimizer outcomes."""
    cap = _cap(max_segments)
    if not isinstance(task, C.TaskSpec):
        raise TypeError("RustCNC seed dispatch requires a typed TaskSpec")
    if task.family == "facing_asymmetric":
        from cam.rustcnc_face_contour import face_seed
        return face_seed(task, spec, max_segments=cap)
    if task.family == "pocket_island":
        return pocket_with_contour_seed(task, spec, max_segments=cap)
    if task.family == "variable_width_depth":
        from cam.rustcnc_adaptive2d import adaptive2d_seed
        return adaptive2d_seed(task, spec, max_segments=cap)
    raise ValueError(f"no registered RustCNC starting strategy for task family {task.family!r}")


__all__ = ["SEED_GENERATION_VERSION", "DEFAULT_MAX_SEGMENTS", "RUSTCNC_COMMIT",
           "cam_seed_for_task", "pocket_with_contour_seed"]
