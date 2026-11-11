# SPDX-License-Identifier: MIT
"""16-task 4-family 3-axis CNC milling benchmark suite (MIT clean-room).

Families (2 dev + 2 test instances each = 16 tasks):
1. ``facing_open_boss``: Open-perimeter billet facing around an asymmetric raised boss.
2. ``pocket_island``: Walled rectangular pocket roughing around an offset retained island.
3. ``adaptive_channel``: Dual-island pocket with left/right chambers and a pinched central throat.
4. ``contour_profile``: Outside-in multi-offset exterior boss contour profiling.

All geometries use the ``box_minus_multi_pocket`` analytic primitive supported identically by
``cam.posted_evaluator``, ``verification.camotics_adapter``, and sampled ``CSGSimulatorDelta``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from algorithms import parameterization as P
from algorithms import run_artifacts as R
from benchmarks import contracts as C
from cam.canonical_strategies import canonical_seed_for_task
from cam.strategies import (
    AdaptiveConfig,
    ContourConfig,
    FacingConfig,
    PocketConfig,
    generate_cleanroom_toolpath,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_MULTI = "box_minus_multi_pocket"
CORNER_RADIUS_KEY = "pocket_unreached_corner_radius_mm"
DEFAULT_VOXEL_MM = 0.25

MILLING_FAMILIES = (
    "facing_open_boss",
    "pocket_island",
    "adaptive_channel",
    "contour_profile",
)


@dataclass(frozen=True)
class MillingCaseSpec:
    family: str
    split: str
    case_id: str
    stock_min_mm: Tuple[float, float, float]
    stock_max_mm: Tuple[float, float, float]
    pocket_min_mm: Tuple[float, float, float]
    pocket_max_mm: Tuple[float, float, float]
    islands_mm: Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]], ...]
    mrr_limit_mm3_s: float = 110.0
    notes: str = ""


@dataclass(frozen=True)
class MillingTaskBundle:
    case: MillingCaseSpec
    task: C.TaskSpec
    nominal_motion: C.MotionPlan
    mrr_limit_mm3_s: float


# 16 deterministic geometric instances (4 families x 2 dev + 2 test).
# All coordinates lie on exact 0.5 mm multiples so 0.25 mm and 0.50 mm grids have integral shapes
# and voxel centres never land on planar SDF=0 boundaries.
MILLING_CASES: Tuple[MillingCaseSpec, ...] = (
    # --- Family 1: facing_open_boss (open outer perimeter, asymmetric standing boss) ---
    MillingCaseSpec(
        family="facing_open_boss",
        split="dev",
        case_id="face-dev-00",
        stock_min_mm=(-16.0, -12.0, -8.0),
        stock_max_mm=(16.0, 12.0, 0.0),
        pocket_min_mm=(-16.0, -12.0, -2.5),
        pocket_max_mm=(16.0, 12.0, 0.0),
        islands_mm=(((3.0, 1.5, -2.5), (11.0, 7.5, 0.0)),),
        mrr_limit_mm3_s=105.0,
        notes="Upper-right offset boss on open face.",
    ),
    MillingCaseSpec(
        family="facing_open_boss",
        split="dev",
        case_id="face-dev-01",
        stock_min_mm=(-18.0, -12.0, -8.0),
        stock_max_mm=(14.0, 12.0, 0.0),
        pocket_min_mm=(-18.0, -12.0, -2.0),
        pocket_max_mm=(14.0, 12.0, 0.0),
        islands_mm=(((-13.0, -7.5, -2.0), (-4.0, -1.0, 0.0)),),
        mrr_limit_mm3_s=95.0,
        notes="Lower-left offset boss on asymmetric billet.",
    ),
    MillingCaseSpec(
        family="facing_open_boss",
        split="test",
        case_id="face-test-00",
        stock_min_mm=(-16.0, -13.0, -8.0),
        stock_max_mm=(18.0, 11.0, 0.0),
        pocket_min_mm=(-16.0, -13.0, -3.0),
        pocket_max_mm=(18.0, 11.0, 0.0),
        islands_mm=(((4.5, -8.0, -3.0), (13.0, -1.5, 0.0)),),
        mrr_limit_mm3_s=115.0,
        notes="Deep 3 mm open face with lower-right rectangular boss.",
    ),
    MillingCaseSpec(
        family="facing_open_boss",
        split="test",
        case_id="face-test-01",
        stock_min_mm=(-17.0, -11.0, -8.0),
        stock_max_mm=(15.0, 13.0, 0.0),
        pocket_min_mm=(-17.0, -11.0, -2.5),
        pocket_max_mm=(15.0, 13.0, 0.0),
        islands_mm=(((-11.5, 2.0, -2.5), (-2.5, 8.5, 0.0)),),
        mrr_limit_mm3_s=105.0,
        notes="Upper-left boss on shifted open billet.",
    ),
    # --- Family 2: pocket_island (walled rectangular pocket with offset island) ---
    MillingCaseSpec(
        family="pocket_island",
        split="dev",
        case_id="pocket-dev-00",
        stock_min_mm=(-16.0, -13.0, -8.0),
        stock_max_mm=(16.0, 13.0, 0.0),
        pocket_min_mm=(-13.0, -10.0, -3.5),
        pocket_max_mm=(13.0, 10.0, 0.0),
        islands_mm=(((-2.5, -2.0, -3.5), (3.5, 3.0, 0.0)),),
        mrr_limit_mm3_s=110.0,
        notes="Walled pocket with slightly offset rectangular island.",
    ),
    MillingCaseSpec(
        family="pocket_island",
        split="dev",
        case_id="pocket-dev-01",
        stock_min_mm=(-17.0, -13.0, -8.0),
        stock_max_mm=(17.0, 13.0, 0.0),
        pocket_min_mm=(-14.0, -10.0, -3.0),
        pocket_max_mm=(14.0, 10.0, 0.0),
        islands_mm=(((-5.0, -3.0, -3.0), (1.0, 2.0, 0.0)),),
        mrr_limit_mm3_s=100.0,
        notes="Wide walled pocket with left-shifted island.",
    ),
    MillingCaseSpec(
        family="pocket_island",
        split="test",
        case_id="pocket-test-00",
        stock_min_mm=(-16.0, -14.0, -8.0),
        stock_max_mm=(16.0, 14.0, 0.0),
        pocket_min_mm=(-13.0, -11.0, -4.0),
        pocket_max_mm=(13.0, 11.0, 0.0),
        islands_mm=(((-1.5, -3.5, -4.0), (4.5, 2.0, 0.0)),),
        mrr_limit_mm3_s=120.0,
        notes="Deeper 4 mm walled pocket with lower-right offset island.",
    ),
    MillingCaseSpec(
        family="pocket_island",
        split="test",
        case_id="pocket-test-01",
        stock_min_mm=(-18.0, -13.0, -8.0),
        stock_max_mm=(16.0, 13.0, 0.0),
        pocket_min_mm=(-15.0, -10.0, -3.0),
        pocket_max_mm=(13.0, 10.0, 0.0),
        islands_mm=(((-4.5, -1.5, -3.0), (2.5, 3.5, 0.0)),),
        mrr_limit_mm3_s=105.0,
        notes="Asymmetric walled pocket with upper-left shifted rectangular island.",
    ),
    # --- Family 3: adaptive_channel (walled pocket with TWO islands & central throat) ---
    MillingCaseSpec(
        family="adaptive_channel",
        split="dev",
        case_id="adapt-dev-00",
        stock_min_mm=(-18.0, -15.0, -8.0),
        stock_max_mm=(18.0, 15.0, 0.0),
        pocket_min_mm=(-15.0, -12.0, -3.0),
        pocket_max_mm=(15.0, 12.0, 0.0),
        islands_mm=(
            ((-3.5, -7.0, -3.0), (3.5, -3.0, 0.0)),
            ((-3.5, 3.0, -3.0), (3.5, 7.0, 0.0)),
        ),
        mrr_limit_mm3_s=105.0,
        notes="Symmetric dual-island pocket with 6 mm pinched central throat.",
    ),
    MillingCaseSpec(
        family="adaptive_channel",
        split="dev",
        case_id="adapt-dev-01",
        stock_min_mm=(-18.0, -15.0, -8.0),
        stock_max_mm=(18.0, 15.0, 0.0),
        pocket_min_mm=(-15.0, -12.0, -2.5),
        pocket_max_mm=(15.0, 12.0, 0.0),
        islands_mm=(
            ((-2.0, -7.0, -2.5), (4.5, -3.0, 0.0)),
            ((-4.5, 3.0, -2.5), (2.0, 7.0, 0.0)),
        ),
        mrr_limit_mm3_s=95.0,
        notes="Staggered dual islands creating asymmetric chambers and diagonal throat.",
    ),
    MillingCaseSpec(
        family="adaptive_channel",
        split="test",
        case_id="adapt-test-00",
        stock_min_mm=(-19.0, -15.0, -8.0),
        stock_max_mm=(19.0, 15.0, 0.0),
        pocket_min_mm=(-16.0, -12.0, -3.5),
        pocket_max_mm=(16.0, 12.0, 0.0),
        islands_mm=(
            ((-4.0, -7.0, -3.5), (4.0, -3.0, 0.0)),
            ((-4.0, 3.0, -3.5), (4.0, 7.0, 0.0)),
        ),
        mrr_limit_mm3_s=115.0,
        notes="Wider 3.5 mm deep dual-island channel clearing task.",
    ),
    MillingCaseSpec(
        family="adaptive_channel",
        split="test",
        case_id="adapt-test-01",
        stock_min_mm=(-18.0, -16.0, -8.0),
        stock_max_mm=(18.0, 16.0, 0.0),
        pocket_min_mm=(-15.0, -13.0, -3.0),
        pocket_max_mm=(15.0, 13.0, 0.0),
        islands_mm=(
            ((-4.5, -7.5, -3.0), (2.5, -3.5, 0.0)),
            ((-2.5, 3.5, -3.0), (4.5, 7.5, 0.0)),
        ),
        mrr_limit_mm3_s=105.0,
        notes="Opposite-staggered dual islands with 7 mm central channel.",
    ),
    # --- Family 4: contour_profile (outside-in multi-offset exterior boss profiling) ---
    MillingCaseSpec(
        family="contour_profile",
        split="dev",
        case_id="contour-dev-00",
        stock_min_mm=(-15.0, -12.0, -8.0),
        stock_max_mm=(15.0, 12.0, 0.0),
        pocket_min_mm=(-15.0, -12.0, -3.0),
        pocket_max_mm=(15.0, 12.0, 0.0),
        islands_mm=(((-6.0, -4.5, -3.0), (6.0, 4.5, 0.0)),),
        mrr_limit_mm3_s=105.0,
        notes="Centred rectangular boss exterior multi-pass profiling.",
    ),
    MillingCaseSpec(
        family="contour_profile",
        split="dev",
        case_id="contour-dev-01",
        stock_min_mm=(-16.0, -12.0, -8.0),
        stock_max_mm=(16.0, 12.0, 0.0),
        pocket_min_mm=(-16.0, -12.0, -2.5),
        pocket_max_mm=(16.0, 12.0, 0.0),
        islands_mm=(((-4.0, -5.0, -2.5), (7.5, 3.5, 0.0)),),
        mrr_limit_mm3_s=95.0,
        notes="Eccentric boss with uneven left/right shoulder widths.",
    ),
    MillingCaseSpec(
        family="contour_profile",
        split="test",
        case_id="contour-test-00",
        stock_min_mm=(-16.0, -13.0, -8.0),
        stock_max_mm=(16.0, 13.0, 0.0),
        pocket_min_mm=(-16.0, -13.0, -3.5),
        pocket_max_mm=(16.0, 13.0, 0.0),
        islands_mm=(((-7.0, -4.0, -3.5), (5.0, 5.5, 0.0)),),
        mrr_limit_mm3_s=115.0,
        notes="Deep 3.5 mm asymmetric boss contour task.",
    ),
    MillingCaseSpec(
        family="contour_profile",
        split="test",
        case_id="contour-test-01",
        stock_min_mm=(-15.0, -13.0, -8.0),
        stock_max_mm=(17.0, 11.0, 0.0),
        pocket_min_mm=(-15.0, -13.0, -3.0),
        pocket_max_mm=(17.0, 11.0, 0.0),
        islands_mm=(((-5.5, -6.0, -3.0), (8.0, 3.0, 0.0)),),
        mrr_limit_mm3_s=105.0,
        notes="Shifted stock and wide rectangular boss contour task.",
    ),
)


def _grid_points(grid: C.GridSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [
        grid.origin_mm[a] + (np.arange(grid.shape[a], dtype=np.float64) + 0.5) * grid.voxel_mm
        for a in range(3)
    ]
    return tuple(np.meshgrid(*axes, indexing="ij"))


def _box_sdf(points: Tuple[np.ndarray, np.ndarray, np.ndarray], lo, hi) -> np.ndarray:
    lower = np.asarray(lo, dtype=np.float64)
    upper = np.asarray(hi, dtype=np.float64)
    q = [np.maximum(lower[a] - points[a], points[a] - upper[a]) for a in range(3)]
    pos = [np.maximum(v, 0.0) for v in q]
    return np.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2) + np.minimum(np.maximum.reduce(q), 0.0)


def multi_pocket_fields(
    stock: C.GeometrySource,
    geometry: C.GeometrySource,
    grid: C.GridSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact mm SDFs (stock_sdf, target_sdf) for ``box_minus_multi_pocket``."""
    points = _grid_points(grid)
    stock_sdf = _box_sdf(points, stock.params["min_mm"], stock.params["max_mm"])
    p = geometry.params
    box_min, box_max = p["box_min_mm"], p["box_max_mm"]
    pocket_min, pocket_max = p["pocket_min_mm"], p["pocket_max_mm"]
    outer_d = _box_sdf(points, box_min, box_max)
    pocket_d = _box_sdf(points, pocket_min, pocket_max)

    # Cavity minus union of retained islands: cavity_eff < 0 where material is removed.
    cavity_eff = pocket_d.copy()
    for isl in p.get("islands_mm", ()):
        isl_d = _box_sdf(points, isl[0], isl[1])
        cavity_eff = np.maximum(cavity_eff, -isl_d)

    part_sdf = np.maximum(outer_d, -cavity_eff)

    radius = float(p.get(CORNER_RADIUS_KEY, 0.0))
    if radius > 0.0:
        fillet_sdfs: List[np.ndarray] = []
        for x_side in (0, 1):
            x_walled = (
                (pocket_min[0] > box_min[0] + 1e-9)
                if x_side == 0
                else (pocket_max[0] < box_max[0] - 1e-9)
            )
            if not x_walled:
                continue
            for y_side in (0, 1):
                y_walled = (
                    (pocket_min[1] > box_min[1] + 1e-9)
                    if y_side == 0
                    else (pocket_max[1] < box_max[1] - 1e-9)
                )
                if not y_walled:
                    continue
                cx = pocket_min[0] + radius if x_side == 0 else pocket_max[0] - radius
                cy = pocket_min[1] + radius if y_side == 0 else pocket_max[1] - radius
                lo = (
                    pocket_min[0] if x_side == 0 else pocket_max[0] - radius,
                    pocket_min[1] if y_side == 0 else pocket_max[1] - radius,
                    pocket_min[2],
                )
                hi = (
                    pocket_min[0] + radius if x_side == 0 else pocket_max[0],
                    pocket_min[1] + radius if y_side == 0 else pocket_max[1],
                    pocket_max[2],
                )
                cbox = _box_sdf(points, lo, hi)
                outside_disk = radius - np.hypot(points[0] - cx, points[1] - cy)
                fillet_sdfs.append(np.maximum(cbox, outside_disk))
        if fillet_sdfs:
            part_sdf = np.minimum.reduce([part_sdf, *fillet_sdfs])

    # Avoid exact 0.0 float32 values on curved corner-circle boundaries so strict <0 (simulator)
    # and closed <=0 (posted/CAMotics evaluators) agree bit-for-bit.
    zero_f32 = np.asarray(part_sdf, dtype=np.float32) == 0.0
    if np.any(zero_f32):
        part_sdf = part_sdf.copy()
        part_sdf[zero_f32] = -1e-5
    return stock_sdf, part_sdf


def derive_multi_pocket_target(
    stock: C.GeometrySource,
    original_part: C.GeometrySource,
    tool: C.ToolAssembly,
    grid: C.GridSpec,
) -> C.GeometrySource:
    """Add finite-cutter corner fillets only at walled concave pocket corners."""
    radius = float(tool.cutter_radius_mm)
    p = original_part.params
    box_min, box_max = p["box_min_mm"], p["box_max_mm"]
    pocket_min, pocket_max = p["pocket_min_mm"], p["pocket_max_mm"]
    walled_corners = 0
    for x_side in (0, 1):
        xw = (pocket_min[0] > box_min[0] + 1e-9) if x_side == 0 else (pocket_max[0] < box_max[0] - 1e-9)
        for y_side in (0, 1):
            yw = (pocket_min[1] > box_min[1] + 1e-9) if y_side == 0 else (pocket_max[1] < box_max[1] - 1e-9)
            if xw and yw:
                walled_corners += 1

    eff_radius = radius if walled_corners > 0 else 0.0
    _, orig_sdf = multi_pocket_fields(stock, original_part, grid)
    candidate_params = dict(p)
    candidate_params[CORNER_RADIUS_KEY] = eff_radius
    cand_geom = C.GeometrySource(kind="analytic", params=candidate_params, frame=C.Frame.WORK)
    _, target_sdf = multi_pocket_fields(stock, cand_geom, grid)

    orig_mask = orig_sdf <= 0.0
    target_mask = target_sdf <= 0.0
    excluded_voxels = int(np.count_nonzero(target_mask & ~orig_mask))
    grid_excl_mm3 = float(excluded_voxels * (grid.voxel_mm ** 3))
    depth = float(pocket_max[2] - pocket_min[2])
    cont_excl_mm3 = float(walled_corners * (1.0 - math.pi / 4.0) * (radius ** 2) * depth)

    candidate_params.update(
        {
            "continuous_excluded_volume_mm3": cont_excl_mm3,
            "grid_excluded_voxels": excluded_voxels,
            "grid_excluded_volume_mm3": grid_excl_mm3,
            "grid_exclusion_tolerance_mm3": 0.0,
        }
    )
    return C.GeometrySource(kind="analytic", params=candidate_params, frame=C.Frame.WORK)


_CLEANROOM_V2_CONFIGS = {
    "facing_open_boss": FacingConfig(),
    "pocket_island": PocketConfig(
        stepover_fraction=0.68,
        nominal_feed_mm_s=4.6,
        plunge_feed_mm_s=2.8,
    ),
    "adaptive_channel": AdaptiveConfig(
        optimal_load_fraction=0.40,
        trochoid_pitch_fraction=0.40,
        nominal_feed_mm_s=4.5,
        plunge_feed_mm_s=2.8,
        chord_step_mm=1.35,
        simplify_tol_mm=0.12,
    ),
    "contour_profile": ContourConfig(rough_stepover_fraction=0.68, corner_roll_arc_segments=3),
}


def build_milling_task_bundle(
    case: MillingCaseSpec,
    *,
    voxel_mm: float = DEFAULT_VOXEL_MM,
    timing_rule: C.TimingRule = C.TimingRule.BLENDED_ACCEL_BOUNDED,
    seed_engine: str = "v2",
) -> MillingTaskBundle:
    """Build one validated TaskSpec + clean-room MIT nominal MotionPlan (``v2`` default, ``v1`` legacy)."""
    shape = tuple(
        int(round((hi - lo) / voxel_mm))
        for lo, hi in zip(case.stock_min_mm, case.stock_max_mm)
    )
    if any(
        not math.isclose(shape[a] * voxel_mm, case.stock_max_mm[a] - case.stock_min_mm[a], abs_tol=1e-9)
        for a in range(3)
    ):
        raise ValueError(f"case {case.case_id} dimensions are not divisible by voxel_mm={voxel_mm}")

    grid = C.GridSpec(shape=shape, origin_mm=case.stock_min_mm, voxel_mm=voxel_mm, frame=C.Frame.WORK)
    stock = C.GeometrySource(
        kind="analytic",
        params={"primitive": "box", "min_mm": list(case.stock_min_mm), "max_mm": list(case.stock_max_mm)},
        frame=C.Frame.WORK,
    )
    original_part = C.GeometrySource(
        kind="analytic",
        params={
            "primitive": SCHEMA_MULTI,
            "box_min_mm": list(case.stock_min_mm),
            "box_max_mm": list(case.stock_max_mm),
            "pocket_min_mm": list(case.pocket_min_mm),
            "pocket_max_mm": list(case.pocket_max_mm),
            "islands_mm": [[list(isl[0]), list(isl[1])] for isl in case.islands_mm],
        },
        frame=C.Frame.WORK,
    )
    tool = C.ToolAssembly(
        cutter_kind=C.CutterKind.FLAT,
        cutter_radius_mm=2.5,
        flute_length_mm=12.0,
        shank_radius_mm=2.5,
        holder_bottom_mm=14.0,
        holder_radius_mm=8.0,
        holder_height_mm=30.0,
        tool_number=1,
    )
    op_geom = derive_multi_pocket_target(stock, original_part, tool, grid)
    excl_mm3 = float(op_geom.params["grid_excluded_volume_mm3"])
    excl_reason = (
        f"Finite 2.5 mm flat cutter walled-corner fillet exclusion ({excl_mm3} mm^3 on {voxel_mm} mm grid)."
        if excl_mm3 > 0.0
        else ""
    )

    tolerances = C.Tolerances(
        completion_residual_mm3=2.0,
        gouge_volume_mm3=0.0,
        local_surface_error_mm=max(0.5, 2.0 * voxel_mm),
        clearance_mm=0.25,
        rapid_penetration_mm=0.0,
    )
    # Envelope allows generous XY overtravel around stock for open-wall facing/contouring
    limits = C.MachineLimits(
        max_feed_mm_s=20.0,
        rapid_mm_s=100.0,
        max_accel_mm_s2=150.0,
        envelope_min_mm=(case.stock_min_mm[0] - 10.0, case.stock_min_mm[1] - 10.0, case.stock_min_mm[2] - 4.0),
        envelope_max_mm=(case.stock_max_mm[0] + 10.0, case.stock_max_mm[1] + 10.0, 60.0),
    )
    semantics = C.MotionSemantics(timing_rule=timing_rule)

    task = C.TaskSpec(
        task_id=f"milling-{case.family}-{case.case_id}",
        family=case.family,
        split=C.Split(case.split),
        source=C.SourceInfo(
            origin=f"milling_suite:1.0:{case.case_id}",
            license="MIT",
            notes=case.notes,
        ),
        length_unit="mm",
        stock=stock,
        original_part=original_part,
        operation_target=C.OperationTarget(
            geometry=op_geom,
            excluded_volume_mm3=excl_mm3,
            exclusion_reason=excl_reason,
        ),
        grid=grid,
        work_to_machine=C.Transform.translation((0.0, 0.0, 0.0)),
        initial_state=C.MachineState(position_mm=(0.0, 0.0, 6.0), frame=C.Frame.MACHINE),
        tool=tool,
        limits=limits,
        tolerances=tolerances,
        motion_semantics=semantics,
    )
    if seed_engine == "v2":
        nominal_motion = generate_cleanroom_toolpath(
            task, config=_CLEANROOM_V2_CONFIGS[case.family]
        )
    elif seed_engine == "v1":
        nominal_motion = canonical_seed_for_task(task)
    else:
        raise ValueError(f"Unknown seed_engine {seed_engine!r}; expected 'v2' or 'v1'")
    return MillingTaskBundle(
        case=case,
        task=task,
        nominal_motion=nominal_motion,
        mrr_limit_mm3_s=case.mrr_limit_mm3_s,
    )


def build_milling_suite(
    *,
    voxel_mm: float = DEFAULT_VOXEL_MM,
    timing_rule: C.TimingRule = C.TimingRule.BLENDED_ACCEL_BOUNDED,
    split: Optional[str] = None,
    families: Optional[Sequence[str]] = None,
    seed_engine: str = "v2",
) -> Tuple[MillingTaskBundle, ...]:
    """Build the 16-task (or filtered) MIT clean-room CNC milling suite."""
    selected = MILLING_CASES
    if split is not None:
        selected = tuple(c for c in selected if c.split == split)
    if families is not None:
        fam_set = set(families)
        selected = tuple(c for c in selected if c.family in fam_set)
    return tuple(
        build_milling_task_bundle(
            c, voxel_mm=voxel_mm, timing_rule=timing_rule, seed_engine=seed_engine
        )
        for c in selected
    )


def materialize_sampled_sources(
    task: C.TaskSpec,
    cache_root: Optional[Path] = None,
) -> Dict[str, C.GeometrySource]:
    """Materialize hash-checked float32 sampled stock & operation_target SDFs for CSGSimulatorDelta."""
    root = (PROJECT_ROOT if cache_root is None else Path(cache_root)).resolve()
    directory = root / "external" / "milling-suite-inputs" / task.task_hash[7:23]
    directory.mkdir(parents=True, exist_ok=True)
    stock_sdf, target_sdf = multi_pocket_fields(task.stock, task.operation_target.geometry, task.grid)
    sources: Dict[str, C.GeometrySource] = {}
    for label, arr in (("stock", stock_sdf), ("target", target_sdf)):
        payload = R.numpy_bytes(np.asarray(arr, dtype=np.float32))
        digest = C.sha256_bytes(payload)
        path = directory / f"{label}.npy"
        if not path.exists() or C.sha256_bytes(path.read_bytes()) != digest:
            R.ArtifactWriter._replace_bytes(path, payload)
        rel = path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)
        sources[label] = C.GeometrySource(
            kind="sampled",
            path=rel,
            sha256=digest,
            grid=task.grid,
            frame=C.Frame.WORK,
        )
    return sources


def compute_waypoint_xy_weights(
    task: C.TaskSpec,
    motion: C.MotionPlan,
    *,
    scale_mm: float = 1.25,
) -> np.ndarray:
    """Compute dimensionless interior-waypoint XY mobility weights ``w in [0, 1]^(T-1)``.

    Pins rapid and vertical plunge/retract transitions (``w = 0``) so vertical plunges stay
    strictly vertical and high-Z rapids remain invariant, and scales horizontal cutting
    waypoints smoothly by analytical clearance to vertical blueprint walls while leaving full
    ``scale_mm`` mobility on roughing, peel, stepover, and air-overtravel waypoints.
    """
    inv = task.work_to_machine.inverse().apply
    pts = np.asarray(
        [inv(motion.initial_state.position_mm)] + [inv(s.end_mm) for s in motion.segments],
        dtype=np.float64,
    )
    deltas = np.diff(pts, axis=0)
    steps = len(motion.segments)
    n_int = max(0, steps - 1)
    weights = np.zeros(n_int, dtype=np.float64)
    if n_int == 0:
        return weights

    r_tool = float(task.tool.cutter_radius_mm)
    part = task.original_part
    has_multi_pocket = (
        part.kind == "analytic" and part.params.get("primitive") == "box_minus_multi_pocket"
    )
    if has_multi_pocket:
        smin = part.params["box_min_mm"]
        smax = part.params["box_max_mm"]
        pmin = part.params["pocket_min_mm"]
        pmax = part.params["pocket_max_mm"]
        islands = part.params.get("islands_mm", ())

    def _pt_to_seg_2d(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        if l2 <= 1e-18:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def _seg_intersects_box_2d(
        ax: float, ay: float, bx: float, by: float, x0: float, y0: float, x1: float, y1: float
    ) -> bool:
        t0, t1 = 0.0, 1.0
        dx, dy = bx - ax, by - ay
        for p_val, q_val in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
            if abs(p_val) <= 1e-15:
                if q_val < 0.0:
                    return False
            else:
                r = q_val / p_val
                if p_val < 0.0:
                    if r > t1:
                        return False
                    if r > t0:
                        t0 = r
                else:
                    if r < t0:
                        return False
                    if r < t1:
                        t1 = r
        return t0 <= t1

    def _seg_wall_clearance_mm(pa: np.ndarray, pb: np.ndarray) -> float:
        if not has_multi_pocket:
            return scale_mm + 1.0
        ax, ay, az = float(pa[0]), float(pa[1]), float(pa[2])
        bx, by, bz = float(pb[0]), float(pb[1]), float(pb[2])
        z_low = min(az, bz)
        d_min = 1e6
        if pmin[0] > smin[0] + 1e-6:
            d_min = min(d_min, max(0.0, min(ax, bx) - float(pmin[0])))
        if pmax[0] < smax[0] - 1e-6:
            d_min = min(d_min, max(0.0, float(pmax[0]) - max(ax, bx)))
        if pmin[1] > smin[1] + 1e-6:
            d_min = min(d_min, max(0.0, min(ay, by) - float(pmin[1])))
        if pmax[1] < smax[1] - 1e-6:
            d_min = min(d_min, max(0.0, float(pmax[1]) - max(ay, by)))
        for imin, imax in islands:
            if float(imax[2]) > z_low + 1e-6:
                ix0, iy0 = float(imin[0]), float(imin[1])
                ix1, iy1 = float(imax[0]), float(imax[1])
                if _seg_intersects_box_2d(ax, ay, bx, by, ix0, iy0, ix1, iy1):
                    d_min = 0.0
                    break
                for qx, qy in ((ax, ay), (bx, by)):
                    dx = max(0.0, ix0 - qx, qx - ix1)
                    dy = max(0.0, iy0 - qy, qy - iy1)
                    d_min = min(d_min, math.hypot(dx, dy))
                for cx, cy in ((ix0, iy0), (ix0, iy1), (ix1, iy0), (ix1, iy1)):
                    d_min = min(d_min, _pt_to_seg_2d(cx, cy, ax, ay, bx, by))
        return d_min - r_tool

    for i in range(n_int):
        seg_in = motion.segments[i]
        seg_out = motion.segments[i + 1]
        if seg_in.kind == C.SegmentKind.RAPID or seg_out.kind == C.SegmentKind.RAPID:
            weights[i] = 0.0
            continue
        xy_in = math.hypot(float(deltas[i, 0]), float(deltas[i, 1]))
        xy_out = math.hypot(float(deltas[i + 1, 0]), float(deltas[i + 1, 1]))
        if xy_in <= 1e-6 or xy_out <= 1e-6:
            weights[i] = 0.0
            continue
        pk = pts[i + 1]
        d_clear = min(_seg_wall_clearance_mm(pts[i], pk), _seg_wall_clearance_mm(pk, pts[i + 2]))
        weights[i] = float(np.clip((d_clear - 0.08) / max(scale_mm, 1e-6), 0.0, 1.0))
    return weights


def make_joint_parameterization(
    task: C.TaskSpec,
    motion: C.MotionPlan,
    *,
    scale_mm: float = 1.25,
    log_feed_scale: float = 1.25,
    num_pos_controls: Optional[int] = None,
    num_feed_controls: Optional[int] = None,
    feed_only: bool = False,
    path_only: bool = False,
    xy_only: bool = True,
) -> P.JointPathFeedSpline:
    """Create a bounded 2.5D XY PositionSpline + per-segment feedrate JointPathFeedSpline."""
    inv = task.work_to_machine.inverse().apply
    pts_work = [inv(motion.initial_state.position_mm)] + [inv(s.end_mm) for s in motion.segments]
    deltas_mm = np.diff(np.asarray(pts_work, dtype=np.float64), axis=0)
    steps = len(motion.segments)
    n_int = max(1, steps - 1)
    pos_ctrl = (
        min(n_int, 40, max(8, (steps * 2) // 3))
        if num_pos_controls is None
        else min(n_int, num_pos_controls)
    )
    feed_ctrl = steps if num_feed_controls is None else min(steps, num_feed_controls)

    is_rapid = np.array([s.kind == C.SegmentKind.RAPID for s in motion.segments], dtype=bool)
    max_mult = np.array(
        [
            1.0 if s.kind == C.SegmentKind.RAPID else (task.limits.max_feed_mm_s * 0.995) / s.feed_mm_s
            for s in motion.segments
        ],
        dtype=np.float64,
    )
    w_xy = compute_waypoint_xy_weights(task, motion, scale_mm=scale_mm)
    eff_pos_scale = 1e-6 if feed_only else scale_mm
    eff_pos_bounds = (-1e-4, 1e-4) if feed_only else (-1.0, 1.0)
    eff_feed_scale = 1e-6 if path_only else log_feed_scale
    eff_feed_bounds = (-1e-4, 1e-4) if path_only else (-1.0, 1.25)
    return P.JointPathFeedSpline(
        deltas_mm,
        scale_mm=eff_pos_scale,
        log_feed_scale=eff_feed_scale,
        num_pos_controls=pos_ctrl,
        num_feed_controls=feed_ctrl,
        pos_bounds=eff_pos_bounds,
        feed_bounds=eff_feed_bounds,
        is_rapid=is_rapid,
        max_feed_mult_per_seg=max_mult,
        anchor_end=True,
        xy_only=xy_only,
        spatial_mode="position_spline",
        waypoint_weights=w_xy if len(w_xy) == steps - 1 else None,
        initial_position_mm=pts_work[0],
    )


__all__ = [
    "MILLING_FAMILIES",
    "MILLING_CASES",
    "MillingCaseSpec",
    "MillingTaskBundle",
    "multi_pocket_fields",
    "derive_multi_pocket_target",
    "build_milling_task_bundle",
    "build_milling_suite",
    "materialize_sampled_sources",
    "compute_waypoint_xy_weights",
    "make_joint_parameterization",
]
