"""One bounded Titan-M1 +Z roughing pilot with native STEP provenance.

This is NOT a whole-part manufacturability claim.  The original STEP solid P
is retained separately.  The operation target T is its native Boolean union
with one selected small through-hole exclusion; a 6.35 mm flat cutter cannot
enter that measured narrower hole without cutting its original wall.  Other
features are not certified accessible.  The nominal NC removes a real, nonzero
amount of stock in a single central plunge.  It is not a complete finishing
program.  Production evaluates sampled P/T only on the exact matching task
grid. The partial-roughing geometry result is expected to be false, and all
safety/clearance outcomes are retained separately without a physical
machining-safety or finished-target claim. No interpolated or analytic-part
substitute is introduced.

Large derived fields and actual run artifacts go in a NEW directory under
``data/external/`` (ignored), or an explicitly supplied new output directory.
The local Titan STEP has unrecorded provenance/license and is never copied
into these outputs or redistributed by this module.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from benchmarks import contracts as C
from benchmarks import external_tasks as E
from cam.config import MachineConfig
from cam.gcode_parser import parse_motion_plan
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import post_motion_plan
from cam.units import inch_to_mm
from simulator.csg_metrics import grid_material_metrics
from verification.reference_carve import run_reference

PROJECT_ROOT = E.PROJECT_ROOT
SOURCE_STEP = PROJECT_ROOT / "parts/Titan-M1.step"
PROVENANCE_LICENSE = "unknown / provenance unrecorded"
PART_SOURCE_BOUNDS_MM = ((0.0, 0.0, -19.05), (101.6, 48.26, 0.0))
SOURCE_TO_WORK_MM = (-50.8, -24.13, 0.0)
WORK_TO_MACHINE_MM = (200.0, 100.0, -300.0)
STOCK_PAD_MM = 3.81
GRID_PITCH_MM = 1.27
HOLE_SOURCE_XY_MM = (3.81, 3.81)
# Measured against this exact source; these are conversion calculations, not
# rounded guessed physical coordinates or substituted analytic primitives.
HOLE_RADII_MM = tuple(inch_to_mm(value) for value in (0.0405, 0.09375))
REFERENCE_RADIUS_MM = inch_to_mm(0.195)
APPROACH_Z_MM = 5.0
SAFE_Z_MM = 12.7
NOMINAL_XY_WORK_MM = (-20.32, 0.0)  # source (30.48, 24.13); clear-air probes
NOMINAL_END_Z_MM = -9.525
FEED_MM_S = 2.5


def nominal_tool() -> C.ToolAssembly:
    """A plausible imperial flat-tool setup, NOT a vendor-qualified recipe."""
    return C.ToolAssembly(
        cutter_kind=C.CutterKind.FLAT,
        cutter_radius_mm=inch_to_mm(0.125),
        flute_length_mm=inch_to_mm(0.875),
        shank_radius_mm=inch_to_mm(0.125),
        holder_bottom_mm=inch_to_mm(1.75),
        holder_radius_mm=inch_to_mm(0.375),
        holder_height_mm=inch_to_mm(1.5),
        tool_number=1,
    )


def stock_bounds() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lo, hi = PART_SOURCE_BOUNDS_MM
    lower = (
        -0.5 * (hi[0] - lo[0]) - STOCK_PAD_MM,
        -0.5 * (hi[1] - lo[1]) - STOCK_PAD_MM,
        lo[2],
    )
    upper = (
        0.5 * (hi[0] - lo[0]) + STOCK_PAD_MM,
        0.5 * (hi[1] - lo[1]) + STOCK_PAD_MM,
        hi[2],
    )
    return lower, upper


def pilot_grid(voxel_mm: float = GRID_PITCH_MM) -> C.GridSpec:
    if not math.isfinite(voxel_mm) or voxel_mm <= 0:
        raise ValueError("Titan voxel_mm must be finite and positive")
    lower, upper = stock_bounds()
    dimensions = [(hi - lo) / voxel_mm for lo, hi in zip(lower, upper)]
    if any(abs(value - round(value)) > 1e-9 for value in dimensions):
        raise ValueError(f"Titan stock must have integral physical grid extents, got {dimensions}")
    return C.GridSpec(
        shape=tuple(int(round(value)) for value in dimensions),
        origin_mm=lower, voxel_mm=voxel_mm, frame=C.Frame.WORK,
    )


def grid_centres(grid: C.GridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [
        grid.origin_mm[a] + (np.arange(n, dtype=np.float64) + 0.5) * grid.voxel_mm
        for a, n in enumerate(grid.shape)
    ]
    return np.meshgrid(*axes, indexing="ij")


def box_signed_field(grid: C.GridSpec, lower: Sequence[float],
                     upper: Sequence[float]) -> np.ndarray:
    """Float64 physical-millimetre bounded-box field; never a source-part proxy."""
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("stock box requires three finite coordinates per corner")
    if any(
        not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or lo >= hi
        for lo, hi in zip(lower, upper)
    ):
        raise ValueError("stock box bounds must be finite and strictly increasing")
    xyz = grid_centres(grid)
    signed = [
        np.maximum(lower[a] - xyz[a], xyz[a] - upper[a]) for a in range(3)
    ]
    outside = np.sqrt(sum(np.maximum(value, 0.0) ** 2 for value in signed))
    inside = np.minimum(np.maximum.reduce(signed), 0.0)
    return (outside + inside).astype(np.float64)


def compare_line_motion(original: C.MotionPlan, parsed: C.MotionPlan) -> dict[str, Any]:
    """Compare executable line semantics, retaining free-form tag differences.

    The dev-pocket comparator accepts only its seed's specific labels.  The NC
    parser can relabel an approach as a plunge without changing its geometry;
    that is not a reason either to rename a rapid or to waive physical checks.
    This pilot is line-only and refuses arcs rather than chord-approximating them.
    """
    if (
        original.frame != parsed.frame
        or original.initial_state != parsed.initial_state
        or original.timing_rule != parsed.timing_rule
        or len(original.segments) != len(parsed.segments)
    ):
        raise AssertionError("posted motion changed frame, initial state, timing rule, or segment count")
    position_error = 0.0
    duration_error = 0.0
    tag_differences = []
    for before, after in zip(original.segments, parsed.segments):
        if (
            before.kind != after.kind
            or before.kind not in (C.SegmentKind.RAPID, C.SegmentKind.FEED)
            or (before.feed_mm_s is None) != (after.feed_mm_s is None)
        ):
            raise AssertionError(f"posted segment {before.id} changed executable kind/feed presence")
        if before.feed_mm_s is not None and abs(before.feed_mm_s - after.feed_mm_s) > 1e-6:
            raise AssertionError(f"posted segment {before.id} changed feed")
        position_error = max(position_error, *(
            math.dist(a, b) for a, b in zip(
                (before.start_mm, before.end_mm), (after.start_mm, after.end_mm)
            )
        ))
        duration_error = max(duration_error, abs(before.duration_s - after.duration_s))
        if before.tag != after.tag:
            tag_differences.append({
                "segment": before.id, "typed_tag": before.tag, "parsed_tag": after.tag,
                "kind": before.kind.value, "physical_segment_preserved": True,
            })
    if position_error > 0.002 or duration_error > 1e-5:
        raise AssertionError(
            f"posted line motion error: position={position_error} mm, duration={duration_error} s"
        )
    return {
        "line_segment_count": len(original.segments),
        "max_position_error_mm": position_error,
        "max_duration_error_s": duration_error,
        "free_form_tag_differences": tag_differences,
        "physical_segment_checks": "kind, start/end, feed presence/value, duration, initial state and frame",
    }


def geometry_request(output_dir: Path, *, step_path: Path = SOURCE_STEP,
                     voxel_mm: float = GRID_PITCH_MM) -> dict[str, Any]:
    """Freeze physical geometry/access choices before any optimizer scores."""
    lower, upper = stock_bounds()
    grid = pilot_grid(voxel_mm)
    if nominal_tool().flute_length_mm - (upper[2] - lower[2]) < 1.0:
        raise ValueError("flutes cannot keep shank above stock for the full-depth +Z setup")
    return {
        "operation": "titan-pilot",
        "step_path": str(step_path),
        "expected_sha256": E.TITAN_M1_SHA256,
        "reference_cylinder_in": 0.195,
        "output_dir": str(output_dir.resolve()),
        "geometry": {
            "stock": {"min_mm": list(lower), "max_mm": list(upper)},
            "source_to_work_mm": list(SOURCE_TO_WORK_MM),
            "grid": {
                "shape": list(grid.shape), "origin_mm": list(grid.origin_mm),
                "voxel_mm": grid.voxel_mm,
            },
            "tool": dataclasses.asdict(nominal_tool()),
            "hole_centre_source_xy_mm": list(HOLE_SOURCE_XY_MM),
            "expected_hole_radii_mm": list(HOLE_RADII_MM),
            "nominal_feed": {
                "start_work_mm": [*NOMINAL_XY_WORK_MM, APPROACH_Z_MM],
                "end_work_mm": [*NOMINAL_XY_WORK_MM, NOMINAL_END_Z_MM],
            },
        },
    }


def _checked_sampled_values(source: C.GeometrySource) -> np.ndarray:
    """Read the exact hash-declared array used for the grid exclusion count."""
    data = Path(source.path).read_bytes()
    if C.sha256_bytes(data) != source.sha256:
        raise ValueError(f"sampled source {source.path} has a SHA-256 mismatch")
    values = np.load(BytesIO(data), allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.dtype.kind not in "fiu":
        raise ValueError("sampled task source must be one real numeric NumPy array")
    if values.shape != source.grid.shape:
        raise ValueError(
            f"sampled source shape {values.shape} != declared GridSpec {source.grid.shape}"
        )
    values = np.ascontiguousarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("sampled task source includes non-finite values")
    return values


def bore_profile_from_native(native: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the backend's nominal void specification, not a hardcoded profile."""
    operation = native["operation_geometry"]
    shift = np.asarray(operation["source_to_work_mm"], dtype=np.float64)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError("source-to-work translation must be three finite millimetres")
    return np.asarray(operation["hole_centre_source_xy_mm"], dtype=np.float64), shift, np.asarray(
        operation["exclusion"]["nominal_void_profile"], dtype=object
    )


def _expected_bore_mask(grid: C.GridSpec, initial: np.ndarray,
                        native: Mapping[str, Any], profiles: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Nominal plug ∩ source-negative void, independently of field-inferred counts."""
    centre, shift, _ = bore_profile_from_native(native)
    exclusion = native["operation_geometry"]["exclusion"]
    plug_radius = float(exclusion["plug_radius_mm"])
    if centre.shape != (2,) or not np.isfinite(centre).all():
        raise ValueError("bore centre must be two finite source-frame millimetres")
    if not math.isfinite(plug_radius) or plug_radius <= 0:
        raise ValueError("plug radius must be finite and positive")
    xyz = grid_centres(grid)
    source_x = xyz[0] - shift[0]
    source_y = xyz[1] - shift[1]
    source_z = xyz[2] - shift[2]
    radial = np.hypot(source_x - centre[0], source_y - centre[1])
    plug_z_min = float(exclusion["plug_min_source_z_mm"] + shift[2])
    plug_z_max = float(exclusion["plug_max_source_z_mm"] + shift[2])
    if not (math.isfinite(plug_z_min) and math.isfinite(plug_z_max) and plug_z_min < plug_z_max):
        raise ValueError("plug z-range must be finite and increasing")
    nominal_void = np.zeros(grid.shape, dtype=bool)
    previous_face_upper = None
    if len(profiles) != 2:
        raise ValueError("bounded bore cross-check requires two nominal profile segments")
    for face in profiles:
        raw_z_min = float(face["source_z_min_mm"] + shift[2])
        raw_open = float(face["source_opening_z_mm"] + shift[2])
        radius = float(face["radius_mm"])
        if not all(math.isfinite(value) and value > 0 for value in (radius,)):
            raise ValueError("bore radii must be finite and positive")
        if radius >= plug_radius:
            raise ValueError("bore exclusion requires a larger encompassing plug")
        if (
            not math.isfinite(raw_z_min)
            or not math.isfinite(raw_open)
            or raw_open <= raw_z_min
        ):
            raise ValueError("nominal void opening must exceed its finite lower z bound")
        z_min = float(face["source_z_min_mm"]) + shift[2]
        opening = float(face["source_opening_z_mm"]) + shift[2]
        face_range = face["face_z_range_mm"]
        if len(face_range) != 2 or not all(math.isfinite(float(v)) for v in face_range):
            raise ValueError("native face z-range must contain two finite bounds")
        face_upper = float(face_range[1]) + shift[2]
        if previous_face_upper is not None and abs(z_min - previous_face_upper) > 1e-6:
            raise ValueError("nominal void depth segments are not contiguous")
        if z_min < plug_z_min - 1e-9 or opening > plug_z_max + 1e-9:
            raise ValueError("nominal void exceeds the recorded plug z-range")
        if face_upper - z_min <= 0 or opening - z_min <= 0:
            raise ValueError("nominal void has invalid native depth bounds")
        native_common = float(face["nominal_cylinder_common_with_original_part_mm3"])
        if not math.isfinite(native_common) or abs(native_common) > 1e-6:
            raise ValueError("nominal void is not source-negative by the native backend proof")
        nominal_void |= (source_z >= z_min) & (source_z <= opening) & (radial <= radius)
        previous_face_upper = face_upper
    expected = initial & (radial <= plug_radius)
    expected &= (source_z >= plug_z_min) & (source_z <= plug_z_max) & nominal_void
    return expected


def independent_bore_crosscheck(native: Mapping[str, Any], grid: C.GridSpec,
                                stock_field: np.ndarray, part_values: np.ndarray,
                                target_values: np.ndarray, *,
                                include_perturbations: bool = False) -> dict[str, Any]:
    """Compare an independent nominal bore raster with actual sampled P and T.

    The nominal mask is retained source-negative void inside the measured plug.
    ``expected_mask`` is an ephemeral NumPy value for tests; the remaining report
    is JSON-safe. It is not a replacement for the full native P or T geometry.
    """
    part = np.asarray(part_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    stock = np.asarray(stock_field, dtype=np.float64)
    if part.shape != grid.shape or target.shape != grid.shape or stock.shape != grid.shape:
        raise ValueError("bore cross-check arrays must share the declared physical grid")
    if not (np.isfinite(part).all() and np.isfinite(target).all() and np.isfinite(stock).all()):
        raise ValueError("bore cross-check requires finite sampled fields")
    original = part <= 0.0
    operation = target <= 0.0
    initial = stock <= 0.0
    if np.any(original & ~operation):
        raise E.CADValidationError("original part is not a subset of the operation target")
    exclusion = native["operation_geometry"]["exclusion"]
    profiles = exclusion["nominal_void_profile"]
    expected_mask = _expected_bore_mask(grid, initial, native, profiles)
    actual_mask = initial & operation & ~original
    mismatch = np.argwhere(expected_mask != actual_mask)
    first_radius = float(profiles[0]["radius_mm"])
    last_radius = float(profiles[-1]["radius_mm"])
    plug_radius = float(exclusion["plug_radius_mm"])
    pitch = float(grid.voxel_mm)
    result: dict[str, Any] = {
        "schema": "independent-titan-bore-raster/1",
        "definition": "initial & plug & nominal source-negative void vs actual initial & T & ~P",
        "expected_mask": expected_mask,
        "expected_cells": int(expected_mask.sum()),
        "actual_cells": int(actual_mask.sum()),
        "expected_volume_mm3": float(expected_mask.sum() * pitch ** 3),
        "actual_volume_mm3": float(actual_mask.sum() * pitch ** 3),
        "mismatched_cells": int(len(mismatch)),
        "sample_mismatched_indices": [list(map(int, row)) for row in mismatch[:16]],
        "sample_mismatched_cell_centres_mm": [
            list(grid.cell_centre_mm(*map(int, row))) for row in mismatch[:8]
        ],
        "nominal_void_profile": [
            {key: value for key, value in face.items() if key != "nominal_cylinder_common_with_original_part_mm3"}
            for face in profiles
        ],
        "source_negative_common_mm3_max": max(
            abs(float(face["nominal_cylinder_common_with_original_part_mm3"])) for face in profiles
        ),
        "same_physical_grid": {"shape": list(grid.shape), "origin_mm": list(grid.origin_mm),
                               "voxel_mm": pitch, "frame": grid.frame.value},
        "resolution": {
            "small_bore_diameter_mm": 2 * first_radius,
            "small_bore_cells_across_diameter": 2 * first_radius / pitch,
            "counterbore_diameter_mm": 2 * last_radius,
            "counterbore_cells_across_diameter": 2 * last_radius / pitch,
            "plug_diameter_mm": 2 * plug_radius,
            "plug_cells_across_diameter": 2 * plug_radius / pitch,
            "small_bore_pitch_for_eight_cells_mm": 2 * first_radius / 8,
            "small_bore_under_resolved": 2 * first_radius / pitch < 8,
            "note": "span ratios, not a guarantee of feature accuracy or minimum occupied-cell count",
        },
        "nominal_mask_match": int(len(mismatch)) == 0,
    }
    if include_perturbations:
        perturbations = []
        for label, radius in (("primary_radius_0.875_mm", 0.875),
                              ("primary_radius_times_1.25", 1.25 * first_radius)):
            variant = [dict(face) for face in profiles]
            variant[0]["radius_mm"] = radius
            variant_mask = _expected_bore_mask(grid, initial, native, variant)
            difference = int(np.count_nonzero(variant_mask != actual_mask))
            perturbations.append({
                "name": label, "radius_mm": radius,
                "expected_cells": int(variant_mask.sum()),
                "mismatched_cells": difference,
                "actual_mask_mismatch_rejected": difference > 0,
                "detectable_at_this_pitch": difference > 0,
                "scope": "counterfactual analytic perturbation, not modified STEP geometry or a native common-volume proof",
            })
        result["perturbations"] = perturbations
    return result


def build_task(part_source: C.GeometrySource, target_source: C.GeometrySource,
               native_excluded_volume_mm3: Optional[float] = None, *,
              independent_bore_check: Optional[Mapping[str, Any]] = None) -> C.TaskSpec:
    """Use grid-resolved exclusion; native continuous volume is diagnostic only."""
    if part_source.kind != "sampled" or target_source.kind != "sampled":
        raise ValueError("the real Titan pilot must retain its sampled native geometry")
    if part_source.frame != C.Frame.WORK or target_source.frame != C.Frame.WORK:
        raise ValueError("Titan sampled sources must use the work frame")
    if part_source.grid != target_source.grid:
        raise ValueError("original part and operation target must share one explicit physical grid")
    grid = part_source.grid
    if grid is None or grid.frame != C.Frame.WORK:
        raise ValueError("Titan sources require physical work-frame grids")
    lower, upper = stock_bounds()
    bound_lo, bound_hi = grid.bounds_mm
    if any(
        abs(actual - expected) > 1e-9
        for pair in zip((bound_lo, bound_hi), (lower, upper))
        for actual, expected in zip(*pair)
    ):
        raise ValueError("Titan grid bounds disagree with the declared top-centred stock")
    if (
        native_excluded_volume_mm3 is not None
        and (
            not math.isfinite(native_excluded_volume_mm3)
            or native_excluded_volume_mm3 <= 0
        )
    ):
        raise ValueError("native excluded-volume diagnostic must be positive and finite when supplied")
    part_values = _checked_sampled_values(part_source)
    target_values = _checked_sampled_values(target_source)
    grid_counts = E.field_statistics(
        box_signed_field(grid, lower, upper), part_values, target_values, grid.voxel_mm,
    )
    # This exact cell count is the declared quantity under docs/contracts.md.
    # The continuous Boolean volume never picks or broadens the zero tolerance.
    excluded_volume_mm3 = float(grid_counts["excluded_volume_voxel_estimate_mm3"])
    source_params = {
        "backend": "cadquery-ocp==7.7.2 isolated native STEPControl_Reader",
        "source_model_unit": "mm",
        "declared_input_unit": "inch",
        "source_sha256": "sha256:" + E.TITAN_M1_SHA256,
        "source_relative_path": "parts/Titan-M1.step",
        "source_to_work_mm": list(SOURCE_TO_WORK_MM),
        "field": "signed native point-to-boundary distance; BRepClass3d IN/ON negative",
    }
    part_source = dataclasses.replace(part_source, params=source_params)
    target_source = dataclasses.replace(target_source, params={
        **source_params,
        "construction": "native B-rep P UNION selected small-hole cutter-radius exclusion plug",
        "not_an_analytic_part_replacement": True,
        "grid_declared_excluded_volume_mm3": excluded_volume_mm3,
        "native_excluded_volume_mm3_diagnostic": native_excluded_volume_mm3,
        "excluded_volume_convention": "initial & T & ~P cell-count at task-grid pitch, tolerance 0",
        "independent_bore_verification": None if independent_bore_check is None else {
            key: independent_bore_check[key]
            for key in (
                "schema", "definition", "nominal_void_profile", "resolution",
                "expected_cells", "actual_cells", "mismatched_cells",
                "nominal_mask_match", "perturbations",
            )
            if key in independent_bore_check
        },
    })
    stock = C.GeometrySource(kind="analytic", params={
        "primitive": "box", "min_mm": list(lower), "max_mm": list(upper),
    })
    transform = C.Transform.translation(WORK_TO_MACHINE_MM)
    return C.TaskSpec(
        task_id="external-titan-m1-single-tool-plusz-roughing-v1",
        family="external_step_roughing", split=C.Split.EXTERNAL,
        source=C.SourceInfo(
            origin="local:parts/Titan-M1.step", license=PROVENANCE_LICENSE,
            checksums={"parts/Titan-M1.step": "sha256:" + E.TITAN_M1_SHA256},
            notes=(
                "Local-only unrecorded-provenance input. One cutter-radius exclusion "
                "for the measured narrow bore. No vendor dimension, general machinability "
                "or redistribution claim; no performance-based exclusion or dataset-membership freeze."
            ),
        ),
        length_unit="mm", stock=stock, original_part=part_source,
        operation_target=C.OperationTarget(
            geometry=target_source,
            excluded_volume_mm3=float(excluded_volume_mm3),
            exclusion_reason=(
                "For this one-tool +Z roughing operation, retain the selected through-hole/"
                "counterbore material: its measured native cylindrical radii 1.0287 and "
                "2.38125 mm are smaller than the 3.175 mm flat cutter radius. A native "
                "additive Boolean fills that narrow hole rather than claiming the selected "
                "tool can reproduce it. Full-depth flutes exceed the stock depth; this "
                "nominal setup keeps shank/holder above stock, and no fixture solids are "
                "modelled. Only this declared feature is excluded; other features are not "
                "certified accessible, and the nominal plunge is not a complete finishing path. "
                f"Declared excluded volume is grid-resolved: "
                f"{grid_counts['excluded_cells_initial_and_T_not_P']} cells × "
                f"{grid.voxel_mm}³ = {excluded_volume_mm3} mm³, with zero tolerance. "
                "The continuous/native excluded volume is diagnostic only and is not used "
                "as the declared volume or as a reason to relax that tolerance."
            ),
        ),
        grid=grid, work_to_machine=transform,
        initial_state=C.MachineState(
            position_mm=transform.apply((0.0, 0.0, SAFE_Z_MM)), frame=C.Frame.MACHINE,
        ),
        tool=nominal_tool(),
        limits=C.MachineLimits(
            max_feed_mm_s=20.0, rapid_mm_s=100.0, max_accel_mm_s2=200.0,
            envelope_min_mm=(0.0, 0.0, -350.0),
            envelope_max_mm=(400.0, 250.0, -200.0),
        ),
        tolerances=C.Tolerances(
            completion_residual_mm3=10.0, gouge_volume_mm3=0.0,
            local_surface_error_mm=grid.voxel_mm, clearance_mm=1.0,
            rapid_penetration_mm=0.0,
        ),
    )


def nominal_motion(task: C.TaskSpec) -> C.MotionPlan:
    """Exact-stop approach, ONE downward central plunge, then retract."""
    m = task.work_to_machine.apply
    points = [
        task.initial_state.position_mm,
        m((*NOMINAL_XY_WORK_MM, APPROACH_Z_MM)),
        m((*NOMINAL_XY_WORK_MM, NOMINAL_END_Z_MM)),
        m((*NOMINAL_XY_WORK_MM, SAFE_Z_MM)),
        task.initial_state.position_mm,
    ]
    kinds = (C.SegmentKind.RAPID, C.SegmentKind.FEED, C.SegmentKind.RAPID, C.SegmentKind.RAPID)
    tags = ("approach", "plunge", "retract", "retract")
    segments = []
    for index, (kind, tag) in enumerate(zip(kinds, tags)):
        start, end = points[index], points[index + 1]
        length = math.dist(start, end)
        speed = task.limits.rapid_mm_s if kind == C.SegmentKind.RAPID else FEED_MM_S
        segments.append(C.Segment(
            id=f"titan-{index:02d}", kind=kind, start_mm=start, end_mm=end,
            duration_s=length / speed,
            feed_mm_s=None if kind == C.SegmentKind.RAPID else FEED_MM_S,
            tag=tag,
        ))
    plan = C.MotionPlan(
        frame=C.Frame.MACHINE, initial_state=task.initial_state,
        segments=tuple(segments), timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
        provenance="explicit nominal central plunge, not optimized CAM or complete finishing",
    )
    plan.validate_for_task(task)
    return plan


def _write_bytes(root: Path, name: str, content: bytes) -> None:
    path = root / name
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(content)


def _source_metadata(source: C.GeometrySource) -> dict[str, Any]:
    return {
        "kind": source.kind, "path": source.path, "sha256": source.sha256,
        "grid": {
            "shape": list(source.grid.shape), "origin_mm": list(source.grid.origin_mm),
            "voxel_mm": source.grid.voxel_mm, "frame": source.grid.frame.value,
            "sign": "negative_inside", "cell_membership": "phi <= 0",
        },
        "params": source.params,
    }


def run_pilot(output_dir: Optional[Path] = None, *, interpreter: Optional[Path] = None,
              step_path: Path = SOURCE_STEP, timeout_s: float = 600.0) -> dict[str, Any]:
    """Execute the bounded native pilot and retain all supported/unsupported outcomes."""
    if output_dir is None:
        bulk = PROJECT_ROOT / "data/external"
        bulk.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="titan-m1-", dir=bulk)).resolve()
    else:
        root = Path(output_dir).resolve()
        if root.exists():
            raise FileExistsError(f"choose a NEW Titan pilot output directory: {root}")
        root.mkdir(parents=True)
    grid = pilot_grid()
    # This one call validates native source, target and sweep before returning
    # any raw fields. A zero subprocess exit is never itself geometric evidence.
    native = E.cad_request(geometry_request(root / "native", step_path=step_path),
                           interpreter=interpreter, timeout_s=timeout_s)
    operation_dir = root / "native"
    fields = E.load_native_fields(operation_dir, grid.shape)
    native_sweep = native["native_nominal_sweep"]
    if not (native_sweep["part_no_gouge_supported"] and native_sweep["operation_target_no_gouge_supported"]):
        raise E.CADValidationError(
            "nominal sweep intersects retained geometry: "
            f"P={native_sweep['original_part_common_volume_mm3']} mm^3, "
            f"T={native_sweep['operation_target_common_volume_mm3']} mm^3"
        )
    stock_field = box_signed_field(grid, *stock_bounds())
    sampled = E.field_statistics(
        stock_field, fields["part"]["values"], fields["operation_target"]["values"],
        grid.voxel_mm,
    )
    native_excluded = native["operation_geometry"]["exclusion"]["native_excluded_volume_mm3"]
    bore_check = independent_bore_crosscheck(
        native, grid, stock_field, fields["part"]["values"],
        fields["operation_target"]["values"], include_perturbations=True,
    )
    if bore_check["mismatched_cells"]:
        raise E.CADValidationError(
            "independent nominal bore raster disagrees with sampled exclusion: "
            f"{bore_check['mismatched_cells']} cells, expected={bore_check['expected_cells']}, "
            f"actual={bore_check['actual_cells']}; "
            f"first={bore_check['sample_mismatched_indices'][:4]}"
        )
    task = build_task(
        C.GeometrySource(
            kind="sampled", path=str(fields["part"]["path"]),
            sha256=fields["part"]["sha256"], grid=grid,
        ),
        C.GeometrySource(
            kind="sampled", path=str(fields["operation_target"]["path"]),
            sha256=fields["operation_target"]["sha256"], grid=grid,
        ),
        native_excluded,
        independent_bore_check=bore_check,
    )
    declared_excluded = task.operation_target.excluded_volume_mm3
    if abs(bore_check["expected_volume_mm3"] - declared_excluded) > 1e-12 * max(1.0, abs(declared_excluded)):
        raise E.CADValidationError(
            "independent bore volume disagrees with grid-resolved declaration: "
            f"expected={bore_check['expected_volume_mm3']}, declared={declared_excluded}"
        )
    bore_check["task_declared_excluded_volume_mm3"] = declared_excluded
    bore_check["native_excluded_volume_diagnostic_only_mm3"] = native_excluded
    motion = nominal_motion(task)
    initial = stock_field <= 0
    original = fields["part"]["values"] <= 0
    target = fields["operation_target"]["values"] <= 0
    evaluations = {}
    for dialect in (C.Dialect.RS274, C.Dialect.HAAS):
        nc = post_motion_plan(motion, task, dialect=dialect)
        nc_bytes = nc.encode("utf-8")
        parsed = parse_motion_plan(nc, task, dialect=dialect)
        round_trip = compare_line_motion(motion, parsed)
        production = evaluate_posted_gcode(task, nc_bytes, dialect, {
            "subdivisions": 128, "expected_nc_hash": C.nc_hash(nc_bytes),
        })
        if (
            production.coverage != C.Coverage.COMPLETE
            or production.geometry_pass is not False
            or production.metrics["residual_volume_mm3"].value is None
            or production.metrics["residual_volume_mm3"].value <= 0
        ):
            raise AssertionError(
                "sampled production result did not retain complete coverage and the "
                "expected incomplete-roughing geometry outcome: "
                f"{production.coverage} / {production.geometry_pass}"
            )
        declared = task.operation_target.excluded_volume_mm3
        measured_excluded = production.metrics["excluded_volume_mm3"].value
        if measured_excluded is None or measured_excluded != declared:
            raise AssertionError("grid-resolved exclusion changed during posted evaluation")
        reference = run_reference(task, parsed, {
            "max_step_mm": grid.voxel_mm / 4,
            "compute_bounds": True, "safety_screen": True,
        })
        material_metrics = None
        operational_geometry_pass = None
        if reference.removed is not None and reference.remaining is not None:
            if reference.removed.shape != grid.shape or reference.remaining.shape != grid.shape:
                raise AssertionError("independent reference grid shape differs from the task grid")
            metrics = grid_material_metrics(
                reference.remaining, target, initial, grid.voxel_mm,
                original_part_mask=original, surface_origin_mm=grid.origin_mm,
            )
            material_metrics = {
                name: {"value": metric.value, "unit": metric.unit,
                       "null_reason": metric.null_reason}
                for name, metric in metrics.items()
            }
            operational_geometry_pass = C.geometry_pass_from_metrics(metrics, task.tolerances)
        evaluations[dialect.value] = {
            "nc": nc, "nc_hash": C.nc_hash(nc_bytes),
            "post_round_trip": round_trip,
            "production": production,
            "reference": reference,
            "grid_material_metrics": material_metrics,
            "operational_geometry_pass_from_grid": operational_geometry_pass,
        }

    native_sweep = native["native_nominal_sweep"]
    files = {
        "task.json": C.dumps(task).encode("utf-8"),
        "motion_plan.json": C.dumps(motion).encode("utf-8"),
        "native_import_and_sweep.json": json.dumps(native, indent=2, sort_keys=True).encode("utf-8"),
        "grid_statistics.json": json.dumps(sampled, indent=2, sort_keys=True).encode("utf-8"),
        "independent_bore_crosscheck.json": (
            json.dumps({key: value for key, value in bore_check.items() if key != "expected_mask"},
                       indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        ),
        "sources.json": json.dumps({
            "original_part": _source_metadata(task.original_part),
            "operation_target": _source_metadata(task.operation_target.geometry),
            "stock": {"kind": "analytic box", "bounds_mm": stock_bounds(),
                      "same_grid_and_units": True},
        }, indent=2, sort_keys=True).encode("utf-8"),
    }
    for name, record in evaluations.items():
        record["production_bytes"] = C.dumps(record["production"]).encode("utf-8")
        files.update({
            f"{name}.nc": record["nc"].encode("utf-8"),
            f"production_{name}.json": record["production_bytes"],
            f"reference_{name}.json": C.dumps(record["reference"].result).encode("utf-8"),
            f"independent_{name}.json": json.dumps({
                "coverage": record["reference"].result.coverage.value,
                "safety_pass_stock_chronology_not_a_certificate": record["reference"].result.safety_pass,
                "reference_geometry_pass": record["reference"].result.geometry_pass,
                "grid_material_metrics": record["grid_material_metrics"],
                "operational_geometry_pass_from_grid": record["operational_geometry_pass_from_grid"],
                "native_one_feed_common_volume_mm3": native_sweep["stock_common_volume_mm3"],
                "post_round_trip": record["post_round_trip"],
            }, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"),
        })
    files["nominal_summary.json"] = json.dumps({
        "status": "incomplete",
        "reason": (
            "One roughing plunge, not a complete finishing path. Production sampled "
            "geometry is complete at the exact task grid and reports this partial "
            "operation as geometry false; no efficiency or physical machining-safety "
            "claim is made."
        ),
        "single_tool_plus_z_only": True,
        "one_selected_hole_excluded": True,
        "independent_bore_crosscheck": {
            "definition": bore_check["definition"],
            "expected_cells": bore_check["expected_cells"],
            "actual_cells": bore_check["actual_cells"],
            "mismatched_cells": bore_check["mismatched_cells"],
            "nominal_mask_match": bore_check["nominal_mask_match"],
            "small_bore_cells_across_diameter": bore_check["resolution"]["small_bore_cells_across_diameter"],
            "small_bore_under_resolved": bore_check["resolution"]["small_bore_under_resolved"],
            "perturbations": bore_check["perturbations"],
        },
        "all_other_features_certified_accessible": False,
        "native_sweep_no_original_or_target_gouge": (
            native_sweep["part_no_gouge_supported"]
            and native_sweep["operation_target_no_gouge_supported"]
        ),
        "production": {
            name: {"coverage": record["production"].coverage.value,
                   "reasons": record["production"].coverage_reasons,
                   "safety_pass": record["production"].safety_pass,
                   "efficiency_eligible": record["production"].efficiency_eligible,
                   "geometry_pass": record["production"].geometry_pass}
            for name, record in evaluations.items()
        },
        "independent": {
            name: {"coverage": record["reference"].result.coverage.value,
                   "safety_pass_stock_chronology_not_a_certificate": record["reference"].result.safety_pass,
                   "operational_geometry_pass_from_grid": record["operational_geometry_pass_from_grid"]}
            for name, record in evaluations.items()
        },
        "sampling_and_nominal_are_not_frozen_benchmark_membership": True,
    }, indent=2, sort_keys=True).encode("utf-8")
    for name, data in files.items():
        _write_bytes(root, name, data)
    # Hash converted arrays and all artifact bytes, without copying the input STEP.
    artifact_names = ["part.npy", "operation_target.npy", *sorted(files)]
    hashes = {}
    for name in artifact_names:
        source = operation_dir / name if name.endswith(".npy") else root / name
        hashes[name] = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    _write_bytes(root, "hashes.json", json.dumps(hashes, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return {
        "status": "incomplete", "output_dir": str(root), "task": task,
        "native": native, "sampled": sampled, "evaluations": evaluations,
        "hashes": hashes,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="NEW output directory; default data/external/titan-m1-*")
    args = parser.parse_args(argv)
    try:
        result = run_pilot(args.output)
    except E.CADUnavailable as error:
        raise SystemExit(f"PILOT BLOCKED: {error}") from error
    print(json.dumps({
        "status": result["status"], "output_dir": result["output_dir"],
        "hashes": result["hashes"], "sampled": result["sampled"],
        "native_excluded_volume_mm3": result["native"]["operation_geometry"]["exclusion"]["native_excluded_volume_mm3"],
        "native_one_feed": result["native"]["native_nominal_sweep"],
        "independent": {
            name: {"coverage": record["reference"].result.coverage.value,
                   "safety_pass_stock_chronology_not_a_certificate": record["reference"].result.safety_pass,
                   "geometry_pass_from_grid": record["operational_geometry_pass_from_grid"]}
            for name, record in result["evaluations"].items()
        },
        "not_a_complete_finished_part_or_machining_certificate": True,
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
