"""Create and independently validate one reachable dev pocket-with-island.

The stock is complete, bounded box stock. The original part retains the full
pocket and island; only finite-cutter unreachable corner fillets are excluded
from the operation, with continuous and grid-resolved exclusion measurements.
The seed is an in-repository heuristic from ``cam.seeds``, not adaptive CAM.
The nominal seed is executed through DeltaSpline, differentiable and hard
stock execution, both posts and typed reparses. Reference chronology and
shared sampled-grid metrics are then cross-checked with the production
posted-path evaluator. ``geometry_pass`` comes from declared completion,
gouge and local-surface tolerances; unavailable or failing coverage is never
rewritten as a pass.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import taichi as ti

from algorithms.parameterization import DeltaSpline
from benchmarks import contracts as C
from cam.gcode_parser import parse_motion_plan
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import post_motion_plan
from cam.seeds import pocket_with_island_seed, seed_deltas_mm
from simulator import csg_metrics
from simulator.csg_simulator import CSGSimulatorDelta
from taichi.lang import impl as ti_impl
from verification.reference_carve import run_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOCK_MIN = (-30.0, -20.0, -12.0)
STOCK_MAX = (30.0, 20.0, 0.0)
POCKET_MIN = (-24.0, -16.0, -6.0)
POCKET_MAX = (24.0, 16.0, 0.0)
ISLAND_MIN = (-3.5, -3.5, -6.0)
ISLAND_MAX = (3.5, 3.5, 0.0)
GRID_PITCH = 0.25
# Full pocket walls. Only cutter-unreachable corner fillets are excluded;
# there is deliberately no whole-side inset/corridor in the revised task.
OP_POCKET_MIN = POCKET_MIN
OP_POCKET_MAX = POCKET_MAX
SCHEMA_NAME = "box_minus_pocket"
CORNER_RADIUS_PARAM = "pocket_unreached_corner_radius_mm"


def build_pilot_task(voxel_mm: float = GRID_PITCH) -> C.TaskSpec:
    """The full-pocket dev task with only finite-tool corner exclusions."""
    stock_size = tuple((hi - lo) / voxel_mm for lo, hi in zip(STOCK_MIN, STOCK_MAX))
    if any(int(round(length)) != length for length in stock_size):
        raise ValueError("pilot requires integral isotropic GridSpec dimensions")
    grid = C.GridSpec(
        shape=tuple(int(round(length)) for length in stock_size),
        origin_mm=STOCK_MIN, voxel_mm=voxel_mm, frame=C.Frame.WORK,
    )
    stock = C.GeometrySource(
        kind="analytic", params={"primitive": "box", "min_mm": list(STOCK_MIN),
                                  "max_mm": list(STOCK_MAX)},
    )
    original_part = C.GeometrySource(
        kind="analytic",
        params={
            "primitive": SCHEMA_NAME,
            "box_min_mm": list(STOCK_MIN), "box_max_mm": list(STOCK_MAX),
            "pocket_min_mm": list(POCKET_MIN), "pocket_max_mm": list(POCKET_MAX),
            "island_min_mm": list(ISLAND_MIN), "island_max_mm": list(ISLAND_MAX),
        },
    )
    tool = C.ToolAssembly(
        cutter_kind=C.CutterKind.FLAT, cutter_radius_mm=2.0,
        flute_length_mm=10.0, shank_radius_mm=2.0,
        holder_bottom_mm=10.0, holder_radius_mm=5.0,
        holder_height_mm=32.0, tool_number=1,
    )
    tolerances = C.Tolerances(
        completion_residual_mm3=2.0,
        gouge_volume_mm3=0.0,
        # 0.5 mm local error is twice the 0.25 mm cell-centre spacing,
        # not a tolerance below the resolved grid pitch.
        local_surface_error_mm=0.5,
        clearance_mm=0.25,
        # No rapid ever cuts or approaches stock: all below-stock transitions
        # are feed, and every high reposition stays strictly above stock top.
        rapid_penetration_mm=0.0,
    )
    operation_part = derive_reachable_target(stock, original_part, tool, grid,
                                             tolerances.clearance_mm)
    return C.TaskSpec(
        task_id="dev-pocket-island-000-v2", family="pocket_island",
        split=C.Split.DEV,
        source=C.SourceInfo(
            origin="procedural:pilot-pocket-with-island@2-reachable",
            license="MIT",
            notes="Full pocket with square island; only finite-tool inaccessible corner fillets excluded.",
        ),
        length_unit="mm",
        stock=stock,
        original_part=original_part,
        operation_target=C.OperationTarget(
            geometry=operation_part,
            excluded_volume_mm3=operation_part.params["continuous_excluded_volume_mm3"],
            exclusion_reason=(
                "Only four corner fillets unreachable by a 2 mm-radius flat cutter "
                "are excluded. At Z=-6 mm, holder/shank bottoms are 4 mm above "
                "stock top, with positive cell-clearance margin. No wall corridor inset."
            ),
        ),
        grid=grid,
        work_to_machine=C.Transform.translation((0.0, 0.0, 0.0)),
        initial_state=C.MachineState(position_mm=(0.0, 0.0, 10.0), frame=C.Frame.MACHINE),
        tool=tool,
        limits=C.MachineLimits(
            max_feed_mm_s=8.0, rapid_mm_s=80.0, max_accel_mm_s2=100.0,
            envelope_min_mm=(-40.0, -30.0, -20.0),
            envelope_max_mm=(40.0, 30.0, 60.0),
        ),
        tolerances=tolerances,
    )


def grid_points(grid: C.GridSpec):
    axes = [
        grid.origin_mm[a] + (np.arange(grid.shape[a]) + 0.5) * grid.voxel_mm
        for a in range(3)
    ]
    return np.meshgrid(*axes, indexing="ij")


def box_field(points, lower, upper):
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("a bounded pilot box requires three lower/upper coordinates")
    if any(
        not math.isfinite(lo) or not math.isfinite(hi) or lo > hi
        for lo, hi in zip(lower, upper)
    ):
        raise ValueError(f"inverted or non-finite box bounds: {lower!r} / {upper!r}")
    distances = [np.maximum(lower[a] - points[a], points[a] - upper[a]) for a in range(3)]
    positive = [np.maximum(distance, 0.0) for distance in distances]
    outside = np.sqrt(sum(component * component for component in positive))
    interior = np.minimum(np.maximum.reduce(distances), 0.0)
    return outside + interior


def _geometry_fields(stock, geometry, grid):
    """Return analytic stock, pocket, island and original-part fields in mm."""
    if stock.kind != "analytic" or stock.params.get("primitive") != "box":
        raise ValueError("pilot requires analytic bounded box stock")
    if geometry.kind != "analytic" or geometry.params.get("primitive") != SCHEMA_NAME:
        raise ValueError("pilot requires the declared box-minus-pocket geometry")
    points = grid_points(grid)
    stock_d = box_field(points, stock.params["min_mm"], stock.params["max_mm"])
    pocket_d = box_field(
        points, geometry.params["pocket_min_mm"], geometry.params["pocket_max_mm"]
    )
    island_d = box_field(
        points, geometry.params["island_min_mm"], geometry.params["island_max_mm"]
    )
    # Same material-sign convention as posted_evaluator._analytic_mask:
    # bounded box minus pocket, UNION the retained island.
    original = np.maximum(stock_d, -np.maximum(pocket_d, -island_d))
    return stock_d, pocket_d, island_d, original


def _corner_fillet_fields(stock, original_part, tool, grid, radius_mm):
    """Four original-pocket corner regions outside the finite cutter sweep."""
    if tool.cutter_kind != C.CutterKind.FLAT:
        raise ValueError("reachable pilot geometry currently requires a flat upright cutter")
    if radius_mm != tool.cutter_radius_mm:
        raise ValueError("fillet radius must be derived from, and match, tool.cutter_radius_mm")
    points = grid_points(grid)
    lower = original_part.params["pocket_min_mm"]
    upper = original_part.params["pocket_max_mm"]
    fields = []
    for sx in (0, 1):
        for sy in (0, 1):
            cx = upper[0] - radius_mm if sx else lower[0] + radius_mm
            cy = upper[1] - radius_mm if sy else lower[1] + radius_mm
            lo = (
                cx if sx else lower[0],
                cy if sy else lower[1],
                lower[2],
            )
            hi = (
                upper[0] if sx else cx,
                upper[1] if sy else cy,
                upper[2],
            )
            corner_box = box_field(points, lo, hi)
            # A cut centred at (cx,cy) reaches a disk of radius r. Retain
            # only the corner-box material outside that closed disk. Its exact
            # per-corner area is (1-pi/4)*r^2, not (1-pi/4) times the pocket.
            outside_disk = radius_mm - np.hypot(points[0] - cx, points[1] - cy)
            fields.append(np.maximum(corner_box, outside_disk))
    return fields


def _holder_floor_margin_mm(stock, original_part, tool, grid, clearance_mm):
    """Non-cutting assembly clearance above stock at the pocket floor."""
    pocket_floor = original_part.params["pocket_min_mm"][2]
    stock_top = stock.params["max_mm"][2]
    shank_height = max(0.0, tool.holder_bottom_mm - tool.flute_length_mm)
    lowest_noncutting_base = pocket_floor + (
        min(tool.holder_bottom_mm, tool.flute_length_mm)
        if shank_height > 0 else tool.holder_bottom_mm
    )
    above_stock = lowest_noncutting_base - stock_top
    allowance = clearance_mm + math.sqrt(3.0) * grid.voxel_mm / 2.0
    if above_stock < allowance:
        raise ValueError(
            "holder/shank no longer clears the stock at the pocket floor: "
            f"{above_stock} mm < qualification+cell margin {allowance} mm"
        )
    return {
        "bottom_at_floor_mm": lowest_noncutting_base,
        "stock_top_mm": stock_top,
        "above_stock_mm": above_stock,
        "qualification_and_cell_margin_mm": allowance,
        "shank_height_mm": shank_height,
    }


def derive_reachable_target(stock, original_part, tool, grid, clearance_mm):
    """Add only physically unreachable corner fillets to the original part.

    Wall access is not inset. At the floor the complete finite cutter can reach
    x_min+r..x_max-r and y_min+r..y_max-r, leaving exactly four quarter-circle
    complements in the original square pocket corners. Non-cutting shank and
    holder must actually clear the stock top there, including cell/qualification
    margin; no holder radius is used to manufacture a wall-wide exclusion.
    Grid and continuous exclusion measurements are fixed before trajectory
    execution. Sampling tolerance is their explicit absolute difference, not
    a performance- or outcome-dependent allowance.
    """
    if original_part.params.get("primitive") != SCHEMA_NAME:
        raise ValueError("derivation requires the unchanged box-minus-pocket original part")
    radius = float(tool.cutter_radius_mm)
    holder = _holder_floor_margin_mm(stock, original_part, tool, grid, clearance_mm)
    _, _, _, original = _geometry_fields(stock, original_part, grid)
    fillets = _corner_fillet_fields(stock, original_part, tool, grid, radius)
    original_mask = original <= 0.0
    target_mask = original_mask | np.logical_or.reduce([field <= 0.0 for field in fillets])
    if np.any(original_mask & ~target_mask):
        raise AssertionError("operation target must retain all original-part cells")
    excluded_cells = int(np.count_nonzero(target_mask & ~original_mask))
    grid_volume = float(excluded_cells * grid.voxel_mm ** 3)
    lo = original_part.params["pocket_min_mm"]
    hi = original_part.params["pocket_max_mm"]
    depth = hi[2] - lo[2]
    continuous_volume = 4.0 * (1.0 - math.pi / 4.0) * radius ** 2 * depth
    sampling_tolerance = abs(grid_volume - continuous_volume)
    params = dict(original_part.params)
    params.update({
        CORNER_RADIUS_PARAM: radius,
        "continuous_excluded_volume_mm3": continuous_volume,
        "grid_excluded_voxels": excluded_cells,
        "grid_excluded_volume_mm3": grid_volume,
        "grid_exclusion_tolerance_mm3": sampling_tolerance,
        "grid_exclusion_tolerance_policy": (
            "absolute difference between exact physical corner-fillet volume "
            "and the predeclared grid-centre exclusion mask; fixed before motion"
        ),
        "holder_clearance": holder,
        "accessibility": (
            "flat upright finite cutter on full original walls; union original "
            "part with four corner boxes outside the radius cutter disks"
        ),
    })
    return C.GeometrySource(kind="analytic", params=params, frame=C.Frame.WORK)


def original_part_field(task):
    """The unmodified original part's analytic field on the declared grid."""
    return _geometry_fields(task.stock, task.original_part, task.grid)[3]


def source_fields(task: C.TaskSpec):
    """Sampled stock and tool-reachable target, with finite corner exclusions."""
    target = task.operation_target.geometry
    if target.params.get("primitive") != SCHEMA_NAME:
        raise ValueError("operation target must retain the box-minus-pocket primitive")
    for key in ("box_min_mm", "box_max_mm", "pocket_min_mm", "pocket_max_mm",
                "island_min_mm", "island_max_mm"):
        if target.params.get(key) != task.original_part.params.get(key):
            raise ValueError(f"operation target changed original geometry: {key}")
    if CORNER_RADIUS_PARAM not in target.params:
        raise ValueError(f"missing derived {CORNER_RADIUS_PARAM}")
    stock_d, pocket_d, island_d, original = _geometry_fields(
        task.stock, task.original_part, task.grid
    )
    fillets = _corner_fillet_fields(
        task.stock, task.original_part, task.tool, task.grid,
        target.params[CORNER_RADIUS_PARAM],
    )
    part = np.minimum.reduce([original, *fillets])
    required = (pocket_d < 0.0) & (island_d > 0.0) & (stock_d < 0.0)
    required &= np.logical_and.reduce([field > 0.0 for field in fillets])
    retained = (part < 0.0) & ~required
    return stock_d, part, required, retained


def evaluate_sampled_geometry(task, sources, initial_mask, remaining_mask):
    """Shared T1.3a grid metrics for actual sampled simulation stock.

    Target and original-part masks deliberately include zero-distance surface
    cells, matching the production analytic evaluator. The declared constant
    physical exclusion is compared with the fixed grid mask using only the
    independently derived sampling policy recorded before nominal execution.
    """
    initial = np.asarray(initial_mask)
    remaining = np.asarray(remaining_mask)
    if initial.dtype != np.dtype(bool) or remaining.dtype != np.dtype(bool):
        raise TypeError("sampled stock metrics require boolean occupancy masks")
    if initial.shape != task.grid.shape or remaining.shape != task.grid.shape:
        raise ValueError("sampled stock masks do not match the declared task grid")
    source = sources["target"]
    if source.kind != "sampled":
        raise ValueError("target geometry evaluator requires an actual sampled GeometrySource")
    target_path = Path(source.path)
    target_bytes = target_path.read_bytes()
    if C.sha256_bytes(target_bytes) != source.sha256:
        raise RuntimeError("sampled target hash changed after immutable source declaration")
    payload = np.load(target_path, allow_pickle=False)
    if payload.shape != task.grid.shape or not np.all(np.isfinite(payload)):
        raise ValueError("sampled target does not match finite declared grid geometry")
    target = payload <= 0.0
    original = original_part_field(task) <= 0.0
    metric_revision = C.sha256_bytes(Path(csg_metrics.__file__).read_bytes())
    metrics = csg_metrics.compute_grid_metrics(
        remaining, target, initial, task.grid.voxel_mm,
        original_part_mask=original, surface_origin_mm=task.grid.origin_mm,
        declared_excluded_volume_mm3=task.operation_target.excluded_volume_mm3,
        exclusion_tolerance_mm3=task.operation_target.geometry.params[
            "grid_exclusion_tolerance_mm3"
        ],
    )
    return {
        "metrics": metrics,
        "geometry_pass": C.geometry_pass_from_metrics(metrics, task.tolerances),
        "metric_library_sha256": metric_revision,
        "target_mask": target,
        "original_part_mask": original,
        "excluded_grid_volume_mm3": metrics["excluded_volume_mm3"].value,
        "declared_continuous_excluded_volume_mm3": (
            task.operation_target.excluded_volume_mm3
        ),
        "grid_exclusion_tolerance_mm3": (
            task.operation_target.geometry.params["grid_exclusion_tolerance_mm3"]
        ),
    }


def production_settings(task):
    """Grid-coherent posted-path refinement policy for this flat-tool task.

    The evaluator applies its own next-power-of-two subdivision after this cap.
    For the 2 mm pilot cutter, max_step_mm=1 supplies at least 64 substeps for
    a 44 mm perimeter edge (four times the original nominal 8-interval test),
    rather than assuming that every long line is covered by only eight intervals.
    Volume/time uncertainty bounds remain enabled by the production defaults.
    """
    return {"max_step_mm": task.tool.cutter_radius_mm / 2.0}


def fit_parameterization(motion: C.MotionPlan, controls: int = 12) -> DeltaSpline:
    """Fit immutable CAM residuals, with a fixed P0. K is only 12 here."""
    deltas = np.asarray(seed_deltas_mm(motion), dtype=np.float64)
    controls = min(controls, len(deltas))
    return DeltaSpline(
        deltas, scale_mm=1.0, num_control_points=controls,
        anchor_end=False, initial_position_mm=motion.initial_state.position_mm,
    )


def compare_motion(left: C.MotionPlan, right: C.MotionPlan,
                   position_tol_mm: float = 0.002, time_tol_s: float = 1e-5):
    """Compare physical semantics, not provenance or newly assigned source IDs."""
    if left.frame != right.frame or left.initial_state != right.initial_state:
        raise AssertionError("initial state/frame changed through posting")
    if left.timing_rule != right.timing_rule or len(left.segments) != len(right.segments):
        raise AssertionError("timing rule or segment count changed through posting")
    max_position_error = 0.0
    max_time_error = 0.0
    semantic_tags = {"approach", "plunge", "ramp", "retract", "home"}
    tag_differences = []
    for a, b in zip(left.segments, right.segments):
        if a.kind != b.kind:
            raise AssertionError(f"segment kind changed: {a.kind} -> {b.kind}")
        distance = max(math.dist(a.start_mm, b.start_mm), math.dist(a.end_mm, b.end_mm))
        max_position_error = max(max_position_error, distance)
        # Parenthesize both booleans: "x is None != y" is a comparison chain,
        # not a comparison of feed-presence predicates.
        if (a.feed_mm_s is None) != (b.feed_mm_s is None):
            raise AssertionError("feed presence changed")
        if a.feed_mm_s is not None and not math.isclose(a.feed_mm_s, b.feed_mm_s, abs_tol=1e-6):
            raise AssertionError("segment feed changed")
        # Semantic tags are preserved and checked. Author-defined provenance
        # labels are recorded separately, not mistaken for geometry changes;
        # motion_plan.json is their canonical metadata representation.
        if a.tag in semantic_tags or b.tag in semantic_tags:
            same_physical_tag = a.tag == b.tag or (
                b.tag in semantic_tags and a.tag.endswith("-" + b.tag)
            )
            if not same_physical_tag:
                raise AssertionError(f"semantic tag changed {a.tag!r} -> {b.tag!r}")
        elif a.tag != b.tag:
            tag_differences.append({"segment": a.id, "seed_tag": a.tag, "posted_tag": b.tag})
        max_time_error = max(max_time_error, abs(a.duration_s - b.duration_s))
    if max_position_error > position_tol_mm or max_time_error > time_tol_s:
        raise AssertionError(
            f"NC semantic loss: positions {max_position_error} mm, timing {max_time_error} s"
        )
    return {
        "max_position_error_mm": max_position_error,
        "max_time_error_s": max_time_error,
        "semantic_tag_mismatches": 0,
        "provenance_tag_differences": tag_differences,
    }


def contact_boundary_diagnostics(task, motion, indices):
    """Classify raw stock-mask disagreements for this line-only pilot.

    ``verification.reference_carve`` samples finite stations; a continuous
    grazing contact can therefore be missed without changing its collision
    certificate. For constant-Z XY or vertical feed lines, compute each
    disputed cell centre's exact distance to the continuous capped-cylinder
    sweep independently of the production kernel. Only such contacts within
    the task's declared local-surface tolerance are classifiable. This is a
    boundary-sensitivity diagnostic, NOT operation-target geometry scoring or
    an independent general flat-cutter evaluator; arbitrary sloped feeds are
    rejected rather than hidden by a numerical tolerance.
    """
    indices = np.asarray(indices, dtype=np.int64).reshape((-1, 3))
    shape = np.asarray(task.grid.shape, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= shape):
        raise ValueError("contact diagnostic cell indices are outside the task grid")
    cells = [
        task.grid.origin_mm[a] + (indices[:, a] + 0.5) * task.grid.voxel_mm
        for a in range(3)
    ]
    best = np.full(len(indices), np.inf, dtype=np.float64)
    supported = task.tool.cutter_kind == C.CutterKind.FLAT
    radius = task.tool.cutter_radius_mm
    height = task.tool.flute_length_mm

    def signed_cylinder_distance(horizontal, vertical):
        horizontal_out = np.maximum(horizontal, 0.0)
        vertical_out = np.maximum(vertical, 0.0)
        outside = np.sqrt(horizontal_out**2 + vertical_out**2)
        inside = np.minimum(np.maximum(horizontal, vertical), 0.0)
        return outside + inside

    for segment in motion.segments:
        if not supported:
            break
        if segment.kind != C.SegmentKind.FEED:
            continue
        a = np.asarray(segment.start_mm, dtype=np.float64)
        b = np.asarray(segment.end_mm, dtype=np.float64)
        dx, dy, dz = b - a
        if abs(dz) <= 1e-9:
            denominator = dx * dx + dy * dy
            projection = (
                (cells[0] - a[0]) * dx + (cells[1] - a[1]) * dy
            ) / max(float(denominator), 1e-30)
            projection = np.clip(projection, 0.0, 1.0)
            horizontal = np.hypot(
                cells[0] - (a[0] + projection * dx),
                cells[1] - (a[1] + projection * dy),
            ) - radius
            vertical = np.maximum(a[2] - cells[2], cells[2] - (a[2] + height))
        elif abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
            horizontal = np.hypot(cells[0] - a[0], cells[1] - a[1]) - radius
            lower = min(a[2], b[2])
            upper = max(a[2], b[2]) + height
            vertical = np.maximum(lower - cells[2], cells[2] - upper)
        else:
            supported = False
            break
        distance = signed_cylinder_distance(horizontal, vertical)
        nearest = np.abs(distance) < np.abs(best)
        best[nearest] = distance[nearest]

    limit = task.tolerances.local_surface_error_mm
    supported = supported and bool(np.isfinite(best).all())
    outside = int(np.count_nonzero(np.abs(best) > limit)) if supported else len(indices)
    max_distance = float(np.max(np.abs(best))) if supported and len(best) else (
        0.0 if not len(best) else None
    )
    contact_only = supported and outside == 0
    return {
        "cell_disagreement_count": int(len(indices)),
        "scope_supported": supported,
        "contact_boundary_only": contact_only,
        "outside_local_tolerance_cells": outside,
        "max_abs_contact_distance_mm": max_distance,
        "local_boundary_tolerance_mm": limit,
        "scope": "constant-Z XY and vertical feed lines; exact continuous capped-cylinder contact distance",
        "status": (
            "identical" if len(indices) == 0 and supported else
            "boundary_contacts_only" if contact_only else
            "unresolved_contact_disagreement"
        ),
        "note": (
            "Finite-station reference removal and the continuous simulator can "
            "disagree at grazing contacts. This is not a raw-stock equality "
            "certificate and does not replace geometry_pass."
        ),
    }


def _git_provenance():
    diff = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "diff", "--binary", "HEAD"],
        capture_output=True, check=False,
    )
    if diff.returncode:
        raise RuntimeError(diff.stderr.decode("utf-8", errors="replace"))
    status = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        capture_output=True, check=False, text=True,
    )
    if status.returncode:
        raise RuntimeError(status.stderr)
    head = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True, check=True, text=True,
    ).stdout.strip()
    owned = ["cam/seeds.py", "benchmarks/pilot_task.py", "tests/task_suite_test.py"]
    dirty_hash = C.content_hash({"diff_sha256": C.sha256_bytes(diff.stdout), "owned": {
        path: C.sha256_bytes((PROJECT_ROOT / path).read_bytes())
        for path in owned if (PROJECT_ROOT / path).exists()
    }}) if status.stdout.strip() else None
    return head, dirty_hash


def run_nominal(output_dir=None):
    """Execute the one declared pilot and return its measurements and records."""
    task = build_pilot_task()
    motion = pocket_with_island_seed(task)
    parameterization = fit_parameterization(motion)
    zero = np.zeros((parameterization.num_control_points, 3), dtype=np.float64)
    cam_deltas = parameterization.forward(zero)
    if not np.array_equal(cam_deltas, np.asarray(seed_deltas_mm(motion))):
        raise AssertionError("zero-residual forward deltas are not nominal CAM deltas")
    expected_points = parameterization.positions(zero)
    expected_deltas = np.diff(expected_points, axis=0)
    if not np.allclose(expected_deltas, seed_deltas_mm(motion), atol=1e-12, rtol=0.0):
        raise AssertionError("zero-residual reconstructed path is not CAM motion")

    if output_dir is None:
        run_root = PROJECT_ROOT / "runs"
        run_root.mkdir(exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="pilot-pocket-", dir=run_root))
    else:
        output_dir = Path(output_dir).resolve()
        if output_dir.exists():
            raise FileExistsError(f"choose a NEW nominal output directory: {output_dir}")
        output_dir.mkdir(parents=True)
    stock_mm, part_mm, required, retained = source_fields(task)
    source_paths = {}
    for name, values in (("stock", stock_mm), ("target", part_mm)):
        path = output_dir / f"source_{name}.npy"
        np.save(path, values.astype(np.float64))
        source_paths[name] = C.GeometrySource(
            kind="sampled", path=str(path),
            sha256=C.sha256_bytes(path.read_bytes()), grid=task.grid,
        )

    return evaluate_nominal(
        task, motion, parameterization, zero, source_paths,
        required, retained, output_dir
    )


def evaluate_nominal(task, motion, parameterization, theta, source_paths,
                     required, retained, output_dir):
    if ti_impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, offline_cache=False)
    extent = np.asarray(STOCK_MAX) - np.asarray(STOCK_MIN)
    normalized_expected = (
        expected_positions_from_motion(motion) - np.asarray(STOCK_MIN)
    ) / extent
    sim = CSGSimulatorDelta(
        voxel_size_mm=task.grid.voxel_mm,
        stock_size_mm=extent,
        stock_box_origin_mm=STOCK_MIN,
        tool_start=normalized_expected[0].tolist(),
        max_steps=len(motion.segments), target_geometry=source_paths["target"],
        stock_geometry=source_paths["stock"],
        enforce_speed_limits=False,
        init_taichi=False,
    )
    sim.tool_radius[None] = task.tool.cutter_radius_mm
    sim.tool_height[None] = task.tool.flute_length_mm
    sim.holder_radius[None] = task.tool.holder_radius_mm
    sim.holder_height[None] = task.tool.holder_height_mm
    deltas = parameterization.forward(theta)
    normalized_deltas = np.asarray(deltas) / extent
    sim.tool_delta.from_numpy(np.asarray(normalized_deltas, dtype=np.float32))
    with ti.ad.Tape(loss=sim.loss):
        sim.forward(len(motion.segments) + 1)
    if not np.isfinite(float(sim.loss[None])) or not np.isfinite(sim.tool_delta.grad.to_numpy()).all():
        raise FloatingPointError("nominal training evaluation has non-finite loss or gradient")
    actual_positions = sim.tool_pos.to_numpy()[: len(motion.segments) + 1]
    np.testing.assert_allclose(
        actual_positions, normalized_expected, atol=2e-6, rtol=0.0,
        err_msg="training execution changed the exact-stop nominal path",
    )
    hard = sim.forward_hard(len(motion.segments) + 1, clip_speeds=False)
    assert hard is None
    stock_history = sim.stock.to_numpy()
    initial_stock = stock_history[0] < 0
    remaining_stock = stock_history[-1] < 0
    by_cell_removed = initial_stock & ~remaining_stock
    sampled_evaluation = evaluate_sampled_geometry(
        task, source_paths, initial_stock, remaining_stock
    )

    # Cellwise checks are explicit semantic expectations, not C.geometry_pass.
    remaining_required = required & ~by_cell_removed
    unwanted_retained = retained & by_cell_removed
    target_residual_mm3 = float(np.count_nonzero(remaining_required) * task.grid.voxel_mm ** 3)
    unwanted_removed_mm3 = float(np.count_nonzero(unwanted_retained) * task.grid.voxel_mm ** 3)
    completion = {
        "target_residual_mm3": target_residual_mm3,
        "retained_part_removed_mm3": unwanted_removed_mm3,
        "target_remaining_voxels": int(np.count_nonzero(remaining_required)),
        "retained_part_removed_voxels": int(np.count_nonzero(unwanted_retained)),
        "residual_tolerance_mm3": task.tolerances.completion_residual_mm3,
        "target_metrics_status": "cell counts plus shared T1.3a sampled-grid metrics",
        "reference_geometry_pass": None,
        "sampled_geometry_pass": sampled_evaluation["geometry_pass"],
        "sampled_grid_metrics": {
            name: json.loads(C.canonical_json(metric))
            for name, metric in sampled_evaluation["metrics"].items()
        },
        "sampled_exclusion": {
            "continuous_volume_mm3": task.operation_target.excluded_volume_mm3,
            "grid_volume_mm3": sampled_evaluation["excluded_grid_volume_mm3"],
            "sampling_tolerance_mm3": sampled_evaluation["grid_exclusion_tolerance_mm3"],
        },
    }
    completion_pass = sampled_evaluation["geometry_pass"] is True
    evaluations = {}
    motion_errors = {}
    for dialect in (C.Dialect.RS274, C.Dialect.HAAS):
        nc = post_motion_plan(motion, task, dialect=dialect)
        nc_bytes = nc.encode("utf-8")
        nc_hash = C.nc_hash(nc_bytes)
        parsed = parse_motion_plan(nc, task, dialect=dialect)
        motion_errors[dialect.value] = compare_motion(motion, parsed)
        trace = run_reference(task, parsed, {"subdivisions": 8})
        if trace.result.coverage != C.Coverage.COMPLETE or trace.result.safety_pass is not True:
            raise RuntimeError(
                f"nominal independent safety failed/blocked ({dialect.value}): "
                f"{trace.result.coverage} / {trace.result.safety_pass} {trace.result.coverage_reasons}"
            )
        lower_clearance = trace.result.metrics["min_clearance_lower_bound_mm"].value
        rapid_penetration = trace.result.metrics[
            "rapid_cutter_penetration_upper_bound_mm"
        ].value
        if (
            lower_clearance is None
            or lower_clearance < task.tolerances.clearance_mm
            or rapid_penetration is None
            or rapid_penetration > task.tolerances.rapid_penetration_mm
        ):
            raise RuntimeError(
                "nominal clearance/rapid-penetration qualification failed: "
                f"lower={lower_clearance} mm, rapid_upper={rapid_penetration} mm"
            )
        if trace.removed is None or trace.removed.shape != by_cell_removed.shape:
            raise RuntimeError("independent evaluator returned no stock-removal mask")
        cell_disagreement = np.logical_xor(by_cell_removed, trace.removed)
        contact_diagnostic = contact_boundary_diagnostics(
            task, parsed, np.argwhere(cell_disagreement)
        )
        if not contact_diagnostic["contact_boundary_only"]:
            raise RuntimeError(
                "raw stock-mask disagreement is outside the supported grazing-"
                f"contact scope: {contact_diagnostic}"
            )
        reference_result = replace(
            trace.result, nc_hash=nc_hash, dialect=dialect, motion_hash=None
        )
        production = evaluate_posted_gcode(
            task, nc_bytes, dialect,
            {
                **production_settings(task),
                "expected_nc_hash": nc_hash,
                "exclusion_tolerance_mm3": task.operation_target.geometry.params[
                    "grid_exclusion_tolerance_mm3"
                ],
            },
        )
        if (
            production.coverage != C.Coverage.COMPLETE
            or production.safety_pass is not True
            or production.geometry_pass is not True
        ):
            raise RuntimeError(
                f"nominal production posted evaluation failed/blocked ({dialect.value}): "
                f"coverage={production.coverage.value}, safety={production.safety_pass}, "
                f"geometry={production.geometry_pass}, reasons={production.coverage_reasons}, "
                f"residual={production.metrics['residual_volume_mm3'].value} mm^3, "
                f"gouge={production.metrics['gouge_volume_mm3'].value} mm^3, "
                f"max_surface={production.metrics['max_surface_error_mm'].value} mm"
            )
        production_clearance = production.metrics["min_clearance_lower_bound_mm"].value
        production_rapid = production.metrics[
            "rapid_cutter_penetration_upper_bound_mm"
        ].value
        if (
            production_clearance is None
            or production_clearance < task.tolerances.clearance_mm
            or production_rapid is None
            or production_rapid > task.tolerances.rapid_penetration_mm
        ):
            raise RuntimeError(
                f"production clearance qualification failed ({dialect.value}): "
                f"lower={production_clearance} mm, rapid={production_rapid} mm"
            )
        summary = {
            "status": "evaluated",
            "coverage": production.coverage.value,
            "safety_pass": production.safety_pass,
            "geometry_pass": production.geometry_pass,
            "residual_volume_mm3": production.metrics["residual_volume_mm3"].value,
            "gouge_volume_mm3": production.metrics["gouge_volume_mm3"].value,
            "max_surface_error_mm": production.metrics["max_surface_error_mm"].value,
            "excluded_volume_mm3": production.metrics["excluded_volume_mm3"].value,
            "min_clearance_lower_bound_mm": production_clearance,
            "rapid_penetration_upper_bound_mm": production_rapid,
        }
        evaluations[dialect.value] = {
            "nc_bytes": nc_bytes, "nc_hash": nc_hash,
            "eval": C.dumps(production).encode("utf-8"),
            "reference_eval": C.dumps(reference_result).encode("utf-8"),
            "production_summary": summary,
            "stock_cell_disagreement": contact_diagnostic,
        }
    if not completion_pass:
        raise RuntimeError(f"nominal pocket target/coverage failed: {completion}")

    return write_artifacts(
        output_dir, task, motion, parameterization, theta,
        source_paths, actual_positions, normalized_deltas, by_cell_removed,
        evaluations, completion, motion_errors,
    )


def _npy(values):
    import io

    stream = io.BytesIO()
    np.save(stream, np.asarray(values, dtype=np.float32))
    return stream.getvalue()


def expected_positions_from_motion(motion):
    return np.asarray([motion.initial_state.position_mm] + [segment.end_mm for segment in motion.segments])


def _source_metadata(source):
    return {
        "path": source.path,
        "sha256": source.sha256,
        "grid": {
            "shape": list(source.grid.shape),
            "origin_mm": list(source.grid.origin_mm),
            "voxel_mm": source.grid.voxel_mm,
            "sign_convention": source.grid.sign_convention,
            "frame": source.grid.frame.value,
        },
    }


def write_artifacts(root, task, motion, parameterization, theta, sources,
                    actual_positions, actual_deltas, removed, evaluations, completion,
                    motion_errors):
    root = Path(root)
    expected_mm = expected_positions_from_motion(motion)
    expected_normalized = (expected_mm - np.asarray(STOCK_MIN)) / (
        np.asarray(STOCK_MAX) - np.asarray(STOCK_MIN)
    )
    parameter_deltas = parameterization.forward(theta)
    np.testing.assert_array_equal(parameter_deltas, np.asarray(seed_deltas_mm(motion)))
    parameter_points = parameterization.positions(theta)
    reconstructed_delta_error_mm = float(np.max(
        np.abs(np.diff(parameter_points, axis=0) - parameter_deltas)
    ))
    if reconstructed_delta_error_mm > 1e-12:
        raise AssertionError("reconstructed path does not preserve the typed CAM deltas")
    np.testing.assert_array_equal(theta, np.zeros_like(theta))
    training_error_mm = float(np.max(
        np.abs(actual_positions - expected_normalized)
        * (np.asarray(STOCK_MAX) - np.asarray(STOCK_MIN))
    ))
    if training_error_mm > 0.002:
        raise AssertionError("training execution exceeds the 0.002 mm path tolerance")
    head, dirty_hash = _git_provenance()
    files = {}
    files["task.json"] = C.dumps(task).encode("utf-8")
    files["config.json"] = (json.dumps({
        "source_commit": head,
        "dirty_patch_hash": dirty_hash,
        "seed_provenance": motion.provenance,
        "parameterization": {
            "kind": "DeltaSpline",
            "scale_mm": parameterization._scale_mm,
            "control_points": parameterization.num_control_points,
            "anchor_end": False,
            "theta_is_all_zero": bool(np.all(np.asarray(theta) == 0.0)),
            "basis_sha256": C.content_hash(parameterization.basis.tolist()),
        },
        "execution": {
            "timing_rule": motion.timing_rule.value,
            "clip_speeds": False,
            "runtime_voxel_mm": task.grid.voxel_mm,
            "grid_shape": list(task.grid.shape),
            "stock_min_mm": list(STOCK_MIN), "stock_max_mm": list(STOCK_MAX),
            "note": "typed exact-stop constant-speed residual execution; no silent speed clipping",
        },
        "independent_evaluation": (
            "cam.posted_evaluator.evaluate_posted_gcode (primary posted evaluation); "
            "verification.reference_carve (independent chronology cross-check); "
            "simulator.csg_metrics (sampled-grid metric cross-check)"
        ),
    }, indent=2, sort_keys=True) + "\n").encode("utf-8")
    files["motion_plan.json"] = C.dumps(motion).encode("utf-8")
    files["history.json"] = (json.dumps([{
        "step": "nominal-zero-residual",
        "candidate": "CAM-Base",
        "theta_nonzero_count": int(np.count_nonzero(theta)),
        "training_execution": "differentiable forward_hard_crosscheck",
    }], indent=2) + "\n").encode("utf-8")
    files["metrics.json"] = (json.dumps({
        "completion": completion,
        "completion_metric_scope": (
            "Shared sampled-grid residual, gouge and local-surface metrics plus "
            "production posted-path evaluation. Independent reference "
            "geometry_pass remains unavailable by its stock-motion-only design."
        ),
        "reference_geometry_pass": None,
        "training_execution": {
            "max_position_error_mm": training_error_mm,
            "position_tolerance_mm": 0.002,
            "reconstructed_delta_error_mm": reconstructed_delta_error_mm,
            "reconstructed_delta_tolerance_mm": 1e-12,
            "differentiable_loss_gradients": "finite (verified before hard forward)",
            "speed_clipping": False,
        },
        "post_parse_equivalence": motion_errors,
        "raw_stock_cell_disagreement": {
            name: record.get("stock_cell_disagreement")
            for name, record in evaluations.items()
        },
        "production_posted_evaluator": {
            name: record.get("production_summary", {"status": "synthetic_unit_fixture"})
            for name, record in evaluations.items()
        },
        "camotics_cross_check": {
            "status": "not_run",
            "reason": "no external CAMotics result was obtained for this T2.2b run",
        },
    }, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    files["initial_trajectory.npy"] = _npy(expected_normalized)
    files["trajectory.npy"] = _npy(actual_positions)
    files["trajectory_deltas.npy"] = _npy(actual_deltas)
    files["training_positions.npy"] = _npy(actual_positions)
    files["sim_removed_mask.npy"] = _npy(removed.astype(np.float32))
    files["sources.json"] = (json.dumps({
        "stock": _source_metadata(sources["stock"]),
        "target": _source_metadata(sources["target"]),
    }, indent=2, sort_keys=True) + "\n").encode("utf-8")

    posted = []
    for name, dialect in (("rs274", C.Dialect.RS274), ("haas", C.Dialect.HAAS)):
        record = evaluations[dialect.value]
        files[f"{name}.nc"] = record["nc_bytes"]
        if "reference_eval" in record:
            files[f"reference_{name}.json"] = record["reference_eval"]
        eval_name = f"eval_posted_{name}.json"
        files[eval_name] = record["eval"]
        posted.append(C.PostedProgram(
            dialect=dialect, status=C.PostStatus.SUPPORTED,
            nc_hash=record["nc_hash"], eval_file=eval_name,
        ))
    index = {"schema_version": 1, "programs": {}}
    for entry in posted:
        key = entry.dialect.value
        index["programs"][key] = json.loads(C.canonical_json(entry))
    files["eval_posted.json"] = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode("utf-8")
    run = C.RunRecord(
        run_id="CAM-Base-dev-pocket-island-000-v2",
        task_hash=task.task_hash, status=C.RunStatus.COMPLETE,
        artifacts=tuple(sorted(files)), posted=tuple(posted),
        source_commit=head, dirty_patch_hash=dirty_hash,
    )
    files["run.json"] = C.dumps(run).encode("utf-8")
    root_files = sorted(files)
    root.joinpath("hashes.json").write_text(json.dumps({
        name: C.sha256_bytes(files[name]) for name in root_files
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, data in files.items():
        (root / name).write_bytes(data)
    # A separate index matches content but is not implicated in NC identity.
    result = {
        "run_id": run.run_id, "output_dir": str(root),
        "status": "complete", "run": run, "task": task, "motion": motion,
        "hashes": {name: C.sha256_bytes(files[name]) for name in root_files},
        "completion": completion,
        "evaluations": evaluations,
        "files": root_files + ["hashes.json"],
    }
    root.joinpath("nominal_result.json").write_text(json.dumps({
        "run_id": result["run_id"], "status": result["status"],
        "hashes": result["hashes"], "completion": completion,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="new, non-existing output directory; default creates a unique runs/pilot-pocket-* directory")
    args = parser.parse_args(argv)
    result = run_nominal(args.output)
    print(json.dumps({
        "run_id": result["run_id"], "output_dir": result["output_dir"],
        "hashes": result["hashes"], "completion": result["completion"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
