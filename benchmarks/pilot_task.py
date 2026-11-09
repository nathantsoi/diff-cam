"""Create and independently validate one reachable dev pocket-with-island.

The stock is complete, bounded box stock. The operation target is a pocket
around a square island; an outer-wall fringe has a declared exclusion and
volume from a conservative access-corridor choice, not a proof of global
inaccessibility. The original part remains a separate geometric record. The seed is
an in-repo heuristic from ``cam.seeds``, not adaptive CAM. The
nominal program is executed through DeltaSpline, differentiable and hard stock
execution, posts, typed parsing, and ``verification.reference_carve``.

Original reference coverage computes stock motion chronology, not operation-
target residual/gouge from ``box_minus_pocket`` (currently deferred to T1.3).
Therefore completion is additionally evaluated by explicit cellwise part-mask
counts using the same physical grid; reference ``geometry_pass=None`` remains
visible rather than being converted into an invented passing result.
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
from cam.posts import post_motion_plan
from cam.seeds import pocket_with_island_seed, seed_deltas_mm
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
GRID_PITCH = 1.0
# Operation-target inset, not an invisible relaxation. This conservative
# line-only dev operation uses the largest assembly radius for its access
# corridor, including 0.5 mm additional margin. It is a declared reachability
# choice for this task, not a proof that every cutter/flute could not finish
# the original boundary. The original part remains separately recorded.
REACH_MARGIN_MM = 5.5
# Inset ONLY the X/Y walls; preserve the six-millimetre floor-to-top depth.
OP_POCKET_MIN = (
    POCKET_MIN[0] + REACH_MARGIN_MM,
    POCKET_MIN[1] + REACH_MARGIN_MM,
    POCKET_MIN[2],
)
OP_POCKET_MAX = (
    POCKET_MAX[0] - REACH_MARGIN_MM,
    POCKET_MAX[1] - REACH_MARGIN_MM,
    POCKET_MAX[2],
)
FRINGE_VOLUME_MM3 = (
    (POCKET_MAX[0] - POCKET_MIN[0]) * (POCKET_MAX[1] - POCKET_MIN[1])
    - (OP_POCKET_MAX[0] - OP_POCKET_MIN[0]) * (OP_POCKET_MAX[1] - OP_POCKET_MIN[1])
) * (POCKET_MAX[2] - POCKET_MIN[2])
SCHEMA_NAME = "box_minus_pocket"


def build_pilot_task(voxel_mm: float = GRID_PITCH) -> C.TaskSpec:
    """The one declared dev task with physically accessible outer clearance."""
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
    def part_source(pocket_min, pocket_max, fringe_volume):
        return C.GeometrySource(
            kind="analytic",
            params={
                "primitive": SCHEMA_NAME,
                "box_min_mm": list(STOCK_MIN), "box_max_mm": list(STOCK_MAX),
                "pocket_min_mm": list(pocket_min), "pocket_max_mm": list(pocket_max),
                "island_min_mm": list(ISLAND_MIN), "island_max_mm": list(ISLAND_MAX),
                "fringe_volume_mm3": fringe_volume,
            },
        )

    # Do not silently rewrite original_part into the inset operation target.
    part = part_source(POCKET_MIN, POCKET_MAX, 0.0)
    operation_part = part_source(OP_POCKET_MIN, OP_POCKET_MAX, FRINGE_VOLUME_MM3)
    return C.TaskSpec(
        task_id="dev-pocket-island-000-v1", family="pocket_island",
        split=C.Split.DEV,
        source=C.SourceInfo(
            origin="procedural:pilot-pocket-with-island@1",
            license="MIT",
            notes="Single bounded dev pocket with square island. No performance-based exclusions.",
        ),
        length_unit="mm",
        stock=stock,
        original_part=part,
        operation_target=C.OperationTarget(
            geometry=operation_part,
            excluded_volume_mm3=FRINGE_VOLUME_MM3,
            exclusion_reason=(
                "The outer-wall fringe between the 48x32 mm original pocket and "
                "37x21 mm inset is excluded for this dev operation. Its access "
                "corridor uses the 5 mm holder radius plus 0.5 mm extra margin "
                "conservatively; this is a declared task reachability choice, not "
                "a proof that the full pocket is inaccessible to every assembly."
            ),
        ),
        grid=grid,
        work_to_machine=C.Transform.translation((0.0, 0.0, 0.0)),
        initial_state=C.MachineState(position_mm=(0.0, 0.0, 10.0), frame=C.Frame.MACHINE),
        tool=C.ToolAssembly(
            cutter_kind=C.CutterKind.FLAT, cutter_radius_mm=2.0,
            flute_length_mm=10.0, shank_radius_mm=2.0,
            holder_bottom_mm=10.0, holder_radius_mm=5.0,
            holder_height_mm=32.0, tool_number=1,
        ),
        limits=C.MachineLimits(
            max_feed_mm_s=8.0, rapid_mm_s=80.0, max_accel_mm_s2=100.0,
            envelope_min_mm=(-40.0, -30.0, -20.0),
            envelope_max_mm=(40.0, 30.0, 60.0),
        ),
        tolerances=C.Tolerances(
            completion_residual_mm3=2.0,
            gouge_volume_mm3=0.0,
            local_surface_error_mm=0.5,
            clearance_mm=0.25,
            # Do not use the band to invent clearance through stock. Ordinary
            # retracts start vertically and rise above every retained surface.
            rapid_penetration_mm=0.0,
        ),
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


def source_fields(task: C.TaskSpec):
    """Exact analytic inputs for sampled stock and the inset operation target."""
    if task.stock.params.get("primitive") != "box":
        raise ValueError("pilot source_fields requires analytic bounded box stock")
    target = task.operation_target.geometry
    if target.params.get("primitive") != SCHEMA_NAME:
        raise ValueError("pilot source_fields requires the declared box-minus-pocket target")
    points = grid_points(task.grid)
    stock = box_field(
        points, task.stock.params["min_mm"], task.stock.params["max_mm"]
    )
    pocket = box_field(
        points, target.params["pocket_min_mm"], target.params["pocket_max_mm"]
    )
    island = box_field(
        points, target.params["island_min_mm"], target.params["island_max_mm"]
    )
    # First subtract the pocket from bounded stock, then UNION the retained
    # island back into that solid. max(stock, -pocket, island) would instead
    # remove all non-island outer stock and is not box-minus-pocket-with-island.
    part = np.minimum(np.maximum(stock, -pocket), island)
    # negative_inside and strict interior match the simulator cell occupancy.
    # These masks are disjoint even if a grid happens to hit a pocket boundary.
    required = (pocket < 0.0) & (island > 0.0) & (stock < 0.0)
    retained = (part < 0.0) & ~required
    return stock, part, required, retained


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
        output_dir = Path(output_dir)
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
    by_cell_removed = (sim.stock.to_numpy()[0] < 0) & ~(sim.stock.to_numpy()[-1] < 0)

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
        "target_metrics_status": "independent-grid_completion_counts; target SDF floor/island params declared in task.json",
        "reference_geometry_pass": None,
    }
    completion_pass = (
        target_residual_mm3 <= task.tolerances.completion_residual_mm3
        and unwanted_removed_mm3 <= task.tolerances.gouge_volume_mm3
    )
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
        result = replace(trace.result, nc_hash=nc_hash, dialect=dialect, motion_hash=None)
        evaluations[dialect.value] = {
            "nc_bytes": nc_bytes, "nc_hash": nc_hash,
            "eval": C.dumps(result).encode("utf-8"),
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
        "independent_evaluation": "verification.reference_carve (production posted evaluator not integrated yet)",
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
            "Cellwise operation-target residual and retained-part removal at the "
            "declared grid; independent reference geometry_pass remains null."
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
            "status": "not_integrated_pending_T1.3",
            "camotics_cross_check": "not_run_pending_T1.3",
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
        run_id="CAM-Base-dev-pocket-island-000-v1",
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
