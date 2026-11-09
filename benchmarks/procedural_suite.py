"""Deterministic bootstrap T2.3 families with production-verified seeds.

Scope deliberately follows the available analytic stock and pocket primitives.
It does not pretend to generate free-form/cross-scale CAD solids, and suite
counts beyond one dev and one test case per family are not frozen yet. A case
may be rejected during generation for a physical/configuration reason, but
never because a solver or optimizer performed poorly.

Original part P is unchanged. Operation target T adds only unreachable corner
material from finite-radius cutter accessibility. Its normative excluded value
is the exact precomputed grid-cell exclusion at the task pitch. The continuous
fillets are diagnostics only, with zero evaluator-exclusion tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import json
import math
from pathlib import Path
import random
import subprocess

import numpy as np

from benchmarks import contracts as C
from benchmarks.pilot_task import (
    derive_reachable_target, expected_positions_from_motion, production_settings,
)
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import post_motion_plan
from cam.seeds import IslandSeedSpec, seed_deltas_mm, seed_for_task, seed_spec_for_task
from verification.reference_carve import run_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_VERSION = "t2.3-bootstrap/1"
GENERATOR_SEED = 133742
PITCH_MM = 0.25


@dataclass(frozen=True)
class Scenario:
    family: str
    split: str
    name: str
    stock: tuple[tuple[float, float, float], tuple[float, float, float]]
    pocket: tuple[tuple[float, float, float], tuple[float, float, float]]
    island: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    topology: str = "centred_box_pocket"


# Explicit schema variants, not a corpus silently curated from solver outcomes.
SCENARIOS = (
    Scenario(
        "facing_asymmetric", "dev", "asymmetric-lipped-face",
        ((-12.0, -8.0, -8.0), (10.0, 8.0, 0.0)),
        ((-8.0, -5.0, -0.5), (8.0, 5.0, 0.0)), topology="lipped_top_facing",
    ),
    Scenario(
        "facing_asymmetric", "test", "offset-shoulder-face",
        ((-16.0, -11.0, -8.0), (12.0, 9.0, 0.0)),
        ((-4.0, -7.0, -0.75), (8.0, 3.0, 0.0)), topology="offset_shouldered_top_facing",
    ),
    Scenario(
        "pocket_island", "dev", "oblong-square-island",
        ((-14.0, -11.0, -8.0), (14.0, 11.0, 0.0)),
        ((-11.0, -9.0, -6.0), (11.0, 9.0, 0.0)),
        ((-2.0, -2.0, -6.0), (2.0, 2.0, 0.0)), topology="oblong_with_square_island",
    ),
    Scenario(
        "pocket_island", "test", "offset-truncated-square-pad",
        ((-16.0, -11.0, -8.0), (16.0, 11.0, 0.0)),
        ((-12.0, -8.0, -5.0), (12.0, 8.0, 0.0)),
        ((0.5, -2.5, -5.0), (3.5, 0.5, -1.0)), topology="oblong_with_offset_step_pad",
    ),
    Scenario(
        "variable_width_depth", "dev", "narrow-deep-clearing",
        ((-8.0, -11.5, -12.0), (8.0, 11.5, 0.0)),
        ((-4.0, -7.0, -8.0), (4.0, 7.0, 0.0)), topology="narrow_deep_clearing",
    ),
    Scenario(
        "variable_width_depth", "test", "wide-shallow-clearing",
        ((-15.0, -9.0, -12.0), (15.0, 9.0, 0.0)),
        ((-8.0, -5.0, -3.0), (8.0, 5.0, 0.0)), topology="wide_shallow_clearing",
    ),
)


@dataclass(frozen=True)
class TaskBundle:
    task: C.TaskSpec
    seed_spec: IslandSeedSpec
    motion: C.MotionPlan
    topology: str



def _source(kind, **params):
    return C.GeometrySource(kind=kind, params=params, frame=C.Frame.WORK)


def _pseudo_island(stock):
    """An explicitly empty retained region, wholly above stock.

    The primitive requires an island box. This nonempty air box has no common
    volume with stock, so it adds no retained material, unlike an artificial
    undescribed empty primitive or zero-area island substituted for geometry.
    """
    lo = (stock[0][0], stock[0][1], stock[1][2] + 1.0)
    hi = (stock[0][0] + 1.0, stock[0][1] + 1.0, stock[1][2] + 2.0)
    return lo, hi


def _dimensions(lo, hi, pitch):
    lengths = tuple((hi[a] - lo[a]) / pitch for a in range(3))
    rounded = tuple(int(round(v)) for v in lengths)
    if any(v <= 0 or not math.isclose(v, rounded, abs_tol=1e-8)
           for v, rounded in zip(lengths, rounded)):
        raise ValueError(f"grid pitch {pitch} does not span bounds {lo}..{hi}")
    return rounded


def make_task(scenario: Scenario) -> TaskBundle:
    """Build a bounded physical task and return its typed nominal seed."""
    for name, bounds in (("stock", scenario.stock), ("pocket", scenario.pocket)):
        if (
            len(bounds) != 2
            or any(len(point) != 3 for point in bounds)
            or any(not math.isfinite(value) for point in bounds for value in point)
            or any(bounds[1][axis] <= bounds[0][axis] for axis in range(3))
        ):
            raise ValueError(f"invalid ordered {name} bounds: {bounds!r}")
    for axis in range(3):
        if scenario.stock[0][axis] > scenario.pocket[0][axis] or scenario.stock[1][axis] < scenario.pocket[1][axis]:
            raise ValueError("target pocket must lie inside the asymmetric bounded stock")
    stock_lo, stock_hi = scenario.stock
    # Deterministic per-case variation. This can widen the outer retaining
    # stock only; a declared target or a ruled-out feature is never dropped.
    rng = random.Random(f"{GENERATION_VERSION}:{GENERATOR_SEED}:{scenario.family}:{scenario.split}")
    pad = rng.choice((0.0, 0.5, 1.0))
    stock_lo = (stock_lo[0] - pad, stock_lo[1], stock_lo[2])
    stock_hi = (stock_hi[0] + pad, stock_hi[1], stock_hi[2])
    stock = (stock_lo, stock_hi)
    pocket = scenario.pocket
    if pocket[0][0] < stock_lo[0] or pocket[1][0] > stock_hi[0]:
        raise ValueError("deterministic margin would leave stock beyond its declared bounds")
    pitch = PITCH_MM
    grid = C.GridSpec(_dimensions(stock_lo, stock_hi, pitch), stock_lo, pitch)
    stock_geometry = _source("analytic", primitive="box", min_mm=list(stock_lo), max_mm=list(stock_hi))
    island = scenario.island or _pseudo_island(stock)
    original = _source(
        "analytic", primitive="box_minus_pocket",
        box_min_mm=list(stock_lo), box_max_mm=list(stock_hi),
        pocket_min_mm=list(pocket[0]), pocket_max_mm=list(pocket[1]),
        island_min_mm=list(island[0]), island_max_mm=list(island[1]),
        **({"empty_island": scenario.island is None}),
    )
    tool = C.ToolAssembly(
        cutter_kind=C.CutterKind.FLAT, cutter_radius_mm=2.0,
        flute_length_mm=10.0, shank_radius_mm=2.0,
        holder_bottom_mm=10.0, holder_radius_mm=5.0,
        holder_height_mm=32.0, tool_number=1,
    )
    tolerances = C.Tolerances(
        completion_residual_mm3=2.0, gouge_volume_mm3=0.0,
        local_surface_error_mm=0.5, clearance_mm=0.25,
        rapid_penetration_mm=0.0,
    )
    operation = derive_reachable_target(stock_geometry, original, tool, grid, tolerances.clearance_mm)
    excluded_grid = operation.params["grid_excluded_volume_mm3"]
    excluded_continuous = operation.params["continuous_excluded_volume_mm3"]
    task = C.TaskSpec(
        task_id=f"procedural-{scenario.family}-{scenario.split}-{scenario.name}",
        family=scenario.family,
        split=C.Split(scenario.split), source=C.SourceInfo(
            origin=f"{GENERATION_VERSION}:{scenario.topology}", license="MIT",
            notes="Bounded analytic procedural seed; no solver-performance exclusions.",
        ),
        length_unit="mm", stock=stock_geometry, original_part=original,
        operation_target=C.OperationTarget(
            geometry=operation,
            excluded_volume_mm3=excluded_grid,
            exclusion_reason=(
                "Finite-radius cutter cannot reach four pocket corner fillets. "
                f"Normative exclusion {excluded_grid} mm^3 on the declared "
                f"{pitch} mm grid. Continuous diagnostic is "
                f"{excluded_continuous} mm^3. Exact zero sampling tolerance."
            ),
        ),
        grid=grid, work_to_machine=C.Transform.translation((0.0, 0.0, 0.0)),
        initial_state=C.MachineState(position_mm=(0.0, 0.0, stock_hi[2] + 10.0), frame=C.Frame.MACHINE),
        tool=tool,
        limits=C.MachineLimits(
            max_feed_mm_s=8.0, rapid_mm_s=80.0, max_accel_mm_s2=100.0,
            envelope_min_mm=(stock_lo[0] - 4.0, stock_lo[1] - 4.0, stock_lo[2] - 4.0),
            envelope_max_mm=(stock_hi[0] + 4.0, stock_hi[1] + 4.0, max(60.0, stock_hi[2] + 44.0)),
        ),
        tolerances=tolerances,
    )
    seed_spec = seed_spec_for_task(task)
    motion = seed_for_task(task)
    motion.validate_for_task(task)
    return TaskBundle(task, seed_spec, motion, scenario.topology)


def generate_tasks():
    return tuple(make_task(scenario) for scenario in SCENARIOS)


def _json(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def build_manifest(tasks=None):
    """Return deterministic task-definition metadata, not claimed seed results."""
    tasks = generate_tasks() if tasks is None else tuple(tasks)
    entries = []
    for bundle in tasks:
        params = bundle.task.operation_target.geometry.params
        entries.append({
            "task_id": bundle.task.task_id,
            "family": bundle.task.family,
            "split": bundle.task.split.value,
            "topology": bundle.topology,
            "task_hash": bundle.task.task_hash,
            "task_record": json.loads(C.dumps(bundle.task)),
            "seed_spec": bundle.seed_spec.as_dict(),
            "motion_hash": bundle.motion.motion_hash,
            "grid_excluded_volume_mm3": bundle.task.operation_target.excluded_volume_mm3,
            "continuous_exclusion_diagnostic_mm3": params["continuous_excluded_volume_mm3"],
            "evaluator_exclusion_tolerance_mm3": params["grid_exclusion_tolerance_mm3"],
            "seed_provenance": bundle.motion.provenance,
        })
    return {
        "schema_version": "t2.3-procedural/1",
        "generation_version": GENERATION_VERSION,
        "generator_seed": GENERATOR_SEED,
        "suite_count_policy": "one dev and one test per family for bootstrap only; full suite counts pending pilot cost",
        "scope": "bounded analytic box and box-minus-pocket families; no free-form CAD claim",
        "grid_pitch_mm": PITCH_MM,
        "families": sorted({bundle.task.family for bundle in tasks}),
        "tasks": entries,
    }


def manifest_text():
    return _json(build_manifest())




def write_manifest(path=None, expected_text=None):
    """Create/verify a deterministic manifest without overwriting a changed file."""
    path = PROJECT_ROOT / "configs" / "task_manifest.json" if path is None else Path(path)
    expected = manifest_text() if expected_text is None else expected_text
    if path.exists() and path.read_text(encoding="utf-8") != expected:
        raise FileExistsError(f"manifest changed since generation; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def verify_manifest(path=None):
    path = PROJECT_ROOT / "configs" / "task_manifest.json" if path is None else Path(path)
    actual = path.read_text(encoding="utf-8")
    if actual != manifest_text():
        raise ValueError(f"manifest is not reproducible byte-for-byte: {path}")
    return True


def _provenance():
    commit = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    owned = (
        "cam/seeds.py", "benchmarks/procedural_suite.py",
        "scripts/generate_task_suite.py", "configs/task_manifest.json",
        "tests/procedural_suite_test.py",
        # Read-only upstream geometry helpers captured by seed evaluation too.
        "benchmarks/pilot_task.py",
    )
    patch = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "diff", "--binary", "HEAD"],
        capture_output=True, check=True,
    ).stdout
    state = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = None
    if state:
        dirty = C.content_hash({
            "diff_sha256": C.sha256_bytes(patch),
            "owned": {
                file: C.sha256_bytes((PROJECT_ROOT / file).read_bytes())
                for file in owned if (PROJECT_ROOT / file).is_file()
            },
        })
    return commit, dirty


def evaluate_seed(bundle: TaskBundle):
    """Evaluate a typed nominal seed with both production posts and reference.

    A ``TaskSpec`` without a qualified production seed is not a qualified task.
    No candidate is promoted on physical uncertainty, unknown coverage or a
    failed safety/geometry decision. Generic seed/feature errors are preserved.
    """
    task = bundle.task
    for dialect in C.Dialect:
        nc = post_motion_plan(bundle.motion, task, dialect=dialect).encode("utf-8")
        nc_hash = C.nc_hash(nc)
        settings = dict(production_settings(task))
        settings.update(expected_nc_hash=nc_hash, exclusion_tolerance_mm3=0.0)
        result = evaluate_posted_gcode(task, nc, dialect, settings)
        if result.coverage != C.Coverage.COMPLETE or result.safety_pass is not True or result.geometry_pass is not True:
            raise RuntimeError(
                f"unqualified procedural seed {task.task_id} ({dialect.value}): "
                f"coverage={result.coverage.value}, safety={result.safety_pass}, "
                f"geometry={result.geometry_pass}, reasons={result.coverage_reasons}"
            )
        if result.nc_hash != nc_hash:
            raise AssertionError("production returned a mismatched NC identity")
        lower = result.metrics["min_clearance_lower_bound_mm"].value
        rapid_upper = result.metrics["rapid_cutter_penetration_upper_bound_mm"].value
        if (
            lower is None or lower < task.tolerances.clearance_mm
            or rapid_upper is None or rapid_upper > task.tolerances.rapid_penetration_mm
        ):
            raise RuntimeError(
                f"unqualified clearance for {task.task_id} ({dialect.value}): "
                f"lower={lower} mm, rapid={rapid_upper} mm"
            )
        yield dialect, nc, nc_hash, result


def _write_case(bundle: TaskBundle, root: Path):
    if root.exists():
        raise FileExistsError(f"existing seed artifact directory will not be overwritten: {root}")
    root.mkdir(parents=False, exist_ok=False)
    evaluations = {}
    files = {
        "task.json": C.dumps(bundle.task).encode("utf-8"),
        "motion_plan.json": C.dumps(bundle.motion).encode("utf-8"),
        "history.json": _json([{
            "step": "nominal-zero-residual", "candidate": "procedural-seed",
            "generation_version": GENERATION_VERSION,
            "seed_spec": bundle.seed_spec.as_dict(),
        }]).encode("utf-8"),
    }
    posted = []
    for dialect, nc, nc_hash, result in evaluate_seed(bundle):
        file = C.DIALECT_ARTIFACTS[dialect][0]
        eval_file = C.DIALECT_ARTIFACTS[dialect][1]
        files[file] = nc
        files[eval_file] = C.dumps(result).encode("utf-8")
        evaluations[dialect.value] = result
        posted.append(C.PostedProgram(
            dialect=dialect, status=C.PostStatus.SUPPORTED,
            nc_hash=nc_hash, eval_file=eval_file,
        ))
    for dialect in C.Dialect:
        if evaluations[dialect.value].nc_hash != C.nc_hash(files[C.DIALECT_ARTIFACTS[dialect][0]]):
            raise AssertionError("retained NC hash mismatch")
    reference = run_reference(bundle.task, bundle.motion, {"subdivisions": 8})
    if reference.result.coverage != C.Coverage.COMPLETE or reference.result.safety_pass is not True:
        raise RuntimeError(f"independent reference failed qualification: {bundle.task.task_id}: {reference.result.coverage_reasons}")
    reference_clearance = reference.result.metrics["min_clearance_lower_bound_mm"].value
    reference_rapid = reference.result.metrics["rapid_cutter_penetration_upper_bound_mm"].value
    if (
        reference_clearance is None
        or reference_clearance < bundle.task.tolerances.clearance_mm
        or reference_rapid is None
        or reference_rapid > bundle.task.tolerances.rapid_penetration_mm
    ):
        raise RuntimeError(
            "unqualified independent-reference clearance: "
            f"lower={reference_clearance} mm, rapid={reference_rapid} mm"
        )
    files["reference.json"] = C.dumps(reference.result).encode("utf-8")
    commit, dirty = _provenance()
    metrics = {
        name: {
            "coverage": evaluations[name].coverage.value,
            "safety_pass": evaluations[name].safety_pass,
            "geometry_pass": evaluations[name].geometry_pass,
            "residual_volume_mm3": evaluations[name].metrics["residual_volume_mm3"].value,
            "gouge_volume_mm3": evaluations[name].metrics["gouge_volume_mm3"].value,
            "max_surface_error_mm": evaluations[name].metrics["max_surface_error_mm"].value,
            "min_clearance_lower_bound_mm": evaluations[name].metrics["min_clearance_lower_bound_mm"].value,
            "rapid_penetration_upper_bound_mm": evaluations[name].metrics["rapid_cutter_penetration_upper_bound_mm"].value,
            "excluded_volume_mm3": evaluations[name].metrics["excluded_volume_mm3"].value,
            "exclusion_tolerance_mm3": evaluations[name].discretization["settings"]["exclusion_tolerance_mm3"],
        }
        for name in evaluations
    }
    metrics["generator"] = {
        "generation_version": GENERATION_VERSION,
        "generator_seed": GENERATOR_SEED,
        "task_hash": bundle.task.task_hash,
        "motion_hash": bundle.motion.motion_hash,
        "seed_spec_sha256": C.content_hash(bundle.seed_spec.as_dict()),
        "reference_safety_pass": reference.result.safety_pass,
        "reference_coverage": reference.result.coverage.value,
        "reference_geometry_pass": reference.result.geometry_pass,
    }
    # Match the existing writer's full normalized-grid trajectory convention.
    # This is the nominal command, not an invented independent simulation. Keep
    # physical machine-frame coordinates separately so neither representation
    # can be mistaken for the other (and never store a one-point "initial path").
    machine_waypoints = np.asarray(expected_positions_from_motion(bundle.motion), dtype=np.float64)
    machine_deltas = np.asarray(seed_deltas_mm(bundle.motion), dtype=np.float64)
    np.testing.assert_allclose(
        np.diff(machine_waypoints, axis=0), machine_deltas, atol=1e-10, rtol=0.0,
    )
    inverse_work = bundle.task.work_to_machine.inverse()
    work_waypoints = np.asarray(
        [inverse_work.apply(point) for point in machine_waypoints], dtype=np.float64,
    )
    grid = bundle.task.grid
    if grid.frame != C.Frame.WORK:
        raise ValueError("normalized nominal artifacts require a work-frame grid")
    origin = np.asarray(grid.origin_mm, dtype=np.float64)
    extent = np.asarray(grid.extent_mm, dtype=np.float64)
    # The transform is homogeneous 4x4; increments use the linear 3x3 block,
    # never the translation column. Absolute waypoints use inverse_work.apply.
    inverse_rotation = np.asarray(inverse_work.matrix, dtype=np.float64)[:3, :3]
    work_deltas = machine_deltas @ inverse_rotation.T
    np.testing.assert_allclose(
        np.diff(work_waypoints, axis=0), work_deltas, atol=1e-12, rtol=0.0,
    )
    trajectories = np.asarray((work_waypoints - origin) / extent, dtype=np.float32)
    diffs = np.diff(trajectories, axis=0).astype(np.float32)
    expected_normalized_deltas = np.asarray(work_deltas / extent, dtype=np.float32)
    np.testing.assert_allclose(diffs, expected_normalized_deltas, atol=1e-6, rtol=0.0)
    files.update({
        "config.json": _json({
            "source_commit": commit,
            "dirty_patch_hash": dirty,
            "trajectory_representation": {
                "kind": "nominal typed seed, not an independent simulator trajectory",
                "normalized_formula": "(work_mm - grid.origin_mm) / grid.extent_mm",
                "initial_trajectory": "full normalized seed; identical to trajectory.npy for this nominal",
                "machine_mm_files": ["machine_waypoints_mm.npy", "machine_deltas_mm.npy"],
                "machine_position_to_work": json.loads(C.canonical_json(
                    bundle.task.work_to_machine.inverse()
                )),
            },
        }).encode("utf-8"),
        "metrics.json": _json(metrics).encode("utf-8"),
        "trajectory.npy": _npy_bytes(trajectories),
        "trajectory_deltas.npy": _npy_bytes(diffs),
        "initial_trajectory.npy": _npy_bytes(trajectories),
        "machine_waypoints_mm.npy": _npy_bytes(machine_waypoints),
        "machine_deltas_mm.npy": _npy_bytes(machine_deltas),
    })
    index = {"schema_version": 1, "programs": {}}
    for program in posted:
        index["programs"][program.dialect.value] = json.loads(C.canonical_json(program))
    files["eval_posted.json"] = _json(index).encode("utf-8")
    run = C.RunRecord(
        run_id=f"seed-{bundle.task.task_id}", task_hash=bundle.task.task_hash,
        status=C.RunStatus.COMPLETE, artifacts=tuple(sorted(files)),
        posted=tuple(posted), source_commit=commit, dirty_patch_hash=dirty,
    )
    files["run.json"] = C.dumps(run).encode("utf-8")
    for name, payload in files.items():
        (root / name).write_bytes(payload)
    manifest = {name: C.sha256_bytes(payload) for name, payload in files.items()}
    (root / "hashes.json").write_text(_json(manifest), encoding="utf-8")
    return run


def _npy_bytes(array):
    import io
    stream = io.BytesIO()
    np.save(stream, np.asarray(array, dtype=np.float32))
    return stream.getvalue()


def verify_all(tasks=None, output_root=None):
    """Materialize and validate one production-qualified seed per task."""
    tasks = generate_tasks() if tasks is None else tuple(tasks)
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "runs" / "procedural-bootstrap"
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    for bundle in tasks:
        case_root = root / bundle.task.task_id
        reports.append(_write_case(bundle, case_root))
    return reports


__all__ = [
    "Scenario", "TaskBundle", "GENERATION_VERSION", "GENERATOR_SEED",
    "PITCH_MM", "make_task", "generate_tasks", "build_manifest", "manifest_text",
    "write_manifest", "verify_manifest", "evaluate_seed", "verify_all",
]
