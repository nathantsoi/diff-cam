"""Deterministic T2.3 task families and independent-reference cheap gates.

Version ``t2.3/2`` declares per-family dev/test counts (default 3/10), with
real bounded geometric/topological design variation from per-case seeds.
The six ``t2.3-bootstrap/1`` definitions remain reproducible through their
own named generator; they are not silently recounted as the v2 suite.

Scope follows the available analytic stock/pocket primitives: open pockets or
zero/one square retained pad, never an unsupported free-form or multiple-pad
claim. No solver outcome selects tasks. The v2 default cheap path uses a fresh
bounded f64 hard-sim worker per task and scores its actual stock masks against
the task grid's reachable operation target. The independent NumPy reference
is a separate diagnostic cross-check, not an automatic per-task claim.
Neither path posts G-code or claims posted qualification.

Original part P is unchanged. Operation target T adds only unreachable corner
material from finite-radius cutter accessibility. Its normative excluded value
is the exact precomputed grid-cell exclusion at the task pitch. The continuous
fillets are diagnostics only, with zero evaluator-exclusion tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import hashlib

import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import taichi as ti
from taichi.lang import impl as ti_impl

from algorithms.run_artifacts import simulator_array, simulator_precision_metadata

from benchmarks import contracts as C
from benchmarks.pilot_task import (
    derive_reachable_target, expected_positions_from_motion, original_part_field,
    production_settings, source_fields,
)
from cam.posted_evaluator import evaluate_posted_gcode
from cam.posts import post_motion_plan
from cam.seeds import IslandSeedSpec, seed_deltas_mm, seed_for_task, seed_spec_for_task
from verification.reference_carve import run_reference
from simulator.csg_simulator import CSGSimulatorDelta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_VERSION = "t2.3-bootstrap/1"
GENERATION_VERSION = "t2.3/2"
GENERATOR_SEED = 133742
DEFAULT_DEV_PER_FAMILY = 3
DEFAULT_TEST_PER_FAMILY = 10
PITCH_MM = 0.25
MIN_DESIGN_DISTANCE = 0.25
CHEAP_GATE_EXECUTOR = "simulator.csg_simulator.CSGSimulatorDelta.forward_hard"
CHEAP_GATE_SETTINGS = {
    "requested_precision": "f64",
    "source_payload_dtype": "float32",
    "clip_speeds": False,
    "source_grid": "exact task GridSpec, physical-mm source geometry",
    "stock_membership": "hard stock SDF < 0 at voxel centres",
    "holder_overlap": "conservative whole-segment interval, summed mm^3, not chronological collision volume",
}
CHEAP_WORKER_TIMEOUT_S = 180.0
CHEAP_WORKER_MEMORY_LIMIT_BYTES = 2_000_000_000
CHEAP_HISTORY_BUDGET_BYTES = 1_500_000_000
REFERENCE_CHECK_SETTINGS = {
    "max_step_mm": PITCH_MM / 2.0,
    "compute_bounds": False,
    "safety_screen": True,
}
PREDECLARED_EXCLUSION_REASONS = (
    "unsupported or contradictory task/seed geometry",
    "unreachable original target or finite-cutter/shank/holder access violation",
    "failed task contract or nominal seed validation",
    "incomplete reference coverage or unavailable/unsafe assembly clearance",
    "residual over task tolerance or operation-target gouge over tolerance",
    "cheap-worker timeout, memory bound or invalid execution evidence",
)
FAMILIES = ("facing_asymmetric", "pocket_island", "variable_width_depth")
# A normalized maximum-coordinate design metric. These declared physical
# scales prevent a 0.25 mm normalizer detail from substituting for real design
# variation. The retained-pad count uses one count as its own unit.
DESIGN_SCALES = {
    "stock_min_x": 4.0, "stock_max_x": 4.0,
    "stock_min_y": 4.0, "stock_max_y": 4.0,
    "stock_depth": 4.0,
    "pocket_cx": 4.0, "pocket_cy": 4.0,
    "pocket_half_x": 4.0, "pocket_half_y": 4.0,
    "pocket_depth": 2.0,
    "island_cx": 2.0, "island_cy": 2.0,
    "island_half_side": 2.0, "island_top_z": 2.0,
    "island_count": 1.0,
}


@dataclass(frozen=True)
class Scenario:
    family: str
    split: str
    name: str
    stock: tuple[tuple[float, float, float], tuple[float, float, float]]
    pocket: tuple[tuple[float, float, float], tuple[float, float, float]]
    island: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    topology: str = "centred_box_pocket"
    generation_version: str = GENERATION_VERSION
    generator_seed: int = GENERATOR_SEED
    design_seed: int = 0
    case_index: int = -1


# Explicit frozen v1 variants, never curated from solver outcomes.
BOOTSTRAP_SCENARIOS = (
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


BOOTSTRAP_SCENARIOS = tuple(
    replace(scenario, generation_version=BOOTSTRAP_VERSION,
            generator_seed=GENERATOR_SEED, design_seed=0, case_index=index)
    for index, scenario in enumerate(BOOTSTRAP_SCENARIOS)
)


@dataclass(frozen=True)
class TaskBundle:
    task: C.TaskSpec
    seed_spec: IslandSeedSpec
    motion: C.MotionPlan
    topology: str
    generation_version: str = GENERATION_VERSION
    generator_seed: int = GENERATOR_SEED
    design_parameters: dict[str, float | int] = field(default_factory=dict)
    design_seed: int = 0
    case_index: int = -1



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


def _resolved_stock(scenario):
    """Preserve the v1 random padding exactly; v2 stock is already explicit."""
    if scenario.generation_version != BOOTSTRAP_VERSION:
        return scenario.stock
    rng = random.Random(
        f"{scenario.generation_version}:{scenario.generator_seed}:"
        f"{scenario.family}:{scenario.split}"
    )
    pad = rng.choice((0.0, 0.5, 1.0))
    lo, hi = scenario.stock
    return ((lo[0] - pad, lo[1], lo[2]), (hi[0] + pad, hi[1], hi[2]))


def _retained_island(scenario):
    if scenario.island is None:
        return None
    stock = _resolved_stock(scenario)
    lo, hi = scenario.island
    if not all(stock[0][axis] < hi[axis] and lo[axis] < stock[1][axis]
               for axis in range(3)):
        return None
    return lo, hi


def _design_parameters(scenario):
    stock = _resolved_stock(scenario)
    pocket = scenario.pocket
    island = _retained_island(scenario)
    half = 0.0 if island is None else (island[1][0] - island[0][0]) / 2.0
    if island is not None and not math.isclose(
            half, (island[1][1] - island[0][1]) / 2.0, abs_tol=1e-9):
        raise ValueError("this declared family supports square, not rectangular, retained pads")
    return {
        "stock_min_x": float(stock[0][0]), "stock_max_x": float(stock[1][0]),
        "stock_min_y": float(stock[0][1]), "stock_max_y": float(stock[1][1]),
        "stock_depth": float(stock[1][2] - stock[0][2]),
        "pocket_cx": float((pocket[0][0] + pocket[1][0]) / 2.0),
        "pocket_cy": float((pocket[0][1] + pocket[1][1]) / 2.0),
        "pocket_half_x": float((pocket[1][0] - pocket[0][0]) / 2.0),
        "pocket_half_y": float((pocket[1][1] - pocket[0][1]) / 2.0),
        "pocket_depth": float(pocket[1][2] - pocket[0][2]),
        "island_cx": 0.0 if island is None else float((island[0][0] + island[1][0]) / 2.0),
        "island_cy": 0.0 if island is None else float((island[0][1] + island[1][1]) / 2.0),
        "island_half_side": float(half),
        "island_top_z": 0.0 if island is None else float(island[1][2]),
        "island_count": int(island is not None),
    }


def design_parameters_from_task(task):
    """Recompute a design vector from the actual typed source geometry."""
    if not isinstance(task, C.TaskSpec) or task.original_part.kind != "analytic":
        raise ValueError("design provenance requires an analytic typed TaskSpec")
    params = task.original_part.params
    if params.get("primitive") != "box_minus_pocket":
        raise ValueError("design provenance requires box-minus-pocket geometry")
    def bounds(lower, upper):
        return tuple(tuple(float(value) for value in params[key])
                     for key in (lower, upper))
    stock = bounds("box_min_mm", "box_max_mm")
    pocket = bounds("pocket_min_mm", "pocket_max_mm")
    island = bounds("island_min_mm", "island_max_mm")
    physical = island if all(stock[0][a] < island[1][a] and island[0][a] < stock[1][a]
                              for a in range(3)) else None
    return _design_parameters(Scenario(
        task.family, task.split.value, task.task_id, stock, pocket, physical,
        generation_version=GENERATION_VERSION,
    ))


def _descriptor(value):
    if isinstance(value, Scenario):
        return _design_parameters(value)
    if isinstance(value, TaskBundle):
        if set(value.design_parameters) != set(DESIGN_SCALES):
            raise ValueError("task bundle lacks the complete declared design vector")
        return dict(value.design_parameters)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError("design descriptor must be a Scenario, TaskBundle or parameter dictionary")


def design_distance(left, right, *, normalized=True):
    """Maximum scaled absolute difference over the complete physical design.

    ``normalized=False`` is a diagnostic in component units. The suite guard
    uses the declared normalized metric and separately catches exact physical
    duplicates, so a different family/case/index cannot mask copied geometry.
    """
    left, right = _descriptor(left), _descriptor(right)
    if set(left) != set(DESIGN_SCALES) or set(right) != set(DESIGN_SCALES):
        raise ValueError("design vectors must contain every declared parameter")
    differences = []
    for name, scale in DESIGN_SCALES.items():
        a, b = left[name], right[name]
        if (isinstance(a, bool) or isinstance(b, bool)
                or not isinstance(a, (int, float)) or not isinstance(b, (int, float))
                or not math.isfinite(a) or not math.isfinite(b)):
            raise ValueError(f"non-finite or nonnumeric design component {name}")
        differences.append(abs(float(a) - float(b)) / (scale if normalized else 1.0))
    return float(max(differences))


def validate_designs(scenarios, *, minimum_distance=MIN_DESIGN_DISTANCE):
    """Fail before construction on near-duplicates; no performance filtering."""
    if (isinstance(minimum_distance, bool) or not isinstance(minimum_distance, (int, float))
            or not math.isfinite(minimum_distance) or minimum_distance <= 0):
        raise ValueError("minimum design distance must be finite and positive")
    scenarios = tuple(scenarios)
    identifiers = []
    for scenario in scenarios:
        if not isinstance(scenario, Scenario):
            raise TypeError("design guard requires typed Scenarios")
        identifiers.append((scenario.family, scenario.split, scenario.case_index))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate family/split/case addressing")
    for index, left in enumerate(scenarios):
        for right in scenarios[index + 1:]:
            distance = design_distance(left, right)
            if distance < minimum_distance:
                raise ValueError(
                    "near-duplicate procedural designs: "
                    f"{left.name} and {right.name}: "
                    f"normalized distance {distance} < {minimum_distance}"
                )
    return True


def make_task(scenario: Scenario) -> TaskBundle:
    """Build a bounded physical task and return its typed nominal seed."""
    if scenario.family not in FAMILIES or scenario.split not in ("dev", "test"):
        raise ValueError("unrecognized family or split")
    if scenario.generation_version not in (BOOTSTRAP_VERSION, GENERATION_VERSION):
        raise ValueError("unsupported procedural generator version")
    for name, value in (("generator_seed", scenario.generator_seed),
                        ("design_seed", scenario.design_seed),
                        ("case_index", scenario.case_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
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
    stock = _resolved_stock(scenario)
    stock_lo, stock_hi = stock
    pocket = scenario.pocket
    if pocket[0][0] < stock_lo[0] or pocket[1][0] > stock_hi[0]:
        raise ValueError("deterministic margin would leave stock beyond its declared bounds")
    pitch = PITCH_MM
    grid = C.GridSpec(_dimensions(stock_lo, stock_hi, pitch), stock_lo, pitch)
    if math.prod(grid.shape) > 1_000_000:
        raise ValueError("cheap per-task qualification is bounded to one million grid cells")
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
            origin=f"{scenario.generation_version}:{scenario.topology}", license="MIT",
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
    return TaskBundle(
        task, seed_spec, motion, scenario.topology,
        generation_version=scenario.generation_version,
        generator_seed=scenario.generator_seed,
        design_parameters=_design_parameters(scenario),
        design_seed=scenario.design_seed, case_index=scenario.case_index,
    )


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer per family")
    return value


def _scenario_seed(generator_seed, family, split, case_index):
    text = f"{GENERATION_VERSION}:{generator_seed}:{family}:{split}:{case_index}"
    return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:8], "big")


def _scenario(family, split, case_index, generator_seed):
    design_seed = _scenario_seed(generator_seed, family, split, case_index)
    rng = random.Random(design_seed)
    # Keep three named structural variants, rotate them from the seed, and
    # derive them from actual retained geometry rather than the index alone.
    variant = (case_index + rng.randrange(3)) % 3
    retained = variant != 0
    if family == "facing_asymmetric":
        half_x = float(rng.choice((6.0, 7.0, 8.0, 9.0, 10.0)))
        half_y = float(rng.choice((5.0, 6.0, 7.0, 8.0)))
        depth = float(rng.choice((0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)))
    elif family == "pocket_island":
        half_x = float(rng.choice((7.0, 8.0, 9.0, 10.0, 11.0, 12.0)))
        half_y = float(rng.choice((6.0, 7.0, 8.0, 9.0)))
        depth = float(rng.choice((1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)))
    else:
        half_x = float(rng.choice((4.0, 5.0, 6.0, 8.0, 10.0, 12.0)))
        half_y = float(rng.choice((5.0, 6.0, 7.0, 8.0, 9.0, 10.0)))
        depth = float(rng.choice((0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)))
    cx = float(rng.choice((-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)))
    cy = float(rng.choice((-2.0, -1.0, 0.0, 1.0, 2.0)))
    stock_depth = float(rng.choice((8.0, 10.0, 12.0)))

    island = None
    topology = "open_oblong_zero_retained_pads"
    if retained:
        half_side = float(rng.choice((2.0, 2.5, 3.0)))
        # Match seed_spec_for_task's wall gap and its extra raster-clearance
        # margin. Offset cases reserve an actual half-millimetre displacement;
        # no impossible rectangle, multi-pad or arbitrary stepped floor is made.
        required = 4.5 + math.sqrt(2.0) * half_side
        if variant == 2:
            required += 0.5
            depth = max(depth, 0.75)
        minimum_half = math.ceil(required * 4.0) / 4.0
        half_x, half_y = max(half_x, minimum_half), max(half_y, minimum_half)
        max_dx = max(0.0, math.floor((half_x - 4.5 - math.sqrt(2.0) * half_side) * 2.0) / 2.0)
        max_dy = max(0.0, math.floor((half_y - 4.5 - math.sqrt(2.0) * half_side) * 2.0) / 2.0)
        if variant == 1:
            ix = iy = 0.0
            pad_height = depth
            topology = "centred_square_full_height_pad"
        else:
            x_choices = np.arange(-max_dx, max_dx + 0.25, 0.5).tolist()
            y_choices = np.arange(-max_dy, max_dy + 0.25, 0.5).tolist()
            ix = float(rng.choice(x_choices or [0.0]))
            iy = float(rng.choice(y_choices or [0.0]))
            if ix == 0.0 and iy == 0.0:
                if max_dx > 0.0:
                    ix = rng.choice((-max_dx, max_dx))
                elif max_dy > 0.0:
                    iy = rng.choice((-max_dy, max_dy))
                else:
                    raise AssertionError("offset topology reserved no physical offset")
            # Strictly below stock top, on a quarter-millimetre grid.
            height_choices = [i / 4.0 for i in range(2, round((depth - 0.25) * 4) + 1)]
            if not height_choices:
                raise AssertionError("step topology needs a positive step height")
            pad_height = float(rng.choice(height_choices))
            topology = "offset_square_lower_step_pad"
        px, py = cx + ix, cy + iy
        island = ((px - half_side, py - half_side, -depth),
                  (px + half_side, py + half_side, -depth + pad_height))
    pocket = ((cx - half_x, cy - half_y, -depth), (cx + half_x, cy + half_y, 0.0))
    left = float(rng.choice((3.0, 4.0, 5.0, 6.0)))
    right = float(rng.choice((3.0, 4.0, 5.0, 6.0)))
    if family == "facing_asymmetric" and left == right:
        right += 1.0
    bottom = float(rng.choice((3.0, 4.0, 5.0, 6.0)))
    top = float(rng.choice((3.0, 4.0, 5.0, 6.0)))
    stock = ((pocket[0][0] - left, pocket[0][1] - bottom, -stock_depth),
             (pocket[1][0] + right, pocket[1][1] + top, 0.0))
    return Scenario(
        family, split, f"case-{case_index:02d}-{topology}", stock, pocket, island,
        topology=topology, generation_version=GENERATION_VERSION,
        generator_seed=generator_seed, design_seed=design_seed, case_index=case_index,
    )


def generate_scenarios(dev_per_family=DEFAULT_DEV_PER_FAMILY,
                       test_per_family=DEFAULT_TEST_PER_FAMILY, *, seed=GENERATOR_SEED):
    dev = _count(dev_per_family, "dev count")
    test = _count(test_per_family, "test count")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("generator seed must be an unsigned 64-bit integer")
    scenarios = tuple(
        _scenario(family, split, index, seed)
        for family in FAMILIES for split, count in (("dev", dev), ("test", test))
        for index in range(count)
    )
    validate_designs(scenarios)
    return scenarios


SCENARIOS = generate_scenarios()


def generate_tasks(dev_per_family=DEFAULT_DEV_PER_FAMILY,
                   test_per_family=DEFAULT_TEST_PER_FAMILY, *, seed=GENERATOR_SEED):
    scenarios = (SCENARIOS
                 if dev_per_family == DEFAULT_DEV_PER_FAMILY
                 and test_per_family == DEFAULT_TEST_PER_FAMILY and seed == GENERATOR_SEED
                 else generate_scenarios(dev_per_family, test_per_family, seed=seed))
    return tuple(make_task(scenario) for scenario in scenarios)


def generate_bootstrap_tasks():
    """The named six-case v1 subset, independent of v2 defaults and globals."""
    return tuple(make_task(replace(scenario, generation_version=BOOTSTRAP_VERSION))
                 for scenario in BOOTSTRAP_SCENARIOS)


def _require_seed(condition, reason):
    if not condition:
        raise RuntimeError(f"unqualified procedural seed: {reason}")


def _reference_qualify_seed(bundle):
    """Run the nominal-motion cheap gate, with no G-code/post evaluation.

    The independent reference supplies its actual remaining/removed masks.
    Custom-target residual/gouge are then scored against source_fields'
    exactly declared grid. Its intentionally-null target verdict is preserved
    verbatim in the returned reference record, not overwritten as a pass.
    """
    task = C.loads(C.dumps(bundle.task))
    motion = bundle.motion
    _require_seed(isinstance(task, C.TaskSpec), "task must round-trip as TaskSpec")
    _require_seed(task == bundle.task, "task JSON round-trip changed its physical declaration")
    _require_seed(task.task_hash == bundle.task.task_hash, "task hash changed round-trip")
    _require_seed(design_parameters_from_task(task) == bundle.design_parameters,
                  "design parameters do not match the typed task source")
    motion.validate_for_task(task)
    regenerated = seed_for_task(task)
    _require_seed(regenerated.motion_hash == motion.motion_hash,
                  "nominal recipe and retained motion identity differ")
    grid = task.grid
    _require_seed(grid.voxel_mm == PITCH_MM, "cheap gate requires the declared 0.25 mm grid")

    trace = run_reference(task, motion, dict(REFERENCE_CHECK_SETTINGS))
    result = trace.result
    _require_seed(result.coverage == C.Coverage.COMPLETE,
                  f"reference coverage is incomplete: {result.coverage_reasons}")
    _require_seed(result.safety_pass is True,
                  f"assembly safety is not established: {result.safety_pass}")
    _require_seed(result.task_hash == task.task_hash and result.motion_hash == motion.motion_hash,
                  "reference changed task or motion identity")
    _require_seed(result.tool_setup_hash == task.tool_setup_hash,
                  "reference changed cutter/shank/holder identity")
    _require_seed(trace.remaining is not None and trace.removed is not None,
                  "reference did not return actual stock masks")
    remaining = np.asarray(trace.remaining, dtype=bool)
    removed = np.asarray(trace.removed, dtype=bool)
    stock_field, target_field, required, target = source_fields(task)
    initial = stock_field <= 0.0
    target = np.asarray(target & initial, dtype=bool)
    original = original_part_field(task) <= 0.0
    _require_seed(initial.shape == remaining.shape == removed.shape == task.grid.shape,
                  "reference and declared task grids differ")
    _require_seed(np.all(remaining <= initial), "reference gained stock")
    _require_seed(np.array_equal(removed, initial & ~remaining),
                  "reference removed/remaining masks disagree")
    _require_seed(not np.any(required & target), "removal target and retained target overlap")
    _require_seed(np.array_equal(initial & ~required, target),
                  "raw required-removal and retained-target masks disagree")
    _require_seed(np.all(original <= target), "reachable operation target omits original P")

    params = task.operation_target.geometry.params
    excluded_cells = int(np.count_nonzero(target & ~original))
    voxel_volume = float(grid.voxel_mm ** 3)
    excluded_volume = float(excluded_cells * voxel_volume)
    _require_seed(excluded_cells == params["grid_excluded_voxels"], "grid exclusion cell count changed")
    _require_seed(excluded_volume == params["grid_excluded_volume_mm3"] ==
                  task.operation_target.excluded_volume_mm3, "exclusion volume changed")
    _require_seed(params["grid_exclusion_tolerance_mm3"] == 0.0,
                  "cheap gate requires exact zero exclusion tolerance")
    _require_seed(excluded_cells > 0, "P and reachable T are not distinct here")

    in_target = int(np.count_nonzero(initial & target))
    removal_target = int(np.count_nonzero(initial & ~target))
    _require_seed(in_target > 0 and removal_target > 0, "degenerate target denominators")
    residual_cells = int(np.count_nonzero(remaining & ~target))
    gouge_cells = int(np.count_nonzero(removed & target))
    residual = float(residual_cells * voxel_volume)
    gouge = float(gouge_cells * voxel_volume)
    _require_seed(residual <= task.tolerances.completion_residual_mm3,
                  f"residual {residual} mm^3 exceeds task tolerance")
    _require_seed(gouge <= task.tolerances.gouge_volume_mm3,
                  f"operation-target gouge {gouge} mm^3 exceeds task tolerance")
    clearance = result.metrics["min_clearance_lower_bound_mm"].value
    rapid_upper = result.metrics["rapid_cutter_penetration_upper_bound_mm"].value
    _require_seed(clearance is not None and clearance >= task.tolerances.clearance_mm,
                  f"assembly clearance lower bound is unavailable or too small: {clearance}")
    _require_seed(rapid_upper is not None and rapid_upper <= task.tolerances.rapid_penetration_mm,
                  f"rapid penetration bound is unavailable or too large: {rapid_upper}")
    holder = params["holder_clearance"]
    _require_seed(holder["above_stock_mm"] >= holder["qualification_and_cell_margin_mm"],
                  "shank/holder height no longer clears the stock")

    return {
        "status": "qualified_cheap_reference",
        "task_hash": task.task_hash,
        "tool_setup_hash": task.tool_setup_hash,
        "motion_hash": motion.motion_hash,
        "residual_volume_mm3": residual,
        "gouge_volume_mm3": gouge,
        "residual_fraction": residual_cells / removal_target,
        "gouge_fraction": gouge_cells / in_target,
        "residual_cells": residual_cells,
        "gouge_cells": gouge_cells,
        "removed_volume_mm3": result.metrics["removed_volume_mm3"].value,
        "excluded_volume_mm3": excluded_volume,
        "excluded_cells": excluded_cells,
        "min_clearance_lower_bound_mm": clearance,
        "rapid_penetration_upper_bound_mm": rapid_upper,
        "reference_evaluator": result.evaluator_id,
        "reference_revision": result.evaluator_revision,
        "reference_settings": dict(REFERENCE_CHECK_SETTINGS),
        "reference_coverage": result.coverage.value,
        "reference_safety_pass": result.safety_pass,
        "residual_and_gouge_method": (
            "actual independent-reference remaining/removed cell masks versus "
            "operation-target masks; task-grid mm^3, not a posted verdict"
        ),
        "reference_result": json.loads(C.dumps(result)),
    }


def _json(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def build_bootstrap_manifest(tasks=None):
    """The original manifest schema and bytes, not a re-versioned v1 amnesty."""
    tasks = generate_bootstrap_tasks() if tasks is None else tuple(tasks)
    if any(bundle.generation_version != BOOTSTRAP_VERSION for bundle in tasks):
        raise ValueError("bootstrap manifest accepts only named bootstrap/1 bundles")
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
        "generation_version": BOOTSTRAP_VERSION,
        "generator_seed": GENERATOR_SEED,
        "suite_count_policy": "one dev and one test per family for bootstrap only; full suite counts pending pilot cost",
        "scope": "bounded analytic box and box-minus-pocket families; no free-form CAD claim",
        "grid_pitch_mm": PITCH_MM,
        "families": sorted({bundle.task.family for bundle in tasks}),
        "tasks": entries,
    }


def build_manifest(tasks=None):
    """Deterministic v2 task/design declarations; not a posted-qualification claim."""
    tasks = generate_tasks() if tasks is None else tuple(tasks)
    if tasks and all(bundle.generation_version == BOOTSTRAP_VERSION for bundle in tasks):
        return build_bootstrap_manifest(tasks)
    if not tasks or any(bundle.generation_version != GENERATION_VERSION for bundle in tasks):
        raise ValueError("v2 manifest rejects empty or mixed generator-version cohorts")
    if len({bundle.generator_seed for bundle in tasks}) != 1:
        raise ValueError("one manifest must have exactly one declared generator seed")
    for left_index, left in enumerate(tasks):
        for right in tasks[left_index + 1:]:
            if design_distance(left, right) < MIN_DESIGN_DISTANCE:
                raise ValueError("manifest bundles violate declared design distance")
    entries = []
    task_details = {}
    for bundle in tasks:
        task = bundle.task
        params = task.operation_target.geometry.params
        actual = design_parameters_from_task(task)
        if actual != bundle.design_parameters:
            raise ValueError(f"design vector does not match task source geometry: {task.task_id}")
        # Keep the existing frozen-protocol task_registry's exact allowed
        # entry fields. Richer versioned declarations live in task_details,
        # keyed by the same task ID and hash rather than smuggled into seed_spec.
        entries.append({
            "task_id": task.task_id,
            "family": task.family,
            "split": task.split.value,
            "task_hash": task.task_hash,
            "task_record": json.loads(C.dumps(task)),
            "seed_spec": bundle.seed_spec.as_dict(),
            "motion_hash": bundle.motion.motion_hash,
            "grid_excluded_volume_mm3": task.operation_target.excluded_volume_mm3,
            "continuous_exclusion_diagnostic_mm3": params["continuous_excluded_volume_mm3"],
            "evaluator_exclusion_tolerance_mm3": params["grid_exclusion_tolerance_mm3"],
            "seed_provenance": bundle.motion.provenance,
        })
        if task.task_id in task_details:
            raise ValueError(f"duplicate task_detail identifier: {task.task_id}")
        task_details[task.task_id] = {
            "schema_version": "t2.3-task-detail/1",
            "task_id": task.task_id,
            "family": task.family,
            "split": task.split.value,
            "case_index": bundle.case_index,
            "topology": bundle.topology,
            "design_seed": bundle.design_seed,
            "design_parameters": dict(bundle.design_parameters),
            "task_hash": task.task_hash,
            "tool_setup_hash": task.tool_setup_hash,
            "motion_hash": bundle.motion.motion_hash,
            "grid_excluded_volume_mm3": task.operation_target.excluded_volume_mm3,
            "continuous_exclusion_diagnostic_mm3": params["continuous_excluded_volume_mm3"],
            "evaluator_exclusion_tolerance_mm3": params["grid_exclusion_tolerance_mm3"],
            "accessibility": {
                "operation_target_rule": params["accessibility"],
                "holder_clearance": params["holder_clearance"],
                "cutter": {
                    "kind": task.tool.cutter_kind.value,
                    "radius_mm": task.tool.cutter_radius_mm,
                    "flute_length_mm": task.tool.flute_length_mm,
                },
                "shank": {
                    "radius_mm": task.tool.shank_radius_mm,
                    "height_mm": max(0.0, task.tool.holder_bottom_mm - task.tool.flute_length_mm),
                },
                "holder": {
                    "bottom_mm": task.tool.holder_bottom_mm,
                    "radius_mm": task.tool.holder_radius_mm,
                    "height_mm": task.tool.holder_height_mm,
                },
                "assembly_check": (
                    "full-motion hard-sim radial cutter/holder geometry and holder interval screen; "
                    "declared shank bottom/height and noncutting clearance; not just vertical visibility "
                    "or a continuous collision proof"
                ),
            },
            "cheap_gate": {
                "executor": CHEAP_GATE_EXECUTOR,
                "settings": dict(CHEAP_GATE_SETTINGS),
                "custom_target_scoring": "actual hard-sim remaining/removed masks on exact declared 0.25 mm task grid",
            },
            "posted_qualification": "pending",
            "predeclared_exclusion_reasons": list(PREDECLARED_EXCLUSION_REASONS),
        }
    distances = [design_distance(left, right) for index, left in enumerate(tasks)
                 for right in tasks[index + 1:]]
    per_family_counts = {
        family: {split: sum(t.task.family == family and t.task.split.value == split for t in tasks)
                 for split in ("dev", "test")}
        for family in sorted({bundle.task.family for bundle in tasks})
    }
    bootstrap = generate_bootstrap_tasks()
    bootstrap_text = _json(build_bootstrap_manifest(bootstrap))
    return {
        "schema_version": "t2.3-procedural/2",
        "generation_version": GENERATION_VERSION,
        "generator_seed": tasks[0].generator_seed,
        "per_family_counts": per_family_counts,
        "default_counts": {
            "dev_per_family": DEFAULT_DEV_PER_FAMILY,
            "test_per_family": DEFAULT_TEST_PER_FAMILY,
        },
        "count_parameterization": "generate_tasks(dev_per_family, test_per_family, seed=...), and matching CLI flags",
        "suite_count_policy": "declared generation defaults; freeze at T4.2, not an immutable v1 cap",
        "scope": "bounded analytic stock/pocket; open oblong or zero/one square retained pad; no free-form or multi-pad claim",
        "grid_pitch_mm": PITCH_MM,
        "design_distance": {
            "metric": "maximum abs(component_delta)/declared_scale over actual stock, pocket and physical retained-pad variants",
            "scales": dict(DESIGN_SCALES),
            "units": "mm per scale entry; retained count uses a one-pad unit",
            "minimum": MIN_DESIGN_DISTANCE,
            "observed_minimum": min(distances) if distances else None,
            "pairs_checked": len(distances),
            "family_and_case_index_in_metric": False,
        },
        "disjoint_split_rule": "within each family, every dev/test design differs physically and meets the global minimum-distance guard",
        "predeclared_exclusion_reasons": list(PREDECLARED_EXCLUSION_REASONS),
        "performance_based_exclusion": "none; geometry/setup/seed qualification only",
        "cheap_gate": {
            "executor": CHEAP_GATE_EXECUTOR,
            "settings": dict(CHEAP_GATE_SETTINGS),
            "posted_evaluation": False,
            "results_location": "fresh generation_report.json from this generator's cheap gate",
        },
        "task_projection": "legacy frozen-protocol-compatible task entries; versioned detail is separate",
        "task_details_schema": "t2.3-task-detail/1",
        "task_details": task_details,
        "pending_posted_qualification": {
            "status": "pending",
            "coordinator_batch": True,
            "grid_pitch_mm": 0.125,
            "dialects": [dialect.value for dialect in C.Dialect],
        },
        "bootstrap_preservation": {
            "mode": "named and reproducible v1 definition subset, not silently recounted into v2",
            "generator": "generate_bootstrap_tasks() / build_bootstrap_manifest()",
            "generation_version": BOOTSTRAP_VERSION,
            "task_hashes": [bundle.task.task_hash for bundle in bootstrap],
            "motion_hashes": [bundle.motion.motion_hash for bundle in bootstrap],
            "manifest_sha256": C.sha256_bytes(bootstrap_text.encode("utf-8")),
        },
        "families": sorted({bundle.task.family for bundle in tasks}),
        "tasks": entries,
    }


def manifest_text(tasks=None):
    return _json(build_manifest(tasks))


def bootstrap_manifest_text():
    return _json(build_bootstrap_manifest())




def write_manifest(path=None, expected_text=None):
    """Create/verify a deterministic manifest without overwriting a changed file."""
    path = PROJECT_ROOT / "configs" / "task_manifest.json" if path is None else Path(path)
    expected = manifest_text() if expected_text is None else expected_text
    if path.exists() and path.read_text(encoding="utf-8") != expected:
        raise FileExistsError(f"manifest changed since generation; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def verify_manifest(path=None, *, expected_text=None):
    path = PROJECT_ROOT / "configs" / "task_manifest.json" if path is None else Path(path)
    expected = manifest_text() if expected_text is None else expected_text
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
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
        "verification/reference_carve.py",
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
            "generation_version": bundle.generation_version,
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
        "generation_version": bundle.generation_version,
        "generator_seed": bundle.generator_seed,
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


def _worker_hwm_bytes(pid=None):
    """Own/current child VmHWM, not ru_maxrss inherited across fork/exec."""
    path = Path("/proc/self/status" if pid is None else f"/proc/{pid}/status")
    try:
        for line in path.read_text().splitlines():
            if line.startswith("VmHWM:"):
                fields = line.split()
                if len(fields) >= 3 and fields[-1] == "kB":
                    peak = int(fields[-2]) * 1024
                    return peak if peak > 0 else None
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def _ensure_f64_cpu():
    runtime = ti_impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f64)
        runtime = ti_impl.get_runtime()
    if runtime.default_fp != ti.f64:
        raise RuntimeError("cheap hard-sim qualification requires a fresh f64 CPU process")


def _finite_stock_history(values, segments):
    history = np.asarray(values)
    _require_seed(history.ndim == 4 and history.shape[0] == segments + 1,
                  "hard-stock history has the wrong dimensions")
    _require_seed(history.dtype.kind in "fiu" and np.isfinite(history).all(),
                  "non-finite or nonnumeric hard-stock history cannot be scored as removal")
    return history


class _CheapWorkerFailure(RuntimeError):
    def __init__(self, message, *, failure_kind, error_type=None,
                 worker_traceback=None, worker_pid=None, own_hwm_bytes=None):
        super().__init__(message)
        self.failure_kind = failure_kind
        self.worker_error_type = error_type
        self.worker_traceback = worker_traceback
        self.worker_pid = worker_pid
        self.worker_own_hwm_bytes = own_hwm_bytes


def _worker_failure(result, kind, message):
    return _CheapWorkerFailure(
        message, failure_kind=kind,
        error_type=result.get("error_type") if isinstance(result.get("error_type"), str) else None,
        worker_traceback=(result.get("traceback")
                          if isinstance(result.get("traceback"), str) else None),
        worker_pid=result.get("worker_pid"),
        own_hwm_bytes=result.get("worker_hwm_bytes"),
    )


def _worker_evaluate(request, request_sha256):
    """Execute one fresh, bounded, canonical typed job without pickle state."""
    task = C.loads(json.dumps(request["task"]))
    motion = C.loads(json.dumps(request["motion"]))
    if not isinstance(task, C.TaskSpec) or not isinstance(motion, C.MotionPlan):
        raise ValueError("worker requires canonical TaskSpec and MotionPlan records")
    _require_seed(request["generation_version"] in (BOOTSTRAP_VERSION, GENERATION_VERSION),
                  "worker generator version is unsupported")
    _require_seed(task.source.origin.startswith(request["generation_version"] + ":"),
                  "task source provenance does not match worker generator version")
    _require_seed(task.grid.frame == C.Frame.WORK and task.grid.voxel_mm == PITCH_MM,
                  "cheap qualification requires the exact 0.25 mm work-frame task grid")
    _require_seed(bool(motion.segments), "nominal seed contains no executable segments")
    for key in ("generator_seed", "design_seed", "case_index"):
        value = request[key]
        _require_seed(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                      f"worker {key} must be a non-negative integer")
    _require_seed(design_parameters_from_task(task) == request["design_parameters"],
                  "worker design vector differs from actual typed geometry")
    motion.validate_for_task(task)
    _require_seed(seed_for_task(task).motion_hash == motion.motion_hash,
                  "worker seed recipe differs from retained motion")
    _require_seed(all(segment.kind != C.SegmentKind.ARC for segment in motion.segments),
                  "cheap hard-sim does not replace arcs with chords")
    _require_seed(task.tool.cutter_kind == C.CutterKind.FLAT,
                  "cheap seed qualification requires a flat cutter")
    _require_seed(task.tool.shank_radius_mm == task.tool.cutter_radius_mm,
                  "cheap hard-sim supports the declared equal-radius shank")
    _require_seed(task.tool.holder_bottom_mm == task.tool.flute_length_mm,
                  "cheap hard-sim requires the declared flute-to-holder junction")
    _require_seed(np.allclose(task.work_to_machine.matrix, np.eye(4), atol=1e-12, rtol=0.0),
                  "cheap hard-sim currently requires the declared identity work transform")
    stock_top = task.stock.params["max_mm"][2]
    for segment in motion.segments:
        if segment.kind == C.SegmentKind.RAPID:
            _require_seed(min(segment.start_mm[2], segment.end_mm[2]) > stock_top,
                          "rapid segments must remain strictly above stock")
    expected_machine = np.asarray(expected_positions_from_motion(motion), dtype=np.float64)
    inverse = task.work_to_machine.inverse()
    expected_work = np.asarray([inverse.apply(point) for point in expected_machine])
    origin = np.asarray(task.grid.origin_mm, dtype=np.float64)
    extent = np.asarray(task.grid.extent_mm, dtype=np.float64)
    expected = (expected_work - origin) / extent
    _require_seed(np.isfinite(expected).all(), "normalized seed motion is non-finite")
    cells = math.prod(task.grid.shape)
    history_bytes = cells * (len(motion.segments) + 1) * 8
    _require_seed(cells <= 1_000_000 and history_bytes <= CHEAP_HISTORY_BUDGET_BYTES,
                  f"cheap grid/history budget exceeded: {cells} cells, {history_bytes} bytes")

    started = time.perf_counter()
    stock_field, target_field, required, retained = source_fields(task)
    _ensure_f64_cpu()
    with tempfile.TemporaryDirectory(prefix="diffcam-t23b-sampled-source-") as directory:
        sources = {}
        for name, field in (("stock", stock_field), ("target", target_field)):
            values = np.ascontiguousarray(field, dtype=np.float32)
            _require_seed(np.isfinite(values).all(), "non-finite sampled source field")
            path = Path(directory) / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            sources[name] = C.GeometrySource(
                kind="sampled", path=str(path), grid=task.grid,
                sha256=C.sha256_bytes(path.read_bytes()),
            )
        sim = CSGSimulatorDelta(
            resolution=max(task.grid.shape), voxel_size_mm=task.grid.voxel_mm,
            stock_size_mm=extent, stock_box_origin_mm=origin,
            max_steps=len(motion.segments), tool_start=tuple(expected[0]),
            target_geometry=sources["target"], stock_geometry=sources["stock"],
            workspace_mm=tuple(task.limits.envelope_max_mm[a] - task.limits.envelope_min_mm[a]
                               for a in range(3)),
            enforce_speed_limits=False, init_taichi=False, precision="f64",
        )
        sim.tool_radius[None] = task.tool.cutter_radius_mm
        sim.tool_height[None] = task.tool.flute_length_mm
        sim.tool_cut_height[None] = task.tool.flute_length_mm
        sim.holder_radius[None] = task.tool.holder_radius_mm
        sim.holder_height[None] = task.tool.holder_height_mm
        sim.enforce_z_floor[None] = 0
        precision = simulator_precision_metadata(sim, requested_precision="f64")
        sim.tool_delta.from_numpy(simulator_array(sim, np.diff(expected, axis=0)))
        sim.bake_target_grid()
        _require_seed(np.array_equal(sim.target.to_numpy() < 0.0, target_field < 0.0),
                      "baked target-grid mask differs from the declared source")
        sim.forward_hard(len(motion.segments) + 1, clip_speeds=False)
        actual_work = sim.tool_pos.to_numpy()[:len(motion.segments) + 1] * extent + origin
        try:
            np.testing.assert_allclose(actual_work, expected_work, atol=1e-10, rtol=0.0)
        except AssertionError as error:
            raise RuntimeError(f"hard-sim changed the exact-stop nominal path: {error}") from error
        history = _finite_stock_history(sim.stock.to_numpy(), len(motion.segments))
        initial = history[0] < 0.0
        remaining = history[-1] < 0.0
        _require_seed(np.array_equal(initial, stock_field <= 0.0),
                      "initial stock-grid mask differs from declared source")
        del history
        holder = float(sim.holder_overlap_total_mm3(len(motion.segments)))
        target_values = sim.target.to_numpy()  # Validate finite grid evidence, not scalar masks alone.
        _require_seed(np.isfinite(target_values).all(), "non-finite baked target grid")
        del target_values, sim
    seconds = time.perf_counter() - started
    initial_mask = stock_field <= 0.0
    target = retained & initial_mask
    removed = initial_mask & ~remaining
    _require_seed(np.all(remaining <= initial_mask), "hard-sim gained stock")
    _require_seed(not np.any(required & target), "removal and retained target masks overlap")
    _require_seed(np.array_equal(initial_mask & ~required, target),
                  "required-removal and retained-target masks disagree")
    original = original_part_field(task) <= 0.0
    _require_seed(np.all(original <= target), "reachable operation target omits original P")
    excluded_cells = int(np.count_nonzero(target & ~original))
    voxel_volume = float(task.grid.voxel_mm ** 3)
    excluded = float(excluded_cells * voxel_volume)
    params = task.operation_target.geometry.params
    _require_seed(excluded_cells == params["grid_excluded_voxels"], "grid exclusion count changed")
    _require_seed(excluded == params["grid_excluded_volume_mm3"] ==
                  task.operation_target.excluded_volume_mm3, "exclusion volume changed")
    _require_seed(params["grid_exclusion_tolerance_mm3"] == 0.0,
                  "cheap qualification requires exact zero exclusion tolerance")
    _require_seed(excluded_cells > 0, "original and reachable target are not distinct")
    in_target = int(np.count_nonzero(initial_mask & target))
    removal_target = int(np.count_nonzero(initial_mask & ~target))
    _require_seed(in_target > 0 and removal_target > 0, "degenerate target denominators")
    residual_cells = int(np.count_nonzero(remaining & ~target))
    gouge_cells = int(np.count_nonzero(removed & target))
    residual, gouge = float(residual_cells * voxel_volume), float(gouge_cells * voxel_volume)
    _require_seed(residual <= task.tolerances.completion_residual_mm3,
                  f"hard-sim residual {residual} mm^3 exceeds task tolerance")
    _require_seed(gouge <= task.tolerances.gouge_volume_mm3,
                  f"hard-sim gouge {gouge} mm^3 exceeds task tolerance")
    _require_seed(math.isfinite(holder) and holder >= 0.0 and holder <= 0.0,
                  f"conservative holder interval is nonzero or invalid: {holder} mm^3")
    _require_seed(all(math.isfinite(value) and value >= 0.0
                      for value in (residual, gouge, excluded, seconds)), "non-finite cheap metrics")
    holder_info = params["holder_clearance"]
    _require_seed(holder_info["above_stock_mm"] >= holder_info["qualification_and_cell_margin_mm"],
                  "holder/shank bottom no longer clears the stock")
    return {
        "status": "qualified_cheap_hard_sim",
        "task_hash": task.task_hash,
        "tool_setup_hash": task.tool_setup_hash,
        "motion_hash": motion.motion_hash,
        "residual_volume_mm3": residual,
        "gouge_volume_mm3": gouge,
        "residual_fraction": residual_cells / removal_target,
        "gouge_fraction": gouge_cells / in_target,
        "residual_cells": residual_cells,
        "gouge_cells": gouge_cells,
        "excluded_volume_mm3": excluded,
        "excluded_cells": excluded_cells,
        "removed_volume_mm3": float(np.count_nonzero(removed) * voxel_volume),
        "holder_overlap_interval_sum_mm3": holder,
        "holder_overlap_semantics": (
            "conservative whole-segment interval sum in mm^3, not unique chronological "
            "collision volume; zero required in addition to the seed's noncutting accessibility guards"
        ),
        "rapid_accessibility": "every rapid segment is strictly above the stock top",
        "source_hashes": {name: source.sha256 for name, source in sources.items()},
        "source_payload_dtype": "float32",
        "source_grid_declaration": json.loads(C.canonical_json(task.grid)),
        "simulator_precision": precision,
        "worker_seconds": seconds,
        "cheap_executor": CHEAP_GATE_EXECUTOR,
        "cheap_settings": dict(CHEAP_GATE_SETTINGS),
    }


def _stop_owned_worker(process):
    """Terminate only the Popen child this controller started, if still live."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _request_document(bundle):
    if not isinstance(bundle, TaskBundle):
        raise TypeError("cheap qualification requires a TaskBundle")
    return {
        "schema_version": "t2.3-cheap-job/1",
        "generation_version": bundle.generation_version,
        "generator_seed": bundle.generator_seed,
        "design_seed": bundle.design_seed,
        "case_index": bundle.case_index,
        "design_parameters": dict(bundle.design_parameters),
        "task": json.loads(C.dumps(bundle.task)),
        "motion": json.loads(C.dumps(bundle.motion)),
    }


def cheap_qualify_seed(bundle, *, log_directory=None):
    """Run one fresh bounded f64-CPU worker for one typed nominal seed."""
    started = time.perf_counter()
    request = _request_document(bundle)
    payload = _json(request).encode("utf-8")
    request_hash = C.sha256_bytes(payload)
    with tempfile.TemporaryDirectory(prefix="diffcam-t23b-job-") as directory:
        job = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        job.write_bytes(payload)
        if log_directory is None:
            log = Path(directory) / "worker.log"
        else:
            log_directory = Path(log_directory)
            log_directory.mkdir(parents=True, exist_ok=True)
            log = log_directory / f"{bundle.task.task_id}.worker.log"
        env = os.environ.copy()
        for name in ("TI_ARCH", "DIFFCAM_REAL_NOMINAL", "DIFFCAM_REAL_PILOT_SMOKE",
                     "DIFFCAM_PRECISION_COST"):
            env.pop(name, None)
        command = [
            sys.executable, "-I", "-m", "benchmarks.procedural_suite", "cheap-worker",
            "--job", str(job), "--output", str(response),
            "--repository", str(PROJECT_ROOT),
        ]
        process = None
        return_code = None
        peak = None
        try:
            with log.open("xb") as child_log:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env,
                                           stdin=subprocess.DEVNULL, stdout=child_log,
                                           stderr=subprocess.STDOUT)
            deadline = time.monotonic() + CHEAP_WORKER_TIMEOUT_S
            while True:
                sample = _worker_hwm_bytes(process.pid)
                if sample is not None:
                    peak = max(sample, peak or 0)
                    if peak > CHEAP_WORKER_MEMORY_LIMIT_BYTES:
                        raise RuntimeError(
                            f"cheap worker exceeded {CHEAP_WORKER_MEMORY_LIMIT_BYTES} own VmHWM bytes")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"cheap worker exceeded {CHEAP_WORKER_TIMEOUT_S} seconds")
                return_code = process.poll()
                if return_code is not None:
                    break
                time.sleep(0.05)
        finally:
            if process is not None:
                _stop_owned_worker(process)
        if not response.is_file():
            tail = log.read_text(encoding="utf-8", errors="replace")[-4096:]
            raise RuntimeError(
                f"cheap worker exited without a typed result (exit {return_code}, "
                f"logs {log}): {tail}"
            )
        try:
            result = json.loads(response.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            raise RuntimeError(f"cheap worker produced malformed evidence: {error}") from error
    if not isinstance(result, dict) or result.get("schema_version") != "t2.3-cheap-result/1":
        raise RuntimeError("unrecognized cheap worker result schema")
    if (result.get("request_sha256") != request_hash
            or result.get("worker_pid") != process.pid):
        raise RuntimeError("cheap worker result/job identity mismatch")
    status = result.get("status")
    if status != "failed" and return_code != 0:
        raise _worker_failure(
            result, "worker_process",
            f"cheap worker exited {return_code}; a qualified response requires exit 0")
    own_peak = result.get("worker_hwm_bytes")
    if own_peak is not None and (isinstance(own_peak, bool) or not isinstance(own_peak, int)):
        raise _worker_failure(result, "worker_protocol", "invalid worker own-VmHWM evidence")
    if status == "failed" and return_code not in (0, 1):
        raise _worker_failure(result, "worker_protocol",
                              f"invalid process status for a typed failure: {return_code}")
    if result.get("status") == "failed":
        kind = "seed_qualification" if return_code == 1 else "worker_protocol"
        raise _worker_failure(result, kind,
                              str(result.get("error", "cheap worker failed without a reason")))
    if status != "qualified_cheap_hard_sim":
        raise _worker_failure(result, "worker_protocol", f"unrecognized worker status: {status}")
    if (isinstance(own_peak, bool) or not isinstance(own_peak, int) or own_peak <= 0
            or own_peak > CHEAP_WORKER_MEMORY_LIMIT_BYTES):
        raise _worker_failure(result, "worker_protocol",
                              "cheap worker own-VmHWM evidence is missing or exceeds its bound")
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, dict):
        raise RuntimeError("cheap worker omitted its evaluation")
    for key in ("task_hash", "motion_hash"):
        if key == "task_hash":
            expected = bundle.task.task_hash
        else:
            expected = bundle.motion.motion_hash
        if evaluation.get(key) != expected:
            raise RuntimeError(f"cheap worker {key} mismatch")
    elapsed = time.perf_counter() - started
    return {
        **evaluation,
        "worker_pid": process.pid,
        "worker_own_hwm_bytes": own_peak,
        "controller_observed_hwm_bytes": peak,
        "memory_limit_bytes": CHEAP_WORKER_MEMORY_LIMIT_BYTES,
        "worker_timeout_s": CHEAP_WORKER_TIMEOUT_S,
        "controller_wall_seconds": elapsed,
        "worker_log": str(log),
        "request_sha256": request_hash,
        "posted_qualification": "pending",
        "fresh_process": True,
        "process_parallelism": 1,
    }


def _cheap_worker_main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Internal bounded T2.3 nominal worker")
    parser.add_argument("command", choices=("cheap-worker",))
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    request_hash = None
    started = time.perf_counter()
    try:
        if Path(__file__).resolve() != (args.repository.resolve() / "benchmarks" /
                                       "procedural_suite.py").resolve():
            raise RuntimeError("worker resolved a different source worktree")
        if not args.job.is_file() or args.job.stat().st_size > 2_000_000:
            raise ValueError("worker request is missing or exceeds its declaration-size bound")
        payload = args.job.read_bytes()
        request_hash = C.sha256_bytes(payload)
        request = json.loads(payload)
        expected = {"schema_version", "generation_version", "generator_seed", "design_seed",
                    "case_index", "design_parameters", "task", "motion"}
        if not isinstance(request, dict) or set(request) != expected:
            raise ValueError("worker request must contain exactly the declared typed fields")
        if request["schema_version"] != "t2.3-cheap-job/1":
            raise ValueError("unrecognized cheap-worker request schema")
        evaluation = _worker_evaluate(request, request_hash)
        report = {
            "schema_version": "t2.3-cheap-result/1",
            "status": "qualified_cheap_hard_sim",
            "request_sha256": request_hash,
            "worker_pid": os.getpid(),
            "worker_hwm_bytes": _worker_hwm_bytes(),
            "worker_wall_seconds": time.perf_counter() - started,
            "evaluation": evaluation,
        }
        exit_code = 0
    except Exception as error:
        report = {
            "schema_version": "t2.3-cheap-result/1",
            "status": "failed",
            "request_sha256": request_hash,
            "worker_pid": os.getpid(),
            "worker_hwm_bytes": _worker_hwm_bytes(),
            "worker_wall_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(_json(report))
    return exit_code


def _npy_bytes(array):
    import io
    stream = io.BytesIO()
    np.save(stream, np.asarray(array, dtype=np.float32))
    return stream.getvalue()


def verify_all(tasks=None, output_root=None):
    """Legacy bootstrap-only posted qualification; never used by the v2 CLI."""
    tasks = generate_bootstrap_tasks() if tasks is None else tuple(tasks)
    if any(bundle.generation_version != BOOTSTRAP_VERSION for bundle in tasks):
        raise ValueError("explicit legacy posted qualification requires bootstrap/1 bundles")
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "runs" / "procedural-bootstrap"
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    for bundle in tasks:
        case_root = root / bundle.task.task_id
        reports.append(_write_case(bundle, case_root))
    return reports


__all__ = [
    "Scenario", "TaskBundle", "GENERATION_VERSION", "BOOTSTRAP_VERSION", "GENERATOR_SEED",
    "DEFAULT_DEV_PER_FAMILY", "DEFAULT_TEST_PER_FAMILY", "FAMILIES", "PITCH_MM",
    "MIN_DESIGN_DISTANCE", "DESIGN_SCALES", "CHEAP_GATE_EXECUTOR", "CHEAP_GATE_SETTINGS",
    "SCENARIOS", "BOOTSTRAP_SCENARIOS", "make_task", "generate_scenarios", "generate_tasks",
    "generate_bootstrap_tasks", "design_distance", "design_parameters_from_task", "validate_designs",
    "cheap_qualify_seed", "build_manifest", "build_bootstrap_manifest", "manifest_text",
    "write_manifest", "verify_manifest", "evaluate_seed", "verify_all",
]


if __name__ == "__main__":
    raise SystemExit(_cheap_worker_main())
