"""One bounded gradient-mechanics dry run on the actual dev pocket with island.

This is tooling validation and cost measurement, not optimization, task
validation, a solver comparison, or equivalence proof of different objectives.
Gaussian/FD noise is in normalized T2.1 theta coordinates (scale_mm=1 here).

    uv run --extra optim --locked python -m scripts.run_gradient_mechanics \
        --output data/gradient-mechanics/t33-cost-probe

Run from the repository root of the T3.3 worktree. A fresh output directory is
required. Only two local physical-mm .npy sources are written alongside JSON;
no CAD source or heavy CAD dependency is needed.

CPU is the required backend. Taichi is initialized once per process and one
CSGSimulatorDelta is shared by all evaluations. The real target is a sampled
 operation field: the simulator has no analytic box_minus_pocket primitive.

The separate explicitly opted-in fixed-path study uses f64 and a configured
CPU thread limit of at most eight. It requires a complete prewritten declaration:
    uv run --extra optim --locked python -m scripts.run_gradient_mechanics \
        --output <fresh-durable-directory> --study-config <declared.json> \
        --study-mode preflight|full|sweep
Preflight performs exactly one real AD evaluation; it is not the matrix study.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence
import time

import numpy as np

from algorithms import gradient_mechanics as gm
from algorithms.run_artifacts import ComputeLedger, source_metadata
from benchmarks import contracts as C
from benchmarks import pilot_task
from cam import seeds as seed_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPSILONS = (0.025, 0.00625)
DEFAULT_INDICES = (0, 1, 2, 35)


@dataclass(frozen=True)
class CostConfig:
    """One explicitly declared, deliberately small cost-probe configuration."""
    pairs: int = 2
    reference_pairs: int = 8
    seed: int = 0
    sigma_theta: float = 0.05
    epsilon_theta: tuple[float, ...] = DEFAULT_EPSILONS
    fd_indices: tuple[int, ...] = DEFAULT_INDICES
    k_smooth: float = 40.0
    voxel_mm: float = pilot_task.GRID_PITCH


def validate_config(config: CostConfig, dimension: int | None = None) -> None:
    """Reject accidental budget expansion, silent sweep selection, or unit drift."""
    gm._positive_int(config.pairs, "pairs")
    if (isinstance(config.reference_pairs, bool)
            or not isinstance(config.reference_pairs, int)
            or config.reference_pairs <= config.pairs):
        raise ValueError("reference_pairs must be a larger positive integer")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, values in (("sigma_theta", np.atleast_1d(config.sigma_theta)),
                         ("epsilon_theta", np.atleast_1d(config.epsilon_theta))):
        a = np.asarray(values, dtype=np.float64)
        if a.size == 0 or not np.isfinite(a).all() or np.any(a <= 0):
            raise ValueError(f"{name} must contain finite positive theta units")
    if len(config.fd_indices) == 0 or len(set(config.fd_indices)) != len(config.fd_indices):
        raise ValueError("fd_indices must be nonempty and unique")
    if any(isinstance(i, bool) or not isinstance(i, int) for i in config.fd_indices):
        raise ValueError("fd_indices must be integers")
    if dimension is not None:
        if min(config.fd_indices) < 0 or max(config.fd_indices) >= dimension:
            raise ValueError(f"fd_indices must be within 0..{dimension - 1}")
        if np.asarray(config.sigma_theta).size not in (1, dimension):
            raise ValueError("sigma must be scalar or one sigma per normalized theta coordinate")
    if not isinstance(config.sigma_theta, (float, int)):
        # The bounded CLI takes one scalar. The generic module supports vectors.
        raise ValueError("this cost probe requires one explicit scalar sigma")
    for name in ("k_smooth", "voxel_mm"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def terminal_geometry_loss_numpy(stock, target, *, gouge_weight: float,
                                  residual_weight: float, loss_shift: float = 0.0):
    """Independent float64 terminal-loss/derivative reference for diagnostics.

    This mirrors the CSG simulator's fixed-target terminal gouge and residual
    terms on a declared grid. It is not a new production objective or a holder,
    trajectory, or machining-safety evaluation. Gradients are with respect to
    each voxel's stock SDF, before its sigmoid clamp. Clamp-tie coordinates are
    returned separately rather than assigned a fabricated classical derivative.
    """
    stock = np.asarray(stock, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if stock.shape != target.shape or stock.size == 0:
        raise ValueError("terminal diagnostic requires matching nonempty stock/target Float64 fields")
    if not np.isfinite(stock).all() or not np.isfinite(target).all():
        raise ValueError("terminal diagnostic fields must be finite")
    if not math.isfinite(loss_shift):
        raise ValueError("loss_shift must be finite")
    for name, weight in (("gouge_weight", gouge_weight), ("residual_weight", residual_weight)):
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    shifted = stock + loss_shift
    stock_occ = 1.0 / (1.0 + np.exp(np.clip(shifted, -50.0, 50.0)))
    target_occ = 1.0 / (1.0 + np.exp(np.clip(target, -50.0, 50.0)))
    gouge = target_occ * (1.0 - stock_occ)
    residual = (1.0 - target_occ) * stock_occ
    g = np.sum(gouge_weight * gouge * gouge) / stock.size
    r = np.sum(residual_weight * residual * residual) / stock.size
    # Use the exact two weighted terms' chain rule; avoid hand-simplified factors.
    apply_jacobian = np.divide(
        (2.0 * gouge_weight * gouge * -target_occ
         + 2.0 * residual_weight * residual * (1.0 - target_occ))
        * (-stock_occ * (1.0 - stock_occ)), stock.size,
    )
    inside_clamp = (shifted > -50.0) & (shifted < 50.0)
    gradient = np.where(inside_clamp, apply_jacobian, 0.0)
    return {"loss": float(g + r), "gouge": float(g), "residual": float(r),
            "gradient": gradient, "clamp_ties": (shifted == -50.0) | (shifted == 50.0),
            "interior_to_clamp": inside_clamp,
            "gouge_weight": float(gouge_weight), "residual_weight": float(residual_weight)}


def _write_source(root: Path, name: str, values, grid: C.GridSpec,
                  params: dict[str, Any]) -> C.GeometrySource:
    path = root / f"{name}.npy"
    array = np.ascontiguousarray(values, dtype="<f8")
    if array.shape != grid.shape or not np.isfinite(array).all():
        raise ValueError(f"sampled {name} must be finite on the exact task grid")
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    actual_hash = C.sha256_bytes(path.read_bytes())
    loaded = np.load(path, allow_pickle=False)
    if loaded.shape != array.shape or not np.array_equal(loaded, array):
        raise ValueError(f"serialized sampled {name} differs from its source")
    return C.GeometrySource(kind="sampled", path=str(path), sha256=actual_hash,
                            grid=grid, frame=C.Frame.WORK, params=dict(params))


def build_pilot_evaluator(root: Path, *, config: CostConfig = CostConfig(),
                          ledger: ComputeLedger | None = None, precision: str | None = None,
                          max_cpu_threads: int | None = None):
    """Construct the real sampled pilot, CPU simulator, and one evaluator.

    The optional voxel_mm allows a separately labelled reduced-grid smoke test;
    it never resamples or relabels a different .npy grid. The CLI's actual pilot
    uses the full 0.25 mm task grid.

    When supplied, ``precision`` selects both the runtime and public constructor.
    ``None`` preserves the prior caller/default-constructor behavior, including
    matched wrapper-provided defaults. The existing CandidateEvaluator continues
    assigning normalized delta inputs as float32.
    """
    t0 = time.perf_counter()
    if precision is not None and precision not in ("f32", "f64"):
        raise ValueError("precision must be explicit 'f32' or 'f64'")
    if max_cpu_threads is not None:
        if isinstance(max_cpu_threads, bool) or not isinstance(max_cpu_threads, int):
            raise ValueError("max_cpu_threads must be an integer in 1..8")
        if not 1 <= max_cpu_threads <= 8:
            raise ValueError("max_cpu_threads must be in 1..8 for this bounded CPU study")
    validate_config(config)
    root = root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing existing geometry/output directory: {root}")
    root.mkdir(parents=True)
    task = pilot_task.build_pilot_task(voxel_mm=config.voxel_mm)
    motion = seed_module.seed_for_task(task)
    parameterization = pilot_task.fit_parameterization(motion)
    validate_config(config, parameterization.num_params)
    if parameterization.scale_mm != 1.0:
        raise ValueError("cost probe expects the pilot's explicit 1 mm/theta-unit scale")
    if abs(task.tool.holder_bottom_mm - task.tool.flute_length_mm) > 1e-9:
        raise ValueError("this legacy simulator scope requires a flush holder at the flute top")
    if task.tool.cutter_kind != C.CutterKind.FLAT:
        raise ValueError("the differentiable pilot requires an upright flat cutter")
    stock, target, required, retained = pilot_task.source_fields(task)
    if np.any(required & retained):
        raise ValueError("pilot required/retained material masks overlap")
    stock_source = _write_source(root, "stock", stock, task.grid, task.stock.params)
    target_source = _write_source(root, "target", target, task.grid,
                                   task.operation_target.geometry.params)
    # Only S/T are sampled simulator inputs. The full analytic original P is
    # preserved unchanged; neither sampling nor fillet exclusions rewrite P.
    evaluated_task = replace(
        task, stock=stock_source,
        operation_target=replace(task.operation_target, geometry=target_source),
    )
    initial = np.asarray(parameterization.initial_position_mm, dtype=np.float64)
    start = (initial - np.asarray(task.grid.origin_mm)) / np.asarray(task.grid.extent_mm)
    import taichi as ti
    ti.init(arch=ti.cpu, offline_cache=False,
            default_fp=ti.f32 if precision is None else getattr(ti, precision),
            **({} if max_cpu_threads is None else {"cpu_max_num_threads": max_cpu_threads}))
    from simulator.csg_simulator import CSGSimulatorDelta

    sim = CSGSimulatorDelta(
        max_steps=parameterization.num_steps,
        stock_size_mm=task.grid.extent_mm,
        stock_box_origin_mm=task.grid.origin_mm,
        voxel_size_mm=task.grid.voxel_mm,
        # The public constructor requires a temporary analytic field shape.
        # target_geometry immediately installs and verifies the actual sampled
        # field; no analytic params are used for the baked target or scoring.
        target_shape="box",
        target_geometry=target_source,
        stock_geometry=stock_source,
        tool_start=start.tolist(),
        enforce_speed_limits=False,
        init_taichi=False,
        k_init=config.k_smooth,
        **({} if precision is None else {"precision": precision}),
    )
    sim.tool_radius[None] = task.tool.cutter_radius_mm
    sim.tool_height[None] = task.tool.flute_length_mm
    sim.tool_cut_height[None] = task.tool.flute_length_mm
    sim.holder_radius[None] = task.tool.holder_radius_mm
    sim.holder_height[None] = task.tool.holder_height_mm
    sim.bake_target_grid()
    if sim.target_shape != "sampled":
        raise RuntimeError("pilot's stored sampled target was not installed")
    from algorithms import objectives as objective_module
    evaluator = objective_module.CandidateEvaluator(
        evaluated_task, motion, parameterization, sim,
        spec=objective_module.ObjectiveSpec(k=config.k_smooth),
        ledger=ledger or ComputeLedger(), seed=config.seed,
    )
    actual_precision = getattr(sim, "precision", precision or "f32")
    from taichi.lang import impl as taichi_impl
    compile_config = taichi_impl.current_cfg()
    actual_threads = getattr(compile_config, "cpu_max_num_threads", None)
    if actual_threads is None:
        raise RuntimeError("Taichi did not expose its configured CPU thread cap")
    actual_threads = int(actual_threads)
    if max_cpu_threads is not None and actual_threads != max_cpu_threads:
        raise RuntimeError(f"Taichi CPU thread cap differs from requested {max_cpu_threads}: {actual_threads}")
    metadata = {
        "task_id": task.task_id, "original_task_hash": task.task_hash,
        "sampled_evaluator_task_hash": evaluated_task.task_hash,
        "motion_hash": motion.motion_hash,
        "grid": {
            "shape": list(evaluated_task.grid.shape),
            "origin_mm": list(evaluated_task.grid.origin_mm),
            "voxel_mm": evaluated_task.grid.voxel_mm,
            "frame": evaluated_task.grid.frame.value,
            "sign_convention": evaluated_task.grid.sign_convention,
        },
        "target_source": {
            "kind": target_source.kind, "path": target_source.path,
            "sha256": target_source.sha256, "frame": target_source.frame.value,
        },
        "stock_source": {
            "kind": stock_source.kind, "path": stock_source.path,
            "sha256": stock_source.sha256, "frame": stock_source.frame.value,
        },
        "original_part_source": {
            "kind": task.original_part.kind,
            "params": dict(task.original_part.params),
            "preserved_not_replaced_by_sampled_target": True,
        },
        "grid_role": ("registered_pilot_grid" if task.grid.voxel_mm == pilot_task.GRID_PITCH
                      else "explicit_reduced_smoke_grid_not_main_cost_probe"),
        "target_sha256": target_source.sha256,
        "stock_sha256": stock_source.sha256,
        "nonzero_removable_stock": bool(np.any((stock <= 0) & (target > 0))),
        "nonzero_retained_target": bool(np.any((stock <= 0) & (target <= 0))),
        "target_shape": "sampled", "initial_position_mm": initial.tolist(),
        "parameterization": {
            "kind": type(parameterization).__name__, "parameter_shape": list(parameterization.parameter_shape),
            "num_params": parameterization.num_params, "num_steps": parameterization.num_steps,
            "scale_mm": parameterization.scale_mm, "bounds": None,
            "sigma_units": "T2.1 normalized theta; multiplied by scale_mm, not normalized stock coordinates",
            "gradient_units": "objective per normalized theta unit, with exact transposed Jacobian",
        },
        "simulator": {
            "backend": "cpu", "taichi": ti.__version__,
            "floating_point": "float32" if actual_precision == "f32" else "float64",
            "precision": actual_precision,
            "taichi_cpu_max_num_threads": actual_threads,
            "taichi_default_fp": str(getattr(compile_config, "default_fp", None)),
            "input_delta_dispatch": f"existing CandidateEvaluator normalized {actual_precision} assignment",
            "atomic_nondeterminism": "use tolerances, never bitwise assertions",
            "immutable_geometry_declared_at_initialization": True,
            "holder_assumption": "flute top == holder bottom in this scoped pilot",
        },
        "setup_wall_time_s": time.perf_counter() - t0,
    }
    metadata["minimum_physical_sdf_checks"] = {
        "stock_min": float(stock.min()), "stock_max": float(stock.max()),
        "target_min": float(target.min()), "target_max": float(target.max()),
    }
    for name in ("nonzero_removable_stock", "nonzero_retained_target"):
        if not metadata[name]:
            raise ValueError(f"not a material-removal task: {name} is false")
    return evaluated_task, motion, parameterization, evaluator, metadata


def _finite_array_json(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return [None if not math.isfinite(x) else float(x) for x in array.reshape(-1)]


def _verify_json(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return _finite_array_json(item)
        if isinstance(item, dict):
            return {key: convert(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(value) for value in item]
        return item
    return convert(value)


def _actual_path(evaluator, theta):
    typed = evaluator.candidate_motion(theta)
    expected, normalized = gm.execute_normalized_and_machine(
        typed, evaluator.task.grid.origin_mm, evaluator.dimensions,
    )
    actual_normalized = np.asarray(evaluator.sim.tool_pos.to_numpy(), dtype=np.float64)
    if actual_normalized.shape != normalized.shape or not np.isfinite(actual_normalized).all():
        raise AssertionError("unexpected actual simulator path shape or non-finite position")
    # CSG positions are normalized on the physical stock box. Convert back to
    # work-mm before comparing against this pilot's identity work->machine plan.
    actual_mm = (np.asarray(evaluator.task.grid.origin_mm)
                 + actual_normalized * np.asarray(evaluator.dimensions))
    result = {
        "max_normalized_error": float(np.linalg.norm(actual_normalized - normalized, axis=1).max()),
        "max_machine_error_mm": float(np.linalg.norm(actual_mm - expected, axis=1).max()),
        "motion_hash": typed.motion_hash,
    }
    result["same_within_tolerance"] = (
        result["max_normalized_error"] <= 2e-6 and result["max_machine_error_mm"] <= 0.002)
    return result


def run_pilot(output: Path, config: CostConfig = CostConfig()) -> dict[str, Any]:
    """Run the once-declared cost probe and write its immutable local record."""
    validate_config(config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite report directory: {output}")
    total_start = time.perf_counter()
    task, motion, parameterization, evaluator, metadata = build_pilot_evaluator(output, config=config)
    ledger = evaluator.ledger
    theta = np.zeros(parameterization.num_params, dtype=np.float64)
    start = time.perf_counter()
    estimate = gm.estimate_with_reference(
        evaluator, theta, config.sigma_theta, config.pairs, config.reference_pairs,
        seed=config.seed, ledger=ledger,
    )
    elapsed_estimator = time.perf_counter() - start
    if any(item.status != "ok" for item in (estimate["primary"], estimate["reference"])):
        raise RuntimeError("pilot antithetic estimator returned invalid candidates; see ledger, no gradient claim")
    start = time.perf_counter()
    verification = gm.verify_smooth_ad(
        evaluator, theta, config.epsilon_theta, indices=config.fd_indices,
        ledger=ledger, require_path_identity=True,
    )
    elapsed_verification = time.perf_counter() - start
    before_base = gm._snapshot(ledger)
    hard_base = evaluator.evaluate_candidate(theta, "pilot-cam-zero-hard-base", "hard")
    if not hard_base.valid or not math.isfinite(hard_base.objective):
        raise RuntimeError(f"hard base failed: {hard_base.invalid_reason}")
    baseline_path = _actual_path(evaluator, theta)
    if not baseline_path["same_within_tolerance"]:
        raise AssertionError("hard execution differs from the declared fixed-theta path")
    baseline_delta = gm.delta_snapshot(before_base, gm._snapshot(ledger))
    cross_surrogate = gm.gradient_alignment(
        verification["ad_gradient"], estimate["primary"].gradient,
    )
    cross_surrogate["meaning"] = (
        "Diagnostic comparison of a sampled-randomized-hard gradient and smooth AD. "
        "Different objectives/noise: NOT a proof of unbiased or equal derivatives."
    )
    config_json = {
        "name": "one-small-pilot-pocket-cost-probe",
        "pairs": config.pairs, "reference_pairs": config.reference_pairs,
        "seed": config.seed, "sigma_theta": config.sigma_theta,
        "epsilon_theta": list(config.epsilon_theta),
        "fd_indices_explicit_subset": list(config.fd_indices),
        "fd_full_gradient_claim": False,
        "k_smooth": config.k_smooth, "voxel_mm": config.voxel_mm,
        "objective_spec": asdict(evaluator.spec),
        "simulator_loss_shift": (float(evaluator.sim.loss_shift[None])
                                 if hasattr(evaluator.sim, "loss_shift") else None),
        "architecture": "cpu", "objective_kind_hard": "hard",
        "objective_kind_ad_and_fd": "one identical CandidateEvaluator smooth objective",
        "full_solver_study": False, "optimization_steps": 0, "validation_calls_requested": 0,
    }
    snapshot = ledger.snapshot()
    expected_hard = 2 * config.pairs + 2 * config.reference_pairs + 1
    if sum(entry["objective_kind"] == "hard" for entry in ledger.entries) != expected_hard:
        raise AssertionError("hard candidate ledger total does not equal 2N + 2M + explicit base")
    if any(entry["objective_kind"] != "smooth" or entry["backward_calls"] for entry in ledger.entries
           if entry["candidate_id"].startswith("fd-axis-")):
        raise AssertionError("FD simulation consumed a backward call or switched objective kind")
    replay = [entry for entry in ledger.entries if entry["candidate_id"] == "fixed-path-smooth-forward-replay"]
    if len(replay) != 1 or replay[0]["objective_kind"] != "smooth" or replay[0]["backward_calls"] != 0:
        raise AssertionError("same-theta replay must be exactly one forward-only smooth evaluation")
    report = {
        "status": "completed_kind_checked_cost_probe", "schema": "diffcam-gradient-mechanics/1",
        "run_id": "t33-pilot-pocket-one-cost-probe",
        "configuration": config_json, "task_and_geometry": metadata,
        "theta": {"values": theta.tolist(), "hash": gm.normalized_theta_hash(theta),
                  "immutable_cam_zero_residual": True, "motion_hash": motion.motion_hash},
        "randomized_hard": {
            "definition": "J_sigma(theta) = E[L_hard(theta + sigma*z)], z standard normal",
            "estimator": "mean antithetic (L(theta+sigma z)-L(theta-sigma z))*z/(2*sigma)",
            "implementation": "NumPy float64 parameter math; real simulator execution is float32",
            "primary": gm.estimate_to_json(estimate["primary"]),
            "reference": gm.estimate_to_json(estimate["reference"]),
            "primary_vs_reference": estimate["comparison"],
            "phase_ledger_delta": estimate["ledger_delta"],
            "phase_wall_time_s": elapsed_estimator,
        },
        "smooth_ad_fd": _verify_json(verification),
        "hard_base": {
            "objective_kind": "hard", "valid": hard_base.valid, "objective": hard_base.objective,
            "coverage": hard_base.coverage.value, "actual_path_match": baseline_path,
            "same_theta_as_smooth_ad": True,
            "purpose": "Only execute-path identity is compared across objective kinds; objective values are different",
            "ledger_delta": baseline_delta,
        },
        "different_objective_alignment": cross_surrogate,
        "smooth_verification_wall_time_s": elapsed_verification,
        "cost": {
            "setup_wall_time_s": metadata["setup_wall_time_s"],
            "randomized_hard_phase_wall_time_s": elapsed_estimator,
            "smooth_ad_fd_phase_wall_time_s": elapsed_verification,
            "elapsed_run_wall_time_s": time.perf_counter() - total_start,
            "counters_including_explicit_hard_base": {k: v for k, v in snapshot.items() if k != "entries"},
            "expected_hard_count_including_base": expected_hard,
            "base_hard_is_an_extra_one_evaluation": True,
            "ad_forward_path_replay_is_an_extra_smooth_forward_evaluation": True,
            "note_on_warmup": "ComputeLedger warmup is the first candidate, not the only compilation-bearing call",
        },
        "per_candidate_cost": ledger.entries,
        "limits": [
            "A bounded cost probe on selected FD coordinates, not the full gradient study",
            "Different smooth and randomized hard surrogates are not interchangeable or proven unbiased",
            "Taichi float32/array atomic nondeterminism is compared with tolerances, not bit equality",
            "This does not run optimization, claim task machinability, or certify physical machining safety",
            "Original part P is not replaced by the sampled operation field T",
        ],
        "source": source_metadata(PROJECT_ROOT),
    }
    C.canonical_json(report)  # Reject non-finite values before artifact commit.
    path = output / "gradient_mechanics.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(C.canonical_json(report) + "\n")
    return report


def ad_order_regression(output: Path, *, voxel_mm: float = 2.0,
                        ordered: bool = True, precision: str = "f32") -> dict[str, Any]:
    """Exercise real AD resets on an explicitly coarse copy of the pilot grid.

    ``ordered=True`` performs same-theta AD, then a different-theta forward,
    then AD at the original theta. ``False`` performs only that original AD.
    Callers must put the latter in a fresh process for a fresh-process control;
    this helper must not re-initialize Taichi or create a second simulator.

    The default 2 mm grid is independently sampled by the existing factory,
    not a resampling of the declared 0.25 mm pilot arrays. It is a regression
    for evaluation-state ordering, not a substitute for the full-grid cost
    artifact, a machining completion check, or an AD/finite-difference oracle.
    """
    if not isinstance(ordered, bool):
        raise TypeError("ordered must select the declared sequence explicitly")
    output = output.resolve()
    _, _, parameterization, evaluator, _ = build_pilot_evaluator(
        output, config=CostConfig(voxel_mm=voxel_mm), precision=precision,
    )
    theta = np.zeros(parameterization.num_params, dtype=np.float64)
    records = {}

    def ad_record(key: str, candidate_id: str):
        evaluated = evaluator.evaluate_candidate_with_gradient(theta, candidate_id)
        if not evaluated.result.valid:
            raise RuntimeError(f"AD regression candidate failed: {evaluated.result.invalid_reason}")
        path = _actual_path(evaluator, theta)
        if not path["same_within_tolerance"]:
            raise AssertionError(f"AD regression path mismatch: {path}")
        gradient = np.asarray(evaluated.gradient_theta, dtype=np.float64)
        if gradient.shape != theta.shape or not np.isfinite(gradient).all():
            raise AssertionError("AD regression requires a finite full theta gradient")
        records[key] = {
            "gradient": gradient.tolist(), "objective": float(evaluated.result.objective),
            "candidate_id": candidate_id, "theta_hash": evaluated.result.theta_hash,
            "actual_path": path,
        }

    ad_record("initial", "order-initial" if ordered else "fresh-process-control")
    if ordered:
        ad_record("repeat_same", "order-repeat-same")
        other = theta.copy()
        other[35] = 0.025
        changed = evaluator.evaluate_candidate(other, "order-different-theta", "smooth")
        if not changed.valid:
            raise RuntimeError(f"different-theta evaluation failed: {changed.invalid_reason}")
        other_path = _actual_path(evaluator, other)
        if not other_path["same_within_tolerance"]:
            raise AssertionError(f"different-theta path mismatch: {other_path}")
        if changed.theta_hash == records["initial"]["theta_hash"]:
            raise AssertionError("the intervening evaluation did not change theta")
        ad_record("after_other", "order-after-different")
    snapshot = evaluator.ledger.snapshot()
    expected = 4 if ordered else 1
    if snapshot["candidate_evaluations"] != expected:
        raise AssertionError("state-order regression consumed an unexpected candidate count")
    report = {
        "schema": "diffcam-ad-order-regression/1",
        "sequence": "same_then_different_then_same" if ordered else "fresh_process_control",
        "grid": {"voxel_mm": voxel_mm, "shape": list(evaluator.task.grid.shape),
                 "independently_sampled": True},
        "theta": theta.tolist(), "different_theta_index": 35,
        "different_theta_delta": 0.025 if ordered else None,
        "loss_dtype": evaluator.sim.loss.to_numpy().dtype.str,
        "stock_dtype": evaluator.sim.stock.to_numpy().dtype.str,
        "stock_gradient_dtype": evaluator.sim.stock.grad.to_numpy().dtype.str,
        "tool_delta_dtype": evaluator.sim.tool_delta.to_numpy().dtype.str,
        "tool_delta_gradient_dtype": evaluator.sim.tool_delta.grad.to_numpy().dtype.str,
        "simulator_precision": evaluator.sim.precision,
        "input_delta_dispatch": f"existing CandidateEvaluator normalized {evaluator.sim.precision} assignment",
        "records": records,
        "nonzero_gradient": any(abs(value) > 1e-10
                                for value in records["initial"]["gradient"]),
        "ledger": {key: value for key, value in snapshot.items() if key != "entries"},
        "candidate_order": [entry["candidate_id"] for entry in evaluator.ledger.entries],
        "source": source_metadata(PROJECT_ROOT),
    }
    if not math.isfinite(records["initial"]["objective"]) or not report["nonzero_gradient"]:
        raise AssertionError("degenerate AD regression cannot test stale state")
    path = output / "ad_order.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(C.canonical_json(report) + "\n")
    return report


# Fixed-path f64 study. This is a separate explicit opt-in, not an optimizer run.
STUDY_SCHEMA = "diffcam-fixed-path-gradient-study/1"
STUDY_MODES = ("full", "preflight", "sweep")
STUDY_MAX_RSS_BYTES = 8589934592


@dataclass(frozen=True)
class FixedStudyConfig:
    task_id: str
    voxel_mm: float
    precision: str
    cpu_max_num_threads: int
    k_values: tuple[float, ...]
    paths: tuple[Mapping[str, Any], ...]
    gouge_weight: float
    residual_weight: float
    holder_weight: float
    holder_margin_mm: float
    time_weight: float
    air_time_weight: float
    efficiency_path_ids: tuple[str, ...]
    zero_weight_path_id: str
    epsilon_theta: tuple[float, ...]
    fd_indices: tuple[int, ...]
    sigma_theta: float
    pairs: int
    reference_pairs: int
    hard_root_seed: int
    path_root_seed: int
    sweep_reference_kind: str
    sweep_fractions: tuple[float, ...]
    sweep_seeds: tuple[int, ...]
    sweep_root_seed: int
    max_wall_time_s: float
    max_rss_bytes: int = STUDY_MAX_RSS_BYTES
    reuse_k_independent_hard: bool = False


def _study_positive(value, where, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{where} must be a finite {'nonnegative' if allow_zero else 'positive'} number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{where} must be finite {'nonnegative' if allow_zero else 'positive'}")
    return value


def _study_index(value, where, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")
    return value


def study_config_from_json(document) -> FixedStudyConfig:
    if not isinstance(document, Mapping):
        raise ValueError("fixed-study config must be a JSON object")
    require = ("schema", "study_only", "task_id", "voxel_mm", "precision", "cpu_max_num_threads",
               "k_values", "paths", "gouge_weight", "residual_weight", "holder_weight", "holder_margin_mm",
               "time_weight", "air_time_weight", "efficiency_path_ids", "zero_weight_path_id",
               "study_only_efficiency", "epsilon_theta", "fd_indices", "sigma_theta", "pairs",
               "reference_pairs", "hard_root_seed", "path_root_seed", "scale_sweep", "max_wall_time_s")
    if set(document) - set(require) - {"reuse_k_independent_hard"}:
        raise ValueError(f"study config requires exactly the declared fields; extra={sorted(set(document)-set(require))}")
    if document["schema"] != STUDY_SCHEMA or document["study_only"] is not True:
        raise ValueError("this is a study-only fixed-path declaration, not a frozen protocol or campaign")
    if document["study_only_efficiency"] is not True:
        raise ValueError("efficiency weights must explicitly be study-only, never protocol weights")
    paths_value = document["paths"]
    if not isinstance(paths_value, list) or not paths_value:
        raise ValueError("study paths must be a nonempty ordered list")
    paths = []
    names = set()
    for index, raw in enumerate(paths_value):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            raise ValueError(f"paths[{index}] requires an id")
        import re
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", raw["id"]) is None or raw["id"] in names:
            raise ValueError(f"paths[{index}].id must be unique and filename-safe")
        names.add(raw["id"])
        kind = raw.get("kind")
        allowed = {"id", "kind", "seed", "scale_fraction", "reference_kind", "distribution", "exploratory"}
        if kind == "cam":
            allowed = {"id", "kind"}
        elif kind == "stationary_line":
            allowed = {"id", "kind"}
        elif kind not in ("cam_perturbed", "delta"):
            raise ValueError(f"unsupported study path kind {kind!r}")
        if set(raw) - allowed or "kind" not in raw:
            raise ValueError(f"paths[{index}] contains undeclared fields")
        if kind in ("cam_perturbed", "delta"):
            if raw.get("exploratory") is not True:
                raise ValueError("exploratory seed/perturbation paths require explicit exploratory=true")
            _study_positive(raw["scale_fraction"], f"paths[{index}].scale_fraction")
            _study_index(raw["seed"], f"paths[{index}].seed")
            if raw["reference_kind"] not in ("cam_delta_norm", "stock_xy_extent"):
                raise ValueError(f"paths[{index}].reference_kind is not executable")
            if raw["distribution"] not in ("gaussian", "uniform"):
                raise ValueError(f"paths[{index}].distribution is not executable")
        if kind == "delta" and raw.get("exploratory") is not True:
            raise ValueError("the D-08 straight-line study path must be explicitly marked exploratory")
        paths.append(dict(raw))
    if sum(path["kind"] == "cam" for path in paths) != 1:
        raise ValueError("exactly one nominal CAM fixed-path reference is required")
    if not any(path["kind"] == "cam_perturbed" for path in paths):
        raise ValueError("declare at least one CAM-fit perturbation")
    if sum(path["kind"] == "delta" for path in paths) != 1:
        raise ValueError("exactly one positive-scale delta/D-08 path is required")
    if not any(path["kind"] == "cam_perturbed" for path in paths):
        raise ValueError("declare at least one normalized CAM perturbation")
    for name in ("efficiency_path_ids", "zero_weight_path_id"):
        value = document[name]
        if name == "zero_weight_path_id":
            if value not in names:
                raise ValueError("zero_weight_path_id must name a retained fixed path")
        elif (not isinstance(value, list) or not value or set(value) - names
              or len(set(value)) != len(value)):
            raise ValueError("efficiency_path_ids must be unique retained fixed-path ids")
    sweep = document["scale_sweep"]
    if not isinstance(sweep, Mapping) or set(sweep) != {"reference_kind", "fractions", "seeds", "root_seed"}:
        raise ValueError("scale_sweep must declare reference, five fractions, thirty seeds and one root seed")
    if len(sweep["fractions"]) != 5 or len(set(sweep["fractions"])) != 5:
        raise ValueError("the small scale probe requires exactly five distinct scale fractions")
    if len(sweep["seeds"]) != 30 or len(set(sweep["seeds"])) != 30:
        raise ValueError("scale_sweep.seeds must contain thirty distinct seeds in declared order")
    config = FixedStudyConfig(
        task_id=document["task_id"], voxel_mm=document["voxel_mm"], precision=document["precision"],
        cpu_max_num_threads=document["cpu_max_num_threads"], k_values=tuple(document["k_values"]),
        paths=tuple(paths), gouge_weight=document["gouge_weight"], residual_weight=document["residual_weight"],
        holder_weight=document["holder_weight"], holder_margin_mm=document["holder_margin_mm"],
        time_weight=document["time_weight"], air_time_weight=document["air_time_weight"],
        efficiency_path_ids=tuple(document["efficiency_path_ids"]), zero_weight_path_id=document["zero_weight_path_id"],
        epsilon_theta=tuple(document["epsilon_theta"]), fd_indices=tuple(document["fd_indices"]),
        sigma_theta=document["sigma_theta"], pairs=document["pairs"], reference_pairs=document["reference_pairs"],
        hard_root_seed=document["hard_root_seed"], path_root_seed=document["path_root_seed"],
        sweep_reference_kind=sweep["reference_kind"], sweep_fractions=tuple(sweep["fractions"]),
        sweep_seeds=tuple(sweep["seeds"]), sweep_root_seed=sweep["root_seed"],
        max_wall_time_s=document["max_wall_time_s"],
        reuse_k_independent_hard=document.get("reuse_k_independent_hard", False),
    )
    validate_fixed_study(config)
    return config


def validate_fixed_study(config: FixedStudyConfig) -> None:
    if config.task_id != "dev-pocket-island-000-v2":
        raise ValueError("the fixed study is scoped to the declared pilot pocket task")
    if config.voxel_mm != pilot_task.GRID_PITCH or config.voxel_mm != 0.25:
        raise ValueError("the real study requires the actual 0.25 mm pilot grid, not a reduced-grid stand-in")
    if config.precision != "f64":
        raise ValueError("real study requires f64 precision")
    if isinstance(config.cpu_max_num_threads, bool) or not isinstance(config.cpu_max_num_threads, int):
        raise ValueError("cpu_max_num_threads must be an integer")
    if not 1 <= config.cpu_max_num_threads <= 8:
        raise ValueError("cpu_max_num_threads must be in 1..8")
    for index, value in enumerate(config.k_values):
        _study_positive(value, f"k_values[{index}]")
    if tuple(config.k_values) != (20.0, 40.0, 80.0, 160.0):
        raise ValueError("the bounded study requires exactly k={20,40,80,160} in declared order")
    for name in ("gouge_weight", "residual_weight", "holder_weight", "holder_margin_mm"):
        _study_positive(getattr(config, name), name, allow_zero=True)
    for name in ("time_weight", "air_time_weight"):
        _study_positive(getattr(config, name), name)
    for name in ("sigma_theta", "max_wall_time_s"):
        _study_positive(getattr(config, name), name)
    for name in ("hard_root_seed", "path_root_seed", "sweep_root_seed", "pairs", "reference_pairs"):
        _study_index(getattr(config, name), name, 1 if name.endswith("pairs") else 0)
    if (config.pairs, config.reference_pairs) != (16, 64):
        raise ValueError("the bounded study requires 16 primary pairs and a 64-pair reference")
    if len(config.epsilon_theta) == 0:
        raise ValueError("central FD needs an explicit epsilon sweep")
    for value in config.epsilon_theta:
        _study_positive(value, "FD epsilon in normalized theta units")
    if not config.fd_indices or len(set(config.fd_indices)) != len(config.fd_indices):
        raise ValueError("FD indices must be unique and nonempty")
    for index in config.fd_indices:
        _study_index(index, "FD index")
    if len(config.efficiency_path_ids) != 1:
        raise ValueError("minimum informative study uses one efficiency path and k=40")
    if config.sweep_reference_kind != "stock_xy_extent":
        raise ValueError("scale sweep uses the explicitly declared stock_XY reference")
    if config.sweep_fractions != tuple(sorted(config.sweep_fractions)):
        raise ValueError("retain declared ascending scale ordering")
    for fraction in config.sweep_fractions:
        _study_positive(fraction, "scale_sweep.fraction")
    for seed in config.sweep_seeds:
        _study_index(seed, "scale_sweep.seed")
    if config.max_rss_bytes != STUDY_MAX_RSS_BYTES:
        raise ValueError("D-07 process memory ceiling cannot be raised in this study")
    if not isinstance(config.reuse_k_independent_hard, bool):
        raise ValueError("reuse_k_independent_hard must be an explicit boolean")


def _straight_declaration() -> dict[str, Any]:
    return {"start": "cam_start", "end": "cam_end", "segment_placement": "uniform",
            "feed_and_rapid_entry": "retain allowed feeds and actual above-stock rapid entry",
            "same_segment_count_and_k": True, "safe_rapid_entry": True,
            "respect_task_feed_limits": True, "efficiency_not_required": True}


def _seed_declaration(path, root_seed: int) -> dict[str, Any]:
    where = "random_init" if path["kind"] == "delta" else "cam_perturbed_init"
    recipe = {
        "parameter_space": "delta" if path["kind"] == "delta" else "theta",
        "distribution": {"kind": path["distribution"], "detail": "iid"},
        "scale": {"fraction": path["scale_fraction"], "reference": {
            "kind": path["reference_kind"], "description": "fixed T3.3 study declaration; not a frozen protocol scale",
            "units": "mm"}},
        "seed_derivation_rule": {"algorithm": "sha256_v1", "root_seed": root_seed,
            "input": "task_hash,declaration_hash,root_seed,seed", "per_seed_rule": "first_8_bytes_little_endian_uint64"},
    }
    if where == "cam_perturbed_init":
        recipe.update(exploratory=True, excluded_from_init_axis_claims=True)
    return {"straight_line_init": _straight_declaration(), where: recipe}


def _process_memory() -> dict[str, Any]:
    values = {}
    with Path("/proc/self/status").open(encoding="ascii") as stream:
        for line in stream:
            if line.startswith(("VmHWM:", "VmRSS:")):
                name, value = line.split(":", 1)
                values[name] = int(value.strip().split()[0]) * 1024
    return {"own_process_vmhwm_bytes": values.get("VmHWM"),
            "own_process_current_vmrss_bytes": values.get("VmRSS"),
            "scope": "whole-process high-water/current RSS, not attributed per-row allocation and not inherited ru_maxrss"}


def _new_json(path: Path, document) -> None:
    # Fresh artifact only. All NaN/unselected values become explicit nulls.
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = C.canonical_json(_verify_json(document)).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded + b"\n")


def _dtype_information(sim) -> dict[str, Any]:
    records = {}
    for name in ("loss", "stock", "tool_delta"):
        for suffix, field in (("", getattr(sim, name)), (".grad", getattr(sim, name).grad)):
            dtype = getattr(field, "dtype", None)
            serialized = getattr(dtype, "str", str(dtype))
            records[name + suffix] = {"dtype": serialized,
                                      "is_float64": serialized in ("<f8", ">f8", "f64", "float64")}
    return records


def _prepare_study_path(path, task, base_evaluator, config: FixedStudyConfig) -> dict[str, Any]:
    declaration = (None if path["kind"] in ("cam", "stationary_line")
                   else _seed_declaration(path, config.path_root_seed))
    result = None
    if path["kind"] == "cam":
        motion = seed_module.seed_for_task(task)
        motion.validate_for_task(task)
        metadata = {"mode": "nominal_cam", "motion_hash": motion.motion_hash,
                    "provenance": motion.provenance, "source": "seed_module.seed_for_task"}
    else:
        if path["kind"] == "delta":
            result = seed_module.random_seed(task, declaration, path["seed"])
        elif path["kind"] == "cam_perturbed":
            result = seed_module.cam_perturbed_seed(task, declaration, path["seed"])
        else:
            result = seed_module.straight_line_seed(task, declaration)
        if not result.valid:
            return {"id": path["id"], "kind": path["kind"], "status": "invalid_source_motion",
                    "reason": result.invalid_reason, "declaration": declaration, "seed_result": result.to_record()}
        motion = result.require_plan()
        metadata = result.to_record()
    parameterization = pilot_task.fit_parameterization(motion)
    theta = np.zeros(parameterization.num_params, dtype=np.float64)
    interim = type(base_evaluator)(task, motion, parameterization, base_evaluator.sim,
                                   spec=base_evaluator.spec, ledger=base_evaluator.ledger,
                                   seed=base_evaluator.seed, requested_precision="f64")
    fixed_motion = interim.candidate_motion(theta)
    source_points = np.asarray(motion.waypoints_mm(), dtype=np.float64)
    fixed_points = np.asarray(fixed_motion.waypoints_mm(), dtype=np.float64)
    shape = source_points.shape == fixed_points.shape
    fit_error = (float(np.linalg.norm(fixed_points - source_points, axis=1).max()) if shape else None)
    declare = dict(path) if declaration is not None else None
    result_derivation = None if result is None else json.loads(result.derivation)
    if declare is not None:
        declare["reference_scale_metadata"] = (None if result is None else result_derivation.get("scale"))
        declare["noise_scale_metadata"] = (None if result is None else result_derivation.get("noise_scale"))
    return {"id": path["id"], "kind": path["kind"], "status": "ready",
            "source_motion": motion, "source_result": result,
            "source_hash": motion.motion_hash, "motion": fixed_motion,
            "parameterization": parameterization, "theta": theta,
            "source_record": metadata, "declaration": declare,
            "source_to_fixed_path_max_error_mm": fit_error,
            "theta_hash": gm.normalized_theta_hash(theta),
            "basis_sha256": C.content_hash(parameterization.linear_map.tolist()),
            "segment_count": len(motion.segments),
            "num_control_points": parameterization.num_control_points,
            "parameter_shape": list(parameterization.parameter_shape),
            "fixed_motion_hash": fixed_motion.motion_hash,
            "source_role": "actual source MotionPlan; fixed path is its parameterization at theta=0",
            "no_source_hash_substitution": True}


def _public_path(path) -> dict[str, Any]:
    return {key: value for key, value in path.items()
            if key not in {"source_motion", "motion", "parameterization", "theta", "source_result"}}


def scale_validity_sweep(task, config: FixedStudyConfig, *, generator=None) -> dict[str, Any]:
    """Only physical/structural MotionPlan validity, never optimality or σ rejection."""
    started = time.perf_counter()
    rows = []
    for fraction in config.sweep_fractions:
        declaration = _seed_declaration({
            "kind": "delta", "distribution": "gaussian", "scale_fraction": fraction,
            "reference_kind": config.sweep_reference_kind, "exploratory": True,
        }, config.sweep_root_seed)
        for seed in config.sweep_seeds:
            result = (generator(declaration, seed) if generator is not None
                      else seed_module.random_seed(task, declaration, seed))
            meta = json.loads(result.derivation)
            evidence = meta.get("evidence", meta)
            stats = evidence.get("deviations", {})
            rows.append({
                "fraction": fraction, "seed": seed, "valid": bool(result.valid),
                "status": "valid_structural_motion" if result.valid else "failed_physical_constraint",
                "invalid_reason": result.invalid_reason,
                "motion_hash": None if result.plan is None else result.plan.motion_hash,
                "declaration_hash": result.declaration_hash,
                "scale": evidence.get("scale"),
                "deviations": stats,
                "noise_scale": evidence.get("noise_scale"),
                "T": evidence.get("parameter_map", {}).get("steps"),
                "seed_derivation": evidence.get("seed_derivation"),
            })
            if evidence.get("noise_scale", {}).get("filter_applied") is not False:
                raise ValueError("scale validity sweep refuses any sigma-based filtering")
    summaries = []
    for fraction in config.sweep_fractions:
        selected = [row for row in rows if row["fraction"] == fraction]
        if tuple(row["seed"] for row in selected) != config.sweep_seeds:
            raise AssertionError("scale sweep changed or dropped requested seed order")
        valid = sum(row["valid"] for row in selected)
        summaries.append({"fraction": fraction, "reference_kind": config.sweep_reference_kind,
                          "per_coordinate_sigma_mm": next(row["scale"]["scale_mm"] for row in selected),
                          "attempted": len(selected), "valid": valid, "failed": len(selected) - valid,
                          "valid_rate": valid / len(selected),
                          "constraints": "envelope, feeds, safe entry, contiguity and validate_for_task; not machining success"})
    if len(rows) != 150 or any(row["T"] != 75 for row in rows):
        raise AssertionError("the declared sweep must retain all five-scale/thirty-seed T=75 outcomes")
    return {"schema": "diffcam-delta-validity-scale-sweep/1", "task_id": task.task_id,
            "task_hash": task.task_hash, "physical_validity_only": True,
            "source_region": "D-08 delta Gaussian, per-free-vertex iid about the CAM start/end line",
            "sigma_is_not_a_validity_filter": True, "no_retry_no_replacement": True,
            "independent_declaration_per_scale_not_paired_common_random_numbers": True,
            "summaries": summaries, "rows": rows, "wall_time_s": time.perf_counter() - started,
            "memory": _process_memory(), "source": source_metadata(PROJECT_ROOT)}


def _coordinate_errors(first, second, indices, *, epsilon_abs=1e-12) -> list[dict[str, Any]]:
    """AD-vs-estimator differences; undefined zero-denominator relative errors are null."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("coordinate-error rows require equal-sized one-dimensional gradients")
    selected = []
    for index in indices:
        if (isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer)) or not 0 <= int(index) < first.size):
            raise ValueError(f"invalid comparison coordinate index {index!r}")
        if index in selected:
            raise ValueError(f"duplicate comparison coordinate index {index}")
        selected.append(int(index))
    for index in selected:
        if not np.isfinite(first[index]) or not np.isfinite(second[index]):
            raise ValueError(f"selected coordinate {index} has no finite gradient pair")
    result = []
    for index in selected:
        a, b = float(first[index]), float(second[index])
        difference = b - a
        result.append({"index": int(index), "first": a, "second": b,
                       "signed_second_minus_first": difference, "absolute_error": abs(difference),
                       "relative_error": (abs(difference) / abs(a) if abs(a) > epsilon_abs else None),
                       "relative_denominator": "abs(AD) when greater than the declared absolute-near-zero threshold",
                       "epsilon_abs": epsilon_abs})
    return result


def _fixed_cases(config: FixedStudyConfig) -> list[dict[str, Any]]:
    cases = []
    counter = 0
    for path_index, path in enumerate(config.paths):
        for k_index, k in enumerate(config.k_values):
            cases.append({"case_index": counter, "path_index": path_index, "path_id": path["id"],
                          "k_index": k_index, "k": k, "objective_mode": "geometry"})
            counter += 1
    for path_index, mode in (
        (next(i for i, p in enumerate(config.paths) if p["id"] == config.zero_weight_path_id), "zero_weights"),
        (next(i for i, p in enumerate(config.paths) if p["id"] in config.efficiency_path_ids), "study_efficiency"),
    ):
        k_index = config.k_values.index(40.0)
        cases.append({"case_index": counter, "path_index": path_index, "k_index": k_index,
                      "path_id": config.paths[path_index]["id"], "k": 40.0, "objective_mode": mode})
        counter += 1
    return cases


def _objective_spec(config: FixedStudyConfig, case, reference_duration_s: float):
    from algorithms.objectives import ObjectiveSpec
    weights = {"gouge_weight": config.gouge_weight, "residual_weight": config.residual_weight,
               "holder_weight": config.holder_weight, "holder_margin_mm": config.holder_margin_mm}
    if case["objective_mode"] == "zero_weights":
        weights = {name: 0.0 for name in weights}
    return ObjectiveSpec(k=case["k"], **weights,
                         time_weight=config.time_weight if case["objective_mode"] == "study_efficiency" else 0.0,
                         air_time_weight=config.air_time_weight if case["objective_mode"] == "study_efficiency" else 0.0,
                         reference_cycle_time_s=reference_duration_s)


def _hard_probe_key(case, path, config: FixedStudyConfig) -> str:
    """Hard-mode identity excludes smoothing k and preserves all changing scopes."""
    return C.content_hash({
        "schema": "diffcam-shared-hard-probe-key/1", "task_id": config.task_id,
        "path_id": path["id"], "fixed_motion_hash": path["fixed_motion_hash"],
        "source_motion_hash": path["source_hash"], "theta_hash": path["theta_hash"],
        "basis_sha256": path["basis_sha256"], "objective_mode": case["objective_mode"],
        "precision": config.precision, "voxel_mm": config.voxel_mm,
        "sigma_theta": config.sigma_theta, "pairs": config.pairs,
        "reference_pairs": config.reference_pairs, "path_index": case["path_index"],
        "source": "same path, fixed theta, objective, sigma, grid and precision",
        "k_is_not_used_by_forward_hard": True,
    })


def _study_budget_guard(config: FixedStudyConfig, started: float, where: str) -> None:
    if time.perf_counter() - started > config.max_wall_time_s:
        raise TimeoutError(f"declared study wall-time budget reached before {where}")
    if _process_memory()["own_process_vmhwm_bytes"] > config.max_rss_bytes:
        raise MemoryError(f"own-process VmHWM exceeded D-07 {config.max_rss_bytes} bytes before {where}")


def _run_fixed_case(case, path, base_evaluator, ledger, config: FixedStudyConfig, *, study_started: float,
                    hard_probes: dict):
    record = {"case": dict(case), "status": "started", "source_path": _public_path(path),
              "errors": [], "objective_mode": case["objective_mode"], "optimizer_steps": 0}
    if path["status"] != "ready":
        record.update(status="invalid_source_motion", reason=path.get("reason"),
                      source_record=path.get("source_record"), actual_candidate_evaluations=0,
                      wall_time_s=None, memory=_process_memory())
        return record
    reference_seconds = math.fsum(segment.duration_s for segment in path["motion"].segments)
    theta = path["theta"].copy()
    evaluator = type(base_evaluator)(base_evaluator.task, path["motion"], path["parameterization"],
                                     base_evaluator.sim,
                                     spec=_objective_spec(config, case, reference_seconds),
                                     ledger=ledger, seed=base_evaluator.seed,
                                     requested_precision="f64")
    setup = {
        "precision": config.precision, "voxel_mm": config.voxel_mm,
        "k": case["k"], "objective_spec": asdict(evaluator.spec),
        "fixed_theta_hash": path["theta_hash"], "fixed_motion_hash": path["fixed_motion_hash"],
        "basis_sha256": path["basis_sha256"],
        "study_only_efficiency": case["objective_mode"] == "study_efficiency",
        "study_only_zero_objective_not_a_machining_result": case["objective_mode"] == "zero_weights",
        "reference_cycle_time_s": reference_seconds,
        "smooth_versus_hard_timing": (
            "native smooth timing versus exact MotionPlan sharp hard timing; do not claim equivalence"
            if case["objective_mode"] == "study_efficiency" else "objective dependent record only"),
    }
    setup["objective_setup_sha256"] = C.content_hash(setup)
    record["setup"] = setup
    record["sigma"] = {"theta_units": config.sigma_theta, "normalized_parameter_space": "T2.1 theta; mm per unit = 1 for these fits"}
    first_index = len(ledger.entries)
    reused = False
    started = time.perf_counter()
    try:
        smooth_started = time.perf_counter()
        _study_budget_guard(config, study_started, "smooth AD/FD")
        smooth = gm.verify_smooth_ad(evaluator, theta, config.epsilon_theta,
                                     indices=config.fd_indices, ledger=ledger,
                                     require_path_identity=True)
        ad = np.asarray(smooth["ad_gradient"], dtype=np.float64)
        record["raw_smooth_ad_result"] = smooth
        observed_precision = _dtype_information(evaluator.sim)
        if not all(field["is_float64"] for field in observed_precision.values()):
            raise TypeError(f"study requires f64 primal and adjoint fields: {observed_precision}")
        fd_errors = []
        for fd in smooth["fd"]:
            epsilon = np.asarray(fd["epsilon"], dtype=np.float64)
            if epsilon.shape != theta.shape:
                raise ValueError(f"FD epsilon shape {epsilon.shape} does not match fixed theta shape {theta.shape}")
            item = {"status": fd["status"], "epsilon_normalized_theta": epsilon[list(config.fd_indices)].tolist()}
            if fd["status"] == "ok":
                item["gradient"] = fd["gradient"].tolist()
                item["coordinate_errors"] = _coordinate_errors(ad, fd["gradient"], config.fd_indices)
                item["alignment_on_selected_coordinates"] = gm.gradient_alignment(
                    ad[list(config.fd_indices)], fd["gradient"][list(config.fd_indices)])
            else:
                item["reason"] = fd.get("reason")
            fd_errors.append(item)
        record["smooth_ad"] = {
            **smooth,
            "full_ad_gradient": ad.tolist(),
            "complete_vector_dimension": int(ad.size),
            "fd_is_selected_subset_not_full_fd": True,
            "fd_alignment_and_coordinate_errors": fd_errors,
            "wall_time_s": time.perf_counter() - smooth_started,
            "precision_fields_after_real_ad": observed_precision,
            "precision_metadata": evaluator.simulator_precision,
        }
        hard_started = time.perf_counter()
        _study_budget_guard(config, study_started, "randomized hard estimator")
        probe_key = _hard_probe_key(case, path, config)
        probe = (hard_probes.get(probe_key)
                 if config.reuse_k_independent_hard else None)
        if probe is None:
            seed_parts = ([config.hard_root_seed, case["path_index"],
                           {"geometry": 0, "zero_weights": 1, "study_efficiency": 2}[case["objective_mode"]]]
                          if config.reuse_k_independent_hard else
                          [config.hard_root_seed, case["path_index"], case["k_index"],
                           {"geometry": 0, "zero_weights": 1, "study_efficiency": 2}[case["objective_mode"]]])
            seed = np.random.SeedSequence(seed_parts)
            estimate = gm.estimate_with_reference(evaluator, theta, config.sigma_theta,
                                                  config.pairs, config.reference_pairs,
                                                  seed=seed, ledger=ledger)
            source_info = {
                "schema": "diffcam-source-hard-probe/1", "case_index": case["case_index"],
                "precision": config.precision, "voxel_mm": config.voxel_mm,
                "segment_count": len(path["motion"].segments),
                "simulator_precision_metadata": evaluator.simulator_precision,
                "path_id": path["id"], "objective_mode": case["objective_mode"],
                "objective_spec": asdict(evaluator.spec),
                "fixed_motion_hash": path["fixed_motion_hash"],
                "fixed_theta_hash": path["theta_hash"], "basis_sha256": path["basis_sha256"],
                "sigma_theta": config.sigma_theta, "epsilon_theta_units": "normalized T2.1 theta",
                "seed_entropy": list(seed.entropy), "k_when_first_evaluated": case["k"],
                "k_is_not_used_by_forward_hard": True,
                "primary_noise_sha256": estimate["primary"].noise_sha256,
                "reference_noise_sha256": estimate["reference"].noise_sha256,
                "primary_gradient": _verify_json(estimate["primary"].gradient),
                "reference_gradient": _verify_json(estimate["reference"].gradient),
                "primary_standard_error": _verify_json(estimate["primary"].standard_error),
                "reference_standard_error": _verify_json(estimate["reference"].standard_error),
            }
            probe = {"source_info": source_info,
                     "probe_hash": C.content_hash(source_info), "estimate": estimate,
                     "execution_wall_time_s": time.perf_counter() - hard_started,
                     "execution_ledger_delta": estimate["ledger_delta"]}
            if config.reuse_k_independent_hard:
                hard_probes[probe_key] = probe
            new_hard_count = 2 * config.pairs + 2 * config.reference_pairs
        else:
            estimate = probe["estimate"]
            new_hard_count = 0
        reused = probe is not None and new_hard_count == 0
        record["declaration_of_streams"] = {
            "tiered_entropy": probe["source_info"]["seed_entropy"],
            "primary_and_reference_are_spawned_independently": True,
            "no_reused_samples_across_probes": True,
            "pair_count_source": "declared primary and 64-pair reference",
            "reused_across_k": reused,
        }
        record["randomized_hard"] = {
            "primary": gm.estimate_to_json(estimate["primary"]),
            "reference": gm.estimate_to_json(estimate["reference"]),
            "primary_reference_comparison": estimate["comparison"],
            "ledger_delta": probe["execution_ledger_delta"],
            "exact_hard_evaluations_primary": 2 * config.pairs,
            "exact_hard_evaluations_reference": 2 * config.reference_pairs,
            "new_hard_evaluations_this_case": new_hard_count,
            "reused_across_k": reused,
            "source_probe_hash": probe["probe_hash"],
            "source_probe": probe["source_info"],
            "source_probe_execution_wall_time_s": probe["execution_wall_time_s"],
            "reuse_lookup_wall_time_s": time.perf_counter() - hard_started,
            "reuse_scope": "same task/path/fixed-theta/objective/sigma/f64/grid; no k dependence",
        }
        if estimate["primary"].status == "ok" and estimate["reference"].status == "ok":
            record["ad_vs_primary_hard"] = gm.gradient_alignment(ad, estimate["primary"].gradient)
            record["ad_vs_reference_hard"] = gm.gradient_alignment(ad, estimate["reference"].gradient)
            for name, gradient in (("primary", estimate["primary"].gradient),
                                   ("reference", estimate["reference"].gradient)):
                matching = [{"estimator": name, "limit_to_selected_fd_indices": True,
                             **_coordinate_errors(ad, gradient, config.fd_indices)[i]}
                            for i in range(len(config.fd_indices))]
                record.setdefault("ad_vs_hard_coordinate_errors", []).extend(matching)
        else:
            record["ad_vs_primary_hard"] = {"status": "undefined", "cosine": None}
            record["ad_vs_reference_hard"] = {"status": "undefined", "cosine": None}
        record["status"] = "complete"
        record["diagnostic_status"] = (
            "all_phases_returned_gradient_records"
            if estimate["primary"].status == estimate["reference"].status == "ok"
            else "invalid_estimator_retained_not_zero_or_dropped")
        record["randomized_hard"]["hard_definition"] = (
            "J_sigma(theta)=E[L_hard(theta+sigma*z)], z~N(0,I); mean of antithetic differences*z/(2*sigma)")
    except Exception as error:  # Preserve partial evidence, never label it a successful numerical result.
        record.update(status="partial_failed", reason=str(error), traceback=traceback.format_exc())
    finally:
        last_index = len(ledger.entries)
        record["ledger_entry_range"] = [first_index, last_index]
        record["actual_candidate_evaluations"] = last_index - first_index
        record["expected_complete_candidate_evaluations"] = (
            2 + 2 * len(config.epsilon_theta) * len(config.fd_indices)
            + (0 if reused else 2 * config.pairs + 2 * config.reference_pairs))
        record["wall_time_s"] = time.perf_counter() - started
        record["memory"] = _process_memory()
        record["rows_unavailable_or_failed_are_not_replaced"] = True
        if (record["status"] == "complete" and
                record["actual_candidate_evaluations"] != record["expected_complete_candidate_evaluations"]):
            record["status"] = "partial_failed"
            record["reason"] = (f"evaluation accounting mismatch: expected {record['expected_complete_candidate_evaluations']}, "
                                f"observed {record['actual_candidate_evaluations']}")
            record["errors"].append({"kind": "candidate_count_mismatch"})
    return record


def _expected_study_evaluations(config: FixedStudyConfig, paths) -> int | None:
    if any(path["status"] != "ready" for path in paths):
        return None
    used_probes = set()
    total = 0
    for case in _fixed_cases(config):
        total += 2 + 2 * len(config.epsilon_theta) * len(config.fd_indices)
        key = _hard_probe_key(case, paths[case["path_index"]], config)
        if not config.reuse_k_independent_hard or key not in used_probes:
            total += 2 * config.pairs + 2 * config.reference_pairs
            used_probes.add(key)
    return total


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def run_fixed_gradient_study(output: Path, config_path: Path, *, mode="full") -> dict[str, Any]:
    """Run the declared fixed-path f64 study, or one explicitly named preflight.

    Only one process/Taichi runtime is used. Every allocated path/case/attempt is
    retained. No optimizer update, physical execution, optimizer seed search or
    alternate grid is performed.
    """
    if mode not in STUDY_MODES:
        raise ValueError(f"mode must be one of {STUDY_MODES}")
    raw_config = config_path.read_bytes()
    document = json.loads(raw_config)
    config = study_config_from_json(document)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fixed-study output: {output}")
    output.mkdir(parents=True)
    raw_sha = C.sha256_bytes(raw_config)
    _new_json(output / "declared-config.json", document)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    plan = _fixed_cases(config)
    _new_json(output / "matrix-plan.json", {"configuration_sha256": raw_sha,
        "study_only": True, "precision": "f64", "cpu_max_num_threads": config.cpu_max_num_threads,
        "k_values": list(config.k_values), "cases": plan,
        "reuse_k_independent_hard": config.reuse_k_independent_hard,
        "per_complete_case_new_probe_evaluations": (
            2 + 2 * len(config.epsilon_theta) * len(config.fd_indices)
            + 2 * config.pairs + 2 * config.reference_pairs),
        "per_case_reused_probe_evaluations": (
            2 + 2 * len(config.epsilon_theta) * len(config.fd_indices)),
        "per_case_with_fresh_hard_probe_evaluations": (
            2 + 2 * len(config.epsilon_theta) * len(config.fd_indices)
            + 2 * config.pairs + 2 * config.reference_pairs)})
    ledger = ComputeLedger()
    cost = CostConfig(pairs=config.pairs, reference_pairs=config.reference_pairs,
                      epsilon_theta=config.epsilon_theta, fd_indices=config.fd_indices,
                      sigma_theta=config.sigma_theta, k_smooth=40.0, voxel_mm=config.voxel_mm)
    # Building real immutable geometry initializes exactly one CPU/f64 runtime.
    task, nominal_motion, parameterization, evaluator, metadata = build_pilot_evaluator(
        output / "geometry", config=cost, ledger=ledger, precision="f64",
        max_cpu_threads=config.cpu_max_num_threads)
    if metadata["task_id"] != config.task_id or parameterization.num_params != 36:
        raise ValueError("the actual prototype/parameter identity differs from the declared matrix")
    paths = []
    for declaration in config.paths:
        try:
            paths.append(_prepare_study_path(declaration, task, evaluator, config))
        except Exception as error:
            paths.append({"id": declaration["id"], "kind": declaration["kind"],
                          "status": "invalid_source_motion", "reason": str(error),
                          "traceback": traceback.format_exc()})
    _new_json(output / "paths.json", [_public_path(path) for path in paths])
    expected_evaluations = _expected_study_evaluations(config, paths)
    _new_json(output / "evaluation-plan.json", {
        "schema": "diffcam-evaluation-plan/1", "reuse_k_independent_hard": config.reuse_k_independent_hard,
        "primary_pairs_per_source": config.pairs, "reference_pairs_per_source": config.reference_pairs,
        "hard_evaluations_per_source_primary": 2 * config.pairs,
        "hard_evaluations_per_source_reference": 2 * config.reference_pairs,
        "planned_total_new_evaluations_if_all_sources_valid": expected_evaluations,
        "accounting": "reused rows contribute zero new evaluations; the source probe's actual 2N evaluations appear exactly once in the raw total",
    })
    sweep = scale_validity_sweep(task, config)
    _new_json(output / "delta-scale-validity.json", sweep)
    if mode == "sweep":
        report = {"schema": STUDY_SCHEMA, "status": "scale_sweep_only",
                  "execution_is_not_gradient_study": True, "config_sha256": raw_sha,
                  "scale_sweep": sweep, "source": source_metadata(PROJECT_ROOT)}
        _new_json(output / "summary.json", report)
        return report
    results = []
    hard_probes = {}
    stop_reason = None
    if mode == "preflight":
        path = next((path for path in paths if path["kind"] == "cam" and path["status"] == "ready"), None)
        if path is None:
            raise RuntimeError("CAM preflight has no valid source MotionPlan")
        case = {"case_index": 0, "path_index": 0, "path_id": path["id"], "k_index": 1,
                "k": 40.0, "objective_mode": "geometry"}
        before = gm._snapshot(ledger)
        ad_started = time.perf_counter()
        preflight_evaluator = type(evaluator)(
            task, path["motion"], path["parameterization"], evaluator.sim,
            spec=_objective_spec(config, case, math.fsum(s.duration_s for s in path["motion"].segments)),
            ledger=ledger, seed=evaluator.seed)
        value, gradient = gm.scalar_result(preflight_evaluator, path["theta"], "t33-preflight-only-one-ad")
        path_check = _actual_path(preflight_evaluator, path["theta"])
        if not path_check["same_within_tolerance"]:
            raise AssertionError(f"preflight actual path differs from its declared typed path: {path_check}")
        phase_delta = gm.delta_snapshot(before, gm._snapshot(ledger))
        if tuple(phase_delta[name] for name in ("candidate_evaluations", "forward_calls", "backward_calls")) != (1, 1, 1):
            raise AssertionError(f"preflight must make exactly one AD forward/backward: {phase_delta}")
        evidence = {"status": "complete", "case": case, "objective": value.objective,
                    "gradient": np.asarray(gradient, dtype=np.float64).tolist(),
                    "path_identity": path_check, "ledger_delta": phase_delta,
                    "wall_time_s": time.perf_counter() - ad_started,
                    "precision_fields_after_real_ad": _dtype_information(preflight_evaluator.sim),
                    "precision_metadata": preflight_evaluator.simulator_precision,
                    "source_path": _public_path(path), "memory": _process_memory()}
        results.append(evidence)
        _new_json(output / "preflight-ad.json", evidence)
        report = {"schema": STUDY_SCHEMA, "status": "preflight_only_not_matrix_study",
                  "config_sha256": raw_sha, "case_count_declared": len(plan), "matrix_cases_run": 0,
                  "optimized_steps": 0, "runtime": metadata.get("simulator"),
                  "precision_fields": _dtype_information(preflight_evaluator.sim),
                  "first_real_ad_case": evidence, "full_scale_sweep": sweep,
                  "setup_wall_time_s": metadata["setup_wall_time_s"],
                  "elapsed_wall_time_s": time.perf_counter() - started,
                  "memory": _process_memory(), "source": source_metadata(PROJECT_ROOT)}
        _new_json(output / "summary.json", report)
        return report
    for case in plan:
        path = paths[case["path_index"]]
        if stop_reason is not None:
            results.append({"case": case, "status": "not_run_after_budget_or_failure",
                            "reason": stop_reason, "source_path": _public_path(path),
                            "actual_candidate_evaluations": 0})
            continue
        try:
            _study_budget_guard(config, started, f"case {case['case_index']}")
        except Exception as error:
            stop_reason = str(error)
            results.append({"case": case, "status": "not_run_after_budget_or_failure",
                            "reason": stop_reason, "source_path": _public_path(path),
                            "actual_candidate_evaluations": 0})
            continue
        result = _run_fixed_case(case, path, evaluator, ledger, config,
                                 study_started=started, hard_probes=hard_probes)
        results.append(result)
        _new_json(output / f"case-{case['case_index']:03d}-{case['path_id']}-k{case['k']:g}-{case['objective_mode']}.json", result)
        print(json.dumps({"case": case["case_index"], "path": case["path_id"], "k": case["k"],
                          "mode": case["objective_mode"], "status": result["status"],
                          "wall_time_s": result.get("wall_time_s"),
                          "candidate_evaluations": result["actual_candidate_evaluations"]}), flush=True)
        if result["status"] == "partial_failed":
            stop_reason = f"case {case['case_index']} failed; subsequent cases were not launched"
        if _process_memory()["own_process_vmhwm_bytes"] > config.max_rss_bytes:
            stop_reason = "own-process VmHWM exceeded the D-07 process cap; subsequent cases stopped"
    rows = []
    for result in results:
        hard = result.get("randomized_hard", {})
        smooth = result.get("smooth_ad", {})
        primary = hard.get("primary", {})
        reference = hard.get("reference", {})
        alignment = result.get("ad_vs_reference_hard", {})
        rows.append({"case_index": result["case"]["case_index"], "path_id": result["case"]["path_id"],
                     "k": result["case"]["k"], "objective_mode": result["case"]["objective_mode"],
                     "status": result["status"], "ad_objective": smooth.get("ad_objective"),
                     "ad_norm": (float(np.linalg.norm(result["smooth_ad"]["full_ad_gradient"]))
                                 if "full_ad_gradient" in result.get("smooth_ad", {}) else None),
                     "primary_hard_status": primary.get("status"), "reference_hard_status": reference.get("status"),
                     "primary_pairs": primary.get("pairs"), "reference_pairs": reference.get("pairs"),
                     "cosine_ad_vs_reference": alignment.get("cosine"),
                     "relative_difference_norm_ad_vs_reference": alignment.get("relative_difference_norm"),
                     "wall_time_s": result.get("wall_time_s"),
                     "candidate_evaluations": result["actual_candidate_evaluations"],
                     "memory_vmhwm_bytes": result.get("memory", {}).get("own_process_vmhwm_bytes")})
    _write_table(output / "results-table.csv", rows,
                 ("case_index", "path_id", "k", "objective_mode", "status", "ad_objective", "ad_norm",
                  "primary_hard_status", "reference_hard_status", "primary_pairs", "reference_pairs",
                  "cosine_ad_vs_reference", "relative_difference_norm_ad_vs_reference", "wall_time_s",
                  "candidate_evaluations", "memory_vmhwm_bytes"))
    report = {"schema": STUDY_SCHEMA, "status": "matrix_complete" if all(item["status"] == "complete" for item in results)
              else "partial_or_invalid_available_not_vector_success",
        "config_sha256": raw_sha, "study_only_efficiency_weights": True,
        "budget_enforcement": {"max_wall_time_s": config.max_wall_time_s,
                               "checkpoints": "before each case and each AD/FD or randomized-hard phase",
                               "limitation": "a phase already in a native simulator evaluation is not interrupted mid-call"},
        "optimizer_steps": 0, "unbiasedness_or_equivalence_claim": False,
        "paths": [_public_path(path) for path in paths], "cases": results,
        "rows_table": str(output / "results-table.csv"),
        "declared_case_count": len(plan), "executed_case_count": sum(item["status"] == "complete" for item in results),
        "failed_or_partial_case_indices": [result["case_index"] for result in results if result["status"] != "complete"],
        "full_gradient_each_complete_case": True, "full_fd_claim": False,
        "ad_fd_same_smooth_objective": True,
        "hard_antithetic_evaluations_per_case": 2 * config.pairs,
        "reference_antithetic_evaluations_per_case": 2 * config.reference_pairs,
        "expected_complete_evaluations_total": expected_evaluations,
        "hard_probe_sources": [probe["source_info"] for probe in hard_probes.values()],
        "actual_candidate_evaluations": ledger.snapshot()["candidate_evaluations"],
        "actual_backward_evaluations": ledger.snapshot()["backward_calls"],
        "setup": metadata, "scale_sweep_summary": sweep["summaries"],
        "elapsed_wall_time_s": time.perf_counter() - started, "started_utc": started_utc,
        "memory": _process_memory(), "source": source_metadata(PROJECT_ROOT),
        "limits": ["structural MotionPlan validity is not physical machining success",
                   "smooth and randomized-hard surrogates need not have equal derivatives",
                   "selected FD coordinates with explicit epsilons, not all coordinates or a convergence proof",
                   "global cosine can hide coordinate errors; per-coordinate rows remain in case JSON",
                   "no optimizer, physical hardware or REAL_* campaign was run",
                   "study-only efficiency weights are not frozen protocol values"],
        "stop_reason": stop_reason,
        "ledger": {key: value for key, value in ledger.snapshot().items() if key != "entries"},
        "per_candidate_ledger": ledger.entries,
    }
    if all(item["status"] == "complete" for item in results):
        if expected_evaluations != report["actual_candidate_evaluations"]:
            raise AssertionError("raw ledger does not match the declared deduplicated evaluation plan")
        fully_complete = len(results)
        if report["actual_backward_evaluations"] != fully_complete:
            raise AssertionError("exactly one true AD backward call is expected per complete fixed case")
    _new_json(output / "summary.json", report)
    _new_json(output / "candidate-ledger.json", ledger.entries)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="fresh local output directory")
    parser.add_argument("--study-config", type=Path, default=None,
                        help="required prewritten JSON for the explicitly opted-in fixed-path f64 study")
    parser.add_argument("--study-mode", choices=STUDY_MODES, default="full",
                        help="complete matrix, one-AD preflight, or only the small physical-validity sweep")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--reference-pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="one scalar sigma in normalized theta coordinates")
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    parser.add_argument("--fd-indices", type=int, nargs="+", default=list(DEFAULT_INDICES))
    args = parser.parse_args(argv)
    if args.study_config is not None:
        report = run_fixed_gradient_study(args.output, args.study_config, mode=args.study_mode)
        print(json.dumps({"status": report["status"], "output": str(args.output),
                          "config_sha256": report["config_sha256"]}, indent=2))
        return 0
    if args.study_mode != "full":
        parser.error("--study-mode preflight/sweep requires --study-config")
    config = CostConfig(pairs=args.pairs, reference_pairs=args.reference_pairs,
                        seed=args.seed, sigma_theta=args.sigma,
                        epsilon_theta=tuple(args.epsilons), fd_indices=tuple(args.fd_indices))
    report = run_pilot(args.output, config)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "ledger": report["cost"]["counters_including_explicit_hard_base"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
