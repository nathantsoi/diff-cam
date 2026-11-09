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
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Sequence
import time

import numpy as np

from algorithms import gradient_mechanics as gm
from algorithms.run_artifacts import ComputeLedger, source_metadata
from benchmarks import contracts as C
from benchmarks import pilot_task
from cam.seeds import pocket_with_island_seed


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
                          ledger: ComputeLedger | None = None):
    """Construct the real sampled pilot, CPU simulator, and one evaluator.

    The optional voxel_mm allows a separately labelled reduced-grid smoke test;
    it never resamples or relabels a different .npy grid. The CLI's actual pilot
    uses the full 0.25 mm task grid.
    """
    t0 = time.perf_counter()
    validate_config(config)
    root = root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing existing geometry/output directory: {root}")
    root.mkdir(parents=True)
    task = pilot_task.build_pilot_task(voxel_mm=config.voxel_mm)
    motion = pocket_with_island_seed(task)
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
    ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f32)
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
            "floating_point": "float32", "atomic_nondeterminism": "use tolerances, never bitwise assertions",
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
                        ordered: bool = True) -> dict[str, Any]:
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
        output, config=CostConfig(voxel_mm=voxel_mm),
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
        "tool_delta_gradient_dtype": evaluator.sim.tool_delta.grad.to_numpy().dtype.str,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="fresh local output directory")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--reference-pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="one scalar sigma in normalized theta coordinates")
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    parser.add_argument("--fd-indices", type=int, nargs="+", default=list(DEFAULT_INDICES))
    args = parser.parse_args(argv)
    config = CostConfig(pairs=args.pairs, reference_pairs=args.reference_pairs,
                        seed=args.seed, sigma_theta=args.sigma,
                        epsilon_theta=tuple(args.epsilons), fd_indices=tuple(args.fd_indices))
    report = run_pilot(args.output, config)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "ledger": report["cost"]["counters_including_explicit_hard_base"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
