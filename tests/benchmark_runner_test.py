"""T4.1a runner tests: frozen plan first, then small actual subprocess evidence."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from benchmarks import audit as A
from benchmarks import contracts as C
from scripts import run_benchmark_suite as runner

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pilot.json"


def _invoke(config, root, *, extra=(), expect=0, timeout=120):
    config_path = root.parent / "fixture-config.json"
    config_path.write_text(json.dumps(config, sort_keys=True, indent=1))
    completed = subprocess.run([
        sys.executable, "scripts/run_benchmark_suite.py",
        "--config", str(config_path), "--output-dir", str(root),
        "--split", "dev", "--experiment", "pilot", *extra,
    ], cwd=CONFIG.parents[1], capture_output=True, text=True, timeout=timeout, check=False)
    assert completed.returncode == expect, completed.stdout + completed.stderr
    return completed


def _base():
    config = json.loads(CONFIG.read_text())
    # The legacy tiny subprocess tests never measure/execute the full pilot.
    config["task_source"] = "prototype_analytic"
    return config


def _pilot_config():
    """Explicit coarse task for source/format tests, never a production timing."""
    config = _base()
    config.update(task_source="pilot_pocket", grid_pitch_mm=2.0,
                  require_grid_exclusion_tolerance=True)
    from benchmarks import pilot_task
    task = pilot_task.build_pilot_task(config["grid_pitch_mm"])
    tolerance = task.operation_target.geometry.params.get("grid_exclusion_tolerance_mm3", 0.0)
    config["exclusion_tolerance_mm3"] = tolerance
    return config


def _named(root, run_id):
    return C.loads((root / run_id / "run.json").read_text())


def test_frozen_config_selects_real_grid_and_declared_zero_tolerance():
    config = json.loads(CONFIG.read_text())
    assert config["task_source"] == "pilot_pocket"
    assert config["grid_pitch_mm"] == 0.25
    assert config["exclusion_tolerance_mm3"] == 0.0
    assert config["require_grid_exclusion_tolerance"] is True
    assert config["arch"] == "cpu"
    assert "estimated_seconds_per_candidate" not in config
    assert isinstance(config["timeout_s"], (int, float)) and config["timeout_s"] > 0
    assert isinstance(config["timing_timeout_s"], (int, float)) and config["timing_timeout_s"] > 0


def test_condition_ids_are_deterministic_and_manifest_nominal_once(tmp_path):
    config = _base()
    config["experiment"] = "pilot"
    first = runner._plan_entries(config, runner._context(config, tmp_path)[0], "dev", None)
    second = runner._plan_entries(config, runner._context(config, tmp_path)[0], "dev", None)
    assert len(first) == len(second) == 7
    assert [entry["run_id"] for entry in first] == [entry["run_id"] for entry in second]
    for condition in ("nominal-cam", "fixed-adam", "annealed-adam"):
        assert sum(entry["method"] == condition for entry in first) == 1
    for condition in ("hard-cmaes", "smooth-cmaes"):
        assert sum(entry["method"] == condition for entry in first) == 2
    assert [entry["method"] for entry in first].count("nominal-cam") == 1
    assert sorted({entry["seed"] for entry in first}) == [0, 1]
    ordinary = [entry for entry in first if entry["method"] != "nominal-cam"]
    assert {entry["seed"] for entry in ordinary} == {0, 1}
    alone = [entry for entry in first if entry["method"] == "nominal-cam"][0]
    assert alone["seed"] == 0  # deterministic nominal result, not a fictitious seed set.


def test_advertised_enum_rejections_do_not_create_empty_runs(tmp_path):
    config = _base()
    # The adapter passes explicit experiments, just as a user would type them.
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(config))
    for experiment in ("mechanics", "all", "solvers", "initialization"):
        completed = subprocess.run([
            sys.executable, "scripts/run_benchmark_suite.py", "--config", str(config_path),
            "--output-dir", str(tmp_path / experiment), "--split", "dev", "--experiment", experiment,
        ], cwd=CONFIG.parents[1], capture_output=True, text=True, check=False)
        assert completed.returncode == 2
        assert "not implemented" in completed.stderr
        assert not (tmp_path / experiment).exists()


def test_dry_run_counts_cost_uncertainty_and_nominal_subset(tmp_path):
    config = _base()
    config["estimated_seconds_per_candidate"] = {"fixed-adam": 1.25}
    output = tmp_path / "dry"
    result = _invoke(config, output, extra=("--dry-run",), timeout=60)
    planned = json.loads((output / "dry_run.json").read_text())
    assert planned["planned_run_count"] == 7
    assert not (output / "nominal-cam").exists()
    per_run = {row["run_id"]: row for row in planned["costs"]}
    fixed = next(row for row in per_run.values() if row["run_id"].startswith("fixed-adam-s0-"))
    assert fixed["candidate_evaluations"] == 4
    assert fixed["validation_calls"] == 2
    assert fixed["estimated_run_wall_time_s"] == pytest.approx(7.5)
    unknown = next(row for row in per_run.values() if row["run_id"].startswith("hard-cmaes-s0-"))
    assert unknown["estimated_run_wall_time_s"] is None
    assert "unknown without a real child measurement" in unknown["jit_warmup"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["runner"]["planned_not_executed"] is True

    one_config = _base()
    one_output = tmp_path / "only-nominal"
    _invoke(one_config, one_output, extra=("--dry-run", "--only", "nominal-cam-*"), timeout=60)
    assert json.loads((one_output / "dry_run.json").read_text())["planned_run_count"] == 1


def test_minimal_real_subprocess_and_t40_reconciliation(tmp_path):
    config = _base()
    config["seeds"] = [0]
    output = tmp_path / "real"
    commanded = _invoke(config, output, extra=("--only", "smooth-cmaes-s0-*"), timeout=180)
    assert '"incomplete": 1' in commanded.stdout
    manifest = A.load_manifest(output / "manifest.json")
    assert len(manifest.runs) == 1
    run_id = next(iter(manifest.runs))
    record = C.loads((output / run_id / "run.json").read_text())
    assert record.status == C.RunStatus.INCOMPLETE

    reports, audit = A.audit_runs(output, manifest)
    assert audit["reconciliation"]["exact_id_reconciliation"] is True
    assert audit["summary"]["auditor_passed"] is False  # posted evaluation unavailable ≠ hidden success.
    assert reports[0]["status"] == "incomplete"
    assert reports[0]["run_id"] == run_id
    assert not (output / run_id / "eval_posted.rs274.json").exists()
    assert (output / run_id / "eval_posted.json").is_file()
    assert (output / run_id / "haas.nc").is_file()
    assert (output / "logs" / f"{run_id}.log").is_file()

    # Same identity resumes with a preserved backup, not a destructive overwrite.
    first = json.loads((output / run_id / "run.json").read_text())
    resumed = _invoke(config, output, extra=("--resume", "--only", run_id), timeout=180)
    assert resumed.returncode == 0
    assert list((output / "resume-backups").iterdir())
    second = json.loads((output / run_id / "run.json").read_text())
    assert second["record"]["run_id"] == run_id


def test_resume_rejects_condition_identity_changes(tmp_path):
    config = _base()
    output = tmp_path / "frozen"
    _invoke(config, output, extra=("--dry-run", "--only", "nominal-cam-*"), timeout=60)
    changed = dict(config, candidate_evaluations=config["candidate_evaluations"] + 1)
    changed_path = tmp_path / "different-root" / "actual"
    changed_path.mkdir(parents=True)
    result = _invoke(changed, changed_path, extra=("--resume",), expect=2, timeout=60)
    assert result.returncode == 2

    output_config = json.loads((output / "frozen_config.json").read_text())
    output_config["identity"] = "sha256:" + "0" * 64
    (output / "frozen_config.json").write_text(json.dumps(output_config))
    result = _invoke(config, output, extra=("--resume",), expect=2, timeout=60)
    assert result.returncode == 2
    assert "identity differs" in result.stderr
    source_config = json.loads((output / "frozen_config.json").read_text())
    assert source_config["raw_config"]["candidate_evaluations"] == config["candidate_evaluations"]


def test_interrupted_child_is_recorded_not_misfiled_as_complete(tmp_path):
    config = _base()
    config["seeds"] = [0]
    planned = runner._plan_entries(config, runner._context(config, tmp_path)[0], "dev", None)
    target = next(entry["run_id"] for entry in planned if entry["method"] == "fixed-adam")
    config["__test_interrupt__"] = target
    output = tmp_path / "interrupted"
    _invoke(config, output, extra=("--only", target), timeout=180)
    record = _named(output, target)
    assert record.status == C.RunStatus.INTERRUPTED
    assert record.failure_reason
    assert not (output / target / "rs274.nc").exists()
    manifest = A.load_manifest(output / "manifest.json")
    _reports, audit = A.audit_runs(output, manifest)
    assert audit["reconciliation"]["exact_id_reconciliation"] is True
    summary = json.loads((output / "execution_summary.json").read_text())
    assert summary["status_counts"] == {"interrupted": 1}


# --- coordinator follow-ups from discriminator re-review (Q1, Q4) -------------------------------
def test_smooth_objective_k_comes_from_config_not_code_default():
    import json, pathlib
    from scripts import run_benchmark_suite as S
    cfg = json.loads(pathlib.Path("configs/pilot.json").read_text())
    assert "k_smooth" in cfg
    src = pathlib.Path(S.__file__).read_text()
    assert 'ObjectiveSpec(k=float(config["k_smooth"]))' in src
    assert "O.ObjectiveSpec()" not in src


def test_config_without_k_smooth_rejected(tmp_path):
    import json, pathlib, pytest
    from scripts import run_benchmark_suite as S
    cfg = json.loads(pathlib.Path("configs/pilot.json").read_text())
    del cfg["k_smooth"]
    p = tmp_path / "c.json"; p.write_text(json.dumps(cfg))
    loader = S._normalized_config
    with pytest.raises(S.RunnerError, match="k_smooth"):
        loader(p)


def test_cma_budget_smaller_than_population_rejected():
    import pytest
    from algorithms import blackbox as B
    with pytest.raises(ValueError, match="smaller than one CMA population"):
        B.OptimizationBudget(candidate_evaluations=3, population_size=4, validation_interval=1, max_validation_calls=0)


# --- T4.1b real-pilot wiring ---------------------------------------------------
def test_pilot_sources_target_exclusion_and_initial_position(tmp_path, monkeypatch):
    from benchmarks import pilot_task

    monkeypatch.setattr(runner, "REPOSITORY", tmp_path / "repository")
    config = _pilot_config()
    task, motion, parameterization, evaluator_factory, posted_factory = runner._context(config, tmp_path)
    sources = runner._sampled_pilot_sources(task)
    assert task.operation_target.geometry.kind == "analytic"
    assert task.original_part is not task.operation_target.geometry
    assert task.initial_state.known and task.initial_state.position_mm is not None
    assert motion.initial_state == task.initial_state
    assert np.allclose(parameterization.initial_position_mm, task.initial_state.position_mm)
    assert evaluator_factory is runner._pilot_evaluator
    assert posted_factory is runner._pilot_post_factory
    assert runner._pilot_exclusion_tolerance(task, config) == pytest.approx(
        config["exclusion_tolerance_mm3"])
    for label in ("stock", "target"):
        source = sources[label]
        path = runner.REPOSITORY / source.path
        assert source.kind == "sampled" and source.frame == C.Frame.WORK
        assert source.sha256 == C.sha256_bytes(path.read_bytes())
        assert path.parent.name == task.task_hash[7:]
        with path.open("rb") as stream:
            values = np.load(stream, allow_pickle=False)
        assert values.dtype == np.float32 and values.shape == task.grid.shape
        assert np.isfinite(values).all()

    # The operation target T, including its corner exclusions, replaces original P.
    stock_sdf, target_sdf, required, retained = pilot_task.source_fields(task)
    assert not np.any(required & retained)
    for values in (stock_sdf, target_sdf):
        assert values.shape == task.grid.shape


def test_pilot_source_cache_rejects_changed_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPOSITORY", tmp_path / "repository")
    task, *_rest = runner._context(_pilot_config(), tmp_path)
    sources = runner._sampled_pilot_sources(task)
    source_path = runner.REPOSITORY / sources["target"].path
    source_path.write_bytes(b"changed immutable test payload")
    with pytest.raises(runner.RunnerError, match="cached source changed"):
        runner._sampled_pilot_sources(task)
    assert source_path.read_bytes() == b"changed immutable test payload"


def test_pilot_rejects_invalid_grid_or_exclusion_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPOSITORY", tmp_path / "repository")
    config = _pilot_config()
    config["grid_pitch_mm"] = float("nan")
    with pytest.raises(runner.RunnerError, match="grid_pitch_mm"):
        runner._context(config, tmp_path)
    task, *_rest = runner._context(_pilot_config(), tmp_path)
    for invalid in (float("nan"), -1.0, True):
        with pytest.raises(runner.RunnerError, match="exclusion tolerance"):
            runner._pilot_exclusion_tolerance(task, {"exclusion_tolerance_mm3": invalid})
    wrong = _pilot_config()
    wrong = _pilot_config()
    small = task.operation_target.geometry.params
    if "grid_exclusion_tolerance_mm3" in small:
        wrong["exclusion_tolerance_mm3"] += 1.0
        with pytest.raises(runner.RunnerError, match="disagrees"):
            runner._pilot_exclusion_tolerance(task, wrong)
    small = task.operation_target.geometry.params
    if "grid_exclusion_tolerance_mm3" not in small:
        with pytest.raises(runner.RunnerError, match="requires config"):
            runner._pilot_exclusion_tolerance(task, {
                "require_grid_exclusion_tolerance": True,
            })


def test_pilot_post_factory_is_task_bound_and_settings_are_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPOSITORY", tmp_path / "repository")
    config = _pilot_config()
    task, *_rest = runner._context(config, tmp_path)
    monkeypatch.setattr(runner.pilot_io, "production_settings",
                        lambda actual: {"max_step_mm": 1.0}, raising=False)
    evaluator = runner._pilot_post_factory(task, config)
    assert evaluator.settings["max_step_mm"] == 1.0
    assert evaluator.settings["record_stations"] is False
    with pytest.raises(TypeError):
        evaluator.settings["compute_bounds"] = False
    with pytest.raises(runner.RunnerError, match="different task"):
        evaluator.evaluate_posted_gcode(runner._prototype()[0], b"", C.Dialect.RS274, {})
    unsafe = {"safety_screen": False}
    with pytest.raises(runner.RunnerError, match="per-call overrides"):
        evaluator.evaluate_posted_gcode(task, b"", C.Dialect.RS274, unsafe)


    """Coordination-only doubles, explicitly not real-task timing evidence."""
    config = _base()
    task, motion, parameterization = runner._prototype()
    calls = []

    class Evaluator:
        geometry_sources = {"stock": SimpleNamespace(path="unit/stock.npy", sha256="sha256:" + "0" * 64),
                            "target": SimpleNamespace(path="unit/target.npy", sha256="sha256:" + "1" * 64)}

        def evaluate_candidate(self, theta, candidate_id, kind):
            calls.append((kind, False))
            return SimpleNamespace(valid=True, objective=1.0, invalid_reason=None)

        def evaluate_candidate_with_gradient(self, theta, candidate_id):
            assert candidate_id == "timing-smooth_gradient"
            calls.append(("smooth", True))
            return SimpleNamespace(result=SimpleNamespace(valid=True, objective=1.0, invalid_reason=None),
                                   gradient_theta=np.zeros(parameterization.num_params))

        def candidate_motion(self, theta):
            return motion

    class Poster:
        def __init__(self):
            self.timings = []

        def evaluate_posted_gcode(self, actual_task, nc_bytes, dialect, settings):
            nc_hash = C.nc_hash(nc_bytes)
            self.timings.append({"dialect": dialect.value, "wall_seconds": 0.0, "nc_hash": nc_hash})
            return SimpleNamespace(dialect=dialect, nc_hash=nc_hash, task_hash=task.task_hash,
                                   efficiency_eligible=True,
                                   metrics={"removed_volume_mm3": SimpleNamespace(value=1.0)})

    monkeypatch.setattr(runner, "_context", lambda cfg, root: (
        task, motion, parameterization, lambda *args: Evaluator(), lambda *args: Poster()))
    report = runner._measure_budget(config, tmp_path)
    assert calls == [("smooth", False), ("smooth", True), ("hard", False)]
    assert {row["dialect"] for row in report["posted_seconds_dialects"]} == {"rs274", "haas"}
    assert set(report["candidate_seconds"]) == {"smooth_forward", "smooth_gradient", "hard_forward"}
    assert len(report["posted_seconds"]) == 2
    assert report["task_hash"] == task.task_hash
    entries = runner._plan_entries(config, task, "dev", None)
    costs = runner._pecent(dict(config, timing_evidence=report), entries)
    assert len(costs) == 7
    assert all(row["estimated_run_wall_time_s"] is not None for row in costs)
    cma = next(row for row in costs if row["run_id"].startswith("hard-cmaes-s0-"))
    assert (cma["candidate_evaluations"], cma["validation_calls"]) == (4, 2)

    # Fixed arithmetic exposes overhead/post double-counting; no JIT or solver.
    known = {
        "schema": "diffcam-pilot-timing/1", "task_hash": task.task_hash,
        "candidate_seconds": {"smooth_forward": 1.0, "smooth_gradient": 2.0, "hard_forward": 3.0},
        "constructor_seconds": 4.0,
        "posted_seconds": [0.5, 0.75],
        "posted_seconds_dialects": [
            {"dialect": "rs274", "wall_seconds": 0.5},
            {"dialect": "haas", "wall_seconds": 0.75},
        ],
        "worker_wall_seconds": 4.0 + 1.0 + 2.0 + 3.0 + 0.5 + 0.75 + 2.0,
    }
    nominal = next(item for item in entries if item["method"] == "nominal-cam")
    cost = runner._pecent(dict(config, timing_evidence=known), [nominal])[0]
    assert cost["other_worker_seconds"] == pytest.approx(2.0)
    assert cost["posted_seconds"] == pytest.approx(1.25)
    assert cost["estimated_run_wall_time_s"] == pytest.approx(10.25)

    missing = dict(report)
    missing.pop("constructor_seconds")
    with pytest.raises(runner.RunnerError, match="timing_evidence must include"):
        runner._pecent(dict(config, timing_evidence=missing), entries)


def test_timing_worker_cannot_measure_prototype_task(tmp_path):
    output = tmp_path / "prototype-timing"
    result = _invoke(_base(), output, extra=("--dry-run", "--timing-worker-run"),
                     expect=2, timeout=60)
    assert "only available for the registered pilot_pocket" in result.stderr
    assert not (output / "budget-timings.json").exists()


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("DIFFCAM_REAL_PILOT_SMOKE") != "1",
                    reason="explicit slow real-pilot smoke: set DIFFCAM_REAL_PILOT_SMOKE=1")
@pytest.mark.parametrize("condition", ["nominal-cam", "fixed-adam", "annealed-adam", "hard-cmaes", "smooth-cmaes"])
def test_real_pilot_one_generation_per_condition(tmp_path, condition):
    """One bounded real task/optimizer generation and two production evaluations.

    This is not the seven-run scientific pilot. Each condition is separately selectable
    so a failure preserves the preceding conditions' independent artifacts and logs.
    CMA cannot legally evaluate a one-member population, hence its two-candidate generation.
    """
    from benchmarks import pilot_task

    config = _base()
    config.update(task_source="pilot_pocket", seeds=[0], grid_pitch_mm=pilot_task.GRID_PITCH,
                  candidate_evaluations=2 if condition.endswith("cmaes") else 1,
                  max_validation_calls=1 if condition.endswith("adam") else 0,
                  validation_interval=1, max_generations=1,
                  require_grid_exclusion_tolerance=True)
    task = pilot_task.build_pilot_task(config["grid_pitch_mm"])
    assert callable(getattr(pilot_task, "production_settings", None)), "T2.2b is required for this smoke"
    config["exclusion_tolerance_mm3"] = task.operation_target.geometry.params["grid_exclusion_tolerance_mm3"]
    output = tmp_path / condition
    _invoke(config, output, extra=("--only", f"{condition}-s0-*"), timeout=3600)
    manifest = A.load_manifest(output / "manifest.json")
    assert len(manifest.runs) == 1
    run_id = next(iter(manifest.runs))
    record = _named(output, run_id)
    assert record.status == C.RunStatus.COMPLETE, record.failure_reason
    metrics = json.loads((output / run_id / "metrics.json").read_text())
    spent = json.loads((output / run_id / "config.json").read_text())["budgets"]["spent"]
    assert spent["candidate_evaluations"] == config["candidate_evaluations"]
    assert spent["generations"] == 1
    assert (output / run_id / "eval_posted.json").is_file()
    for dialect in C.Dialect:
        nc_name, eval_name = C.DIALECT_ARTIFACTS[dialect]
        evaluation = C.loads((output / run_id / eval_name).read_text())
        assert evaluation.nc_hash == C.nc_hash((output / run_id / nc_name).read_bytes())
        assert evaluation.task_hash == task.task_hash
        assert evaluation.efficiency_eligible
    reports, audit = A.audit_runs(output, manifest)
    assert audit["reconciliation"]["exact_id_reconciliation"] is True
    assert len(reports) == 1 and reports[0]["status"] == "complete"
