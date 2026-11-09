"""T4.1a runner tests: frozen plan first, then small actual subprocess evidence."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from algorithms import run_artifacts as R
from algorithms import blackbox as B
from algorithms import optimize as adam
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


def _precision_metadata_stub(precision="f64"):
    """A labelled logic-fixture declaration, not a measurement of a runtime backend."""
    body = {
        "schema_version": "simulator-precision-identity/1",
        "requested_precision": precision,
        "declared_default_fp": precision,
        "runtime_default_fp": precision,
        "backend_arch": "test_fixture_not_a_runtime_measurement",
        "process_constraint": "one precision per process",
        "policy": {
            "schema_version": "simulator-precision-uniform/1",
            "variant_revision": f"uniform-{precision}/1",
            "default_fp": precision,
            "field_scope": "all_float_fields",
            "kernel_scope": "all_float_kernels_and_adjoints",
        },
    }
    return dict(body, precision_identity_sha256=C.content_hash(body))


def _named(root, run_id):
    return C.loads((root / run_id / "run.json").read_text())


def test_frozen_config_selects_real_grid_and_declared_zero_tolerance():
    config = json.loads(CONFIG.read_text())
    from benchmarks import pilot_task
    from cam.posted_evaluator import RefinementSettings, _subdivisions
    from cam.seeds import pocket_with_island_seed

    assert config["task_source"] == "pilot_pocket"
    assert config["grid_pitch_mm"] == 0.25
    refinement = config["posted_refinement"]
    assert refinement["max_step_mm"] == 1.0
    assert refinement["subdivisions"] is None
    assert refinement["compute_bounds"] is True
    assert refinement["safety_screen"] is True
    assert refinement["max_samples_per_segment"] == 4096
    assert refinement["record_stations"] is False
    assert config["exclusion_tolerance_mm3"] == 0.0
    assert config["require_grid_exclusion_tolerance"] is True
    assert config["arch"] == "cpu"
    assert "estimated_seconds_per_candidate" not in config
    assert isinstance(config["timeout_s"], (int, float)) and config["timeout_s"] > 0
    assert isinstance(config["timing_timeout_s"], (int, float)) and config["timing_timeout_s"] > 0
    phases = config["refinement_check"]
    assert phases["enabled"] is True
    assert {role: phases[f"{role}_step_mm"] for role in ("screen", "verdict", "convergence")} == {
        "screen": 1.0, "verdict": 0.125, "convergence": 0.25}
    assert phases["parallel_jobs"] == 2
    assert phases["max_rss_bytes_per_job"] == 2 * 1024 ** 3
    assert config["precision"] == "f64"

    task = pilot_task.build_pilot_task()
    motion = pocket_with_island_seed(task)
    producer = pilot_task.production_settings(task)
    assert refinement["max_step_mm"] == producer["max_step_mm"]
    default_refinement = RefinementSettings(exclusion_tolerance_mm3=config["exclusion_tolerance_mm3"])
    configured_refinement = RefinementSettings.from_mapping(
        dict(refinement, exclusion_tolerance_mm3=config["exclusion_tolerance_mm3"]))
    assert default_refinement.max_step_mm is None
    assert configured_refinement.max_step_mm == 1.0
    default_effective_step = min(task.grid.voxel_mm / 2, task.tool.cutter_radius_mm / 4,
                                 task.tool.flute_length_mm / 4)
    assert default_effective_step == 0.125
    longest_index = max(range(len(motion.segments)), key=lambda i: motion.segments[i].path_length_mm)
    default_counts = [_subdivisions(segment, task, default_refinement) for segment in motion.segments]
    configured_counts = [_subdivisions(segment, task, configured_refinement) for segment in motion.segments]
    assert all(explicit <= baseline for explicit, baseline in zip(configured_counts, default_counts))
    assert default_counts[longest_index] == 8 * configured_counts[longest_index]
    materialized = runner._pilot_refinement_settings(task, config)
    for name in refinement:
        assert materialized[name] == getattr(configured_refinement, name)
    for role in ("screen", "verdict", "convergence"):
        dialects = (C.Dialect.RS274,) if role == "convergence" else tuple(C.Dialect)
        for dialect in dialects:
            from cam.posts import post_motion_plan
            nc = post_motion_plan(motion, task, dialect=dialect).encode("utf-8")
            settings = runner._phase_settings(task, config, dialect, nc, role)
            assert settings["max_step_mm"] == phases[f"{role}_step_mm"]
            assert settings["expected_nc_hash"] == C.nc_hash(nc)
            assert settings["compute_bounds"] is True
            assert settings["safety_screen"] is True
            assert settings["exclusion_tolerance_mm3"] == 0.0

    for key, bad_value in (("max_step_mm", 0.125), ("max_step_mm", True),
                           ("subdivisions", 8), ("compute_bounds", False),
                           ("safety_screen", False), ("record_stations", True)):
        altered = dict(refinement)
        altered[key] = bad_value
        with pytest.raises(runner.RunnerError):
            runner._pilot_refinement_settings(task, dict(config, posted_refinement=altered))


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
    pytest.importorskip("cma", reason="runner CMA subprocess requires --extra optim")
    config = _base()
    config["seeds"] = [0]
    output = tmp_path / "real"
    commanded = _invoke(config, output, extra=("--only", "smooth-cmaes-s0-*"), timeout=config["refinement_check"]["timeout_s"])
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
    resumed = _invoke(config, output, extra=("--resume", "--only", run_id), timeout=config["refinement_check"]["timeout_s"])
    assert resumed.returncode == 0
    assert list((output / "resume-backups").iterdir())
    second = json.loads((output / run_id / "run.json").read_text())
    assert second["record"]["run_id"] == run_id


def test_nominal_one_hard_call_is_not_an_optimizer_generation():
    """Bookkeeping-only spy; no physical simulation is substituted here."""
    class Evaluator:
        parameterization = SimpleNamespace(num_params=2)

        def __init__(self):
            self.ledger = SimpleNamespace(snapshot=lambda: {
                "candidate_evaluations": 1, "forward_calls": 1, "backward_calls": 0,
            })
            self.calls = []

        def evaluate_candidate(self, theta, candidate_id, kind):
            self.calls.append((np.asarray(theta).tolist(), candidate_id, kind))
            return C.CandidateResult(candidate_id=candidate_id,
                                     theta_hash=C.content_hash(theta.tolist()),
                                     objective_kind=kind, objective=0.0, valid=True,
                                     coverage=C.Coverage.INCOMPLETE)

    evaluator = Evaluator()
    outcome = runner._evaluate_outcome("nominal-cam", evaluator, {}, 0)
    assert evaluator.calls == [([0.0, 0.0], "nominal-once", "hard")]
    assert (outcome.evaluations, outcome.generations, outcome.forward_calls,
            outcome.backward_calls) == (1, 0, 1, 0)
    assert outcome.invalid_candidates == 0
    evaluator.ledger.snapshot = lambda: {
        "candidate_evaluations": 1, "forward_calls": 0, "backward_calls": 0,
    }
    with pytest.raises(runner.RunnerError, match="does not match one hard"):
        runner._evaluate_outcome("nominal-cam", evaluator, {}, 0)


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
    _invoke(config, output, extra=("--only", target), timeout=config["refinement_check"]["timeout_s"])
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
    assert evaluator_factory.func is runner._pilot_evaluator
    assert evaluator_factory.keywords["repository"] == runner.REPOSITORY
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


def test_pilot_explicit_repository_is_used_and_zero_sdf_rejected(tmp_path, monkeypatch):
    config = _pilot_config()
    repository = tmp_path / "chosen-repository"
    task, _motion, _params, evaluator_factory, _posted = runner._pilot_context(config, repository)
    assert evaluator_factory.keywords["repository"] == repository
    sources = runner._sampled_pilot_sources(task, repository)
    for source in sources.values():
        path = Path(source.path)
        assert path.is_absolute() and path.is_relative_to(repository.resolve())
        assert path.is_file() and C.sha256_bytes(path.read_bytes()) == source.sha256

    shape = task.grid.shape
    zeros = np.zeros(shape, dtype=np.float32)
    monkeypatch.setattr(runner.pilot_io, "source_fields",
                        lambda _task: (zeros, zeros, np.zeros(shape, dtype=bool),
                                       np.zeros(shape, dtype=bool)))
    with pytest.raises(runner.RunnerError, match="zero-valued SDF"):
        runner._sampled_pilot_sources(task, tmp_path / "zero-sdf")


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


def test_timing_probe_calls_each_kind_and_projects_all_seven(tmp_path, monkeypatch):
    """Coordination-only doubles, explicitly not real-task timing evidence."""
    config = _pilot_config()
    from benchmarks import pilot_task
    from cam.seeds import pocket_with_island_seed
    task = pilot_task.build_pilot_task(voxel_mm=config["grid_pitch_mm"])
    motion = pocket_with_island_seed(task)
    parameterization = pilot_task.fit_parameterization(motion)
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

        def evaluate_posting(self, actual_task, posting, workspace, *, role, single_dialect=False):
            results = {}
            for post in posting.posted:
                nc = posting.files[C.DIALECT_ARTIFACTS[post.dialect][0]]
                self.timings.append({"dialect": post.dialect.value, "wall_seconds": 0.0,
                                     "nc_hash": C.nc_hash(nc)})
                results[post.dialect] = _typed_evaluation(
                    actual_task, nc, dialect=post.dialect,
                    step=1.0 if role == "screen" else 0.125 if role == "verdict" else 0.25)
            return results

    monkeypatch.setattr(runner, "_context", lambda cfg, root: (
        task, motion, parameterization, lambda *args: Evaluator(),
        lambda *args: pytest.fail("the typed override must drive this timing probe")))
    report = runner._measure_budget(config, tmp_path, posted_evaluator_override=Poster())
    assert calls == [("smooth", False), ("smooth", True), ("hard", False)]
    assert {row["dialect"] for row in report["posted_seconds_dialects"]} == {"rs274", "haas"}
    assert set(report["candidate_seconds"]) == {"smooth_forward", "smooth_gradient", "hard_forward"}
    assert len(report["posted_seconds"]) == 2
    assert report["task_hash"] == task.task_hash
    assert report["constructor_seconds"] >= 0.0
    assert set(report["thread_environment"]) == {
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TAICHI_NUM_THREADS"}
    assert report["code_sha256"]["runner"] == C.sha256_bytes(Path(runner.__file__).read_bytes())
    assert set(report["code_sha256"]) == {
        "runner", "posted_evaluator", "strict_parser", "post_writer", "task_builder",
        "objectives", "simulator", "run_artifacts"}
    assert report["config_sha256"].startswith("sha256:")
    entries = runner._plan_entries(config, task, "dev", None)
    costs = runner._pecent(dict(config, timing_evidence=report), entries)
    assert len(costs) == 7
    assert all(row["estimated_run_wall_time_s"] is not None for row in costs)
    cma = next(row for row in costs if row["run_id"].startswith("hard-cmaes-s0-"))
    assert (cma["candidate_evaluations"], cma["validation_calls"]) == (4, 2)

    # Explicit arithmetic: screen posts overlap, so their batch wall time is
    # max(0.5, 0.75) rather than their serial 1.25 s sum. Fine and convergence
    # each have their own phase cost; a further 2 s belongs to primary overhead.
    known = {
        "schema": "diffcam-pilot-timing/2", "task_hash": task.task_hash,
        "candidate_seconds": {"smooth_forward": 1.0, "smooth_gradient": 2.0, "hard_forward": 3.0},
        "constructor_seconds": 4.0,
        "posted_seconds": [0.5, 0.75],
        "posted_seconds_dialects": [
            {"dialect": "rs274", "wall_seconds": 0.5},
            {"dialect": "haas", "wall_seconds": 0.75},
        ],
        "posted_evaluation_levels": {
            "screen": {"phase_wall_seconds": 0.75},
            "verdict": {"phase_wall_seconds": 1.0},
            "convergence": {"phase_wall_seconds": 0.5},
        },
        "worker_wall_seconds": 14.25,
    }
    nominal = next(item for item in entries if item["method"] == "nominal-cam")
    cost = runner._pecent(dict(config, timing_evidence=known), [nominal])[0]
    assert cost["other_worker_seconds"] == pytest.approx(2.0)
    assert cost["posted_seconds"] == pytest.approx(0.75)
    assert cost["estimated_run_wall_time_s"] == pytest.approx(9.75)

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


def test_ineligible_final_program_is_complete_and_excluded_from_efficiency(tmp_path, monkeypatch):
    """Use a typed gouging trajectory to test result/status/audit separation."""
    import numpy as np
    from algorithms import objectives as O
    from algorithms import run_artifacts as R

    config = _pilot_config()
    task, motion, parameterization, _, posted_factory = runner._pilot_context(
        config, tmp_path / "repository")
    assert motion.segments[0].kind == C.SegmentKind.RAPID
    assert motion.segments[1].kind == C.SegmentKind.FEED
    theta = np.zeros(parameterization.num_params)
    displaced = np.asarray(parameterization.positions(theta)).copy()
    # The approach remains safely high. The following feed cuts retained outer-wall
    # stock at X=28mm, well outside the 24mm pocket, then continues into the seed.
    # This is a genuine geometry violation, not a fabricated evaluator verdict.
    displaced[1] = (28.0, 0.0, 8.0)
    displaced[2] = (28.0, 0.0, -3.0)

    def displaced_positions(_self, _theta):
        return displaced.copy()

    monkeypatch.setattr(type(parameterization), "positions", displaced_positions)

    class ToolStart:
        def __getitem__(self, key):
            assert key is None
            return self

        def to_list(self):
            return ((np.asarray(task.initial_state.position_mm) - np.asarray(task.grid.origin_mm))
                    / np.asarray(task.grid.extent_mm)).tolist()

    # No optimizer or simulator is invoked in this unit fixture. The final typed
    # motion, typed posts and production evaluator are all real.
    evaluator = SimpleNamespace(
        task=task, motion=motion, parameterization=parameterization,
        spec=O.ObjectiveSpec(k=config["k_smooth"]), ledger=R.ComputeLedger(),
        sim=SimpleNamespace(tool_start=ToolStart(), precision="f64",
                            precision_metadata=lambda: _precision_metadata_stub()),
    )
    evaluator.candidate_motion = lambda current: O.CandidateEvaluator.candidate_motion(evaluator, current)
    candidate = C.CandidateResult(
        candidate_id="unit-retained-wall-cut", theta_hash=C.content_hash(theta.tolist()),
        objective_kind="hard", objective=0.0, valid=True,
        coverage=C.Coverage.INCOMPLETE, components={"unit_fixture": 1.0},
    )
    outcome = runner.Outcome(
        seed=0, best=candidate, best_theta=theta, initial_theta=theta,
        budget=runner.B.OptimizationBudget(candidate_evaluations=1, population_size=None),
        evaluations=1, validation_calls=0, generations=0,
        forward_calls=1, backward_calls=0, invalid_candidates=0,
        termination_reason="unit_retained_wall_cut", history=[],
    )
    entry = runner._plan_entries(config, task, "dev", "nominal-cam-*")[0]
    root = tmp_path / "ineligible-run"
    record = runner._write_outcome(entry, evaluator, outcome, posted_factory,
                                   config, root, runner.REPOSITORY)
    assert record.status == C.RunStatus.COMPLETE
    assert record.failure_reason is None
    for dialect in C.Dialect:
        nc_name, eval_name = C.DIALECT_ARTIFACTS[dialect]
        evaluation = C.loads((root / record.run_id / eval_name).read_text())
        assert evaluation.coverage == C.Coverage.COMPLETE
        assert evaluation.geometry_pass is False
        assert not evaluation.efficiency_eligible
        assert evaluation.nc_hash == C.nc_hash((root / record.run_id / nc_name).read_bytes())
        assert evaluation.metrics["removed_volume_mm3"].value > 0

    runner._atomic(root / "manifest.json",
                   runner._manifest(config, task, [entry], root, planned=False))
    manifest = A.load_manifest(root / "manifest.json")
    reports, audit = A.audit_runs(root, manifest)
    assert audit["summary"]["auditor_passed"] is True
    assert audit["summary"]["all_runs_completed"] is True
    assert audit["summary"]["evidence_complete"] is True
    assert reports[0]["usable_for_statistics"] is True
    for dialect in C.Dialect:
        cycle = A._metric_by_task(manifest.runs.values(), audit["_audits_by_id"],
                                  dialect, "cycle_time_s", manifest.tasks)
        assert cycle[task.task_id]["observations_used"] == 0
        assert cycle[task.task_id]["value"] is None
        assert cycle[task.task_id]["exclusions"]["not_efficiency_eligible"] == 1
        # Non-efficiency evidence is retained; this is not a missing/crashed run.
        removed = A._metric_by_task(manifest.runs.values(), audit["_audits_by_id"],
                                    dialect, "removed_volume_mm3", manifest.tasks)
        assert removed[task.task_id]["observations_used"] == 1
        assert removed[task.task_id]["value"] > 0


def _typed_evaluation(task, nc_bytes, dialect=C.Dialect.RS274, *, geometry_pass=True, step=1.0):
    """Typed data-format fixture, not a production geometry computation."""
    return C.EvaluationResult(
        evaluator_id="unit-typed-result", evaluator_revision="unit",
        task_hash=task.task_hash, tool_setup_hash=task.tool_setup_hash,
        discretization={"settings": {
            "max_step_mm": step, "expected_nc_hash": C.nc_hash(nc_bytes),
            "exclusion_tolerance_mm3": 0.0,
            "compute_bounds": True, "safety_screen": True,
            "max_samples_per_segment": 4096, "subdivisions": None,
            "record_stations": False,
        }, "segments": []},
        coverage=C.Coverage.COMPLETE, coverage_reasons=(),
        metrics={
            "cycle_time_s": C.Metric(value=20.0, unit="s"),
            "residual_volume_mm3": C.Metric(value=0.5 if geometry_pass else 0.25, unit="mm^3"),
            "gouge_volume_mm3": C.Metric(value=0.0 if geometry_pass else 0.25, unit="mm^3"),
            "max_surface_error_mm": C.Metric(value=0.0 if geometry_pass else 0.25, unit="mm"),
            "removed_volume_mm3": C.Metric(value=8904.0, unit="mm^3"),
        },
        safety_pass=True, geometry_pass=geometry_pass,
        dialect=dialect, nc_hash=C.nc_hash(nc_bytes),
    )


def _pilot_refinement_fixture(root: Path, *, method="fixed-adam", seed=0, residual_mm3=None,
                              primary_geometry_pass=True):
    """Make real typed task/motion/NC/artifact files with a typed input result."""
    import dataclasses
    from algorithms import objectives as O
    from algorithms import run_artifacts as R
    from benchmarks import pilot_task
    from cam import posts
    from cam.posts import post_motion_plan
    from cam.seeds import pocket_with_island_seed

    config = _pilot_config()
    config["experiment"] = "pilot"
    config["seeds"] = [seed]
    task = pilot_task.build_pilot_task(config["grid_pitch_mm"])
    motion = pocket_with_island_seed(task)
    parameterization = pilot_task.fit_parameterization(motion)
    posting = R.post_motion(task, motion)
    entry = runner._plan_entries(config, task, "dev", f"{method}-s{seed}-*")[0]
    primary = _typed_evaluation(task, posting.files["rs274.nc"], geometry_pass=primary_geometry_pass)
    if residual_mm3 is not None:
        primary = dataclasses.replace(primary, metrics={
            **primary.metrics,
            "residual_volume_mm3": C.Metric(value=residual_mm3, unit="mm^3"),
        })
    haas = _typed_evaluation(task, posting.files["haas.nc"], dialect=C.Dialect.HAAS,
                             geometry_pass=primary_geometry_pass)
    if residual_mm3 is not None:
        haas = dataclasses.replace(haas, metrics=primary.metrics)
    files = dict(posting.files)
    position = parameterization.positions(np.zeros(parameterization.num_params))
    files.update({
        "config.json": R.json_bytes({
            **R.build_run_config(
                entry["run_id"], task, motion, seed=0, parameterization=parameterization,
                objective=O.ObjectiveSpec(k=40.0),
                budgets={"limits": {}, "spent": {"candidate_evaluations": 1}},
                repository=runner.REPOSITORY),
            "simulator": _precision_metadata_stub()}),
        "history.json": R.json_bytes({"history": [], "ledger": {"candidate_evaluations": 1,
                                                               "forward_calls": 1,
                                                               "backward_calls": 0,
                                                               "validation_calls": 0}}),
        "metrics.json": R.json_bytes({"best_candidate": None, "execution": {"subprocess": 0}}),
        "motion_plan.json": R.record_bytes(motion),
        "initial_trajectory.npy": R.numpy_bytes(position),
        "trajectory.npy": R.numpy_bytes(position),
        "trajectory_deltas.npy": R.numpy_bytes(np.diff(position, axis=0)),
        "eval_posted_rs274.json": R.record_bytes(primary),
        "eval_posted_haas.json": R.record_bytes(haas),
    })
    metadata = R.source_metadata(runner.REPOSITORY)
    record = R.assemble_record(
        entry["run_id"], task, C.RunStatus.COMPLETE, posting=posting,
        artifact_names=tuple(sorted(files)), source_commit=metadata["source_commit"],
        dirty_patch_hash=metadata["dirty_patch_hash"])
    directory = root / entry["run_id"]
    original_run = (directory / "run.json").read_bytes() if (directory / "run.json").exists() else None
    R.ArtifactWriter().write(str(directory), record, files)
    if original_run is None:
        original_run = (directory / "run.json").read_bytes()
    runner._atomic(root / "manifest.json", runner._manifest(config, task, [entry], root, planned=False))
    runner._atomic(root / "frozen_config.json", {"raw_config": config, "identity": entry["identity"]})
    return root, config, record, original_run, primary


def test_refinement_sidecar_metadata_and_audit_report_are_real_files(tmp_path):
    root, config, record, immutable_run, primary = _pilot_refinement_fixture(tmp_path / "primary")
    aux_root = tmp_path / "shared-refined"
    auxiliary_id = record.run_id + "-verdict"
    original_metrics = (root / record.run_id / "metrics.json").read_bytes()
    seen = []

    def evaluate(task, nc_bytes, dialect, settings):
        seen.append(dict(settings))
        expected_nc = root / record.run_id / C.DIALECT_ARTIFACTS[dialect][0]
        assert nc_bytes == expected_nc.read_bytes()
        return _typed_evaluation(task, nc_bytes, dialect=dialect, geometry_pass=False, step=settings["max_step_mm"])

    result = runner._refinement_worker(config, root, record.run_id, aux_root, evaluator_override=evaluate)
    assert len(seen) == 3
    expected = dict(primary.discretization["settings"])
    assert [item["max_step_mm"] for item in seen] == [0.125, 0.125, 0.25]
    rs_bytes = (root / record.run_id / "rs274.nc").read_bytes()
    haas_bytes = (root / record.run_id / "haas.nc").read_bytes()
    assert seen == [
        dict(expected, max_step_mm=0.125, expected_nc_hash=C.nc_hash(rs_bytes)),
        dict(expected, max_step_mm=0.125, expected_nc_hash=C.nc_hash(haas_bytes)),
        dict(expected, max_step_mm=0.25, expected_nc_hash=C.nc_hash(rs_bytes)),
    ]
    assert result["comparison"]["gouge_volume_mm3"]["delta"] == pytest.approx(0.25)
    assert set(result["screen_disagreement_fields"]) == {"geometry_pass", "efficiency_eligible"}
    assert (root / record.run_id / "run.json").read_bytes() == immutable_run
    assert (root / record.run_id / "metrics.json").read_bytes() == original_metrics

    directory = aux_root / auxiliary_id
    for dialect in C.Dialect:
        evaluation = C.loads((directory / C.DIALECT_ARTIFACTS[dialect][1]).read_text())
        assert evaluation.dialect is dialect
        assert evaluation.discretization["settings"]["max_step_mm"] == 0.125
        expected_nc = root / record.run_id / C.DIALECT_ARTIFACTS[dialect][0]
        assert evaluation.nc_hash == C.nc_hash(expected_nc.read_bytes())
        assert evaluation.geometry_pass is False
        screen = directory / f"screen_{C.DIALECT_ARTIFACTS[dialect][1]}"
        assert screen.read_bytes() == (root / record.run_id / C.DIALECT_ARTIFACTS[dialect][1]).read_bytes()
    convergence = C.loads((directory / "eval_convergence_rs274.json").read_text())
    assert convergence.dialect is C.Dialect.RS274
    assert convergence.discretization["settings"]["max_step_mm"] == 0.25

    link = json.loads((aux_root / f"{record.run_id}-refinement-link.json").read_text())
    assert link["schema"] == "diffcam-refinement-link/4"
    assert link["primary_run_path"] == record.run_id
    assert link["primary_metrics_sha256"] == C.sha256_bytes(original_metrics)
    for dialect in C.Dialect:
        assert link["primary_run_json_sha256"] == C.sha256_bytes((root / record.run_id / "run.json").read_bytes())
    aux_record = directory / "run.json"
    assert link["auxiliary_record"]["sha256"] == C.sha256_bytes(aux_record.read_bytes())
    assert A._audit_run(aux_record).complete_and_audited
    assert link["roles"]["verdict"]["evaluations"]["rs274"]["sha256"] == C.sha256_bytes(
        (directory / "eval_posted_rs274.json").read_bytes())
    assert link["roles"]["screen"]["evaluations"]["haas"]["sha256"] == C.sha256_bytes(
        (directory / "screen_eval_posted_haas.json").read_bytes())
    assert not (root / record.run_id / "eval_posted_rs274_refined.json").exists()

    result = runner._refinement_worker(config, root, record.run_id, aux_root,
                                       evaluator_override=lambda *args: pytest.fail("re-evaluated immutable cache"))
    assert result["status"] == "skipped_complete"
    comparison = directory / "comparison.json"
    original_comparison = comparison.read_bytes()
    for corrupted in (b"{not valid JSON", b"{\"valid_json\":\"but altered\"}\n"):
        comparison.write_bytes(corrupted)
        with pytest.raises(runner.RunnerError, match="refusing to overwrite"):
            runner._refinement_worker(config, root, record.run_id, aux_root,
                                      evaluator_override=lambda *args: pytest.fail("must not re-evaluate"))
        assert comparison.read_bytes() == corrupted
    comparison.write_bytes(original_comparison)
    screen = directory / "screen_eval_posted_rs274.json"
    corrupted_screen = screen.read_bytes()
    screen.write_bytes(b"deliberately corrupted typed screen")
    with pytest.raises(runner.RunnerError):
        runner._refinement_worker(config, root, record.run_id, aux_root,
                                  evaluator_override=lambda *args: pytest.fail("must not re-evaluate"))
    assert screen.read_bytes() == b"deliberately corrupted typed screen"
    screen.write_bytes(corrupted_screen)


def test_refinement_audit_exposes_baseline_refined_and_verdict_change(tmp_path):
    root, config, record, _original, _primary = _pilot_refinement_fixture(tmp_path / "primary")
    aux_root = tmp_path / "audit"
    case = runner._refinement_worker(
        config, root, record.run_id, aux_root,
        evaluator_override=lambda task, nc, dialect, settings:
        _typed_evaluation(task, nc, dialect=dialect, geometry_pass=False, step=settings["max_step_mm"]))
    case["manifest"] = A.load_manifest(root / "manifest.json")
    document = runner._audit_refinement_root([case], aux_root)
    assert document["case_count"] == 1
    assert document["baseline_and_refined_records"] == 2
    assert document["manifest"]["auditor_passed"] is True
    assert document["manifest"]["all_runs_completed"] is True
    path = aux_root / "refinement-audit.json"
    assert path.is_file()
    assert document["schema"] == "diffcam-fidelity-audit/4"
    assert document["reported_outcomes"][record.run_id]["outcomes"]["rs274"]["geometry_pass"] is False
    assert document["reported_outcomes"][record.run_id]["outcomes"]["rs274"]["efficiency_eligible"] is False
    assert len(document["comparisons"]) == 1
    assert set(document["comparisons"][0]["metrics"]) == {
        "residual_volume_mm3", "gouge_volume_mm3", "max_surface_error_mm"}
    assert "cycle_time_s" not in document["comparisons"][0]["metrics"]
    condition = case["manifest"].runs[case["run_id"]].method
    assert document["screen_disagreement_by_condition"][condition]["count"] == 1
    audited_by_id = {entry["run_id"]: entry for entry in document["records"]}
    assert audited_by_id[case["refined_run_id"]]["dialects"]["rs274"]["evaluation"]["geometry_pass"] is False
    assert audited_by_id[record.run_id]["dialects"]["rs274"]["evaluation"]["geometry_pass"] is True


def test_refinement_audit_isolates_one_disagreement_in_same_condition(tmp_path):
    import dataclasses

    cases = []
    for seed, geometry_pass in ((0, True), (1, False)):
        root, config, record, _, _ = _pilot_refinement_fixture(
            tmp_path / f"primary-{seed}", method="hard-cmaes", seed=seed)
        case = runner._refinement_worker(
            config, root, record.run_id, tmp_path / "audit-multiseed",
            evaluator_override=lambda task, nc, dialect, settings, passed=geometry_pass:
            _typed_evaluation(task, nc, dialect=dialect, geometry_pass=passed,
                              step=settings["max_step_mm"]))
        case["manifest"] = A.load_manifest(root / "manifest.json")
        cases.append(case)
    doc = runner._audit_refinement_root(cases, tmp_path / "audit-multiseed")
    audited = A.load_manifest(tmp_path / "audit-multiseed" / "manifest.json")
    for case in cases:
        assert audited.runs[case["run_id"]].method == "hard-cmaes"
        assert audited.runs[case["refined_run_id"]].method == "hard-cmaes-fine-verdict"
    assert len(doc["comparisons"]) == 1
    assert set(doc["comparisons"][0]["metrics"]) == {
        "residual_volume_mm3", "gouge_volume_mm3", "max_surface_error_mm"}
    assert doc["screen_disagreement_by_condition"]["hard-cmaes"]["count"] == 1
    assert doc["reported_outcomes"][cases[1]["run_id"]]["outcomes"]["rs274"]["efficiency_eligible"] is False
    assert set(doc["reported_outcomes"][cases[1]["run_id"]]["screen_disagreement_fields"]) == {
        "geometry_pass", "efficiency_eligible"}

    primary_tasks = dict(cases[0]["manifest"].tasks)
    task_id = next(iter(primary_tasks))
    primary_tasks[task_id] = dataclasses.replace(
        primary_tasks[task_id], completion_residual_mm3=cases[0]["completion_residual_mm3"])
    primary_runs, primary_audits, fine = {}, {}, {}
    for case in cases:
        primary_runs[case["run_id"]] = case["manifest"].runs[case["run_id"]]
        primary_audits[case["run_id"]] = A._audit_run(case["root"] / case["run_id"] / "run.json")
        fine[case["run_id"]] = {}
        for dialect in C.Dialect:
            fine[case["run_id"]][dialect.value] = C.loads(
                (tmp_path / "audit-multiseed" / case["refined_run_id"] /
                 C.DIALECT_ARTIFACTS[dialect][1]).read_text())
    collapsed = A._outcome_by_task(
        list(primary_runs.values()), primary_audits, C.Dialect.RS274,
        "efficiency_eligible", primary_tasks, fine)
    assert collapsed[task_id]["value"] == pytest.approx(0.5)
    assert collapsed[task_id]["known_observations"] == 2
    assert collapsed[task_id]["unknown_or_missing_observations"] == 0
    summary = A._rate_summary(collapsed, [task_id], primary_tasks, "D04-two-seeds", 20261108, 100, 0.95)
    assert summary["task_mean_rate"] == pytest.approx(0.5)
    effect = A._metric_by_task(
        list(primary_runs.values()), primary_audits, C.Dialect.RS274,
        "cycle_time_s", primary_tasks, fine)
    assert effect[task_id]["observations_used"] == 1
    assert effect[task_id]["exclusions"] == {"not_efficiency_eligible": 1}


def test_refinement_completion_flip_updates_primary_pilot_report(tmp_path):
    import dataclasses

    root, config, record, immutable_run, primary = _pilot_refinement_fixture(
        tmp_path / "primary", method="fixed-adam", residual_mm3=0.0)
    auxiliary_root = tmp_path / "auxiliary"

    def evaluate(task, nc, dialect, settings):
        result = _typed_evaluation(task, nc, dialect=dialect, geometry_pass=True, step=settings["max_step_mm"])
        return dataclasses.replace(result, metrics={
            **result.metrics, "residual_volume_mm3": C.Metric(value=3.0, unit="mm^3")})

    case = runner._refinement_worker(
        config, root, record.run_id, auxiliary_root, evaluator_override=evaluate)
    assert case["comparison"]["geometry_pass"]["changed"] is False
    assert case["comparison"]["safety_pass"]["changed"] is False
    assert case["comparison"]["completion_pass"]["primary"] is True
    assert case["comparison"]["completion_pass"]["refined"] is False
    assert case["screen_disagreement_fields"] == ["completion_pass"]
    assert case["comparison"]["any_outcome_verdict_change"] is True
    assert case["comparison"]["completion_pass"]["changed"] is True
    case["manifest"] = A.load_manifest(root / "manifest.json")
    auxiliary = runner._audit_refinement_root([case], auxiliary_root)
    assert auxiliary["screen_disagreement_by_condition"]["fixed-adam"]["count"] == 1
    assert auxiliary["reported_outcomes"][record.run_id]["outcomes"]["rs274"]["efficiency_eligible"] is False
    report = runner._pilot_report(config, root, auxiliary_root, bootstrap_replicates=100)
    row = next(row for row in report["runs"] if row["run_id"] == record.run_id)
    assert row["verdict_evaluations"]["rs274"]["geometry_pass"] is True
    assert row["verdict_evaluations"]["rs274"]["completion_pass"] is False
    assert row["verdict_evaluations"]["rs274"]["efficiency_eligible"] is False
    assert report["refinement"]["screen_disagreements_by_run_id"][record.run_id] == ["completion_pass"]
    threshold = primary.metrics["residual_volume_mm3"].value
    assert threshold <= case["completion_residual_mm3"] < 3.0
    assert (root / record.run_id / "run.json").read_bytes() == immutable_run


def test_primary_pilot_report_uses_only_fine_dialects_and_fails_closed(tmp_path):
    import dataclasses
    import shutil

    nominal_root, nominal_config, nominal, nominal_bytes, _ = _pilot_refinement_fixture(
        tmp_path / "nominal-source", method="nominal-cam", residual_mm3=0.0)
    fixed_root, fixed_config, fixed, fixed_bytes, _ = _pilot_refinement_fixture(
        tmp_path / "fixed-source", method="fixed-adam", residual_mm3=9.375,
        primary_geometry_pass=False)
    root = tmp_path / "primary"
    shutil.copytree(nominal_root, root)
    shutil.copytree(fixed_root / fixed.run_id, root / fixed.run_id)
    document = json.loads((root / "manifest.json").read_text())
    document["expected_runs"].extend(json.loads((fixed_root / "manifest.json").read_text())["expected_runs"])
    document["comparisons"] = [{
        "name": "fine eligibility", "method": "fixed-adam", "baseline": "nominal-cam",
        "dialect": "rs274", "metrics": ["cycle_time_s"],
    }]
    runner._atomic(root / "manifest.json", document)
    aux_root = tmp_path / "auxiliary"
    cases = []
    for source_root, record, config in ((nominal_root, nominal, nominal_config),
                                         (fixed_root, fixed, fixed_config)):
        case = runner._refinement_worker(
            config, source_root, record.run_id, aux_root,
            evaluator_override=lambda task, nc, dialect, settings:
            _typed_evaluation(task, nc, dialect=dialect, geometry_pass=True,
                              step=settings["max_step_mm"]))
        case["manifest"] = A.load_manifest(source_root / "manifest.json")
        cases.append(case)
    runner._audit_refinement_root(cases, aux_root)
    report = runner._pilot_report(nominal_config, root, aux_root, bootstrap_replicates=100)
    comparison = report["comparisons"][0]
    assert comparison["outcomes"]["efficiency_eligible"]["method"]["task_mean_rate"] == 1.0
    task_id = next(iter(document["tasks"]))["task_id"]
    metric = comparison["metrics"]["cycle_time_s"]["method_task_availability"][task_id]
    assert metric["observations_used"] == 1
    assert metric["value"] == 20.0
    assert metric["exclusions"] == {}
    assert report["refinement"]["screen_disagreement_by_condition"]["fixed-adam"]["count"] == 1
    rows = {row["run_id"]: row for row in report["runs"]}
    assert rows[fixed.run_id]["dialects"]["rs274"]["evaluation"]["geometry_pass"] is False
    assert rows[fixed.run_id]["verdict_evaluations"]["rs274"]["geometry_pass"] is True
    assert rows[fixed.run_id]["verdict_evaluations"]["rs274"]["completion_pass"] is True
    for dialect in C.Dialect:
        assert rows[fixed.run_id]["verdict_evaluations"][dialect.value]["efficiency_eligible"] is True
    assert rows[fixed.run_id]["verdict_evaluations"]["haas"]["max_step_mm"] == 0.125
    assert fixed_bytes == (root / fixed.run_id / "run.json").read_bytes()
    assert nominal_bytes == (root / nominal.run_id / "run.json").read_bytes()

    haas_path = aux_root / f"{fixed.run_id}-verdict" / "eval_posted_haas.json"
    haas_original = haas_path.read_bytes()
    haas_path.unlink()
    with pytest.raises(runner.RunnerError, match="auxiliary audit is incomplete or invalid"):
        runner._pilot_report(nominal_config, root, aux_root, bootstrap_replicates=100)
    haas_path.write_bytes(haas_original)
    link_path = aux_root / f"{fixed.run_id}-refinement-link.json"
    stale = json.loads(link_path.read_text())
    stale["schema"] = "diffcam-refinement-link/3"
    link_path.write_text(json.dumps(stale))
    with pytest.raises(runner.RunnerError, match="schema /4"):
        runner._pilot_report(nominal_config, root, aux_root, bootstrap_replicates=100)


def test_refinement_driver_serial_programs_parallel_dialects(tmp_path, monkeypatch):
    import threading

    left, left_config, left_record, _, _ = _pilot_refinement_fixture(tmp_path / "left", method="fixed-adam")
    right, right_config, right_record, _, _ = _pilot_refinement_fixture(tmp_path / "right", method="annealed-adam")
    lock = threading.Lock()
    active, peak = [], []
    workers = set()
    phase_sequences = {left_record.run_id: [], right_record.run_id: []}
    barrier = threading.Barrier(2)

    def compute(task, job, config):
        if job.role == "verdict":
            barrier.wait(timeout=10)
        with lock:
            program = next(
                run_id for run_id in phase_sequences
                if any(part.startswith(f"{run_id}-") for part in job.workspace.parts)
            )
            phase_sequences[program].append(job.role)
            workers.add(threading.get_ident())
        job.workspace.mkdir(parents=True, exist_ok=True)
        evaluation = _typed_evaluation(task, job.nc_bytes, dialect=job.dialect,
                                       step=job.settings["max_step_mm"])
        source = job.workspace / "provided-typed-result.json"
        payload = R.record_bytes(evaluation)
        source.write_bytes(payload)
        return evaluation, {
            "source": str(source), "evaluation_path": str(source),
            "evaluation_sha256": C.sha256_bytes(payload), "role": job.role,
            "dialect": job.dialect.value, "status": "reused_typed_evaluation",
            "worker_peak_rss_kib": None, "worker_pid": None,
            "evaluator_wall_seconds": 0.0, "process_wall_seconds": 0.0,
            "max_step_mm": evaluation.discretization["settings"]["max_step_mm"],
            "expected_nc_hash": evaluation.nc_hash, "thread_environment": None,
        }

    def coordinate(config_path, root, run_id, auxiliary_root, timeout):
        del config_path, timeout  # The in-process coordinator uses the same frozen fixture config.
        with lock:
            active.append(1)
            peak.append(len(active))
        try:
            return runner._refinement_worker(left_config, root, run_id, auxiliary_root)
        finally:
            with lock:
                active.pop()

    monkeypatch.setattr(runner, "_posted_job_process", compute)
    monkeypatch.setattr(runner, "_refine_process", coordinate)
    aux_root = tmp_path / "audit"
    result = runner._refinement_driver(left_config, "unit-test-scheduler.json", [left, right], None, aux_root)
    assert result["status_counts"] == {"complete": 2}
    assert max(peak) == 1
    assert len(workers) == 2
    assert phase_sequences[left_record.run_id] == ["verdict", "verdict", "convergence"]
    assert phase_sequences[right_record.run_id] == ["verdict", "verdict", "convergence"]
    assert result["program_level_parallel_jobs"] == 1
    assert result["posted_job_parallel_jobs"] == 2
    assert result["audit"]["baseline_and_refined_records"] == 4
    assert result["audit"]["manifest"]["auditor_passed"] is True


def test_refinement_process_preserves_cached_history_and_link_paths(tmp_path):
    root, config, record, _, _ = _pilot_refinement_fixture(tmp_path / "primary")
    auxiliary_root = tmp_path / "refined"
    auxiliary_directory = auxiliary_root / f"{record.run_id}-verdict"
    runner._refinement_worker(
        config, root, record.run_id, auxiliary_root,
        evaluator_override=lambda task, nc, dialect, settings:
        _typed_evaluation(task, nc, dialect=dialect, step=settings["max_step_mm"]))
    config_path = tmp_path / "fixture-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    history_path = auxiliary_directory / "history.json"
    original_history = history_path.read_bytes()
    first = runner._refine_process(str(config_path), root, record.run_id, auxiliary_root, timeout=config["refinement_check"]["timeout_s"])
    second = runner._refine_process(str(config_path), root, record.run_id, auxiliary_root, timeout=config["refinement_check"]["timeout_s"])
    assert first["status"] == second["status"] == "skipped_complete"
    assert history_path.read_bytes() == original_history
    for result in (first, second):
        assert result["auxiliary_link_path"] == auxiliary_root / f"{record.run_id}-refinement-link.json"
        assert result["auxiliary_directory"] == auxiliary_directory
        assert Path(result["log"]) == history_path
        assert len(result["resources"]["processes"]) == 3
        resources = json.loads(history_path.read_text())["resources"]
        assert isinstance(resources["phase_wall_seconds"], (int, float))
        assert len(resources["phase_records"]) == 2
    driven = runner._refinement_driver(config, str(config_path), [root], None, auxiliary_root)
    assert driven["status_counts"] == {"skipped_complete": 1}
    assert driven["audit"]["baseline_and_refined_records"] == 2
    assert driven["audit"]["manifest"]["auditor_passed"] is True
    table = runner._pilot_report(config, root, auxiliary_root, bootstrap_replicates=100)
    assert table["refinement"]["screen_disagreement_by_condition"] == {"fixed-adam": {"count": 0}}
    reported = next(row for row in table["runs"] if row["run_id"] == record.run_id)
    assert reported["verdict_evaluations"][C.Dialect.RS274.value]["geometry_pass"] is True
    assert table["precision"]["required_precision"] == "f64"
    assert table["precision"]["evidence_complete"] is True
    assert table["precision"]["metadata_by_run_id"][record.run_id] == _precision_metadata_stub()


def test_refinement_timing_is_separate_from_primary_and_parallel_forecast(tmp_path, monkeypatch):
    from benchmarks import pilot_task
    from cam.seeds import pocket_with_island_seed

    config = _pilot_config()
    task = pilot_task.build_pilot_task(voxel_mm=config["grid_pitch_mm"])
    motion = pocket_with_island_seed(task)
    parameterization = pilot_task.fit_parameterization(motion)

    class Candidate:
        geometry_sources = {"stock": SimpleNamespace(path="stock.npy", sha256="sha256:" + "0" * 64),
                            "target": SimpleNamespace(path="target.npy", sha256="sha256:" + "1" * 64)}

        def evaluate_candidate(self, theta, candidate_id, kind):
            return SimpleNamespace(valid=True, objective=1.0, invalid_reason=None)

        def evaluate_candidate_with_gradient(self, theta, candidate_id):
            return SimpleNamespace(result=SimpleNamespace(valid=True, objective=1.0, invalid_reason=None),
                                   gradient_theta=np.zeros(parameterization.num_params))

        def candidate_motion(self, theta):
            return motion

    class Primary:
        def __init__(self):
            self.timings = []

        def evaluate_posting(self, actual_task, posting, workspace, *, role, single_dialect=False):
            self.timings = []
            results = {}
            for post in posting.posted:
                nc = posting.files[C.DIALECT_ARTIFACTS[post.dialect][0]]
                step = {"screen": 1.0, "verdict": 0.125, "convergence": 0.25}[role]
                results[post.dialect] = _typed_evaluation(
                    actual_task, nc, dialect=post.dialect, geometry_pass=True, step=step)
                seconds = {"screen": 0.5, "convergence": 0.5}[role] if role != "verdict" else (
                    1.5 if post.dialect == C.Dialect.RS274 else 2.0)
                self.timings.append({"dialect": post.dialect.value,
                                     "wall_seconds": seconds, "nc_hash": C.nc_hash(nc)})
            return results

    monkeypatch.setattr(runner, "_context", lambda cfg, root: (
        task, motion, parameterization, lambda *args: Candidate(), lambda *args: Primary()))
    evidence = runner._measure_budget(
        config, tmp_path, capture_dir=tmp_path / "timing-example",
        posted_evaluator_override=Primary())
    assert evidence["in_process_unit_override"] is True
    with pytest.raises(runner.RunnerError, match="independent posted workers"):
        runner._validate_production_timing_evidence(config, evidence, task)
    with pytest.raises(runner.RunnerError, match="not a valid f64 simulator measurement"):
        runner._validate_production_timing_evidence(
            config, dict(evidence, in_process_unit_override=False,
                         simulator_precision=_precision_metadata_stub("f32")), task)
    with pytest.raises(runner.RunnerError, match="independent posted-worker records"):
        runner._validate_production_timing_evidence(
            config, dict(evidence, in_process_unit_override=False,
                         simulator_precision=_precision_metadata_stub("f64")), task)
    entries = runner._plan_entries(config, task, "dev", None)
    costs = runner._pecent(dict(config, timing_evidence=evidence), entries)
    projection = runner._refinement_projection(config, entries, evidence, tmp_path)
    assert projection["expected_program_count"] == 7
    # Rule stages explicitly differentiated for the arithmetic unit fixture:
    phases = evidence["posted_evaluation_levels"]
    phases["screen"]["phase_wall_seconds"] = 2.0
    phases["verdict"]["phase_wall_seconds"] = 3.0
    phases["convergence"]["phase_wall_seconds"] = 1.0
    evidence["candidate_seconds"] = {name: 0.0 for name in evidence["candidate_seconds"]}
    evidence["constructor_seconds"] = 0.0
    evidence["worker_wall_seconds"] = 30.0
    costs = runner._pecent(dict(config, timing_evidence=evidence), entries)
    nominal = next(row for row in costs if row["run_id"].startswith("nominal-cam-"))
    assert nominal["estimated_run_wall_time_s"] == pytest.approx(26.0)
    projection = runner._refinement_projection(config, entries, evidence, tmp_path)
    assert projection["levels"]["screen"]["estimated_phase_wall_time_s"] == 14.0
    assert projection["levels"]["verdict"]["estimated_phase_wall_time_s"] == 21.0
    assert projection["levels"]["convergence"]["estimated_phase_wall_time_s"] == 7.0
    assert projection["estimated_phase_wall_time_s"] == 28.0
    serial = sum(row["serial_evaluation_work_seconds"] for row in projection["levels"].values())
    assert serial == 5.0 * 7
    assert projection["screen_in_primary_estimates"] is True
    assert projection["serial_evaluation_work_seconds"] == pytest.approx(serial)
    for role in ("screen", "verdict", "convergence"):
        assert projection["levels"][role]["actual_independent_job_count"] == 0
        for item in projection["levels"][role]["artifacts"].values():
            assert (tmp_path / item["evaluation_path"]).is_file()
            assert C.sha256_bytes((tmp_path / item["evaluation_path"]).read_bytes()) == item["evaluation_sha256"]
    damaged = dict(phases["verdict"]["artifacts"]["rs274"])
    damaged["evaluation_sha256"] = "sha256:" + "0" * 64
    phases["verdict"]["artifacts"]["rs274"] = damaged
    with pytest.raises(runner.RunnerError, match="hash changed"):
        runner._refinement_projection(config, entries, evidence, tmp_path)


def test_fine_outcome_null_semantics():
    import dataclasses
    from benchmarks import pilot_task
    from cam.seeds import pocket_with_island_seed
    from cam.posts import post_motion_plan

    task = pilot_task.build_pilot_task(voxel_mm=2.0)
    nc = post_motion_plan(pocket_with_island_seed(task), task, dialect=C.Dialect.RS274).encode("utf-8")
    primary = _typed_evaluation(task, nc, geometry_pass=True)
    unknown = _typed_evaluation(task, nc, geometry_pass=None, step=0.125)
    outcome = runner._evaluation_outcome(unknown, task)
    assert outcome["geometry_pass"] is None
    assert outcome["completion_pass"] is True
    assert outcome["efficiency_eligible"] is False
    comparison = runner._fidelity_deltas(primary, unknown, task)
    assert comparison["geometry_pass"]["refined"] is None
    assert set(comparison["changed_outcome_fields"]) == {"geometry_pass", "efficiency_eligible"}
    missing = dataclasses.replace(
        unknown, geometry_pass=True, metrics={
            **unknown.metrics,
            "residual_volume_mm3": C.Metric(value=None, unit="mm^3", null_reason="bounds unresolved"),
        })
    incomplete = runner._evaluation_outcome(missing, task)
    assert incomplete["geometry_pass"] is True
    assert incomplete["completion_pass"] is None
    assert incomplete["efficiency_eligible"] is None


def test_report_rejects_disabled_refinement_and_conflicting_task_threshold(tmp_path):
    root, config, record, _, _ = _pilot_refinement_fixture(tmp_path / "primary")
    manifest_path = root / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text())
    tampered_manifest = json.loads(manifest_path.read_text())
    tampered_manifest["tasks"][0]["completion_residual_mm3"] = 0.0
    manifest_path.write_bytes(R.json_bytes(tampered_manifest))
    aux_root = tmp_path / "verdict"
    case = runner._refinement_worker(
        config, root, record.run_id, aux_root,
        evaluator_override=lambda task, nc, dialect, settings:
        _typed_evaluation(task, nc, dialect=dialect, step=settings["max_step_mm"]))
    case["manifest"] = A.load_manifest(manifest_path)
    runner._audit_refinement_root([case], aux_root)
    with pytest.raises(runner.RunnerError, match="threshold differs from the typed task"):
        runner._pilot_report(config, root, aux_root, bootstrap_replicates=20)
    manifest_path.write_bytes(R.json_bytes(original_manifest))
    config["refinement_check"]["enabled"] = False
    with pytest.raises(runner.RunnerError, match="never substitute the primary screen"):
        runner._pilot_report(config, root, aux_root, bootstrap_replicates=20)


def test_audit_fine_evaluation_must_match_primary_nc_and_tool(tmp_path):
    import dataclasses

    root, config, record, _, _ = _pilot_refinement_fixture(tmp_path / "primary")
    fine = {}
    for dialect in C.Dialect:
        nc = (root / record.run_id / C.DIALECT_ARTIFACTS[dialect][0]).read_bytes()
        fine[dialect.value] = _typed_evaluation(
            C.TaskSpec.from_dict(json.loads((root / record.run_id / "config.json").read_text())["task"]),
            nc, dialect=dialect, step=0.125)
    for field, invalid_hash, reason in (
        ("nc_hash", C.nc_hash(b"different exact NC"), "exact-NC identity"),
        ("tool_setup_hash", C.content_hash("different tool"), "tool setup identity"),
    ):
        supplied = dict(fine, rs274=dataclasses.replace(fine["rs274"], **{field: invalid_hash}))
        with pytest.raises(A.AuditInputError, match=reason):
            A.generate_report(
                root, A.load_manifest(root / "manifest.json"), bootstrap_replicates=20,
                verdict_evaluations={record.run_id: supplied},
                screen_disagreements={record.run_id: []}, required_precision="f64")


def test_configured_typed_reuse_is_explicit_and_hash_checked(tmp_path):
    root, config, record, _, _ = _pilot_refinement_fixture(tmp_path / "primary")
    nc = (root / record.run_id / "rs274.nc").read_bytes()
    primary_task = C.TaskSpec.from_dict(json.loads((root / record.run_id / "config.json").read_text())["task"])
    cached_bytes = R.record_bytes(_typed_evaluation(primary_task, nc, step=0.125))
    source = tmp_path / "explicit-historical-rs274.json"
    source.write_bytes(cached_bytes)
    config["reuse_typed_evaluations"] = {
        "verdict/rs274": {"path": str(source.resolve()), "sha256": C.sha256_bytes(cached_bytes)}}
    overrides = {}
    for role, dialect in (("verdict", C.Dialect.HAAS), ("convergence", C.Dialect.RS274)):
        nc_bytes = (root / record.run_id / C.DIALECT_ARTIFACTS[dialect][0]).read_bytes()
        overrides[(role, dialect.value)] = _typed_evaluation(
            primary_task,
            nc_bytes, dialect=dialect, geometry_pass=True,
            step=0.125 if role == "verdict" else 0.25)
    auxiliary_root = tmp_path / "reuse-check"
    result = runner._refinement_worker(
        config, root, record.run_id, auxiliary_root, evaluator_override=overrides)
    cached = result["resources"]["processes"][0]
    assert (cached["role"], cached["dialect"]) == ("verdict", "rs274")
    assert cached["status"] == "reused_typed_evaluation"
    assert cached["source"] == str(source.resolve())
    assert cached["evaluation_sha256"] == C.sha256_bytes(cached_bytes)
    assert cached["worker_pid"] is None
    assert cached["evaluator_wall_seconds"] is None
    assert [row["phase"] for row in result["resources"]["phase_records"]] == ["verdict", "convergence"]

    config_path = tmp_path / "configured-reuse.json"
    config_path.write_bytes(R.json_bytes(config))
    assert runner._normalized_config(config_path)["reuse_typed_evaluations"] == config["reuse_typed_evaluations"]
    direct = runner._refinement_worker(
        config, root, record.run_id, tmp_path / "direct-object-reuse",
        evaluator_override=overrides,
        reuse_sources={("verdict", "rs274"): C.loads(cached_bytes.decode("utf-8"))})
    direct_processes = direct["resources"]["processes"]
    assert direct_processes[0]["status"] == "reused_typed_evaluation"
    direct_source = Path(direct_processes[0]["source"])
    assert direct_source.is_file() and direct_source.read_bytes() == cached_bytes
    damaged_resources = json.loads(json.dumps(direct["resources"]))
    damaged_resources["processes"][0]["process_wall_seconds"] = 0.0
    with pytest.raises(runner.RunnerError, match="cannot supply fresh .* durations"):
        runner._validate_resource_record(damaged_resources, config,
                                         {d.value: C.nc_hash((root / record.run_id / C.DIALECT_ARTIFACTS[d][0]).read_bytes())
                                          for d in C.Dialect}, tmp_path / "direct-object-reuse")
    before = source.read_bytes()
    invalid = json.loads(config_path.read_text())
    invalid["reuse_typed_evaluations"]["verdict/rs274"]["sha256"] = "sha256:" + "0" * 64
    config_path.write_bytes(R.json_bytes(invalid))
    with pytest.raises(runner.RunnerError, match="declared source hash differs"):
        runner._normalized_config(config_path)
    assert source.read_bytes() == before


@pytest.mark.parametrize("case", ["missing", "mixed"])
def test_precision_report_fails_closed_for_required_run_metadata(tmp_path, case):
    root, config, record, _, _ = _pilot_refinement_fixture(tmp_path / case)
    config_path = root / record.run_id / "config.json"
    original = config_path.read_bytes()
    document = json.loads(original)
    if case == "missing":
        document.pop("simulator")
        expected_reason = "config.simulator"
    else:
        document["simulator"] = _precision_metadata_stub("f32")
        expected_reason = "differs from the required campaign"
    # An explicit new test copy: no historical primary or malformed evidence is rewritten.
    config_path.write_bytes(R.json_bytes(document))
    auxiliary_root = tmp_path / f"{case}-verdict"
    result = runner._refinement_worker(
        config, root, record.run_id, auxiliary_root,
        evaluator_override=lambda task, nc, dialect, settings:
        _typed_evaluation(task, nc, dialect=dialect, step=settings["max_step_mm"]))
    result["manifest"] = A.load_manifest(root / "manifest.json")
    runner._audit_refinement_root([result], auxiliary_root)
    with pytest.raises(A.AuditInputError, match="no verdict report is published") as caught:
        runner._pilot_report(config, root, auxiliary_root, bootstrap_replicates=100)
    assert expected_reason in str(caught.value)
    assert record.run_id in str(caught.value)


def test_precision_identity_run_config_and_checkpoint_separation(tmp_path):
    """Explicit precision stubs test identity/checkpoint logic, not simulator accuracy."""
    class PrecisionStub:
        def __init__(self, precision):
            self.precision = precision

        def precision_metadata(self):
            body = {
                "schema_version": "simulator-precision-identity/1",
                "requested_precision": self.precision,
                "declared_default_fp": self.precision,
                "runtime_default_fp": self.precision,
                "backend_arch": "test-backend-not-a-runtime-measurement",
                "process_constraint": "one precision per process",
                "policy": {
                    "schema_version": "simulator-precision-uniform/1",
                    "variant_revision": f"uniform-{self.precision}/1",
                    "default_fp": self.precision,
                    "field_scope": "all_float_fields",
                    "kernel_scope": "all_float_kernels_and_adjoints",
                },
            }
            return dict(body, precision_identity_sha256=C.content_hash(body))

    task, motion, parameterization = runner._prototype()
    def evaluator(precision):
        return SimpleNamespace(
            sim=PrecisionStub(precision), task=task, motion=motion,
            parameterization=parameterization, spec={"kind": "identity_unit"})

    f32, f64 = evaluator("f32"), evaluator("f64")
    precision = R.simulator_precision_metadata(f64.sim, requested_precision="f64")
    assert precision["runtime_default_fp"] == "f64"
    assert precision["precision_identity_sha256"].startswith("sha256:")
    assert B._setup_signature(f32, "hard") != B._setup_signature(f64, "hard")
    with pytest.raises(ValueError, match="precision mismatch"):
        R.simulator_precision_metadata(f32.sim, requested_precision="f64")
    damaged_metadata = dict(precision, precision_identity_sha256="sha256:" + "0" * 64)
    damaged = SimpleNamespace(precision="f64", precision_metadata=lambda: damaged_metadata)
    with pytest.raises(ValueError, match="identity hash mismatch"):
        R.simulator_precision_metadata(damaged)

    config = R.build_run_config(
        "precision-unit", task, motion, seed=0, parameterization=parameterization,
        objective={"kind": "identity_unit"}, budgets={"candidate_evaluations": 1},
        repository=runner.REPOSITORY, simulator=f64.sim)
    assert config["simulator"] == precision

    document = {"schema": B.CHECKPOINT_SCHEMA, "objective_kind": "hard",
                "setup_hash": B._setup_signature(f64, "hard")}
    path = tmp_path / "precision-checkpoint.json"
    path.write_bytes(R.json_bytes(dict(document, payload_hash=C.content_hash(document))))
    assert B._read_checkpoint(path, f64, "hard")["setup_hash"] == document["setup_hash"]
    with pytest.raises(B.BlackBoxError, match="simulator-precision identity differs"):
        B._read_checkpoint(path, f32, "hard")

    adam_document = {"schema": "diffcam-adam-checkpoint/1", "seed": 0,
                      "task_setup_hash": B._setup_signature(f64, "smooth"),
                      "simulator_precision": precision}
    path.write_bytes(R.json_bytes(dict(adam_document, payload_hash=C.content_hash(adam_document))))
    assert adam._read_adam_checkpoint(path, f64)["simulator_precision"] == precision
    adam_document = dict(adam_document, simulator_precision=R.simulator_precision_metadata(f32.sim))
    path.write_bytes(R.json_bytes(dict(adam_document, payload_hash=C.content_hash(adam_document))))
    with pytest.raises(B.BlackBoxError, match="simulator-precision identity differs"):
        adam._read_adam_checkpoint(path, f64)


def test_f64_pilot_tool_delta_preserves_full_float64_inputs(tmp_path):
    """Real small-grid f64 candidate; this remains a CPU unit execution, not pilot optimization."""
    config = _pilot_config()
    config_path = tmp_path / "f64-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "f64-roundtrip"
    code = r'''
import json, pathlib, sys
import numpy as np
repository, config_path, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
sys.path.insert(0, str(repository))
from algorithms import run_artifacts as R
from scripts import run_benchmark_suite as runner
config = runner._normalized_config(config_path)
task, motion, parameterization, _, _ = runner._context(config, repository)
evaluator = runner._pilot_evaluator(task, motion, parameterization, config, 0, repository=repository)
theta = np.linspace(np.nextafter(0.0001, 1.0), np.nextafter(0.001, 1.0),
                    parameterization.num_params, dtype=np.float64)
expected = evaluator._deltas(theta)
result = evaluator.evaluate_candidate(theta, "f64-roundtrip", "smooth")
actual = evaluator.sim.tool_delta.to_numpy()
assert result.valid, result.invalid_reason
assert actual.dtype == np.float64
assert np.array_equal(actual, expected), np.max(np.abs(actual - expected))
assert np.any(actual != actual.astype(np.float32).astype(np.float64)), "fixture cannot distinguish f32 input narrowing"
precision = R.simulator_precision_metadata(evaluator.sim, requested_precision=config["precision"])
output.mkdir()
np.save(output / "actual.npy", actual, allow_pickle=False)
np.save(output / "expected.npy", expected, allow_pickle=False)
(output / "identity.json").write_text(json.dumps({"precision": precision, "dtype": str(actual.dtype)},
                                                indent=2, sort_keys=True) + "\n")
'''
    environment = dict(os.environ, DIFFCAM_RUNNER_SIMULATOR="0")
    environment.pop("TI_ARCH", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(runner.REPOSITORY), str(config_path), str(output)],
        cwd=runner.REPOSITORY, env=environment, capture_output=True, text=True, timeout=240, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = np.load(output / "actual.npy", allow_pickle=False)
    expected = np.load(output / "expected.npy", allow_pickle=False)
    assert actual.dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
    identity = json.loads((output / "identity.json").read_text())
    assert identity["dtype"] == "float64"
    assert identity["precision"]["requested_precision"] == "f64"
    assert identity["precision"]["runtime_default_fp"] == "f64"


def test_posted_children_have_independent_pids_and_bounded_rss(tmp_path):
    from benchmarks import pilot_task
    from cam.seeds import pocket_with_island_seed

    config = _pilot_config()
    task = pilot_task.build_pilot_task(voxel_mm=config["grid_pitch_mm"])
    posting = R.post_motion(task, pocket_with_island_seed(task))
    evaluator = runner._pilot_post_factory(task, config)
    result = evaluator.evaluate_posting(task, posting, tmp_path / "children")
    assert set(result) == set(C.Dialect)
    phase = evaluator.phase_records[-1]
    assert len(phase["processes"]) == 2
    assert phase["job_policy"]["total_posted_rss_ceiling_bytes"] == 4 * 1024 ** 3
    assert phase["job_policy"]["parallel_jobs"] == 2
    for index, item in enumerate(phase["processes"]):
        assert item["worker_pid"] > 0
        peak = item["worker_peak_rss_kib"]
        assert peak > 0
        assert peak * 1024 <= config["refinement_check"]["max_rss_bytes_per_job"]
        assert item["role"] == "screen"
        assert item["dialect"] in {"rs274", "haas"}
        assert item["evaluation_sha256"] == C.sha256_bytes(Path(item["evaluation_path"]).read_bytes())
        assert item["process_wall_seconds"] >= item["evaluator_wall_seconds"] >= 0.0
        assert Path(item["log"]).is_file()
        assert set(item["thread_environment"]) == {
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TAICHI_NUM_THREADS"}
        assert item["worker_pid"] != phase["processes"][1 - index]["worker_pid"]


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
    if condition.endswith("cmaes"):
        pytest.importorskip("cma", reason="explicit CMA smoke requires --extra optim")
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
    # Preserve genuinely evaluated artifacts if a harness assertion needed a
    # correction. Resume audits and skips only matching complete evaluations;
    # incomplete/failed artifacts are preserved in resume-backups, never erased.
    resume = ("--resume",) if (output / "frozen_config.json").is_file() else ()
    _invoke(config, output, extra=(*resume, "--only", f"{condition}-s0-*"), timeout=10801)
    manifest = A.load_manifest(output / "manifest.json")
    assert len(manifest.runs) == 1
    run_id = next(iter(manifest.runs))
    record = _named(output, run_id)
    assert record.status == C.RunStatus.COMPLETE, record.failure_reason
    metrics = json.loads((output / run_id / "metrics.json").read_text())
    spent = json.loads((output / run_id / "config.json").read_text())["budgets"]["spent"]
    assert spent["candidate_evaluations"] == config["candidate_evaluations"]
    # Nominal is one direct hard evaluation, not an optimizer generation.
    expected_calls = (0, 1, 0) if condition == "nominal-cam" else (
        (1, 2, 1) if condition.endswith("adam") else (1, 2, 0))
    assert (spent["generations"], spent["forward_calls"], spent["backward_calls"]) == expected_calls
    assert (output / run_id / "eval_posted.json").is_file()
    for dialect in C.Dialect:
        nc_name, eval_name = C.DIALECT_ARTIFACTS[dialect]
        evaluation = C.loads((output / run_id / eval_name).read_text())
        assert evaluation.nc_hash == C.nc_hash((output / run_id / nc_name).read_bytes())
        assert evaluation.task_hash == task.task_hash
        # Eligibility is derived from coverage/safety/geometry by contracts;
        # it need not be a standalone serialized key. Preserve False outcomes.
        assert isinstance(evaluation.efficiency_eligible, bool)
    aux_root = output.resolve().parent / f"{output.resolve().name}-refinement"
    auxiliary_id = f"{run_id}-verdict"
    auxiliary_directory = aux_root / auxiliary_id
    checks = json.loads((aux_root / f"{run_id}-refinement-link.json").read_text())
    assert checks["schema"] == "diffcam-refinement-link/4"
    assert checks["primary_run_path"] == run_id
    assert "primary_run_directory" not in checks
    for dialect in C.Dialect:
        nc_name, eval_name = C.DIALECT_ARTIFACTS[dialect]
        evaluation = C.loads((auxiliary_directory / eval_name).read_text())
        assert evaluation.task_hash == task.task_hash
        assert evaluation.dialect is dialect
        assert evaluation.discretization["settings"]["max_step_mm"] == 0.125
        assert evaluation.nc_hash == C.nc_hash((output / run_id / nc_name).read_bytes())
        assert checks["roles"]["verdict"]["evaluations"][dialect.value]["sha256"] == C.sha256_bytes(
            (auxiliary_directory / eval_name).read_bytes())
        screen = auxiliary_directory / f"screen_{eval_name}"
        assert screen.read_bytes() == (output / run_id / eval_name).read_bytes()
        assert checks["roles"]["screen"]["evaluations"][dialect.value]["sha256"] == C.sha256_bytes(screen.read_bytes())
    convergence = C.loads((auxiliary_directory / "eval_convergence_rs274.json").read_text())
    assert convergence.discretization["settings"]["max_step_mm"] == 0.25
    assert convergence.nc_hash == C.nc_hash((output / run_id / "rs274.nc").read_bytes())
    assert not (output / run_id / "eval_posted_rs274_refined.json").exists()
    assert "refinement_checks" not in metrics["execution"]
    aux_record_path = auxiliary_directory / "run.json"
    aux_record = C.loads(aux_record_path.read_text())
    assert aux_record.task_hash == task.task_hash
    assert aux_record.status == C.RunStatus.COMPLETE
    assert checks["auxiliary_record"]["sha256"] == C.sha256_bytes(aux_record_path.read_bytes())
    assert checks["auxiliary_record"]["path"] == os.path.relpath(aux_record_path, aux_root)
    history = json.loads((auxiliary_directory / "history.json").read_text())
    assert len(history["resources"]["processes"]) == 3
    assert [phase["phase"] for phase in history["resources"]["phase_records"]] == ["verdict", "convergence"]
    for item in history["resources"]["processes"]:
        if item["status"] == "complete":
            assert item["worker_pid"] > 0
            assert 0 < item["worker_peak_rss_kib"] * 1024 <= config["refinement_check"]["max_rss_bytes_per_job"]
    aux_report = json.loads((aux_root / "refinement-audit.json").read_text())
    assert aux_report["schema"] == "diffcam-fidelity-audit/4"
    assert aux_report["manifest"]["auditor_passed"] is True
    assert aux_report["case_count"] == 1
    assert aux_report["baseline_and_refined_records"] == 2

    table_path = output / "pilot_report.json"
    table = json.loads(table_path.read_text())
    execution = json.loads((output / "execution_summary.json").read_text())
    assert execution["pilot_report"]["sha256"] == C.sha256_bytes(table_path.read_bytes())
    assert table["refinement"]["applied_run_ids"] == [run_id]
    assert table["refinement"]["links_by_run_id"][run_id]["sha256"] == C.sha256_bytes(
        (aux_root / f"{run_id}-refinement-link.json").read_bytes())
    row = next(row for row in table["runs"] if row["run_id"] == run_id)
    comparison = json.loads((aux_root / f"{run_id}-verdict" / "comparison.json").read_text())
    for dialect in C.Dialect:
        verdict = C.loads((aux_root / f"{run_id}-verdict" / C.DIALECT_ARTIFACTS[dialect][1]).read_text())
        for name in ("safety_pass", "geometry_pass"):
            assert row["verdict_evaluations"][dialect.value][name] == getattr(verdict, name)
        assert row["verdict_evaluations"][dialect.value]["max_step_mm"] == 0.125
    disagreement_fields = comparison["screen_to_verdict_rs274"]["changed_outcome_fields"]
    assert table["refinement"]["screen_disagreements_by_run_id"][run_id] == disagreement_fields
    assert table["refinement"]["screen_disagreement_by_condition"][condition]["count"] == int(bool(disagreement_fields))

    reports, audit = A.audit_runs(output, manifest)
    assert audit["reconciliation"]["exact_id_reconciliation"] is True
    assert len(reports) == 1 and reports[0]["status"] == "complete"
