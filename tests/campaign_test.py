"""T4.3a suite binding and campaign orchestration tests.

No real suite optimizer is launched here. Only the explicit two-condition
prototype uses actual runner workers with a typed posted-evaluation test stub.
"""
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from algorithms import objectives as O
from algorithms import run_artifacts as R
from benchmarks import audit
from benchmarks import contracts as C
from scripts import run_benchmark_suite as runner
from scripts import run_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]


def _precision_metadata(_simulator):
    body = {
        "schema_version": "simulator-precision-identity/1",
        "requested_precision": "f64", "declared_default_fp": "f64", "runtime_default_fp": "f64",
        "backend_arch": "test_fixture_not_a_runtime_measurement",
        "process_constraint": "one precision per process",
        "policy": {"schema_version": "simulator-precision-uniform/1",
                   "variant_revision": "uniform-f64/1", "default_fp": "f64",
                   "field_scope": "all_float_fields", "kernel_scope": "all_float_kernels_and_adjoints"},
    }
    return dict(body, precision_identity_sha256=C.content_hash(body))


def _runner_template():
    result = json.loads((ROOT / "configs/pilot2.json").read_text())
    result.update(task_source="prototype_analytic", experiment="pilot",
                  candidate_evaluations=1, population_size=2,
                  validation_interval=1, max_validation_calls=0,
                  precision="f64")
    result["refinement_check"]["enabled"] = False
    return result


def _prototype_campaign(*, methods=("nominal-cam", "fixed-adam"), concurrency=2):
    conditions = [{"condition": method, "seeds": [0]} for method in methods]
    return {
        "schema": "diffcam-campaign/1", "campaign_id": "explicit-prototype-unit-test",
        "task_source": "prototype_analytic", "prototype_unit_test": True,
        "task_filter": {"splits": ["dev"]}, "concurrency": concurrency,
        "conditions": conditions, "runner_config": _runner_template(),
        "cost_model": {
            "scope": "synthetic arithmetic units for a unit-test schedule",
            "seconds_per_candidate_by_condition": {method: 10.0 for method in methods},
            "fixed_overhead_seconds_by_condition": {method: 2.0 for method in methods},
            "aggregate_bootstrap_replicates": 100,
        },
    }


def _write_config(tmp_path, document):
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_all_39_suite_bindings_match_embedded_tasks_and_nominal_seed():
    manifest = json.loads((ROOT / "configs/task_manifest.json").read_text())
    assert len(manifest["tasks"]) == 39
    observed = {"dev": 0, "test": 0}
    for row in manifest["tasks"]:
        config = dict(_runner_template(), task_source="suite_manifest", task_id=row["task_id"])
        task, motion, parameterization, metadata = runner._suite_task_binding(config)
        assert task.task_id == row["task_id"] and task.task_hash == row["task_hash"]
        assert motion.motion_hash == row["motion_hash"]
        assert parameterization.num_steps == len(motion.segments)
        assert metadata["reference_cycle_time_source"] == "task_nominal_cam_cycle_time"
        spec, emitted = runner._objective_spec(config, task)
        assert spec.reference_cycle_time_s == metadata["reference_cycle_time_s"]
        assert emitted["reference_motion_hash"] == motion.motion_hash
        observed[task.split.value] += 1
    assert observed == {"dev": 9, "test": 30}
    held_out = next(row for row in manifest["tasks"] if row["split"] == "test")
    config = dict(_runner_template(), task_source="suite_manifest", task_id=held_out["task_id"])
    task, _motion, _params, _factory, _posted = runner._context(config, ROOT)
    assert task.split == C.Split.TEST
    entries = runner._plan_entries(config, task, "test", None)
    assert entries and all(entry["task_id"] == held_out["task_id"] for entry in entries)
    assert runner._identity(task, config, "test") == entries[0]["identity"]
    with pytest.raises(runner.RunnerError, match="grid_pitch_mm"):
        runner._suite_context(dict(config, grid_pitch_mm=task.grid.voxel_mm * 2))


def _typed_evaluation(task, nc_bytes, dialect=C.Dialect.RS274, *, geometry_pass=True, step=0.125):
    """Typed test output for the prototype's explicitly stubbed posted child seam."""
    return C.EvaluationResult(
        evaluator_id="campaign-unit-post-stub", evaluator_revision="unit-only",
        task_hash=task.task_hash, tool_setup_hash=task.tool_setup_hash,
        discretization={"settings": {
                "max_step_mm": step, "expected_nc_hash": C.nc_hash(nc_bytes),
                "exclusion_tolerance_mm3": 0.0, "compute_bounds": True, "safety_screen": True,
                "max_samples_per_segment": 4096, "subdivisions": None, "record_stations": False,
            }, "segments": []},
        coverage=C.Coverage.COMPLETE, coverage_reasons=(),
        metrics={"cycle_time_s": C.Metric(value=20.0, unit="s"),
                 "residual_volume_mm3": C.Metric(value=0.5, unit="mm^3"),
                 "gouge_volume_mm3": C.Metric(value=0.0, unit="mm^3"),
                 "max_surface_error_mm": C.Metric(value=0.0, unit="mm"),
                 "removed_volume_mm3": C.Metric(value=8904.0, unit="mm^3")},
        safety_pass=True, geometry_pass=geometry_pass, dialect=dialect, nc_hash=C.nc_hash(nc_bytes),
    )


def test_campaign_primary_post_role_is_fine_verdict_without_aux_phases(tmp_path):
    document = _prototype_campaign()
    config = document["runner_config"]
    config["task_source"] = "suite_manifest"
    config["campaign_verdict_only"] = True
    task, motion, parameterization = runner._prototype()
    reference = runner._nominal_cycle_metadata(task, motion)
    theta = np.zeros(parameterization.num_params)
    posted_roles = []

    class Vector:
        def __getitem__(self, key): return self
        def to_list(self): return [0.0, 0.0, 0.0]

    evaluator = O.CandidateEvaluator.__new__(O.CandidateEvaluator)
    evaluator.task, evaluator.motion = task, motion
    evaluator.parameterization = parameterization
    evaluator.spec = O.ObjectiveSpec(reference_cycle_time_s=reference["reference_cycle_time_s"])
    evaluator.reference_cycle_time_s = reference["reference_cycle_time_s"]
    evaluator.efficiency_metadata = {"time_weight": 0.0, "air_time_weight": 0.0, **reference}
    evaluator.sim = type("Sim", (), {
        "tool_start": Vector(), "precision": "f64",
        "precision_metadata": _precision_metadata})()
    evaluator.ledger = R.ComputeLedger()
    candidate = C.CandidateResult(candidate_id="campaign-role-unit", theta_hash=C.content_hash([0.0]),
                                  objective_kind="hard", objective=0.0, valid=True)
    outcome = runner.Outcome(
        seed=0, best=candidate, best_theta=theta, initial_theta=theta,
        budget=runner.B.OptimizationBudget(candidate_evaluations=1, population_size=None),
        evaluations=1, validation_calls=0, generations=0, forward_calls=1, backward_calls=0,
        invalid_candidates=0, termination_reason="campaign_role_unit", history=[])

    def posted_factory(actual_task, actual_config):
        poster = object.__new__(runner.PilotPostedEvaluator)
        poster.timings, poster.phase_records = [], []

        def evaluate_posting(task_arg, posting, workspace, *, role="screen", single_dialect=False):
            posted_roles.append((role, Path(workspace).name))
            return {post.dialect: _typed_evaluation(
                task_arg, posting.files[C.DIALECT_ARTIFACTS[post.dialect][0]], dialect=post.dialect)
                for post in posting.posted}

        poster.evaluate_posting = evaluate_posting
        return poster

    record = runner._write_outcome(
        {"run_id": "campaign-fine-unit", "method": "nominal-cam", "seed": 0},
        evaluator, outcome, posted_factory, config, tmp_path, runner.REPOSITORY)
    assert record.status == C.RunStatus.COMPLETE
    assert posted_roles == [("verdict", "primary")]
    for dialect in C.Dialect:
        evaluation = C.loads((tmp_path / record.run_id / C.DIALECT_ARTIFACTS[dialect][1]).read_text())
        assert evaluation.discretization["settings"]["max_step_mm"] == 0.125
    config["refinement_check"]["enabled"] = True
    with pytest.raises(runner.RunnerError, match="disable auxiliary"):
        runner._campaign_verdict_only(config)
    config["campaign_verdict_only"] = False
    assert runner._campaign_verdict_only(config) is False


def _spawn_fail_without_artifacts(state):
    def spawn(command, cwd, env, stdout, stderr):
        with state["lock"]:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["commands"].append(command)

        class Process:
            pid = 900001

            def wait(self, timeout=None):
                time.sleep(0.03)
                with state["lock"]:
                    state["active"] -= 1
                return 1

            def terminate(self): return None
            def kill(self): return None

        return Process()
    return spawn


def test_concurrency_cap_and_failed_records_preserved_on_resume(tmp_path):
    methods = ("nominal-cam", "fixed-adam", "annealed-adam", "hard-cmaes", "smooth-cmaes")
    document = _prototype_campaign(methods=methods, concurrency=2)
    del document["cost_model"]  # Scheduling/failure fixture, not an init-only CMA cost-model case.
    path = _write_config(tmp_path, document)
    state = {"active": 0, "peak": 0, "commands": [], "lock": threading.Lock()}
    root = tmp_path / "campaign"
    result = campaign.run_campaign(path, root, spawn=_spawn_fail_without_artifacts(state))
    assert result["run_count"] == 5 and 1 < state["peak"] <= 2
    rows = json.loads((root / "manifest.json").read_text())["expected_runs"]
    assert rows and all(row["status"] == "interrupted" for row in rows)
    for row in rows:
        record = C.loads((root / "tasks" / row["task_id"] / row["run_id"] / "run.json").read_text())
        assert record.status == C.RunStatus.INTERRUPTED and record.failure_reason
    assert json.loads((root / "audit.json").read_text())["reconciliation"]["exact_id_reconciliation"]

    state["commands"].clear()
    rerun = campaign.run_campaign(path, root, resume=True,
                                   spawn=_spawn_fail_without_artifacts(state))
    assert rerun["run_count"] == 5 and len(state["commands"]) == 5
    task_root = next((root / "tasks").iterdir())
    assert list((task_root / "resume-backups").iterdir())  # Every failed attempt is retained.
    assert len(list((task_root / "logs").iterdir())) == 10


def _prototype_child_code():
    # The child, not just a patched parent, replaces only its posted factory.
    # Optimizer, parameterization, TaskSpec and artifact writing all stay real.
    return """import sys
from benchmarks import contracts as C
from scripts import run_benchmark_suite as runner
""" + inspect.getsource(_typed_evaluation) + """
class UnitPosted:
    def evaluate_posted_gcode(self, task, nc_bytes, dialect, settings):
        return _typed_evaluation(task, nc_bytes, dialect=dialect, step=0.125)
original_context = runner._context
def unit_context(config, repository):
    task, motion, parameterization, evaluator_factory, _ = original_context(config, repository)
    return task, motion, parameterization, evaluator_factory, lambda task, config: UnitPosted()
runner._context = unit_context
raise SystemExit(runner.main(sys.argv[1:]))
"""


def test_one_task_two_conditions_end_to_end_resume_skip_and_audit_projection(tmp_path):
    path = _write_config(tmp_path, _prototype_campaign(concurrency=2))
    def spawn(command, **kwargs):
        modified = list(command)
        modified[1:2] = ["-c", _prototype_child_code()]
        return subprocess.Popen(modified, **kwargs)

    root = tmp_path / "campaign"
    planned = campaign.run_campaign(path, root, dry_run=True)
    assert planned["run_count"] == 2 and planned["launched"] is False
    result = campaign.run_campaign(path, root, resume=True, spawn=spawn)
    assert result["status_counts"] == {"complete": 2}
    assert result["audit_summary"]["all_runs_completed"] is True
    assert result["audit_summary"]["auditor_passed"] is True
    manifest = audit.load_manifest(root / "manifest.json")
    reports, checked = audit.audit_runs(root / "audit-runs", manifest)
    assert len(reports) == 2 and checked["summary"]["all_runs_completed"]
    for row in reports:
        actual = root / "tasks" / manifest.runs[row["run_id"]].task_id / row["run_id"] / "run.json"
        projected = root / "audit-runs" / row["run_id"] / "run.json"
        assert actual.samefile(projected)
    record_hashes = {path: path.read_bytes() for path in root.glob("tasks/*/*/run.json")}

    def must_not_spawn(command, **kwargs):
        raise AssertionError("complete audited runs must be skipped on resume")

    again = campaign.run_campaign(path, root, resume=True, spawn=must_not_spawn)
    assert again["status_counts"] == {"complete": 2}
    assert all(row["resumed_skipped"] for row in json.loads(
        (root / "manifest.json").read_text())["expected_runs"])
    assert all(path.read_bytes() == payload for path, payload in record_hashes.items())
    with pytest.raises(campaign.CampaignError, match="cannot be reset"):
        campaign.run_campaign(path, root, resume=True, dry_run=True)


def test_prototype_dry_run_reports_declared_workload_not_execution(tmp_path):
    path = _write_config(tmp_path, _prototype_campaign())
    result = campaign.run_campaign(path, tmp_path / "dry", concurrency=1, dry_run=True)
    assert result["run_count"] == 2 and result["launched"] is False
    assert result["cost_estimate"]["estimated_wall_seconds"] == 34.0
    assert not list((tmp_path / "dry").glob("tasks/*/*/run.json"))


def _exp1_prototype_config():
    document = json.loads((ROOT / "configs/campaign_exp1_dev.json").read_text())
    document.update(task_source="prototype_analytic", prototype_unit_test=True)
    document["runner_config"]["task_source"] = "prototype_analytic"
    document["runner_config"]["campaign_verdict_only"] = False
    document.pop("task_manifest")
    return document


def test_experiment1_selector_excludes_plain_cam_sidecars_and_cmaes(tmp_path):
    document = _exp1_prototype_config()
    document["initialization"]["exploratory_cam_perturbation"] = [{
        "variant_suffix": "cam_perturbed_test",
        "parent": {"condition": "annealed-adam", "init": "cam", "seed": 0},
        "cam_perturbed_init": {
            "parameter_space": "theta",
            "distribution": {"kind": "gaussian", "detail": "iid"},
            "scale": {"fraction": 0.0001, "reference": {
                "kind": "cam_delta_norm", "description": "unit-only sidecar exclusion",
                "units": "mm"}},
            "seed_derivation_rule": {
                "algorithm": "sha256_v1", "root_seed": 101,
                "input": "task_hash,declaration_hash,root_seed,seed",
                "per_seed_rule": "first_8_bytes_little_endian_uint64"},
            "exploratory": True, "excluded_from_init_axis_claims": True,
        },
    }]
    path = _write_config(tmp_path, document)
    prepared = campaign._prepare(campaign._load(path), tmp_path / "exp1", resume=False, dry_run=True)
    assert len(prepared["jobs"]) == 8
    footprints = {(job["entry"]["method"], job["entry"]["init"], job["entry"]["seed"])
                  for job in prepared["jobs"]}
    methods = ("annealed-adam", "continuation-adam")
    assert footprints == {
        (method, init, seed) for method in methods for init, seed in (
            ("experiment1_cam", 0), ("straight_line", 0), ("random", 0), ("random", 1))
    }
    assert all(job["entry"].get("scope") != "exploratory_cam_perturbation"
               for job in prepared["jobs"])
    assert prepared["plan_core"]["experiment"] == "experiment1"
    assert prepared["plan_core"]["comparisons"] == []
    assert all(job["entry"]["run_id"].endswith("-" + job["entry"]["init"])
               for job in prepared["jobs"])
    raw_entries = runner._plan_entries(
        prepared["jobs"][0]["config"], prepared["jobs"][0]["task"], "dev", None)
    assert any(entry["init"] == "cam" for entry in raw_entries)
    assert any(entry["scope"] == "exploratory_cam_perturbation" for entry in raw_entries)
    assert all(job["entry"]["config_override"] == {} for job in prepared["jobs"])
    invalid = _exp1_prototype_config()
    invalid["conditions"].append({"condition": "hard-cmaes", "seeds": [0]})
    with pytest.raises(campaign.CampaignError, match="experiment1 allows only"):
        campaign._load(_write_config(tmp_path, invalid))


def test_experiment1_dev_dry_run_has_72_runs_and_exp2_still_has_312(tmp_path):
    exp1 = campaign._prepare(campaign._load(ROOT / "configs/campaign_exp1_dev.json"),
                             tmp_path / "exp1", resume=False, dry_run=True)
    assert len(exp1["tasks"]) == 9 and len(exp1["jobs"]) == 72
    assert {row["task_split"] for row in exp1["tasks"]} == {"dev"}
    per_task = {}
    for job in exp1["jobs"]:
        task_id = job["entry"]["task_id"]
        per_task.setdefault(task_id, []).append(job["entry"])
        assert job["entry"]["init"] in ("experiment1_cam", "straight_line", "random")
        assert job["entry"]["method"] in ("annealed-adam", "continuation-adam")
        assert job["entry"]["run_id"].endswith("-" + job["entry"]["init"])
    assert len(per_task) == 9
    for entries in per_task.values():
        assert len(entries) == 8
        for method in ("annealed-adam", "continuation-adam"):
            counter = {(entry["init"], entry["seed"]) for entry in entries
                       if entry["method"] == method}
            assert counter == {("experiment1_cam", 0), ("straight_line", 0),
                               ("random", 0), ("random", 1)}
    exp2 = campaign._prepare(campaign._load(ROOT / "configs/campaign_suite.json"),
                             tmp_path / "exp2", resume=False, dry_run=True)
    assert len(exp2["jobs"]) == 312 and len(exp2["tasks"]) == 39
    continuations = [job for job in exp2["jobs"] if job["entry"]["method"] == "continuation-adam"]
    assert len(continuations) == 39
    assert len({job["entry"]["task_id"] for job in continuations}) == 39
    assert all(job["entry"]["init"] == "cam" and job["entry"]["seed"] == 0
               for job in continuations)
    assert all(job["config"]["time_weight"] == 0.01
               and job["config"]["air_time_weight"] == 0.01 for job in exp2["jobs"])
    assert "experiment" not in exp2["plan_core"]


def test_experiment1_all_real_dev_initializer_slots_reconstruct_in_shared_space():
    """Production seed construction only: no optimizer, physics or posted score claim."""
    loaded = campaign._load(ROOT / "configs/campaign_exp1_dev.json")
    assert len(campaign._task_rows(loaded)) == 9
    checked = 0
    for task_id in campaign._task_rows(loaded):
        config = dict(loaded["template"], task_id=task_id)
        task, cam_motion, cam_params, *_ = runner._context(config, ROOT)
        arms = {}
        hashes = set()
        for method in ("annealed-adam", "continuation-adam"):
            for init, seed in (("experiment1_cam", 0), ("straight_line", 0),
                               ("random", 0), ("random", 1)):
                entry = {"method": method, "init": init, "seed": seed, "scope": "optimizer"}
                motion, params, seed_record = runner._initializer_motion(
                    config, task, cam_motion, cam_params, entry)
                assert motion is not None and params is not None
                assert seed_record["experiment1"] == params.experiment1_metadata
                raw_positions = np.asarray([motion.initial_state.position_mm]
                                           + [segment.end_mm for segment in motion.segments])
                work_positions = np.asarray([
                    task.work_to_machine.inverse().apply(point) for point in raw_positions
                ])
                meta = params.experiment1_metadata
                theta = np.asarray(meta["initial_theta"], dtype=np.float64)
                steps = len(motion.segments)
                assert params.num_control_points == params.num_steps == steps
                assert params.num_params == steps * 3
                assert params.scale_mm == 1.0 and params.bounds is None
                assert params.anchor_end is False
                assert raw_positions.shape == (steps + 1, 3)
                np.testing.assert_allclose(theta.reshape(steps, 3),
                                           np.diff(raw_positions, axis=0) / params.scale_mm,
                                           rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(params.positions(theta), work_positions,
                                           rtol=0.0, atol=1e-10)
                np.testing.assert_allclose(params.positions(np.zeros(params.num_params)),
                                           np.repeat(work_positions[:1], steps + 1, axis=0),
                                           rtol=0.0, atol=1e-12)
                assert meta["initial_theta_sha256"] == C.content_hash(theta.tolist())
                assert meta["reference_start_mm_work"] == work_positions[0].tolist()
                assert meta["source_motion_hash"] == motion.motion_hash
                assert len(meta["feed_and_kind_template"]) == steps
                for segment, item in zip(motion.segments, meta["feed_and_kind_template"]):
                    assert item == {"id": segment.id, "kind": segment.kind.value,
                                    "feed_mm_s": segment.feed_mm_s, "tag": segment.tag,
                                    "source_line": segment.source_line}
                common_hash = C.content_hash(R.parameterization_metadata(params))
                assert meta["parameterization_sha256"] == common_hash
                hashes.add(common_hash)
                if method == "annealed-adam":
                    arms[(init, seed)] = (motion.motion_hash, meta["initial_theta_sha256"])
                else:
                    assert arms[(init, seed)] == (motion.motion_hash, meta["initial_theta_sha256"])
                checked += 1
        assert len(hashes) == 1
        assert set(arms) == {("experiment1_cam", 0), ("straight_line", 0),
                             ("random", 0), ("random", 1)}
        assert len(set(arms.values())) == 4
    assert checked == 72
