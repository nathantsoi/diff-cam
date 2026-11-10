"""T4.2a frozen-protocol tooling tests.

Real prototype dry-run invocations, but structural synthetic configuration only.
Coordinator-owned frozen values are not chosen here.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V = load_module("t42_validator", "scripts/validate_frozen_protocol.py")
R = load_module("t42_runner", "scripts/run_benchmark_suite.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_case(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    task, motion, parameterization = R._prototype()
    manifest = repo / "task-manifest.json"
    write(manifest, {
        "schema_version": 1, "families": ["pocket_island"], "generation_version": "t42-synthetic/1",
        "tasks": [{"task_id": task.task_id, "task_hash": task.task_hash,
                   "family": task.family, "task_split": task.split.value,
                   "task_record": task.to_dict(), "motion_hash": motion.motion_hash}],
    })
    registry, splits = V.task_registry(read(manifest), repo)
    decision_records = tmp_path / "decisions.md"
    shutil.copyfile(ROOT / "reports/decisions.md", decision_records)
    tuning_file = repo / "dev/tuning-history.json"
    write(tuning_file, {"recorded_for": "synthetic protocol test only", "round": 1})
    return {
        "schema": "diffcam-frozen-benchmark/1",
        "decision_ids": ["D-2026-11-09-01", "D-2026-11-09-02", "D-2026-11-09-03",
                         "D-2026-11-09-04", "D-2026-11-09-05", "D-2026-11-09-06", "D-2026-11-09-07",
                         "D-2026-11-10-08"],
        "task_manifest": {"path": str(manifest.relative_to(repo)), "sha256": sha(manifest),
                          "split_hashes": splits,
                          "acquisition": "registered prototype task for synthetic validation only"},
        "primary_task_id": task.task_id,
        "precision": "f64",
        "dimensionality_subset": [
            {"name": "lower-k", "task_id": task.task_id, "condition": "fixed-adam",
             "seed": 0, "k": 20.0, "rationale": "one bounded synthetic fixed-K variation"},
        ],
        "methods": [
            {"condition": "nominal-cam", "seeds": [0]},
            {"condition": "fixed-adam", "seeds": [0]},
            {"condition": "annealed-adam", "seeds": [0]},
            {"condition": "hard-cmaes", "seeds": [0, 1]},
            {"condition": "smooth-cmaes", "seeds": [0, 1]},
        ],
        "objective": {
            "primary_k": 40.0, "anneal_start": 20.0, "anneal_end": 120.0,
            "gouge_weight": 2.0, "residual_weight": 1.0, "holder_weight": 50.0,
            "holder_margin_mm": 0.0, "loss_shift_voxels": 0.0,
        },
        "thresholds": dict(task.to_dict()["tolerances"]),
        "posted_evaluation": {
            "verdict_max_step_mm": 0.125, "screen_max_step_mm": 1.0,
            "convergence_max_step_rs274_mm": 0.25,
            "screen_is_diagnostic_only": True, "convergence_is_rs274_only": True,
            "record_memory_by_phase_and_dialect": True,
        },
        "memory": {"max_rss_bytes_per_job": 8589934592,
                   "parallel_jobs": 2, "stop_and_optimize_when_exceeded": True,
                   "vram_estimate_bytes_per_job": 0},
        "budget": {
            "candidate_evaluations": 4, "population_size": 2,
            "validation_interval": 2, "max_validation_calls": 2,
            "max_generations": None, "learning_rate": 0.005,
            "max_gradient_norm": 0.5, "sigma0": 0.01,
            "timeout_s": 30.0, "total_wall_time_s": 50.0,
        },
        "estimates": {
            "run_count": 8, "candidate_evaluations": 29, "validation_calls": 14,
            "seconds_per_candidate_by_method": {"nominal-cam": 0.5, "fixed-adam": 1.25,
                                               "annealed-adam": 0.5, "hard-cmaes": 0.5,
                                               "smooth-cmaes": 0.5},
            "worker_seconds": 30.5,
            "cpu_rss_bytes_per_job": 1073741824,
            "uncertainty": "synthetic arithmetic estimates only; no universal timings or guaranteed phase-memory bound",
        },
        "tuning": {
            "total_candidate_evaluations": 2,
            "only_dev": True,
            "rounds": [{"split": "dev", "candidate_evaluations": 2,
                        "conditions": ["fixed-adam"],
                        "changes": ["synthetic tuning-spend record; no real tuning was run"],
                        "evidence": [{"path": "dev/tuning-history.json",
                                     "sha256": sha(tuning_file), "split": "dev"}]}],
        },
        "comparisons": [
            {"name": "candidate_vs_nominal", "method": "smooth-cmaes", "baseline": "nominal-cam",
             "metrics": ["completion_pass", "cycle_time_s"], "dialects": ["rs274", "haas"]},
        ],
        "multiplicity": {
            "independent_unit": "task", "seed_aggregation": "within_task_before_pairing",
            "sign_flip_two_sided": True,
            "holm_scope": "all_comparisons_metrics_outcomes_and_families",
            "confidence": 0.95, "bootstrap_replicates": 2000,
            "minimum_ci_tasks": 2, "descriptive_below_tasks": 5,
        },
        "external_baselines": {
            "cnc_net_executable": {"status": "unsupported", "allow_success_claim": False},
            "camotics": {"status": "supplementary", "primary_oracle": False},
            "cem": {"status": "unavailable", "allow_optimizer_condition": False,
                    "claim_scope": "zero-order (CMA-ES), not CEM",
                    "reason": "no CEM implementation; all synthetic zero-order runs are hard CMA-ES"},
            "physical_claims": "omit when no physical trials are provided",
            "policy_claims": "omit when no trained-policy evidence is provided",
        },
        "acceptance": {
            "strict_improvement_required": False,
            "allowed_conclusions": ["accept", "reject", "narrow"],
        },
        "execution": {
            "task_source": "prototype_analytic", "experiment": "pilot", "split": "dev",
            "architecture": "cpu", "output_root": "out/frozen-probe",
            "prototype_task_id": task.task_id, "prototype_task_hash": task.task_hash,
        },
    }


def make_init_case(tmp_path):
    """Structural D-08 fixture; these toy values are not coordinator decisions."""
    case = make_case(tmp_path)
    case["methods"] = [
        {"condition": "nominal-cam", "seeds": [0]},
        {"condition": "fixed-adam", "seeds": [0]},
        {"condition": "annealed-adam", "init": "cam", "seeds": [0]},
        {"condition": "annealed-adam", "init": "straight_line", "seeds": [0]},
        {"condition": "annealed-adam", "init": "random", "seeds": [2, 7]},
        {"condition": "hard-cmaes", "init": "cam", "seeds": [0, 1]},
        {"condition": "hard-cmaes", "init": "straight_line", "seeds": [0]},
        {"condition": "hard-cmaes", "init": "random", "seeds": [0, 1]},
        {"condition": "smooth-cmaes", "seeds": [0, 1]},
    ]
    case["dimensionality_subset"].append({
        "name": "straight-line-lower-k", "task_id": case["primary_task_id"],
        "condition": "annealed-adam", "init": "straight_line", "seed": 0, "k": 25.0,
        "rationale": "one bounded initializer/parameterization declaration; not an executable runner ablation",
    })
    case["random_init"] = {
        "parameter_space": "delta",
        "distribution": {"kind": "gaussian", "detail": "independent zero-mean normal draws per coordinate"},
        "scale": {"fraction": 0.05, "reference": {
            "kind": "cam_delta_norm",
            "description": "L2 norm of each CAM delta under the selected parameterization",
            "units": "dimensionless fraction of that norm",
        }},
        "seed_derivation_rule": {
            "algorithm": "declared SeedSequence child streams",
            "root_seed": 17, "input": "root seed, condition, initializer and listed run seed",
            "per_seed_rule": "derive one independent child stream per explicitly listed run seed in declared order",
        },
    }
    case["straight_line_init"] = {
        "start": "CAM entry station after its safe rapid approach",
        "end": "CAM final cutting station before its terminal retract",
        "segment_placement": "uniform station resampling at the CAM segment count with unchanged K parameterization",
        "feed_and_rapid_entry": "safe clearance-Z rapid entry, then feed bounded by the selected TaskSpec; no efficiency promise",
        "same_segment_count_and_k": True, "safe_rapid_entry": True,
        "respect_task_feed_limits": True, "efficiency_not_required": True,
    }
    case["collapse_criterion"] = {
        "geometry_or_completion_failure": True,
        "residual_multiplier": 1.5, "residual_operator": "gt",
        "require_no_hard_loss_improvement_over_initial": True,
        "collapse_and_infeasibility_are_valid_recorded_outcomes": True,
        "predicate": "0.125 mm verdict fails geometry OR completion OR residual > 1.5 times CAM-initialization residual, AND hard loss does not improve over this run's initial path",
    }
    case["comparisons"].append({
        "name": "straight_line_annealed_vs_cam_annealed", "method": "annealed-adam",
        "baseline": "annealed-adam", "method_init": "straight_line", "baseline_init": "cam",
        "metrics": ["completion_pass"], "dialects": ["rs274", "haas"],
    })
    # Seven no-init declarations are added to the eight native CAM cases.
    case["estimates"].update({"run_count": 15, "candidate_evaluations": 57,
                              "validation_calls": 28, "worker_seconds": 51.5})
    case["budget"]["total_wall_time_s"] = 60.0
    return case


def run_case(case, tmp_path):
    config_path = tmp_path / "protocol.json"
    write(config_path, case)
    destination = tmp_path / "validation-output"
    return V.validate(case, repo_root=tmp_path / "repo",
                      decisions_path=tmp_path / "decisions.md", output_dir=destination,
                      protocol_sha256=sha(config_path))


def test_valid_schema_runs_every_primary_and_variant_case_with_exact_provenance(tmp_path):
    case = make_case(tmp_path)
    out = tmp_path / "validation-output"
    report = run_case(case, tmp_path)
    doc = read(out / "validation-report.json")
    assert doc["mode"] == "tooling_only_not_a_campaign_launch"
    assert doc["dry_run_performed"] is True
    assert doc["validated_run_count"] == 8
    assert doc["declared_not_yet_runnable_count"] == 0
    assert all(row["init"] == "cam" and row["native_dry_run_performed"] for row in report["dry_runs"])
    assert len(report["dry_runs"]) == 8
    assert doc["cost_accounting"]["candidate_evaluations"] == 29
    assert doc["cost_accounting"]["validation_calls"] == 14
    assert doc["cost_accounting"]["worker_seconds"] == 30.5
    assert doc["prototype_adapter_compatible"] is True
    nominal = next(row for row in report["dry_runs"] if row["condition"] == "nominal-cam")
    assert nominal["candidate_evaluations"] == 1
    variant = next(row for row in report["dry_runs"] if row["scope"] == "dimensionality_subset")
    assert variant["variant"] == "lower-k"
    fixed = next(row for row in report["dry_runs"] if row["condition"] == "fixed-adam" and row["variant"] is None)
    assert fixed["candidate_evaluations"] == 4 and fixed["validation_calls"] == 2
    assert fixed["estimated_wall_time_s"] == 7.5
    assert variant["run_id"] != fixed["run_id"]
    assert doc["protocol_sha256"] == sha(tmp_path / "protocol.json")
    assert doc["primary_task_hash"] == R._prototype()[0].task_hash
    assert doc["split_hashes"]["dev"] == V.split_hash(
        [{"task_id": R._prototype()[0].task_id, "task_hash": R._prototype()[0].task_hash}])
    assert not (out / "plan" / "manifest.json").exists()
    with (ROOT / "configs/frozen_benchmark.schema.json").open() as stream:
        json.load(stream)


def test_d08_init_axis_has_explicit_previews_not_cam_runner_evidence(tmp_path):
    report = run_case(make_init_case(tmp_path), tmp_path)
    out = tmp_path / "validation-output"
    assert report["dry_run_performed"] is True
    assert report["validated_run_count"] == 8
    assert report["declared_not_yet_runnable_count"] == 7
    assert len(report["dry_runs"]) == 15
    assert report["cost_accounting"]["jobs"] == 15
    assert report["cost_accounting"]["candidate_evaluations"] == 57
    assert report["cost_accounting"]["validation_calls"] == 28
    assert report["cost_accounting"]["worker_seconds"] == 51.5
    pending = [row for row in report["dry_runs"] if row["declared_not_yet_runnable"]]
    assert len(pending) == 7
    assert {row["seed"] for row in pending
            if row["condition"] == "annealed-adam" and row["init"] == "random"} == {2, 7}
    for row in pending:
        assert row["native_dry_run_performed"] is False
        assert row["run_id"] is None and row["candidate_evaluations"] is None
        assert row["validation_calls"] is None and row["planned_run_count"] == 0
        root = out / row["output_subdirectory"]
        assert not (root / "plan").exists()
        assert not (root / "runner-config.json").exists()
        preview = read(root / "initialization-declaration.json")
        assert preview["declared_not_yet_runnable"] is True
        assert preview["schema"] == "diffcam-initialization-declaration/1"
        assert preview["random_init"] and preview["straight_line_init"]
        assert preview["collapse_criterion"]["require_no_hard_loss_improvement_over_initial"] is True
        assert row["initialization_declaration_sha256"] == sha(root / "initialization-declaration.json")
    cam = next(row for row in report["dry_runs"] if not row["declared_not_yet_runnable"])
    config = read(out / cam["output_subdirectory"] / "runner-config.json")
    assert config["init"] == "cam" and config["init_implementation_available"] is True
    assert config["precision"] == "f64" and config["simulator_precision"] == "f64"
    assert (out / cam["output_subdirectory"] / "plan/manifest.json").is_file()


@pytest.mark.parametrize("key", ["random_init", "straight_line_init", "collapse_criterion"])
def test_d08_missing_initializer_declarations_fail_closed(tmp_path, key):
    case = make_init_case(tmp_path)
    del case[key]
    with pytest.raises(V.ProtocolError, match=key + ".*nonempty predeclared object"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case[key] = {}
    with pytest.raises(V.ProtocolError, match=key + ".*nonempty predeclared object"):
        run_case(case, tmp_path)


def test_d08_init_solver_and_deterministic_run_rules(tmp_path):
    case = make_init_case(tmp_path)
    case["methods"][0]["init"] = "random"
    with pytest.raises(V.ProtocolError, match="nominal-cam requires init=cam"):
        run_case(case, tmp_path)
    for condition in ("fixed-adam", "smooth-cmaes"):
        case = make_init_case(tmp_path)
        case["methods"].append({"condition": condition, "init": "random", "seeds": [0]})
        with pytest.raises(V.ProtocolError, match="only with annealed-adam or hard-cmaes"):
            run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    row = next(row for row in case["methods"]
               if row["condition"] == "hard-cmaes" and row["init"] == "straight_line")
    row["seeds"] = [0, 1]
    with pytest.raises(V.ProtocolError, match="deterministic.*exactly one"):
        run_case(case, tmp_path)


def test_d08_placeholder_scale_and_collapse_rules_are_enforced(tmp_path):
    case = make_init_case(tmp_path)
    case["random_init"]["distribution"]["detail"] = "TO-BE-FROZEN"
    with pytest.raises(V.ProtocolError, match="unresolved placeholder"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case["straight_line_init"]["segment_placement"] = "   "
    with pytest.raises(V.ProtocolError, match="segment_placement.*nonempty"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case["random_init"]["scale"]["fraction"] = 0.0
    with pytest.raises(V.ProtocolError, match="scale.fraction.*> 0"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case["random_init"]["parameter_space"] = "unspecified"
    with pytest.raises(V.ProtocolError, match="parameter_space must be theta or delta"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case["collapse_criterion"]["residual_multiplier"] = 0.0
    with pytest.raises(V.ProtocolError, match="residual_multiplier.*> 0"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case["random_init"]["seed_derivation_rule"]["root_seed"] = -1
    with pytest.raises(V.ProtocolError, match="root_seed.*integer >= 0"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case["collapse_criterion"]["collapse_and_infeasibility_are_valid_recorded_outcomes"] = False
    with pytest.raises(V.ProtocolError, match="valid_recorded_outcomes.*true"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case["random_init"]["seed_derivation_rule"]["per_seed_rule"] = ""
    with pytest.raises(V.ProtocolError, match="per_seed_rule.*nonempty"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case["collapse_criterion"]["residual_multiplier"] = None
    with pytest.raises(V.ProtocolError, match="residual_multiplier.*finite number"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case["collapse_criterion"]["require_no_hard_loss_improvement_over_initial"] = False
    with pytest.raises(V.ProtocolError, match="require_no_hard_loss_improvement_over_initial.*true"):
        run_case(case, tmp_path)


def test_d08_init_specific_comparisons_tuning_and_unavailable_cem(tmp_path):
    case = make_init_case(tmp_path)
    case["comparisons"][1]["baseline_init"] = "random"
    case["comparisons"][1]["baseline"] = "fixed-adam"
    with pytest.raises(V.ProtocolError, match="baseline selection.*not a retained"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case["tuning"]["rounds"][0].update({"conditions": ["hard-cmaes"], "init": "cam"})
    declaration = V.validate_declaration(case, tmp_path / "repo", tmp_path / "decisions.md")
    assert declaration["tuning_rounds"][0]["init"] == "cam"
    case["tuning"]["rounds"][0].update({"conditions": ["fixed-adam"], "init": "random"})
    with pytest.raises(V.ProtocolError, match="retained primary.*condition, init"):
        V.validate_declaration(case, tmp_path / "repo", tmp_path / "decisions.md")
    case = make_init_case(tmp_path)
    case["methods"].append({"condition": "cem", "seeds": [0]})
    with pytest.raises(V.ProtocolError, match="not supported by the runner"):
        run_case(case, tmp_path)
    case = make_init_case(tmp_path)
    case["external_baselines"]["cem"]["status"] = "implemented"
    with pytest.raises(V.ProtocolError, match="CEM as an unavailable baseline"):
        run_case(case, tmp_path)



def test_validate_all_root_fields_and_nested_additional_property_blocking(tmp_path):
    case = make_case(tmp_path)
    case["unexpected_protocol_knob"] = 1
    with pytest.raises(V.ProtocolError, match="unknown field"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["objective"]["extra_knob"] = 1
    with pytest.raises(V.ProtocolError, match="unknown field"):
        run_case(case, tmp_path)


def test_reject_f32_and_wrong_fidelity_or_memory_declarations(tmp_path):
    case = make_case(tmp_path)
    case["precision"] = "f32"
    with pytest.raises(V.ProtocolError, match="D-05"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["posted_evaluation"]["verdict_max_step_mm"] = 1.0
    with pytest.raises(V.ProtocolError, match="0.125"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["memory"]["max_rss_bytes_per_job"] += 1
    with pytest.raises(V.ProtocolError, match="D-07"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["memory"]["parallel_jobs"] = 3
    with pytest.raises(V.ProtocolError, match="parallel jobs"):
        run_case(case, tmp_path)


def test_reject_mismatched_primary_task_and_thresholds(tmp_path):
    case = make_case(tmp_path)
    case["task_manifest"]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(V.ProtocolError, match="declared hash"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["primary_task_id"] = "missing-task"
    with pytest.raises(V.ProtocolError, match="primary_task_id"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["thresholds"]["completion_residual_mm3"] += 1.0
    with pytest.raises(V.ProtocolError, match="task tolerance"):
        run_case(case, tmp_path)


def test_split_tuning_reference_is_rejected_before_becoming_evidence(tmp_path):
    case = make_case(tmp_path)
    case["tuning"]["rounds"][0]["evidence"][0]["path"] = "external/tuned-result.json"
    with pytest.raises(V.ProtocolError, match="prohibited test/external tuning data"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["tuning"]["rounds"][0]["split"] = "test"
    with pytest.raises(V.ProtocolError, match="split must be dev"):
        run_case(case, tmp_path)


def test_reject_strict_improvement_gate_and_ranked_acceptance_only(tmp_path):
    case = make_case(tmp_path)
    case["acceptance"]["strict_improvement_required"] = True
    with pytest.raises(V.ProtocolError, match="strict-improvement"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["acceptance"]["allowed_conclusions"] = ["accept"]
    with pytest.raises(V.ProtocolError, match="accept, reject, and narrow"):
        run_case(case, tmp_path)


def test_condition_seed_and_variant_rules_do_not_create_fake_replicates(tmp_path):
    case = make_case(tmp_path)
    case["methods"][0]["seeds"] = [0, 1]
    with pytest.raises(V.ProtocolError, match="deterministic seed 0"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["dimensionality_subset"][0]["k"] = case["objective"]["primary_k"]
    with pytest.raises(V.ProtocolError, match="not a distinct"):
        run_case(case, tmp_path)


def test_estimate_reconciliation_rejects_bad_counts_and_incomplete_timing(tmp_path):
    case = make_case(tmp_path)
    case["estimates"]["candidate_evaluations"] += 1
    with pytest.raises(V.ProtocolError, match="candidate_evaluations"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["estimates"]["worker_seconds"] = 1.0
    with pytest.raises(V.ProtocolError, match="worker_seconds"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["tuning"]["total_candidate_evaluations"] += 1
    with pytest.raises(V.ProtocolError, match="sum of declared rounds"):
        run_case(case, tmp_path)


def test_execution_mode_rejections_do_not_write_output_or_launch_wrong_experiment(tmp_path):
    case = make_case(tmp_path)
    case["execution"]["experiment"] = "all"
    with pytest.raises(V.ProtocolError, match="implements only pilot"):
        run_case(case, tmp_path)
    assert not (tmp_path / "validation-output").exists()
    case = make_case(tmp_path)
    case["execution"]["split"] = "test"
    with pytest.raises(V.ProtocolError, match="split"):
        run_case(case, tmp_path)


def test_non_prototype_registry_can_validate_without_substitution(tmp_path):
    import dataclasses
    case = make_case(tmp_path)
    task = R._prototype()[0]
    changed = dataclasses.replace(task, task_id="other-registered-task")
    manifest = (tmp_path / "repo") / case["task_manifest"]["path"]
    raw = read(manifest)
    raw["tasks"][0]["task_id"] = changed.task_id
    raw["tasks"][0]["task_hash"] = changed.task_hash
    raw["tasks"][0]["task_record"] = changed.to_dict()
    raw["tasks"][0]["motion_hash"] = R._prototype()[1].motion_hash
    write(manifest, raw)
    case["task_manifest"]["sha256"] = sha(manifest)
    case["primary_task_id"] = changed.task_id
    case["dimensionality_subset"][0]["task_id"] = changed.task_id
    # Split hashes are computed from typed task identities, not a task-file refresh URL.
    _, split_hashes = V.task_registry(read(manifest), tmp_path / "repo")
    case["task_manifest"]["split_hashes"] = split_hashes
    with pytest.raises(V.ProtocolError, match="validate-only|task-source adapter"):
        run_case(case, tmp_path)
    config_path = tmp_path / "protocol.json"
    write(config_path, case)
    result = V.validate(case, repo_root=tmp_path / "repo", decisions_path=tmp_path / "decisions.md",
                        dry_run=False, protocol_sha256=sha(config_path))
    assert result["prototype_adapter_compatible"] is False
    assert result["dry_run_performed"] is False


def test_real_cli_and_unsafe_destination_rules(tmp_path):
    case = make_case(tmp_path)
    path = tmp_path / "protocol.json"
    write(path, case)
    out = tmp_path / "cli-out"
    run = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/validate_frozen_protocol.py"),
         "--config", str(path), "--output-dir", str(out), "--repo-root", str(tmp_path / "repo"),
         "--decisions", str(tmp_path / "decisions.md")],
        cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout[run.stdout.index("{"):])["status"] == "tooling_only_not_a_campaign_launch"
    repeated = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_frozen_protocol.py"),
         "--config", str(path), "--output-dir", str(out), "--repo-root", str(tmp_path / "repo"),
         "--decisions", str(tmp_path / "decisions.md")],
        cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    assert repeated.returncode == 2
    assert "fresh directory" in repeated.stderr
    manuscript = tmp_path / "manuscript" / "not-allowed"
    with pytest.raises(V.ProtocolError, match="manuscript"):
        V.execute_dry_runs(V.validate_declaration(case, tmp_path / "repo", tmp_path / "decisions.md"),
                           manuscript, "sha256:" + "0" * 64)


def test_cli_validate_only_never_creates_output_directory(tmp_path):
    case = make_case(tmp_path)
    config = tmp_path / "protocol.json"
    write(config, case)
    out = tmp_path / "validation-no-output"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_frozen_protocol.py"),
         "--config", str(config), "--output-dir", str(out),
         "--repo-root", str(tmp_path / "repo"), "--decisions", str(tmp_path / "decisions.md"),
         "--validate-only"], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not out.exists()


def test_uncertainty_and_negative_counts_fail_closed(tmp_path):
    case = make_case(tmp_path)
    case["budget"]["candidate_evaluations"] = 0
    with pytest.raises(V.ProtocolError, match="candidate_evaluations"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["estimates"]["worker_seconds"] = None
    del case["estimates"]["seconds_per_candidate_by_method"]["nominal-cam"]
    # The update is valid but leaves the aggregate unknown; another method still
    # produces a valid largest-workload bound and a listed unknown job.
    report = run_case(case, tmp_path)
    assert report["cost_accounting"]["worker_seconds"] is None
    assert report["cost_accounting"]["timing_unknown_jobs"]


def test_memory_estimate_overrun_and_split_hash_tamper_fail_closed(tmp_path):
    case = make_case(tmp_path)
    case["estimates"]["cpu_rss_bytes_per_job"] = case["memory"]["max_rss_bytes_per_job"] + 1
    with pytest.raises(V.ProtocolError, match="memory optimization"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["task_manifest"]["split_hashes"]["dev"] = "sha256:" + "0" * 64
    with pytest.raises(V.ProtocolError, match="typed task selection"):
        run_case(case, tmp_path)


def test_validate_only_accepts_future_experiment_metadata_without_execution(tmp_path):
    case = make_case(tmp_path)
    case["execution"]["experiment"] = "all"
    config_path = tmp_path / "protocol.json"
    write(config_path, case)
    report = V.validate(case, repo_root=tmp_path / "repo",
                        decisions_path=tmp_path / "decisions.md", dry_run=False,
                        protocol_sha256=sha(config_path))
    assert "all" in V.runner.EXPERIMENTS
    assert report["dry_run_performed"] is False
    assert report["prototype_adapter_compatible"] is False


def test_anneal_interval_and_evidence_file_constraints_are_checked(tmp_path):
    case = make_case(tmp_path)
    case["objective"]["anneal_end"] = case["objective"]["anneal_start"]
    with pytest.raises(V.ProtocolError, match="anneal_end must exceed"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    evidence = tmp_path / "repo/dev/tuning-history.json"
    assert evidence.is_file()
    evidence.unlink()  # Exact created fixture artifact, not shared worktree data.
def test_protocol_output_root_and_notes_are_validated(tmp_path):
    case = make_case(tmp_path)
    case["execution"]["output_root"] = "../escape"
    with pytest.raises(V.ProtocolError, match="absolute/traversal path"):
        run_case(case, tmp_path)
    case = make_case(tmp_path)
    case["notes"] = "not-a-list-of-strings"
    with pytest.raises(V.ProtocolError, match="list of strings"):
        run_case(case, tmp_path)


def test_missing_tuning_evidence_file_fails_closed(tmp_path):
    case = make_case(tmp_path)
    evidence = tmp_path / "repo/dev/tuning-history.json"
    assert evidence.is_file()
    evidence.unlink()  # Exact fixture file created by make_case; safe to remove once.
    with pytest.raises(V.ProtocolError, match="required evidence file"):
        run_case(case, tmp_path)
