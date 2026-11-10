"""T7.2b real-results package tests.

All task outcomes and gradient numbers in tiny_case are explicitly synthetic
input fixtures. They test audited data plumbing and are not simulation results.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
import shutil
import pytest

from benchmarks import audit as A
from benchmarks import contracts as C


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B = load_module("results_builder", ROOT / "scripts/build_paper_artifacts.py")
FIXTURES = load_module("results_audit_fixtures", ROOT / "tests/eval_aggregator_test.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path):
    return list(csv.DictReader(Path(path).read_text().splitlines()))


import csv


def complete_dialect(run_dir, record, evaluation, nc_bytes):
    settings = {
        "max_step_mm": 0.125, "expected_nc_hash": evaluation.nc_hash,
        "exclusion_tolerance_mm3": 0.0, "compute_bounds": True, "safety_screen": True,
        "max_samples_per_segment": 4096, "subdivisions": None, "record_stations": False,
    }
    new_eval = dataclasses.replace(
        evaluation, discretization={"settings": settings, "segments": []})
    (run_dir / "eval_posted_rs274.json").write_text(C.dumps(new_eval))
    haas = dataclasses.replace(new_eval, dialect=C.Dialect.HAAS)
    (run_dir / "eval_posted_haas.json").write_text(C.dumps(haas))
    (run_dir / "haas.nc").write_bytes(nc_bytes)
    posted = tuple(dataclasses.replace(item, status=C.PostStatus.SUPPORTED,
                                       nc_hash=haas.nc_hash, eval_file="eval_posted_haas.json",
                                       reason=None)
                   if item.dialect is C.Dialect.HAAS else item for item in record.posted)
    record = dataclasses.replace(record, posted=posted,
                                 artifacts=record.artifacts + ("haas.nc", "eval_posted_haas.json"))
    (run_dir / "run.json").write_text(C.dumps(record))
    write_json(run_dir / "eval_posted.json", {
        "schema_version": 1, "programs": {
            "rs274": {"dialect": "rs274", "status": "supported",
                      "nc_hash": haas.nc_hash, "eval_file": "eval_posted_rs274.json"},
            "haas": {"dialect": "haas", "status": "supported",
                     "nc_hash": haas.nc_hash, "eval_file": "eval_posted_haas.json"},
        }})


def tiny_case(tmp_path):
    task = FIXTURES._task("task-a", "pocket")
    root = tmp_path / "campaign"
    cases = [
        ("cam-a0", "nominal-cam", 0, 6.0, 0.0, 0.0, True, "cam", "primary", None),
        ("cam-a1", "nominal-cam", 1, 8.0, 0.0, 0.0, True, "cam", "primary", None),
        ("opt-a0", "annealed-adam", 0, 4.0, 0.0, 0.0, True, "cam", "primary", 0.645),
        ("opt-a1", "annealed-adam", 1, 5.0, 0.5, 0.5, True, "cam", "primary", 0.7),
        ("line-a0", "annealed-adam", 0, 2.0, 0.7, 0.8, False, "straight_line", "optimizer", 0.5),
        ("random-a0", "annealed-adam", 0, 3.0, 0.7, 0.6, False, "random", "optimizer", 0.5),
        ("perturb-a0", "annealed-adam", 0, 7.0, 0.5, 0.4, True, "cam", "exploratory_cam_perturbation", 0.5),
    ]
    expected = []
    for run_id, method, seed, cycle, hard0, hard1, geometry, init, scope, soft in cases:
        run_dir, record, evaluation = FIXTURES._write_run(
            root, run_id, task, method=method, seed=seed, cycle=cycle, geometry=geometry)
        complete_dialect(run_dir, record, evaluation, b"G1 X1 F10\n")
        initial = {"candidate_id": run_id + ":init", "theta_hash": C.content_hash([0.0]),
                   "objective": hard0, "objective_kind": "hard", "valid": True,
                   "components": {"feasible": float(run_id != "random-a0")}}
        initialization = {"init": init, "scope": scope,
                          "exploratory": scope == "exploratory_cam_perturbation",
                          "excluded_from_init_axis_claims": scope == "exploratory_cam_perturbation"}
        if scope == "exploratory_cam_perturbation":
            initialization["parent_run_id"] = "opt-a0"
        config = {"task": task.to_dict(), "task_hash": task.task_hash, "seed": seed,
                  "simulator_precision": "f64", "decision_ids": ["D-2026-11-09-05"],
                  "initialization": initialization,
                  "hard_selection_config_hash": C.content_hash({"synthetic_fixture": "selection-policy"})}
        write_json(run_dir / "config.json", config)
        best_candidate = dict(initial, objective=hard1)
        if run_id == "random-a0":
            best_candidate["components"] = dict(best_candidate["components"], feasible=1.0)
        write_json(run_dir / "metrics.json", {
            "best_candidate": best_candidate,
            "selection": {"initial_candidate": initial,
                          "no_improvement_over_initialization": not (hard1 < hard0)},
            "execution": {"posted_evaluation_phases": []},
        })
        write_json(run_dir / "history.json", {
            "history": [{"theta": [0.0], "soft_objective": soft}],
            "ledger": {"candidate_evaluations": 1, "forward_calls": 1, "backward_calls": 1,
                       "validation_calls": 1, "segment_steps": 1, "voxel_steps": 1,
                       "warmup_wall_time_s": 1.0, "steady_state_wall_time_s": 0.5,
                       "end_to_end_candidate_wall_time_s": 1.5,
                       "process_peak_rss_bytes": 1048576,
                       "memory_scope": "synthetic resource fixture; not real execution",
                       "entries": [{"candidate_id": run_id + ":init", "wall_time_s": 1.5,
                                    "warmup": True, "validation": False}]},
        })
        expected.append({"run_id": run_id, "task_id": task.task_id, "method": method, "seed": seed})
    manifest = {"schema_version": 1,
                "tasks": [{"task_id": task.task_id, "family": task.family, "task_hash": task.task_hash,
                           "task_split": "test",
                           "completion_residual_mm3": task.tolerances.completion_residual_mm3}],
                "expected_runs": expected,
                "comparisons": [{"name": "annealed-vs-nominal-rs274", "method": "annealed-adam",
                                 "baseline": "nominal-cam", "dialect": "rs274",
                                 "metrics": ["cycle_time_s", "air_fraction"]},
                                {"name": "annealed-vs-nominal-haas", "method": "annealed-adam",
                                 "baseline": "nominal-cam", "dialect": "haas",
                                 "metrics": ["cycle_time_s", "air_fraction"]}]}
    write_json(root / "manifest.json", manifest)
    report, typed_manifest = A.generate_report(root, A.load_manifest(manifest), bootstrap_replicates=100)
    assert report["summary"]["auditor_passed"] and report["all_runs_completed"]
    write_json(root / "audit.json", {"schema": "diffcam-campaign-audit/1",
                                     "summary": report["summary"] | {"evidence_complete": True},
                                     "reconciliation": report["expected_vs_observed"],
                                     "reports": report["runs"]})
    write_json(root / "frozen_config.json", {"raw_config": {
        "collapse_criterion": {"geometry_or_completion_failure": True,
                               "residual_multiplier": 1.5, "residual_operator": "gt",
                               "require_no_hard_loss_improvement_over_initial": True,
                               "predicate": "explicit unit-test D-08 example rule"}}})
    return root


def build(campaign, gradient=None, tmp_path=None):
    absent = tmp_path / "absent"
    return B.build_results_package(
        pilot_1_root=absent, pilot_2_root=None, pilot_init_root=None,
        campaign_root=campaign, gradient_root=gradient,
        decisions_path=ROOT / "reports/decisions.md", decision_ids=["D-2026-11-09-05"])


def test_real_results_package_main_init_disagreement_resources_and_summary(tmp_path):
    files, manifest = build(tiny_case(tmp_path), tmp_path=tmp_path)
    for name in ("main_results.csv", "family_breakdown.csv", "init_axis.csv",
                 "selection_disagreement.csv", "resource_costs.csv", "gradient_alignment.csv",
                 "summary.md", "claim_evidence_results.json", "results_manifest.json"):
        assert name in files
    headline = next(row for row in read_csv_from_bytes(files["main_results.csv"])
                    if row["dataset"] == "campaign"
                    and row["method"] == "annealed-adam"
                    and row["family"] == "overall" and row["dialect"] == "rs274")
    assert float(headline["delta_cycle_s"]) == pytest.approx(-1.666666666666667)
    assert float(headline["delta_air_fraction"]) == 0.0
    assert headline["cycle_paired_n"] == "1"
    assert headline["air_paired_n"] == "1"
    assert float(headline["geometry_pass_rate"]) == pytest.approx(0.6)
    assert float(headline["no_improvement_rate"]) == pytest.approx(0.6)
    assert headline["cycle_holm_p"] != "NA"
    assert any(row["family"] == "pocket" and float(row["delta_cycle_s"]) == pytest.approx(-1.666666666666667)
               and float(row["geometry_pass_rate"]) == pytest.approx(0.6)
               for row in read_csv_from_bytes(files["family_breakdown.csv"]))
    initial = read_csv_from_bytes(files["init_axis.csv"])
    straight = next(row for row in initial if row["init"] == "straight_line")
    random = next(row for row in initial if row["init"] == "random")
    perturbed = next(row for row in initial if row["scope"] == "exploratory_cam_perturbation")
    assert straight["collapse_count"] == "1" and straight["collapse_rate"] == "1"
    assert random["collapse_count"] == "0"
    assert perturbed["included_in_init_axis"] == "false"
    disagreement = read_csv_from_bytes(files["selection_disagreement.csv"])
    assert any(row["run_id"] == "random-a0" and row["agree"] == "false" for row in disagreement)
    assert any(row["run_id"] == "opt-a0" and row["soft_minus_hard"] == "0.645" for row in disagreement)
    resource = next(row for row in read_csv_from_bytes(files["resource_costs.csv"])
                    if row["run_id"] == "opt-a0" and row["role"] == "optimizer_total")
    assert resource["candidate_evaluations"] == "1"
    assert resource["wall_time_s"] == "1.5"
    assert resource["process_peak_rss_bytes"] == "1048576"
    text = files["summary.md"].decode()
    assert "CEM is unavailable" in text and "Titan evidence is partial" in text
    assert "CNC-Net is native-route only" in text and "simulation" in text
    assert "pilot_2" in text and "pending" in text


def read_csv_from_bytes(payload):
    return list(csv.DictReader(payload.decode().splitlines()))


def test_all_missing_inputs_produce_pending_rows_not_synthetic_success(tmp_path):
    files, _ = build(tmp_path / "missing-campaign", gradient=None, tmp_path=tmp_path)
    states = read_csv_from_bytes(files["input_status.csv"])
    assert {row["state"] for row in states} == {"absent"}
    assert all(row["method"] == "nominal-cam" for row in read_csv_from_bytes(files["main_results.csv"]))
    assert all(row["delta_cycle_s"] == "NA" for row in read_csv_from_bytes(files["main_results.csv"]))
    assert read_csv_from_bytes(files["resource_costs.csv"]) == []
    assert read_csv_from_bytes(files["gradient_alignment.csv"])[0]["status"] == "absent"


def test_mismatched_post_fidelity_is_pending_not_misrepresented(tmp_path):
    root = tiny_case(tmp_path)
    path = root / "cam-a0/eval_posted_rs274.json"
    typed = C.loads(path.read_text())
    settings = dict(typed.discretization["settings"], max_step_mm=1.0)
    path.write_text(C.dumps(dataclasses.replace(
        typed, discretization={"settings": settings, "segments": []})))
    files, _ = build(root, tmp_path=tmp_path)
    row = next(row for row in read_csv_from_bytes(files["input_status.csv"]) if row["dataset"] == "campaign")
    assert row["state"] == "pending"
    assert any(row["delta_cycle_s"] == "NA" for row in read_csv_from_bytes(files["main_results.csv"]))


def tiny_gradient(tmp_path):
    root = tmp_path / "t33/full-reuse-v3"
    write_json(root / "matrix-plan.json", {
        "schema": "synthetic-gradient-matrix/1", "cases": [
            {"case_index": 0, "path_id": "cam", "k": 20.0, "objective_mode": "geometry"},
            {"case_index": 1, "path_id": "cam", "k": 40.0, "objective_mode": "efficiency"},
        ]})
    case = {
        "status": "complete", "objective_mode": "geometry",
        "case": {"case_index": 0, "path_id": "cam", "k": 20.0},
        "diagnostic_status": "complete typed fixture only",
        "setup": {"precision": "f64"},
        "ad_vs_primary_hard": {"cosine": 0.4, "relative_difference_norm": 0.3},
        "ad_vs_reference_hard": {"cosine": 0.5, "relative_difference_norm": 0.2},
        "ad_vs_reference_hard_status": "ok",
        "smooth_ad": {"fd_alignment_and_coordinate_errors": [{
            "alignment_on_selected_coordinates": {"cosine": 0.99, "relative_difference_norm": 0.01},
            "epsilon_normalized_theta": 0.025, "evaluated_indices": [0, 1, 2, 35],
        }]},
        "randomized_hard": {
            "primary": {"pairs": 16, "standard_error_rms": 0.3},
            "reference": {"pairs": 64, "standard_error_rms": 0.2},
            "reused_across_k": True,
        },
        "actual_candidate_evaluations": 178, "wall_time_s": 80.0,
        "memory": {"own_process_vmhwm_bytes": 1024},
    }
    write_json(root / "case-000-cam-k20-geometry.json", case)
    return root.parent


def test_gradient_alignment_and_pending_k_case(tmp_path):
    files, _ = build(tiny_case(tmp_path), gradient=tiny_gradient(tmp_path), tmp_path=tmp_path)
    rows = read_csv_from_bytes(files["gradient_alignment.csv"])
    assert len(rows) == 2
    assert float(rows[0]["ad_primary_cosine"]) == pytest.approx(0.4)
    assert rows[0]["fd_indices"] == "[0,1,2,35]"
    assert rows[0]["hard_reused_across_k"] == "true"
    assert rows[1]["status"] == "pending"
    assert "AD/primary-randomized-hard cosine=0.4" in files["summary.md"].decode()


def test_actual_retained_pilot_and_gradient_inputs_build_when_available(tmp_path):
    runs = Path("/home/ntsoi/src/cnc/diffcam/runs")
    required = [runs / "pilot-pocket-20261109-t41b-pilot",
                runs / "retained/t33-real-study-20261110-w3-r1"]
    if not all(path.is_dir() for path in required):
        pytest.skip("local retained pilot/T3.3 evidence is unavailable")
    files, _ = B.build_results_package(
        pilot_1_root=runs / "pilot-pocket-20261109-t41b-pilot",
        pilot_2_root=runs / "pilot-pocket-20261110-pilot2",
        pilot_init_root=runs / "pilot-pocket-20261110-pilot2_init",
        campaign_root=None, gradient_root=runs / "retained/t33-real-study-20261110-w3-r1",
        decisions_path=ROOT / "reports/decisions.md", decision_ids=["D-2026-11-09-05"])
    available = next(row for row in read_csv_from_bytes(files["input_status.csv"]) if row["dataset"] == "pilot_1")
    assert available["state"] == "available"
    real = next(row for row in read_csv_from_bytes(files["main_results.csv"])
                if row["dataset"] == "pilot_1" and row["method"] == "nominal-cam" and row["dialect"] == "rs274")
    assert real["geometry_pass_rate"] == "1"
    assert len(read_csv_from_bytes(files["gradient_alignment.csv"])) >= 12


def test_failed_init_retains_original_seed_declaration_and_invalid_reason(tmp_path):
    root = tiny_case(tmp_path)
    task = FIXTURES._task("task-a", "pocket")
    record = C.RunRecord(run_id="random-a0", task_hash=task.task_hash, status=C.RunStatus.FAILED,
                         artifacts=("history.json",), failure_reason="synthetic failed")
    write_json(root / "random-a0/history.json", {"evaluations": []})
    (root / "random-a0/run.json").write_text(C.dumps(record))
    config = read_json(root / "random-a0/config.json")
    config["initialization_seed_result_file"] = "initialization_seed_result.json"
    write_json(root / "random-a0/config.json", config)
    seed_document = {
        "schema": "diffcam-initialization-seed-result/2", "valid": False, "status": "failed_instance",
        "seed": 0, "declaration_hash": "sha256:" + "a" * 64,
        "derivation": {"requested_seed": 0, "no_redraw": True},
        "motion": None, "invalid_reason": "synthetic declared free vertex outside the envelope",
    }
    write_json(root / "random-a0/initialization_seed_result.json", seed_document)
    manifest = A.load_manifest(root / "manifest.json")
    _reports, audit = A.audit_runs(root, manifest)
    document = read_json(root / "audit.json")
    document["summary"] = dict(audit["summary"], evidence_complete=audit["summary"]["evidence_complete"])
    document["reconciliation"] = audit["reconciliation"]
    write_json(root / "audit.json", document)
    files, _ = build(root, tmp_path=tmp_path)
    row = next(row for row in read_csv_from_bytes(files["init_axis.csv"])
               if row["init"] == "random")
    assert row["observed_runs"] == "1" and row["failed_runs"] == "1"
    details = json.loads(row["failure_details"])
    assert details[0]["status"] == "failed"
    assert details[0]["seed_result"] == seed_document
    assert details[0]["failure_reason"] == "synthetic failed"
    assert details[0]["seed_result"]["derivation"]["no_redraw"] is True
    assert details[0]["seed_result"]["motion"] is None
    claim = json.loads(files["claim_evidence_results.json"])
    assert any(row_data["table"] == "init_axis"
               and "initialization_seed_result:random-a0" in row_data["source_file_identity"]
               for row_data in claim["rows"])


def test_results_are_byte_identical_after_input_relocation(tmp_path):
    original = tiny_case(tmp_path)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    left, _ = build(original, tmp_path=tmp_path)
    right, _ = build(relocated, tmp_path=tmp_path)
    assert left == right
    assert left["results_manifest.json"] == right["results_manifest.json"]
