"""T7.2 publication-scaffold tests.

Audit fixtures are structural synthetic inputs, not machining evidence. Reuse
T4.0's typed-record fixtures, then exercise the actual immutable audit report.
No simulator, external baseline executable or optional image renderer is needed.
"""
from __future__ import annotations

import csv
import dataclasses
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from benchmarks import audit as A
from benchmarks import contracts as C

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "reports/decisions.md"
DECISION_05 = "D-2026-11-09-05"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B = load_module("t72_builder", ROOT / "scripts/build_paper_artifacts.py")
FIXTURES = load_module("t72_audit_fixtures", ROOT / "tests/eval_aggregator_test.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False, sort_keys=True) + "\n",
                          encoding="utf-8")


def read_csv(path):
    return list(csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8"))))


def make_case(tmp_path):
    root = tmp_path / "runs"
    tasks = FIXTURES._tasks(("task-a", "pocket"), ("task-b", "other"), ("task-c", "other"))
    expected = [
        ("cam-a0", "task-a", "cam", 0), ("cam-a1", "task-a", "cam", 1),
        ("cam-b0", "task-b", "cam", 0), ("cam-c0", "task-c", "cam", 0),
        ("cand-a0", "task-a", "candidate", 0), ("cand-a1", "task-a", "candidate", 1),
        ("cand-b0", "task-b", "candidate", 0), ("cand-c0", "task-c", "candidate", 0),
    ]
    FIXTURES._write_run(root, "cam-a0", tasks["task-a"], method="cam", cycle=6)
    FIXTURES._write_run(root, "cam-a1", tasks["task-a"], method="cam", seed=1, cycle=8)
    FIXTURES._write_run(root, "cam-b0", tasks["task-b"], method="cam", cycle=7)
    FIXTURES._write_run(root, "cam-c0", tasks["task-c"], method="cam", cycle=9)
    FIXTURES._write_run(root, "cand-a0", tasks["task-a"], cycle=4)
    FIXTURES._write_run(root, "cand-a1", tasks["task-a"], seed=1, cycle=5)
    FIXTURES._write_failed_run(root, "cand-b0", tasks["task-b"], status=C.RunStatus.BLOCKED)
    FIXTURES._write_run(root, "cand-c0", tasks["task-c"], removed=0, residual=0, cycle=3)
    for run_id, task_id, method, seed in expected:
        config_path = root / run_id / "config.json"
        config = (read_json(config_path) if config_path.is_file() else {
            "task_hash": tasks[task_id].task_hash, "method": method, "seed": seed,
        })
        # Campaign metadata declarations for this structural fixture; not a claim
        # that the underlying T4.0 synthetic evaluator was numerically executed f64.
        config["simulator_precision"] = "f64"
        config["decision_ids"] = [DECISION_05]
        write_json(config_path, config)
    history = root / "cam-a0" / "history.json"
    ledger = {
        "candidate_evaluations": 1, "forward_calls": 2, "backward_calls": 1,
        "validation_calls": 1, "segment_steps": 8, "voxel_steps": 64,
        "warmup_wall_time_s": 2.5, "steady_state_wall_time_s": 1.0,
        "end_to_end_candidate_wall_time_s": 3.5,
        "process_peak_rss_bytes": 1048576,
        "python_traced_memory_current_bytes": 0,
        "python_traced_memory_peak_bytes": 1000,
        "memory_scope": "synthetic process-budget fixture; not device-only memory",
        "entries": [{"candidate_id": "synthetic-candidate", "wall_time_s": 2.5,
                     "warmup": True, "validation": False},
                    {"candidate_id": "synthetic-validation", "wall_time_s": 1.0,
                     "warmup": False, "validation": True}],
        "simulator_precision": "f64",
    }
    write_json(history, {"history": [], "ledger": ledger, "simulator_precision": "f64"})
    manifest = FIXTURES._manifest(tasks, expected, metrics=("cycle_time_s", "air_fraction"))
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    report, _ = A.generate_report(root, manifest, bootstrap_replicates=100)
    assert report["evidence_complete"] and not report["all_runs_completed"]
    audit_path = tmp_path / "pilot_report.json"
    write_json(audit_path, report)
    coverage_path = tmp_path / "baseline-coverage.json"
    cam_nc = next(row["dialects"]["rs274"]["declared_nc_hash"]
                  for row in report["runs"] if row["run_id"] == "cam-a0")
    write_json(coverage_path, {
        "schema": "diffcam-paper-coverage/1", "items": [
            {"kind": "external_baseline", "scope": "CNC-Net executable-NC route",
             "status": "unsupported", "reason": "declared unsupported route in this synthetic coverage fixture",
             "run_id": None, "task_hash": None, "nc_hash": None},
            {"kind": "soft_hard_posted", "scope": "synthetic declared fidelity disagreement",
             "status": "disagreement", "reason": "structural negative-result preservation fixture",
             "run_id": "cam-a0", "nc_hash": cam_nc, "disagreement_fields": ["completion_pass"]},
        ],
    })
    return {
        "root": root, "audit": audit_path, "manifest": manifest_path, "coverage": coverage_path,
        "configs": {run_id: root / run_id / "config.json" for run_id, *_ in expected},
        "ledger": history,
        "kwargs": {
            "audit_path": audit_path, "manifest_path": manifest_path,
            "decisions_path": DECISIONS, "precision": "f64", "decision_ids": [DECISION_05],
            "input_kind": "synthetic",
            "run_configs": {run_id: root / run_id / "config.json" for run_id, *_ in expected},
            "ledger_paths": {"cam-a0": history}, "baseline_coverage_path": coverage_path,
        },
    }


def build(case):
    return B.build_artifacts(**case["kwargs"])


def cli_args(case, output):
    args = [
        "--audit", str(case["audit"]), "--manifest", str(case["manifest"]),
        "--decisions", str(DECISIONS), "--campaign-precision", "f64",
        "--decision", DECISION_05, "--input-kind", "synthetic",
        "--run-root", str(case["root"]),
        "--ledger", f"cam-a0={case['ledger']}",
        "--baseline-coverage", str(case["coverage"]), "--output", str(output),
    ]
    return args


def claims(files):
    return json.loads(files["claim_evidence.json"])["claims"]


def test_complete_scaffold_traces_every_cell_and_preserves_unfavorable_rows(tmp_path):
    case = make_case(tmp_path)
    files, metadata = build(case)
    assert metadata["input_kind"] == "synthetic"
    assert metadata["audit_evidence_complete"] and not metadata["all_runs_completed"]
    assert metadata["final_frozen_campaign_validated"] is False
    assert metadata["campaign_precision"] == "f64"
    assert metadata["builder_source_sha256"] == B.digest((ROOT / "scripts/build_paper_artifacts.py").read_bytes())
    for entry in metadata["files"]:
        if entry["path"] != "artifact_manifest.json":
            assert B.digest(files[entry["path"]]) == entry["sha256"]
            assert len(files[entry["path"]]) == entry["bytes"]
    rows = list(csv.DictReader(io.StringIO(files["tables/run_status.csv"].decode())))
    assert len(rows) == 16
    no_cut = next(row for row in rows if row["run_id"] == "cand-c0" and row["dialect"] == "rs274")
    assert no_cut["no_cut"] == "true" and no_cut["completion_pass"] == "false"
    failed = next(row for row in rows if row["run_id"] == "cand-b0" and row["dialect"] == "rs274")
    assert failed["run_status"] == "blocked" and failed["safety_pass"] == "NA"
    assert all(row["precision"] == "f64" for row in rows)
    assert any(row["nominal_anchor"] == "true" for row in rows if row["method"] == "cam")
    metrics = read_csv_from_bytes(files["tables/metric_comparisons.csv"])
    overall = next(row for row in metrics if row["family"] == "overall" and row["metric"] == "cycle_time_s")
    # Four candidate/cam seed runs on task-a do not create four independent tasks.
    # The blocked and no-cut tasks are excluded from cycle-time efficiency data.
    assert overall["paired_tasks"] == "1"
    assert "fewer than 5" in overall["small_n_warning"]
    assert overall["raw_sign_flip_p"] != "NA" and overall["holm_p"] != "NA"
    source_ptr = next(c for c in claims(files) if c["row_id"].endswith("overall/cycle_time_s"))
    assert source_ptr["statistical_unit"] == "task"
    for claim in claims(files):
        assert len(claim["cells"]) == len(claim["column_order"])
        for cell in claim["cells"].values():
            assert "value" in cell and "display" in cell and "source_pointer" in cell
        assert claim["source_file_identity"]["audit"] == B.digest(case["audit"].read_bytes())
    coverage = read_csv_from_bytes(files["tables/coverage.csv"])
    assert any(row["status"] == "unsupported" and "CNC-Net" in row["scope"] for row in coverage)
    assert any(row["category"] == "no_cut" for row in coverage)
    assert any(row["run_id"] == "cand-b0" and row["status"] == "recorded_noncomplete" for row in coverage)
    assert any(row["category"] == "refinement" and row["status"] == "not_provided" for row in coverage)
    decisions = read_csv_from_bytes(files["tables/decisions.csv"])
    assert {"D-2026-11-09-03", "D-2026-11-09-05"}.issubset({row["id"] for row in decisions})
    resources = read_csv_from_bytes(files["tables/resource_cost.csv"])
    available = next(row for row in resources if row["run_id"] == "cam-a0")
    missing = next(row for row in resources if row["run_id"] == "cand-a0")
    assert available["candidate_evaluations"] == "1" and available["ledger_availability"] == "linked"
    assert missing["ledger_availability"] == "unavailable"
    assert missing["end_to_end_candidate_wall_time_s"] == "NA"
    assert missing["candidate_evaluations"] == "NA"
    assert "f64" in files["tables/decisions.csv"].decode()
    assert b"\\toprule" in files["tables/metric_comparisons.tex"]
    assert (ROOT / "reports/final_audit_template.md").read_text().startswith("# Final audit TEMPLATE")


def read_csv_from_bytes(payload):
    return list(csv.DictReader(io.StringIO(payload.decode())))


def test_byte_identical_regeneration_when_inputs_relocate_and_cli_invocation(tmp_path):
    left = make_case(tmp_path / "left")
    right_root = tmp_path / "right"
    shutil.copytree(left["root"], right_root / "runs")
    for name in ("audit", "manifest", "coverage", "ledger"):
        shutil.copyfile(left[name], right_root / left[name].name)
    # Preserve unchanged evidence bytes, including the frozen report's original
    # runs_root string. The builder must not hide source-path edits to get a
    # matching hash. Explicit config bindings point at the archived copies.
    right = {key: right_root / left[key].name for key in ("audit", "manifest", "coverage", "ledger")}
    right["root"] = right_root / "runs"
    right["kwargs"] = {
        **left["kwargs"], "audit_path": right["audit"], "manifest_path": right["manifest"],
        "baseline_coverage_path": right["coverage"],
        "run_configs": {run_id: right["root"] / run_id / "config.json"
                        for run_id in left["kwargs"]["run_configs"]},
        "ledger_paths": {"cam-a0": right["ledger"]},
    }
    files_left, _ = build(left)
    files_right, _ = build(right)
    assert set(files_left) == set(files_right)
    assert all(files_left[name] == files_right[name] for name in files_left)
    out = tmp_path / "cli-output"
    assert B.main(cli_args(right, out)) == 0
    actual = {path.relative_to(out): path.read_bytes()
              for path in out.rglob("*") if path.is_file()}
    assert actual == {Path(name): data for name, data in files_right.items()}
    help_run = subprocess.run(
        [sys.executable, str(out / "figure_data/plot_resource_cost.py"), "--help"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert help_run.returncode == 0
    assert "--output" in help_run.stdout
    # Renderer is optional; no claim of a rendered image in the locked core env.
    assert "not_rendered" in actual[Path("figure_data/figure_plan.json")].decode()


def test_reject_precision_or_decision_mismatch_before_publishing(tmp_path):
    case = make_case(tmp_path)
    config_path = case["configs"]["cand-a0"]
    for field, invalid, expected in (
        ("simulator_precision", "f32", "conflicts with run precision"),
        ("decision_ids", ["D-2026-11-09-01"], "decision IDs"),
    ):
        config = read_json(config_path)
        original = config[field]
        config[field] = invalid
        write_json(config_path, config)
        out = tmp_path / ("output-" + field)
        with pytest.raises(B.PaperArtifactError, match=expected):
            build(case)
        with pytest.raises(SystemExit) as exit_info:
            B.main(cli_args(case, out))
        assert exit_info.value.code == 2
        assert not out.exists()
        config[field] = original
        write_json(config_path, config)


def test_missing_or_ambiguous_precision_metadata_fails_closed(tmp_path):
    case = make_case(tmp_path)
    path = case["configs"]["cam-a0"]
    config = read_json(path)
    del config["simulator_precision"]
    write_json(path, config)
    with pytest.raises(B.PaperArtifactError, match="simulator_precision"):
        build(case)
    config["simulator_precision"] = "f64"
    config["nested"] = {"simulator_precision": "f32"}
    write_json(path, config)
    with pytest.raises(B.PaperArtifactError, match="conflicting precision"):
        build(case)


def test_evidence_complete_and_all_identity_links_are_mandatory(tmp_path):
    case = make_case(tmp_path)
    report = read_json(case["audit"])
    report["evidence_complete"] = False
    write_json(case["audit"], report)
    with pytest.raises(B.PaperArtifactError, match="evidence_complete=false"):
        build(case)
    case = make_case(tmp_path / "missing-run")
    report = read_json(case["audit"])
    removed = report["runs"].pop()
    assert removed["run_id"] in report["expected_vs_observed"]["observed_run_ids"]
    write_json(case["audit"], report)
    with pytest.raises(B.PaperArtifactError, match="run"):
        build(case)
    case = make_case(tmp_path / "bad-nc-link")
    report = read_json(case["audit"])
    target = next(row for row in report["runs"] if row["run_id"] == "cam-a0")
    target["dialects"]["rs274"]["actual_nc_hash"] = None
    write_json(case["audit"], report)
    with pytest.raises(B.PaperArtifactError, match="NC identity"):
        build(case)


def test_seed_replication_cannot_be_disguised_as_independent_tasks(tmp_path):
    case = make_case(tmp_path)
    report = read_json(case["audit"])
    row = next(row for row in report["comparisons"] if row["family"] == "overall")
    row["metrics"]["cycle_time_s"]["paired_tasks"] = 4
    write_json(case["audit"], report)
    with pytest.raises(B.PaperArtifactError, match="paired task"):
        build(case)


def test_bad_ledger_metadata_does_not_become_complete_resource_cost(tmp_path):
    case = make_case(tmp_path)
    for change, expected in (
        ({"forward_calls": -1}, "nonnegative integer"),
        ({"simulator_precision": "f32"}, "ledger/config precision"),
    ):
        history = read_json(case["ledger"])
        history["ledger"].update(change)
        if "simulator_precision" in change:
            history["simulator_precision"] = change["simulator_precision"]
        write_json(case["ledger"], history)
        with pytest.raises(B.PaperArtifactError, match=expected):
            build(case)
        case = make_case(tmp_path / ("ledger-" + next(iter(change))))


def test_output_refuses_nonempty_or_manuscript_destinations(tmp_path):
    case = make_case(tmp_path)
    files, _ = build(case)
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "user-data.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(B.PaperArtifactError, match="fresh/empty"):
        B.publish_outputs(files, target)
    assert marker.read_text() == "preserve"
    assert len(list(target.iterdir())) == 1
    forbidden = tmp_path / "manuscript" / "never-write-here"
    with pytest.raises(B.PaperArtifactError, match="manuscript"):
        B.publish_outputs(files, forbidden)
    assert not forbidden.exists()


def test_optional_plotter_is_explicitly_blocked_without_optional_dependency(tmp_path):
    case = make_case(tmp_path)
    out = tmp_path / "output"
    files, _ = build(case)
    destination = B.publish_outputs(files, out)
    if importlib.util.find_spec("matplotlib") is not None:
        pytest.skip("this negative test specifically exercises the locked core without matplotlib")
    run = subprocess.run(
        [sys.executable, str(destination / "figure_data/plot_resource_cost.py"),
         "--output", str(tmp_path / "figure.png")],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert run.returncode == 2
    assert "separately provisioned" in run.stderr
    assert not (tmp_path / "figure.png").exists()


def test_comparison_free_audited_input_retains_explicit_empty_figure_schema(tmp_path):
    case = make_case(tmp_path)
    report = read_json(case["audit"])
    report["comparisons"] = []
    write_json(case["audit"], report)
    files, _ = build(case)
    assert not read_csv_from_bytes(files["figure_data/metric_comparisons.csv"])
    assert len(read_csv_from_bytes(files["tables/run_status.csv"])) == 16


def make_refinement(case):
    """Synthetic schema-3 bundle; no real CNC/refinement execution is implied."""
    primary_root = case["root"] / "cam-a0"
    task = FIXTURES._task("task-a", "pocket")
    task_info = A.load_manifest(case["manifest"]).tasks["task-a"]
    primary = C.loads((primary_root / "eval_posted_rs274.json").read_text())
    settings = {"voxel_mm": primary.discretization.get("voxel_mm", 0.5),
                "max_step_mm": 1.0, "expected_nc_hash": primary.nc_hash,
                "compute_bounds": True, "safety_screen": True}
    parsed_hash = C.sha256_bytes(b"synthetic-paper-motion-identity")
    primary = dataclasses.replace(
        primary, evaluator_revision="synthetic-primary/2",
        metrics=FIXTURES._metrics(cycle=6, removed=20, residual=0),
        discretization={"voxel_mm": 0.5, "parsed_motion_hash": parsed_hash,
                        "settings": settings},
    )
    nc_bytes = (primary_root / "rs274.nc").read_bytes()
    (primary_root / "eval_posted_rs274.json").write_text(C.dumps(primary))
    # Structural second-dialect support, not a Haas linguistic oracle test.
    (primary_root / "haas.nc").write_bytes(nc_bytes)
    haas = dataclasses.replace(primary, dialect=C.Dialect.HAAS)
    (primary_root / "eval_posted_haas.json").write_text(C.dumps(haas))
    record = C.loads((primary_root / "run.json").read_text())
    posted = tuple(dataclasses.replace(item, status=C.PostStatus.SUPPORTED,
                                       nc_hash=haas.nc_hash, eval_file="eval_posted_haas.json",
                                       reason=None)
                   if item.dialect is C.Dialect.HAAS else item for item in record.posted)
    record = dataclasses.replace(
        record, posted=posted, artifacts=record.artifacts + ("haas.nc", "eval_posted_haas.json"))
    (primary_root / "run.json").write_text(C.dumps(record))
    index = read_json(primary_root / "eval_posted.json")
    index["programs"]["haas"] = {"dialect": "haas", "status": "supported",
                                  "nc_hash": haas.nc_hash, "eval_file": "eval_posted_haas.json"}
    write_json(primary_root / "eval_posted.json", index)

    refined_id = "cam-a0-refined"
    aux_root = case["audit"].parent / "aux"
    auxiliary_root, _, _ = FIXTURES._write_run(aux_root, refined_id, task)
    threshold = task_info.completion_residual_mm3
    refined_eval = C.loads((auxiliary_root / "eval_posted_rs274.json").read_text())
    refined_eval = dataclasses.replace(
        refined_eval, evaluator_id="posted_hard/production",
        evaluator_revision="synthetic-refined/1",
        metrics=FIXTURES._metrics(cycle=3, removed=20, residual=2 * threshold),
        discretization={"voxel_mm": settings["voxel_mm"], "parsed_motion_hash": parsed_hash,
                        "settings": dict(settings, max_step_mm=0.125)},
    )
    refined_file = auxiliary_root / "eval_posted_rs274_refined.json"
    refined_file.write_text(C.dumps(refined_eval))
    primary_efficiency = primary.efficiency_eligible
    refined_efficiency = refined_eval.efficiency_eligible
    comparison = {
        "safety_pass": {"primary": True, "refined": True, "changed": False},
        "geometry_pass": {"primary": True, "refined": True, "changed": False},
        "coverage": {"primary": "complete", "refined": "complete", "changed": False},
        "efficiency_eligible": {"primary": primary_efficiency, "refined": refined_efficiency,
                                "changed": primary_efficiency != refined_efficiency},
        "residual_volume_mm3": {"primary": 0.0, "refined": 2 * threshold,
                                "delta": 2 * threshold, "unit": "mm^3"},
        "gouge_volume_mm3": {"primary": 0.0, "refined": 0.0, "delta": 0.0, "unit": "mm^3"},
        "max_surface_error_mm": {"primary": 0.05, "refined": 0.05, "delta": 0.0, "unit": "mm"},
        "completion_pass": {"primary": True, "refined": False, "changed": True},
        "any_outcome_verdict_change": False,
        "changed_outcome_fields": ["completion_pass"],
    }
    conservative = {
        "safety_pass": True, "geometry_pass": True, "completion_pass": False,
        "efficiency_eligible": False, "refinement_disagreement": True,
        "disagreement_fields": ["completion_pass"], "excluded_from_efficiency": True,
        "rule": "synthetic schema-3 false-worst/unknown conservative rule; disagreements exclude efficiency",
    }
    comparison_path = auxiliary_root / "comparison.json"
    write_json(comparison_path, {
        "schema": "synthetic-refinement-comparison/1", "primary_run_id": "cam-a0",
        "comparison": comparison, "conservative_outcome": conservative,
    })
    link = {
        "schema": "diffcam-refinement-link/3", "primary_run_id": "cam-a0",
        "primary_run_path": "cam-a0", "refined_run_id": refined_id,
        "primary_metrics_sha256": C.sha256_bytes((primary_root / "metrics.json").read_bytes()),
        "primary_rs274_evaluation_sha256": C.sha256_bytes(
            (primary_root / "eval_posted_rs274.json").read_bytes()),
        "primary_haas_evaluation_sha256": C.sha256_bytes(
            (primary_root / "eval_posted_haas.json").read_bytes()),
        "primary_run_json_sha256": C.sha256_bytes((primary_root / "run.json").read_bytes()),
        "refined_artifact_path": f"{refined_id}/eval_posted_rs274_refined.json",
        "refined_evaluation_sha256": C.sha256_bytes(refined_file.read_bytes()),
        "comparison_path": f"{refined_id}/comparison.json",
        "comparison_sha256": C.sha256_bytes(comparison_path.read_bytes()),
        "auxiliary_record": {"path": f"{refined_id}/run.json",
                             "sha256": C.sha256_bytes((auxiliary_root / "run.json").read_bytes())},
        "comparison": comparison, "conservative_outcome": conservative,
    }
    link_path = aux_root / "cam-a0-refinement-link.json"
    write_json(link_path, link)
    # Audit only after editing the fixture's primary typed bytes.
    write_json(case["audit"], A.generate_report(case["root"], case["manifest"],
                                                 bootstrap_replicates=100)[0])
    return link_path


def test_schema3_refinement_disagreement_and_immutable_links(tmp_path):
    case = make_case(tmp_path)
    link = make_refinement(case)
    options = dict(case["kwargs"], refinement_paths=[link])
    files, _ = B.build_artifacts(**options)
    coverage = read_csv_from_bytes(files["tables/coverage.csv"])
    row = next(row for row in coverage if row["category"] == "refinement_disagreement")
    assert row["run_id"] == "cam-a0"
    assert row["status"] == "disagreement"
    assert json.loads(row["disagreement_fields"]) == ["completion_pass"]
    fact = json.loads(row["reason"])
    assert fact["conservative_outcome"]["completion_pass"] is False
    assert fact["conservative_outcome"]["excluded_from_efficiency"] is True
    assert fact["rule"] == "typed_metrics_and_schema3_links_recomputed"
    assert fact["comparison"]["residual_volume_mm3"]["delta"] > 0
    assert (fact["comparison"]["residual_volume_mm3"]["refined"]
            > fact["comparison"]["residual_volume_mm3"]["primary"])
    assert any(row["run_id"] == "cand-a0" and row["status"] == "not_provided"
               for row in coverage if row["category"] == "refinement")
    claim = next(claim for claim in claims(files) if claim["claim_id"].endswith("#coverage/refinement/cam-a0"))
    assert claim["source_file_identity"]["refinement"] == B.digest(link.read_bytes())


def test_refinement_rejects_inconsistent_verdict_or_missing_refined_bytes(tmp_path):
    case = make_case(tmp_path)
    link = make_refinement(case)
    raw = read_json(link)
    raw["conservative_outcome"]["efficiency_eligible"] = True
    raw["conservative_outcome"]["excluded_from_efficiency"] = False
    write_json(link, raw)
    options = dict(case["kwargs"], refinement_paths=[link])
    with pytest.raises(B.PaperArtifactError, match="conservative outcome efficiency_eligible"):
        B.build_artifacts(**options)
    case = make_case(tmp_path / "missing")
    link = make_refinement(case)
    refined = link.parent / "cam-a0-refined/eval_posted_rs274_refined.json"
    assert refined.is_file()
    refined.unlink()  # Exact fixture file created above, not a wildcard cleanup.
    options = dict(case["kwargs"], refinement_paths=[link])
    with pytest.raises(B.PaperArtifactError, match="refined_artifact.*missing"):
        B.build_artifacts(**options)


def test_refinement_rejects_modified_primary_nc_bytes(tmp_path):
    case = make_case(tmp_path)
    link = make_refinement(case)
    nc_file = case["root"] / "cam-a0/rs274.nc"
    before = nc_file.read_bytes()
    nc_file.write_bytes(before + b"; synthetic immutability tamper\n")
    with pytest.raises(B.PaperArtifactError, match="NC bytes differ from the audited identity"):
        B.build_artifacts(**dict(case["kwargs"], refinement_paths=[link]))


def test_reported_nested_outcomes_and_inline_refinement_resources(tmp_path):
    case = make_case(tmp_path)
    report = read_json(case["audit"])
    inline_links = {}
    outcome = {"safety_pass": None, "geometry_pass": None, "completion_pass": False,
               "efficiency_eligible": False, "excluded_from_efficiency": True,
               "refinement_disagreement": False, "disagreement_fields": [],
               "rule": "synthetic inline refinement declaration; raw bundle not re-read"}
    for run in report["runs"]:
        run_id = run["run_id"]
        inline_links[run_id] = {
            "primary_run_id": run_id, "refined_run_id": run_id + "-refined",
            "refined_evaluation_sha256": C.sha256_bytes(run_id.encode()),
            "refined_artifact_path": run_id + "-refined/eval_posted_rs274_refined.json",
            "conservative_outcome": outcome,
        }
    report["refinement"] = {
        "links_by_run_id": inline_links,
        "resources_by_run_id": {"cam-a0": {"stage": "synthetic refinement", "worker_seconds": 2.75}},
    }
    row = next(row for row in report["runs"] if row["run_id"] == "cam-a0")
    row["reported_outcomes"] = {
        "outcomes": {
            "rs274": {"safety_pass": True, "geometry_pass": True, "completion_pass": False,
                      "efficiency_eligible": False, "coverage": "complete"},
            "haas": {"safety_pass": None, "geometry_pass": None, "completion_pass": None,
                     "efficiency_eligible": False, "coverage": "unknown"},
        },
        "screen_disagreement_fields": ["completion_pass"],
    }
    write_json(case["audit"], report)
    files, _ = build(case)
    row = next(row for row in read_csv_from_bytes(files["tables/run_status.csv"])
               if row["run_id"] == "cam-a0" and row["dialect"] == "rs274")
    assert row["completion_pass"] == "true"
    assert row["reported_completion_pass"] == "false"
    assert row["disagreement_fields"] == '["completion_pass"]'
    trace = next(c for c in claims(files)
                 if c["row_id"] == "cam-a0/rs274")
    assert trace["cells"]["reported_completion_pass"]["source_pointer"].endswith(
        "/reported_outcomes/outcomes/rs274/completion_pass")
    coverage = read_csv_from_bytes(files["tables/coverage.csv"])
    resource = next(row for row in coverage if row["category"] == "refinement_resources")
    assert json.loads(resource["reason"])["worker_seconds"] == 2.75
    assert any(row["category"] == "refinement" and row["status"] == "no_recorded_disagreement"
               for row in coverage)


def test_inline_refinement_declaration_cannot_omit_primary_links(tmp_path):
    case = make_case(tmp_path)
    report = read_json(case["audit"])
    report["refinement"] = {"links_by_run_id": {"cam-a0": {"primary_run_id": "cam-a0"}}}
    write_json(case["audit"], report)
    with pytest.raises(B.PaperArtifactError, match="every expected run"):
        build(case)


def declared_exclusion(case):
    manifest = read_json(case["manifest"])
    item = next(task for task in manifest["tasks"] if task["task_id"] == "task-b")
    item.update({"grid_excluded_volume_mm3": 1.5, "evaluator_exclusion_tolerance_mm3": 0.0,
                 "continuous_exclusion_diagnostic_mm3": 1.7168146928204138})
    item["task_record"] = {"record": {"operation_target": {
        "excluded_volume_mm3": 1.5, "exclusion_reason": "synthetic declared finite-cutter exclusion",
        "geometry": {"kind": "analytic", "params": {
            "grid_excluded_volume_mm3": 1.5, "grid_exclusion_tolerance_mm3": 0.0,
            "continuous_excluded_volume_mm3": 1.7168146928204138,
        }},
    }}}
    return manifest, item


def test_task_exclusions_preserve_grid_zero_and_continuous_diagnostic(tmp_path):
    case = make_case(tmp_path)
    manifest, item = declared_exclusion(case)
    write_json(case["manifest"], manifest)
    files, _ = build(case)
    columns = read_csv_from_bytes(files["tables/task_scope.csv"])
    assert len(columns) == 3
    row = next(row for row in columns if row["task_id"] == "task-b")
    assert row["excluded_volume_mm3"] == "1.5"
    assert row["grid_excluded_volume_mm3"] == "1.5"
    assert row["grid_exclusion_tolerance_mm3"] == "0"
    assert row["continuous_exclusion_diagnostic_mm3"] == "1.71681469282"
    assert row["geometry_source_kind"] == "analytic"
    assert "NA" in next(row for row in columns if row["task_id"] == "task-a").values()
    scope_claim = next(c for c in claims(files) if c["row_id"] == "scope/task-b")
    assert scope_claim["cells"]["continuous_exclusion_diagnostic_mm3"]["value"] == item[
        "continuous_exclusion_diagnostic_mm3"]
    assert scope_claim["evidence"]["task_hashes"] == [item["task_hash"]]


def test_zero_tolerance_exclusion_mismatch_fails_closed(tmp_path):
    case = make_case(tmp_path)
    manifest, _ = declared_exclusion(case)
    item = next(task for task in manifest["tasks"] if task["task_id"] == "task-b")
    item["grid_excluded_volume_mm3"] = 2.0
    write_json(case["manifest"], manifest)
    with pytest.raises(B.PaperArtifactError, match="exclusion"):
        build(case)


def test_reported_refinement_flags_without_link_are_rejected(tmp_path):
    case = make_case(tmp_path)
    report = read_json(case["audit"])
    report["runs"][0]["reported_outcomes"] = {"geometry_pass": False, "completion_pass": False}
    write_json(case["audit"], report)
    with pytest.raises(B.PaperArtifactError, match="refinement/link evidence"):
        build(case)
