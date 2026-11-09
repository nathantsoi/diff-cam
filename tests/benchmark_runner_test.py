"""T4.1a runner tests: frozen plan first, then small actual subprocess evidence."""

import json
import subprocess
import sys
from pathlib import Path

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
    return json.loads(CONFIG.read_text())


def _named(root, run_id):
    return C.loads((root / run_id / "run.json").read_text())


def test_condition_ids_are_deterministic_and_manifest_nominal_once(tmp_path):
    config = _base()
    config["experiment"] = "pilot"
    first = runner._plan_entries(config, runner._context(config, tmp_path)[0], "dev", None)
    second = runner._plan_entries(config, runner._context(config, tmp_path)[0], "dev", None)
    assert len(first) == len(second) == 9
    assert [entry["run_id"] for entry in first] == [entry["run_id"] for entry in second]
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
    assert planned["planned_run_count"] == 9
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
