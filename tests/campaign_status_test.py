"""Focus: status reader on synthetic typed records, not campaign evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import contracts as C
from scripts import campaign_status


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def _task():
    return C.example_pocket_task()


def _seed_run(root, task, run_id, method, status, *, seed=0, init="cam", no_improvement=...):
    directory = root / "tasks" / task.task_id / run_id
    directory.mkdir(parents=True, exist_ok=True)
    record = C.RunRecord(run_id=run_id, task_hash=task.task_hash, status=status,
                         artifacts=(C.REQUIRED_RUN_ARTIFACTS if status is C.RunStatus.COMPLETE
                                    else ("history.json",)),
                         source_commit="test-base-hash", dirty_patch_hash=None,
                         posted=tuple(C.PostedProgram(
                             dialect, C.PostStatus.UNSUPPORTED,
                             eval_file=None,
                             reason="synthetic status fixture: no posted evaluation available")
                             for dialect in C.Dialect),
                         failure_reason=(None if status is C.RunStatus.COMPLETE else "synthetic terminal state"))
    if status is C.RunStatus.COMPLETE:
        _write_json(directory / "metrics.json", {})
        (directory / "motion_plan.json").write_text(C.dumps(C.example_motion_plan(task)))
        for name, shape in (("initial_trajectory.npy", (2, 3)), ("trajectory.npy", (2, 3)),
                            ("trajectory_deltas.npy", (0, 3))):
            np.save(directory / name, np.zeros(shape, dtype=np.float64), allow_pickle=False)
        _write_json(directory / "eval_posted.json", {"schema_version": 1, "programs": {}})
    (directory / "run.json").write_text(C.dumps(record))
    _write_json(directory / "config.json", {"task_hash": task.task_hash,
                                            "initialization": {"init": init}})
    _write_json(directory / "history.json", {"entries": []})
    if no_improvement is not ...:
        _write_json(directory / "metrics.json", {"selection": {
            "initial_candidate": {"valid": True},
            "no_improvement_over_initialization": no_improvement}})


def fixture(tmp_path):
    task = _task()
    root = tmp_path / "campaign"
    expected = [
        ("cam-s0", "nominal-cam", 0, C.RunStatus.COMPLETE, "cam"),
        ("fixed-s0", "fixed-adam", 0, C.RunStatus.COMPLETE, "cam"),
        ("hard-s0", "hard-cmaes", 0, C.RunStatus.FAILED, "cam"),
        ("smooth-s0", "smooth-cmaes", 0, C.RunStatus.COMPLETE, "cam"),
        ("smooth-s1", "smooth-cmaes", 1, C.RunStatus.COMPLETE, "cam"),
        ("future-s0", "fixed-adam", 0, C.RunStatus.COMPLETE, "cam"),
        ("straight-s0", "fixed-adam", 0, C.RunStatus.INTERRUPTED, "straight_line"),
    ]
    for run_id, method, seed, status, init in expected:
        if run_id == "future-s0":
            continue  # No run.json yet: this expected invocation is pending.
        no_improvement = {
            "fixed-s0": True, "smooth-s0": False, "smooth-s1": ...,
        }.get(run_id, ...)
        _seed_run(root, task, run_id, method, status, seed=seed,
                  init=init, no_improvement=no_improvement)
    _write_json(root / "manifest.json", {
        "schema_version": 1,
        "tasks": [{"task_id": task.task_id, "family": task.family, "task_hash": task.task_hash,
                   "task_split": task.split.value,
                   "completion_residual_mm3": task.tolerances.completion_residual_mm3}],
        "expected_runs": [{"run_id": run_id, "task_id": task.task_id,
                           "method": method, "seed": seed,
                           "status": "running"} for run_id, method, seed, _status, _init in expected],
        "comparisons": [],
        "campaign": {"source_metadata": {"source_commit": "test-base-hash", "dirty_patch_hash": None}},
        "initialization_cases": [
            {"run_id": run_id, "init": init, "scope": "optimizer"}
            for run_id, _method, _seed, _status, init in expected
        ],
    })
    backups = root / "tasks" / task.task_id / "resume-backups" / "hard-s0-previous"
    backups.mkdir(parents=True)
    _write_json(backups / "history.json", {"entries": []})
    retained = C.RunRecord(
        run_id="hard-s0-previous", task_hash=task.task_hash,
        status=C.RunStatus.FAILED, artifacts=("history.json",),
        failure_reason="synthetic retained failure")
    (backups / "run.json").write_text(C.dumps(retained))
    return root


def test_complete_failed_pending_and_early_selection_rates(tmp_path):
    root = fixture(tmp_path)
    before = {path.relative_to(root): path.read_bytes()
              for path in root.rglob("*") if path.is_file()}
    result = campaign_status.campaign_status(root)
    assert result["expected_runs"] == 7
    assert result["counts"] == {"expected": 7, "complete": 4, "failed": 2,
                                "pending": 1, "invalid": 0, "missing": 0}
    assert result["no_improvement"]["numerator"] == 1
    assert result["no_improvement"]["denominator"] == 2
    assert result["no_improvement"]["value"] == 0.5
    cohort = {(row["method"], row["init"]): row for row in result["conditions"]}
    adam = cohort[("fixed-adam", "cam")]
    assert adam["complete"] == 1 and adam["pending"] == 1
    smooth = cohort[("smooth-cmaes", "cam")]
    assert smooth["complete"] == 2
    assert smooth["no_improvement_count"] == 0
    assert smooth["no_improvement_known"] == 1  # missing metrics is not free improvement
    assert cohort[("fixed-adam", "straight_line")]["failed"] == 1
    nominal = cohort[("nominal-cam", "cam")]
    assert nominal["no_improvement_rate"] is None  # no anchor flag, not 100% success
    assert result["no_improvement"]["unknown"] == 3  # failed, interrupted, missing metric document
    assert cohort[("fixed-adam", "cam")]["no_improvement_cases"][0]["source_commit"] == "test-base-hash"
    assert nominal["no_improvement_known"] == 0  # classic no-improvement rate is undefined, not CAM success
    for path, payload in before.items():
        assert (root / path).read_bytes() == payload


def test_missing_expected_is_pending_and_no_optimistic_rate(tmp_path):
    root = fixture(tmp_path)
    (root / "tasks" / _task().task_id / "smooth-s0/metrics.json").unlink()
    result = campaign_status.campaign_status(root)
    assert result["no_improvement"]["numerator"] == 1
    assert result["no_improvement"]["denominator"] == 1
    assert result["no_improvement"]["value"] == 1.0
    assert any(issue["run_id"] == "smooth-s0" and issue["kind"] == "missing.metrics.json"
               for issue in result["issues"])


def test_malformed_duplicate_and_unexpected_records_are_not_hidden(tmp_path):
    root = fixture(tmp_path)
    task_dir = next((root / "tasks").iterdir())
    (task_dir / "smooth-s1").joinpath("metrics.json").write_text("{not-json")
    duplicate = root / "fixed-s0"
    duplicate.mkdir()
    (duplicate / "run.json").write_bytes((task_dir / "fixed-s0/run.json").read_bytes())
    unexpected = root / "rogue-run"
    unexpected.mkdir()
    (unexpected / "run.json").write_text("an unrelated file")
    result = campaign_status.campaign_status(root)
    kinds = [issue["kind"] for issue in result["issues"]]
    assert "duplicate_run_id" in kinds
    assert "unexpected_run" in kinds
    assert "invalid.metrics.json" in kinds
    counts = {(row["method"], row["init"]): row for row in result["conditions"]}
    assert counts[("fixed-adam", "cam")]["invalid"] == 1
    assert counts[("fixed-adam", "cam")]["complete"] == 0
    assert counts[("fixed-adam", "cam")]["no_improvement_count"] == 0


def test_missing_input_fails_closed(tmp_path):
    with pytest.raises(campaign_status.StatusError):
        campaign_status.campaign_status(tmp_path / "nonexistent")


def test_cli_json_and_strict_status(tmp_path, capsys):
    result = campaign_status.main([str(fixture(tmp_path)), "--json"])
    document = json.loads(capsys.readouterr().out)
    assert result == 0
    assert document["expected_runs"] == 7
    assert document["provisional"] is True


def test_cli_strict_returns_two_on_duplicate_or_unknown(tmp_path, capsys):
    root = fixture(tmp_path)
    directory = root / "fixed-s0"
    directory.mkdir()
    (directory / "run.json").write_text("not a record")
    (root / "rogue").mkdir(exist_ok=True)
    (root / "rogue" / "run.json").write_text("another unexpected file")
    exit_code = campaign_status.main([str(root), "--json", "--strict"])
    assert exit_code == 2
    document = json.loads(capsys.readouterr().out)
    assert any(issue["kind"] == "duplicate_run_id" for issue in document["issues"])
    assert any(issue["kind"] == "unexpected_run" for issue in document["issues"])
