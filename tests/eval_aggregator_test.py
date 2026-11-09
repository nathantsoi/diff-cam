"""Synthetic contract-record tests for T4.0 auditing and task-clustered statistics."""

import dataclasses
import itertools
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from benchmarks import audit as A
from benchmarks import contracts as C


def _task(task_id="task-a", family="pocket"):
    return dataclasses.replace(C.example_pocket_task(), task_id=task_id, family=family)


def _tasks(*specs):
    return {task_id: _task(task_id, family) for task_id, family in specs}


def _manifest(tasks, expected, metrics=("cycle_time_s",)):
    return {
        "schema_version": 1,
        "tasks": [
            {
                "task_id": task.task_id, "family": task.family, "task_hash": task.task_hash,
                "completion_residual_mm3": task.tolerances.completion_residual_mm3,
                "removal_required": True,
            }
            for task in tasks.values()
        ],
        "expected_runs": [
            {"run_id": run_id, "task_id": task_id, "method": method, "seed": seed}
            for run_id, task_id, method, seed in expected
        ],
        "comparisons": [
            {
                "name": "candidate_vs_cam", "method": "candidate", "baseline": "cam",
                "dialect": "rs274", "metrics": list(metrics),
            }
        ],
    }


def _metrics(cycle=5.0, removed=20.0, residual=1.0):
    return {
        "removed_volume_mm3": C.Metric(removed, "mm^3"),
        "residual_volume_mm3": C.Metric(residual, "mm^3"),
        "gouge_volume_mm3": C.Metric(0.0, "mm^3"),
        "max_surface_error_mm": C.Metric(0.05, "mm"),
        "cycle_time_s": C.Metric(cycle, "s"),
        "air_fraction": C.Metric(0.25, "1"),
    }


def _write_run(root, run_id, task, *, cycle=5.0, removed=20.0, residual=1.0,
               coverage=C.Coverage.COMPLETE, safety=True, geometry=True,
               nc_bytes=b"G1 X1 F10\n", seed=0, method="candidate"):
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True)
    nc_hash = C.nc_hash(nc_bytes)
    metrics = _metrics(cycle, removed, residual)
    evaluation = C.EvaluationResult(
        evaluator_id="synthetic/test", evaluator_revision="synthetic-eval/1",
        task_hash=task.task_hash, tool_setup_hash=task.tool_setup_hash,
        discretization={"voxel_mm": 0.5}, coverage=coverage,
        coverage_reasons=() if coverage == C.Coverage.COMPLETE else ("synthetic gap",),
        metrics=metrics,
        safety_pass=safety if coverage == C.Coverage.COMPLETE else None,
        geometry_pass=geometry, nc_hash=nc_hash, dialect=C.Dialect.RS274,
    )
    posted = (
        C.PostedProgram(C.Dialect.RS274, C.PostStatus.SUPPORTED, nc_hash=nc_hash,
                        eval_file="eval_posted_rs274.json"),
        C.PostedProgram(C.Dialect.HAAS, C.PostStatus.UNSUPPORTED, reason="synthetic unsupported"),
    )
    artifacts = C.REQUIRED_RUN_ARTIFACTS + ("rs274.nc", "eval_posted_rs274.json")
    record = C.RunRecord(run_id=run_id, task_hash=task.task_hash, status=C.RunStatus.COMPLETE,
                         artifacts=artifacts, posted=posted)
    for name, value in (
        ("config.json", {"task_hash": task.task_hash, "method": method, "seed": seed}),
        ("history.json", {"evaluations": []}),
        ("metrics.json", {"wall_time_s": 1.0}),
    ):
        (run_dir / name).write_text(json.dumps(value, allow_nan=False))
    for name in ("initial_trajectory.npy", "trajectory.npy", "trajectory_deltas.npy"):
        np.save(run_dir / name, np.zeros((2, 3), dtype=np.float64))
    (run_dir / "motion_plan.json").write_text(C.dumps(C.example_motion_plan(task)))
    (run_dir / "rs274.nc").write_bytes(nc_bytes)
    (run_dir / "eval_posted_rs274.json").write_text(C.dumps(evaluation))
    index = {
        "schema_version": 1,
        "programs": {
            "rs274": {"dialect": "rs274", "status": "supported", "nc_hash": nc_hash,
                      "eval_file": "eval_posted_rs274.json"},
            "haas": {"dialect": "haas", "status": "unsupported", "nc_hash": None, "eval_file": None},
        },
    }
    (run_dir / "eval_posted.json").write_text(json.dumps(index, indent=1))
    (run_dir / "run.json").write_text(C.dumps(record))
    return run_dir, record, evaluation


def _write_failed_run(root, run_id, task, status=C.RunStatus.FAILED):
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True)
    record = C.RunRecord(run_id=run_id, task_hash=task.task_hash, status=status,
                         artifacts=("history.json",), failure_reason=f"synthetic {status.value}")
    (run_dir / "history.json").write_text(json.dumps({"evaluations": []}))
    (run_dir / "run.json").write_text(C.dumps(record))
    return run_dir


def _issue_codes(run):
    return {issue["code"] for issue in run.issues} | {
        issue["code"] for dialect in (run.dialects or {}).values() for issue in dialect.issues
    }


def _overall(report, metric="cycle_time_s"):
    row = next(row for row in report["comparisons"] if row["family"] == "overall")
    return row["metrics"][metric]


def _binomial_probability(n, k, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def _tail_root(increasing, at_least, n, k, target=0.025):
    low, high = 0.0, 1.0
    for _ in range(100):
        mid = (low + high) / 2
        indices = range(k, n + 1) if at_least else range(0, k + 1)
        probability = sum(_binomial_probability(n, index, mid) for index in indices)
        if (probability > target) == increasing:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def test_clopper_pearson_known_values():
    zero = A.clopper_pearson(0, 10)
    all_ten = A.clopper_pearson(10, 10)
    three = A.clopper_pearson(3, 10)
    assert zero["lower"] == 0.0
    assert zero["upper"] == pytest.approx(1.0 - 0.025 ** (1 / 10), rel=1e-12)
    assert all_ten["upper"] == 1.0
    assert all_ten["lower"] == pytest.approx(0.025 ** (1 / 10), rel=1e-12)
    assert three["lower"] == pytest.approx(_tail_root(True, True, 10, 3), abs=1e-12)
    assert three["upper"] == pytest.approx(_tail_root(False, False, 10, 3), abs=1e-12)
    assert A.clopper_pearson(0, 0) is None
    with pytest.raises(A.AuditInputError):
        A.clopper_pearson(1, 0, confidence=1.0)


def test_paired_sign_flip_one_task_null_ci_and_exact_counts():
    single = A._paired_stats([-2.0], [0.0], ["pocket"], "single", 1, 100, 0.95)
    assert single["paired_tasks"] == 1
    assert single["interval"] is None
    assert "insufficient independent tasks" in single["interval_null_reason"]
    assert single["bootstrap_p"] == 1.0
    assert single["sign_flip_method"] == "exact_paired_sign_flip"
    assert (single["sign_flip_extreme"], single["sign_flip_total"]) == (2, 2)
    assert "fewer than 5" in single["small_n_warning"]

    # All eight sign assignments for [-2,-2,-2]: only the identity and its mirror
    # attain |mean| >= 2, hence exactly 2/8 = 0.25, never 1/(B+1).
    identical = A._paired_stats([-2.0] * 3, [0.0] * 3, ["pocket"] * 3,
                                "identical", 1, 100, 0.95)
    assert identical["bootstrap_p"] == 0.25
    assert (identical["sign_flip_extreme"], identical["sign_flip_total"]) == (2, 8)

    # [5,4,-1]: means 8/3,10/3,0,2/3 and their negatives. Four of the eight
    # assignments reach |mean| >= 8/3, hence p = 4/8 = 1/2.
    hand = A._paired_stats([5.0, 4.0, -1.0], [0.0] * 3, ["pocket"] * 3,
                           "hand", 1, 100, 0.95)
    assert hand["mean_difference"] == pytest.approx(8 / 3)
    assert hand["bootstrap_p"] == 0.5
    assert (hand["sign_flip_extreme"], hand["sign_flip_total"]) == (4, 8)


def test_monte_carlo_sign_flip_above_enumeration_limit_is_seeded():
    values = [float(index) for index in range(17)]
    kwargs = (["pocket"] * 17, "over-limit", 42, 100, 0.95)
    first = A._paired_stats(values, [0.0] * 17, *kwargs)
    second = A._paired_stats(values, [0.0] * 17, *kwargs)
    assert first["sign_flip_method"] == "seeded_monte_carlo_paired_sign_flip"
    assert first["sign_flip_total"] == 100
    assert first["bootstrap_p"] == second["bootstrap_p"]
    assert first["sign_flip_extreme"] == second["sign_flip_extreme"]


def test_clean_audit_paired_difference_and_markdown(tmp_path):
    tasks = _tasks(("task-a", "pocket"))
    _write_run(tmp_path, "cam-a", tasks["task-a"], cycle=5.0, residual=1.0, method="cam")
    _write_run(tmp_path, "candidate-a", tasks["task-a"], cycle=3.0, residual=1.0)
    expected = [("cam-a", "task-a", "cam", 0), ("candidate-a", "task-a", "candidate", 0)]
    report, _ = A.generate_report(tmp_path, _manifest(tasks, expected), bootstrap_replicates=100)
    assert report["summary"]["auditor_passed"] is True
    assert report["expected_vs_observed"]["exact_id_reconciliation"] is True
    assert report["summary"]["run_status_counts"] == {"complete": 2}
    assert _overall(report)["mean_difference"] == pytest.approx(-2.0)
    assert _overall(report)["paired_tasks"] == 1
    assert _overall(report)["bootstrap_p"] == 1.0
    assert _overall(report)["interval"] is None
    assert _overall(report)["interval_null_reason"] == "insufficient independent tasks (need at least 2)"
    row = next(row for row in report["comparisons"] if row["family"] == "overall")
    assert row["outcomes"]["completion_pass"]["method"]["estimate_method"] == "clopper_pearson_independent_tasks"
    markdown = A.render_markdown(report)
    assert "candidate_vs_cam" in markdown and "-2" in markdown and "NaN" not in markdown
    json.dumps(report, allow_nan=False)


def test_replicating_seeds_does_not_shrink_task_cluster_intervals(tmp_path):
    tasks = _tasks(("task-a", "pocket"), ("task-b", "pocket"))
    values = {
        "task-a": {"candidate": [3.0, 5.0], "cam": [1.0, 1.0]},
        "task-b": {"candidate": [0.0, 2.0], "cam": [0.0, 0.0]},
    }

    def build(root, copies):
        expected = []
        for task_id, methods in values.items():
            for method, numbers in methods.items():
                seeds = list(range(copies))
                for copy_index, seed in enumerate(seeds):
                    run_id = f"{method}-{task_id}-{copy_index}"
                    _write_run(root, run_id, tasks[task_id], cycle=numbers[seed % len(numbers)],
                               method=method, seed=seed)
                    expected.append((run_id, task_id, method, seed))
        return A.generate_report(root, _manifest(tasks, expected), bootstrap_replicates=200)[0]

    twice = build(tmp_path / "twice", 2)
    replicated = build(tmp_path / "replicated", 4)
    for left, right in ((_overall(twice), _overall(replicated)),):
        assert left["mean_difference"] == pytest.approx(2.0)
        assert left["mean_difference"] == right["mean_difference"]
        assert left["interval"] == right["interval"]
        assert left["bootstrap_p"] == right["bootstrap_p"]
        assert left["method_task_availability"]["task-a"]["observations_used"] == 2
        assert right["method_task_availability"]["task-a"]["observations_used"] == 4

    def method_completion(report):
        row = next(row for row in report["comparisons"] if row["family"] == "overall")
        return row["outcomes"]["completion_pass"]["method"]

    left, right = method_completion(twice), method_completion(replicated)
    assert left["estimate_method"] == right["estimate_method"] == "cluster_bootstrap_over_tasks"
    assert left["interval"] == right["interval"]


def test_no_cut_fails_completion_and_cannot_contaminate_efficiency_metrics(tmp_path):
    tasks = _tasks(("task-a", "pocket"))
    _write_run(tmp_path, "cam-a", tasks["task-a"], cycle=5.0, removed=20, residual=1, method="cam")
    # Deliberately inconsistent producer flags ensure the auditor cannot rely on them.
    _write_run(tmp_path, "candidate-a", tasks["task-a"], cycle=0.1, removed=0, residual=0,
               safety=True, geometry=True)
    expected = [("cam-a", "task-a", "cam", 0), ("candidate-a", "task-a", "candidate", 0)]
    report, _ = A.generate_report(tmp_path, _manifest(tasks, expected), bootstrap_replicates=100)
    summary = _overall(report)
    assert summary["paired_tasks"] == 0
    availability = summary["method_task_availability"]["task-a"]
    assert availability["value"] is None
    assert availability["exclusions"] == {"not_efficiency_eligible": 1}
    row = next(row for row in report["comparisons"] if row["family"] == "overall")
    assert row["outcomes"]["completion_pass"]["method"]["task_mean_rate"] == 0.0
    assert row["outcomes"]["efficiency_eligible"]["method"]["task_mean_rate"] == 0.0
    assert "no-cut" in report["policy"]["completion_rule"]


def test_incomplete_coverage_is_unknown_not_completion_success(tmp_path):
    tasks = _tasks(("task-a", "pocket"))
    _write_run(tmp_path, "cam-a", tasks["task-a"], method="cam")
    _write_run(tmp_path, "candidate-a", tasks["task-a"], coverage=C.Coverage.INCOMPLETE,
               geometry=True, residual=0.0)
    expected = [("cam-a", "task-a", "cam", 0), ("candidate-a", "task-a", "candidate", 0)]
    report, _ = A.generate_report(tmp_path, _manifest(tasks, expected), bootstrap_replicates=100)
    row = next(row for row in report["comparisons"] if row["family"] == "overall")
    outcomes = row["outcomes"]["completion_pass"]["method"]
    assert outcomes["known_task_units"] == 0
    assert outcomes["task_mean_rate"] is None
    assert _overall(report)["paired_tasks"] == 0
    assert report["summary"]["dialect_status_counts"]["incomplete"] == 1


def test_missing_failed_blocked_and_unexpected_ids_reconcile_without_imputation(tmp_path):
    tasks = _tasks(("task-a", "pocket"), ("task-b", "pocket"))
    _write_run(tmp_path, "cam-a", tasks["task-a"], method="cam")
    _write_failed_run(tmp_path, "candidate-a", tasks["task-a"], C.RunStatus.FAILED)
    _write_failed_run(tmp_path, "candidate-b", tasks["task-b"], C.RunStatus.BLOCKED)
    _write_run(tmp_path, "surprise", tasks["task-a"])
    expected = [
        ("cam-a", "task-a", "cam", 0), ("candidate-a", "task-a", "candidate", 0),
        ("candidate-b", "task-b", "candidate", 0), ("cam-b", "task-b", "cam", 0),
    ]
    report, _ = A.generate_report(tmp_path, _manifest(tasks, expected), bootstrap_replicates=100)
    reconciliation = report["expected_vs_observed"]
    assert reconciliation["missing_run_ids"] == ["cam-b"]
    assert reconciliation["unexpected_run_ids"] == ["surprise"]
    assert reconciliation["exact_id_reconciliation"] is False
    counts = report["summary"]["run_status_counts"]
    assert counts == {"blocked": 1, "complete": 2, "failed": 1, "missing": 1}
    assert report["summary"]["auditor_passed"] is False


def test_duplicate_run_ids_are_rejected(tmp_path):
    tasks = _tasks(("task-a", "pocket"))
    _write_run(tmp_path / "one", "duplicate", tasks["task-a"])
    _write_run(tmp_path / "two", "duplicate", tasks["task-a"])
    manifest = _manifest(tasks, [("duplicate", "task-a", "candidate", 0)])
    # The comparison also names cam, so add an expected-but-missing baseline to make the manifest valid.
    manifest["expected_runs"].append({"run_id": "cam-a", "task_id": "task-a", "method": "cam", "seed": 0})
    reports, audit = A.audit_runs(tmp_path, A.load_manifest(manifest))
    assert audit["reconciliation"]["duplicate_run_ids"] == ["duplicate"]
    duplicate = [report for report in reports if report["run_id"] == "duplicate"]
    assert len(duplicate) == 2
    assert all(report["duplicate_id"] and not report["usable_for_statistics"] for report in duplicate)


def test_hash_corruption_and_mismatched_evaluation_identity_are_rejected(tmp_path):
    tasks = _tasks(("task-a", "pocket"))
    run_dir, record, _ = _write_run(tmp_path, "candidate-a", tasks["task-a"])
    (run_dir / "rs274.nc").write_bytes(b"G1 X2\n")
    audited = A._audit_run(run_dir / "run.json")
    assert {"nc_hash_mismatch", "eval_actual_nc_mismatch"} <= _issue_codes(audited)

    run_dir_2, _, _ = _write_run(tmp_path, "candidate-b", tasks["task-a"])
    evaluation_path = run_dir_2 / "eval_posted_rs274.json"
    raw = json.loads(evaluation_path.read_text())
    raw["record"]["task_hash"] = "sha256:" + "0" * 64
    evaluation_path.write_text(json.dumps(raw))
    audited_2 = A._audit_run(run_dir_2 / "run.json")
    assert "eval_task_hash_mismatch" in _issue_codes(audited_2)
    manifest = A.load_manifest(_manifest(tasks, [
        ("candidate-a", "task-a", "candidate", 0), ("cam-a", "task-a", "cam", 0),
    ]))
    _, audit = A.audit_runs(tmp_path, manifest)
    assert audit["reconciliation"]["unexpected_run_ids"] == ["candidate-b"]


def test_malformed_json_arrays_run_records_and_unsafe_artifacts_are_rejected(tmp_path):
    tasks = _tasks(("task-a", "pocket"))
    run_dir, _, _ = _write_run(tmp_path, "candidate-a", tasks["task-a"])
    (run_dir / "history.json").write_text('{"x": NaN}')
    np.save(run_dir / "trajectory.npy", np.array([np.nan]))
    raw = json.loads((run_dir / "run.json").read_text())
    raw["record"]["artifacts"].append("../escape")
    (run_dir / "run.json").write_text(json.dumps(raw))
    audited = A._audit_run(run_dir / "run.json")
    assert {"malformed_artifact", "malformed_array", "invalid_artifact_path"} <= _issue_codes(audited)

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "run.json").write_text("{broken")
    broken = A._audit_run(bad_dir / "run.json")
    assert broken.status == "invalid" and _issue_codes(broken) == {"malformed_run_json"}


def test_holm_and_input_validation():
    rows = [{"bootstrap_p": 0.04}, {"bootstrap_p": 0.01}, {"bootstrap_p": None}]
    A._apply_holm(rows)
    assert rows[0]["holm_p"] == pytest.approx(0.04)
    assert rows[1]["holm_p"] == pytest.approx(0.02)
    assert rows[2].get("holm_p") is None
    tasks = _tasks(("task-a", "pocket"))
    manifest = A.load_manifest(_manifest(tasks, [
        ("cam-a", "task-a", "cam", 0), ("candidate-a", "task-a", "candidate", 0),
    ]))
    with pytest.raises(A.AuditInputError):
        A.aggregate_evaluations(manifest, {}, bootstrap_replicates=100.5)
    with pytest.raises(A.AuditInputError):
        A.aggregate_evaluations(manifest, {}, bootstrap_seed=True)


def test_missing_comparisons_and_null_metrics_render_as_na(tmp_path):
    tasks = _tasks(("task-a", "pocket"))
    expected = [("cam-a", "task-a", "cam", 0), ("candidate-a", "task-a", "candidate", 0)]
    report, _ = A.generate_report(tmp_path, _manifest(tasks, expected), bootstrap_replicates=100)
    summary = _overall(report)
    assert summary["paired_tasks"] == 0 and summary["mean_difference"] is None
    markdown = A.render_markdown(report)
    assert "candidate_vs_cam" in markdown and "NA" in markdown


def test_recorded_failures_count_in_outcomes_metric_drops_and_complete_data(tmp_path):
    tasks = _tasks(("a", "pocket"), ("b", "pocket"))
    expected = [
        ("c-a-0", "a", "candidate", 0), ("c-a-1", "a", "candidate", 1),
        ("c-b-0", "b", "candidate", 0), ("c-b-1", "b", "candidate", 1),
        ("m-a-0", "a", "cam", 0), ("m-a-1", "a", "cam", 1),
        ("m-b-0", "b", "cam", 0), ("m-b-1", "b", "cam", 1),
    ]
    _write_failed_run(tmp_path, "c-a-0", tasks["a"])
    _write_run(tmp_path, "c-a-1", tasks["a"], cycle=1.0)
    _write_failed_run(tmp_path, "c-b-0", tasks["b"])
    _write_failed_run(tmp_path, "c-b-1", tasks["b"])
    for run_id in ("m-a-0", "m-a-1", "m-b-0", "m-b-1"):
        _write_run(tmp_path, run_id, tasks[run_id[2]], cycle=5.0, method="cam")
    report, _ = A.generate_report(tmp_path, _manifest(tasks, expected), bootstrap_replicates=100)
    row = next(row for row in report["comparisons"] if row["family"] == "overall")
    for outcome in ("safety_pass", "completion_pass", "efficiency_eligible"):
        result = row["outcomes"][outcome]
        assert result["method"]["task_mean_rate"] == pytest.approx(0.25)
        assert result["method"]["known_task_units"] == result["method"]["expected_task_units"] == 2
        assert result["paired_difference"]["mean_difference"] == pytest.approx(-0.75)
        assert result["paired_difference"]["paired_tasks"] == 2
    metric = row["metrics"]["cycle_time_s"]
    assert metric["mean_difference"] == pytest.approx(-4.0)
    assert metric["paired_tasks"] == 1
    assert metric["dropped_because_side_failed_or_ineligible"] == 3
    assert metric["dropped_missing_or_null_evidence"] == 0
    assert metric["exclusions_by_side"]["method"] == {"recorded_noncomplete": 3}
    # Mechanically valid terminal failures do not make the data complete or successful.
    assert report["summary"]["auditor_passed"] is True
    assert report["summary"]["complete_data"] is report["complete_data"] is False
    markdown = A.render_markdown(report)
    assert "Failed/ineligible drops" in markdown
    assert "complete_data (expected run/evaluation artifacts complete and clean; coverage limits still reported): False" in markdown


def test_exact_sign_flip_ties_match_4000_rational_enumerations():
    # Copy of the reviewed regression, now self-contained and intentionally not
    # depending on any external reviewer script/path.
    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(4000):
        n = int(rng.integers(3, 9))
        differences = np.round(rng.uniform(1, 500, size=n), 1)
        k = int(rng.integers(1, n))
        differences[k] = -differences[0] if rng.random() < 0.5 else differences[0]
        result = A._sign_flip_p_value(differences, "k", 1, 1000)
        exact = [Fraction(float(value)) for value in differences]
        observed = abs(sum(exact))
        extreme = sum(
            1 for signs in itertools.product((1, -1), repeat=n)
            if abs(sum(sign * value for sign, value in zip(signs, exact))) >= observed
        )
        mismatches += extreme != result["sign_flip_extreme"]
    assert mismatches == 0


def test_singleton_strata_pool_interval_and_keep_small_sample_warning():
    summary = A._paired_stats([1, 9, -3, 20], [0] * 4, ["p", "s", "t", "u"],
                              "singletons", 1, 100, 0.95)
    assert summary["ci_resampling_method"] == "pooled_tasks_due_to_singleton_stratum"
    assert summary["interval"]["lower"] < summary["interval"]["upper"]
    assert (summary["interval"]["lower"], summary["interval"]["upper"]) != (6.75, 6.75)
    assert "fewer than 5" in summary["small_n_warning"]


def test_single_family_emits_one_holm_input_and_overall_row(tmp_path):
    tasks = _tasks(("task-a", "pocket"))
    _write_run(tmp_path, "cam-a", tasks["task-a"], cycle=5, method="cam")
    _write_run(tmp_path, "candidate-a", tasks["task-a"], cycle=3)
    expected = [("cam-a", "task-a", "cam", 0), ("candidate-a", "task-a", "candidate", 0)]
    report, _ = A.generate_report(tmp_path, _manifest(tasks, expected), bootstrap_replicates=100)
    assert len(report["comparisons"]) == 1
    assert report["comparisons"][0]["family"] == "overall"
    assert report["comparisons"][0]["metrics"]["cycle_time_s"]["holm_p"] == 1.0


def test_monte_carlo_keeps_identity_without_forcing_mirror():
    differences = np.arange(1.0, 18.0)  # 17 tasks selects the Monte Carlo branch.
    result = A._sign_flip_p_value(differences, "manual", 42, 100)
    rng = A._rng(42, "manual|sign-flip")
    signs = rng.choice((-1.0, 1.0), size=(100, 17))
    signs[0, :] = 1.0  # only the stipulated identity, not an artificial second mirror.
    means = np.mean(signs * differences[None, :], axis=1)
    observed = abs(float(np.mean(differences)))
    tolerance = 1e-9 * max(1.0, float(np.sum(np.abs(differences))))
    expected = int(np.count_nonzero(np.abs(means) >= observed - tolerance))
    assert result["sign_flip_extreme"] == expected
    assert result["sign_flip_total"] == 100


def test_cli_writes_json_and_markdown_and_returns_audit_status(tmp_path):
    tasks = _tasks(("task-a", "pocket"))
    _write_run(tmp_path / "runs", "cam-a", tasks["task-a"], cycle=6, method="cam")
    _write_run(tmp_path / "runs", "candidate-a", tasks["task-a"], cycle=4)
    expected = [("cam-a", "task-a", "cam", 0), ("candidate-a", "task-a", "candidate", 0)]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(tasks, expected)))
    output_json, output_md = tmp_path / "out/report.json", tmp_path / "out/report.md"
    command = [
        sys.executable, "scripts/aggregate_eval_results.py",
        "--runs-root", str(tmp_path / "runs"), "--manifest", str(manifest_path),
        "--output-json", str(output_json), "--output-md", str(output_md),
        "--bootstrap-replicates", "100",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_json.read_text())
    assert report["summary"]["auditor_passed"] is True
    assert output_md.exists()
    assert _overall(report)["mean_difference"] == pytest.approx(-2.0)
