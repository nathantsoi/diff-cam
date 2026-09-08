import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "direct_feedback_evaluation", ROOT / "scripts/evaluate_direct_feedback.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluation)


def test_trajectory_metrics_use_physical_stock_scale(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    np.save(a / "trajectory.npy", np.array([[0, 0, 0], [1, 0, 0]], dtype=float))
    np.save(b / "trajectory.npy", np.array([[0, 0, 0], [0, 0, 0]], dtype=float))
    result = evaluation._trajectory_metrics(a, b, [1, 2, 3])
    assert result["mean_pointwise_displacement_mm"] == 12.7
    assert result["max_pointwise_displacement_mm"] == 25.4


def test_all_gates_are_required(tmp_path, monkeypatch):
    task = tmp_path / "autoresearch/tasks/train_csg"
    work = task / "direct_feedback/df_x"
    parent = tmp_path / "runs/parent"
    run_a = tmp_path / "runs/a"
    run_b = tmp_path / "runs/b"
    for path in (work, parent, run_a, run_b):
        path.mkdir(parents=True)
    base_metrics = {"hard_dice": .8, "gouge": 10, "residual": 20, "air_time": 2,
                    "total_time": 10, "break_prob_any": .001,
                    "air_time_frac": .2, "air_time_frac_late": .1}
    (parent / "metrics.json").write_text(json.dumps(base_metrics))
    (run_a / "metrics.json").write_text(json.dumps(base_metrics))
    metrics_b = dict(base_metrics, total_time=9)
    (run_b / "metrics.json").write_text(json.dumps(metrics_b))
    traj_a = np.zeros((2, 3)); traj_b = np.ones((2, 3))
    np.save(run_a / "trajectory.npy", traj_a); np.save(run_b / "trajectory.npy", traj_b)
    event = {"parent_args_path": "runs/parent/args.json", "fixed_controls": {"stock_size_in": [1,1,1]},
             "variant_a": {"before": {"w_break": .1}, "changes": {"w_break": .05}},
             "variant_b": {"before": {"w_break": .1}, "changes": {"w_break": 0}}}
    (work / "event.json").write_text(json.dumps(event))
    plan = {"execution": "completed", "run_a": "runs/a", "run_b": "runs/b",
            "meaningful_difference_threshold": {"targeted_metric_relative_change_min": .2,
             "trajectory_mean_displacement_mm_min": .5, "hard_dice_drop_max": .03,
             "gouge_relative_increase_max": .1}}
    (work / "launch_plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(evaluation, "REPO", tmp_path)
    monkeypatch.setattr(evaluation, "TASK", task)
    monkeypatch.setattr(evaluation, "ITERATIONS", task / "direct_feedback")
    result, _ = evaluation.evaluate("df_x", "distinguishable")
    assert result["meaningful_difference"]["trajectory_pass"] is True
    assert result["meaningful_difference"]["target_metric_pass"] is False
    assert result["meaningful_difference"]["passed"] is False


def test_retry_explanation_uses_changed_time_loss():
    metrics = {"total_time": 8, "air_time": 1, "hard_dice": .8,
               "gouge": 2, "residual": 3, "break_prob_any": .0001}
    text = evaluation._explanation("b", "w_time", .001, .2, metrics)
    assert "w_time changed from 0.001 to 0.2" in text
    assert "directly increasing the differentiable time penalty" in text


def test_weak_demo_pair_is_explicitly_ineligible(tmp_path, monkeypatch):
    task = tmp_path / "autoresearch/tasks/train_csg"
    work = task / "direct_feedback/df_retry"
    work.mkdir(parents=True)
    (task / "pairwise.json").write_text("[]")
    event = {"status": "weak_intervention", "variant_a": {"changes": {"w_time": .02}},
             "variant_b": {"changes": {"w_time": .2}}}
    result = {"runs": {"a": "runs/x/run_a", "b": "runs/x/run_b"},
              "generated_explanations": {"a": "facts a", "b": "facts b"},
              "meaningful_difference": {"passed": False, "classification": "weak_intervention"}}
    (work / "event.json").write_text(json.dumps(event))
    (work / "phase5_6_evaluation.json").write_text(json.dumps(result))
    monkeypatch.setattr(evaluation, "TASK", task)
    monkeypatch.setattr(evaluation, "ITERATIONS", task / "direct_feedback")
    monkeypatch.setattr(evaluation, "PAIR_STORE", task / "pairwise.json")
    monkeypatch.setattr(evaluation, "EVENT_STORE", task / "direct_feedback_events.jsonl")
    pair = evaluation.enqueue_weak_demo("df_retry")
    assert pair["experimental_evidence_eligible"] is False
    assert pair["meaningful_difference_passed"] is False
    assert pair["presentation_override"] == "weak_intervention_loop_demo"
