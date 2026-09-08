import importlib.util
import json
import shlex
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "direct_feedback_variants", ROOT / "scripts/run_direct_feedback_variants.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def _fixture(tmp_path, monkeypatch):
    iteration_id = "df_p_0001_1000"
    task = tmp_path / "autoresearch/tasks/train_csg"
    work = task / "direct_feedback" / iteration_id
    parent = tmp_path / "runs/parent"
    work.mkdir(parents=True)
    parent.mkdir(parents=True)
    (parent / "args.json").write_text("{}")
    (parent / "reproduce_command.sh").write_text(
        "#!/bin/bash\npython -m algorithms.train_csg --iters 5000 --seed 1 "
        "--target_shape sphere --w_prox 0.0 --w_gouge 4.0 --runs_subdir old "
        "--track --use_feedback --eval\n"
    )
    event = {
        "iteration_id": iteration_id,
        "status": "objective_decided",
        "git_before": "abc123",
        "parent_run": "parent",
        "parent_args_path": "runs/parent/args.json",
        "fixed_controls": {"seed": 1, "iters": 5000, "target_shape": "sphere"},
        "variant_a": {"hypothesis": "small", "changes": {"w_prox": 1.0}, "before": {"w_prox": 0.0}},
        "variant_b": {"hypothesis": "large", "changes": {"w_prox": 3.0}, "before": {"w_prox": 0.0}},
    }
    (work / "event.json").write_text(json.dumps(event))
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "TASK", task)
    monkeypatch.setattr(runner, "ITERATIONS", task / "direct_feedback")
    monkeypatch.setattr(runner, "EVENT_STORE", task / "direct_feedback_events.jsonl")
    monkeypatch.setattr(
        runner.subprocess, "check_output",
        lambda cmd, **k: "" if "status" in cmd else "abc123\n",
    )
    return iteration_id


def test_plan_clones_parent_and_changes_only_loss_and_output(tmp_path, monkeypatch):
    iteration_id = _fixture(tmp_path, monkeypatch)
    plan, work = runner.build_plan(iteration_id)
    a = runner._option_map(shlex.split(plan["variant_a"]["command"]))
    b = runner._option_map(shlex.split(plan["variant_b"]["command"]))
    assert a["--w_prox"] == ["1.0"]
    assert b["--w_prox"] == ["3.0"]
    assert a["--seed"] == b["--seed"] == ["1"]
    assert "--track" not in a and "--use_feedback" not in a
    assert a["--no-track"] == [True]
    assert plan["allowed_command_differences"] == ["--runs_subdir", "--w_prox"]
    assert (work / "launch_plan.json").is_file()


def test_plan_rejects_code_drift(tmp_path, monkeypatch):
    iteration_id = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner.subprocess, "check_output",
        lambda cmd, **k: "" if "status" in cmd else "different\n",
    )
    with pytest.raises(ValueError, match="code drift"):
        runner.build_plan(iteration_id)


def test_execute_requires_explicit_gpu(monkeypatch):
    monkeypatch.setattr("sys.argv", ["runner", "--iteration-id", "x", "--execute"])
    with pytest.raises(SystemExit, match="--gpu is required"):
        runner.main()
