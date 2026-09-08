import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


agent = _module("direct_feedback_agent", "scripts/direct_feedback_agent.py")
server = _module("serve_web_https", "scripts/serve_web_https.py")


def test_extract_json_accepts_fenced_response():
    assert agent._extract_json('```json\n{"ok": true}\n```') == {"ok": True}


def test_variant_validation_builds_effective_loss():
    base = {key: 0.0 for key in agent.LOSS_BOUNDS}
    result = agent._validate_variant(
        {"hypothesis": "reduce air", "changes": {"w_air_time": 0.003}},
        base,
        "variant_a",
    )
    assert result["before"] == {"w_air_time": 0.0}
    assert result["effective_loss"]["w_air_time"] == 0.003


@pytest.mark.parametrize(
    "changes",
    [
        {"learning_rate": 0.1},
        {"w_air_time": 9.0},
        {"w_air_time": 0.0},
        {"w_air": 1.0, "w_time": 0.01, "w_break": 0.1},
    ],
)
def test_variant_validation_rejects_unbounded_or_unchanged_edits(changes):
    base = {key: 0.0 for key in agent.LOSS_BOUNDS}
    with pytest.raises(ValueError):
        agent._validate_variant({"hypothesis": "bad", "changes": changes}, base, "variant")


def test_answer_is_immutable_and_requires_critique_for_agent_queue(tmp_path):
    store = tmp_path / "autoresearch/tasks/train_csg/pairwise.json"
    store.parent.mkdir(parents=True)
    store.write_text(json.dumps([{"id": "p_0001", "answer": None}]))

    first = server.record_pair_answer(tmp_path, "p_0001", "a", "less air")
    second = server.record_pair_answer(tmp_path, "p_0001", "b", "overwrite")
    assert first["answer"] == "a"
    assert first["direct_feedback_status"] == "queued"
    assert second["answer"] == "a"
    assert second["note"] == "less air"
    assert server.update_pair_note(tmp_path, "p_0001", "late edit") is None


def test_process_consumes_raw_pair_and_writes_append_only_event(tmp_path, monkeypatch):
    task = tmp_path / "autoresearch/tasks/train_csg"
    task.mkdir(parents=True)
    pair = {
        "id": "p_0042", "run_a": "run-a", "run_b": "run-b",
        "answer": "a", "answer_ts": 1234.5, "note": "reduce late air cutting",
        "display_order": ["a", "b"],
    }
    pair_store = task / "pairwise.json"
    pair_store.write_text(json.dumps([pair]))
    for name in ("run-a", "run-b"):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        args = {key: 0.0 for key in agent.LOSS_BOUNDS}
        args.update({"seed": 1, "iters": 50, "max_steps": 32, "target_shape": "sphere"})
        (run / "args.json").write_text(json.dumps(args))
        (run / "metrics.json").write_text(json.dumps({"hard_dice": 0.6}))

    captured = {}
    def fake_model(payload, base_url, model):
        captured.update(payload)
        decision = {
            "interpretation": {"target_behavior": "less late air", "rationale": "penalize it", "uncertainty": "proxy"},
            "variant_a": {"hypothesis": "small", "changes": {"w_air_late": 0.002}},
            "variant_b": {"hypothesis": "large", "changes": {"w_air_late": 0.006}},
        }
        return decision, json.dumps(decision)

    monkeypatch.setattr(agent, "REPO", tmp_path)
    monkeypatch.setattr(agent, "TASK", task)
    monkeypatch.setattr(agent, "PAIR_STORE", pair_store)
    monkeypatch.setattr(agent, "EVENT_STORE", task / "direct_feedback_events.jsonl")
    monkeypatch.setattr(agent, "ITERATIONS", task / "direct_feedback")
    monkeypatch.setattr(agent, "_call_model", fake_model)
    monkeypatch.setattr(agent.subprocess, "check_output", lambda *a, **k: "deadbeef\n")

    event = agent.process("p_0042", "http://local/v1", "local-model")
    assert captured["raw_critique"] == "reduce late air cutting"
    assert event["variant_a"]["changes"] == {"w_air_late": 0.002}
    assert event["variant_b"]["changes"] == {"w_air_late": 0.006}
    lines = (task / "direct_feedback_events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["raw_critique"] == "reduce late air cutting"
