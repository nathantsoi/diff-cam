import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "direct_feedback_retry", ROOT / "scripts/prepare_direct_feedback_retry.py"
)
retry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(retry)


def _source():
    return {
        "status": "weak_intervention", "iteration_id": "df_x", "parent_run": "parent",
        "parent_side": "a", "parent_args_path": "runs/parent/args.json",
        "fixed_controls": {"seed": 1}, "raw_choice": "a", "raw_critique": "quicker",
        "pair_snapshot": {"id": "p_1"},
        "variant_a": {"before": {"w_break": .001},
                      "effective_loss": {"w_time": .001, "w_break": .0005}},
    }


def test_retry_changes_only_direct_time_loss():
    event = retry.build_retry(
        _source(), {"meaningful_difference": {"retry_recommendation": {
            "action": "redesign_contrast_once"}}}, "abc", .02, .2)
    assert event["variant_a"]["changes"] == {"w_time": .02}
    assert event["variant_b"]["changes"] == {"w_time": .2}
    assert event["variant_a"]["effective_loss"]["w_break"] == .001
    assert event["git_before"] == "abc"


def test_retry_requires_ordered_bounded_values():
    with pytest.raises(ValueError, match="conservative"):
        retry.build_retry(
            _source(), {"meaningful_difference": {"retry_recommendation": {
                "action": "redesign_contrast_once"}}}, "abc", .2, .02)
