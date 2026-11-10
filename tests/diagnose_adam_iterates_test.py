"""Focused record-selection checks; no simulator or campaign execution."""
import numpy as np
import pytest

from scripts.diagnose_adam_iterates import DiagnosticError, _validation_points


def _history():
    return {"history": [
        {"iteration": index, "theta": [(index - 1) / 100.0, 0.0],
         "theta_after_step": [index / 100.0, 0.005],
         "k": 20.0 + index, "valid": True}
        for index in range(1, 25)
    ]}


def test_validation_uses_exact_post_step_theta_and_candidate_identity():
    points = _validation_points(_history(), "adam-test", {
        "validation_interval": 4, "max_validation_calls": 6,
        "candidate_evaluations": 24,
    })
    assert [point["iteration"] for point in points] == [4, 8, 12, 16, 20, 24]
    assert [point["original_candidate_id"] for point in points] == [
        f"adam-test:validate:{number}" for number in range(1, 7)]
    np.testing.assert_allclose(points[0]["theta"], [0.04, 0.005])
    assert points[0]["step_l2"] == pytest.approx(np.sqrt(0.01 ** 2 + 0.005 ** 2))
    # The first row's pre-step zero is the actual CAM initialization.
    np.testing.assert_array_equal(points[-1]["pre_step_theta"], [0.23, 0.0])


def test_incomplete_validation_theta_fails_closed():
    document = _history()
    document["history"][3].pop("theta_after_step")
    with pytest.raises(DiagnosticError, match="validation 1"):
        _validation_points(document, "adam-test", {
            "validation_interval": 4, "max_validation_calls": 6,
            "candidate_evaluations": 24,
        })
