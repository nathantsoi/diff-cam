"""Focused D-13/BB1 reach arithmetic and real-initializer construction tests."""
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import contracts as C
from scripts import exp1_reach as reach


CONFIG = reach.ROOT / "configs/campaign_exp1_dev.json"


def test_componentwise_distance_and_reach_ratio_have_declared_units():
    cam = np.asarray([0.5, -1.0, 2.0, 4.0], dtype=np.float64)
    initial = np.asarray([0.0, 1.0, 6.0, 2.0], dtype=np.float64)
    stats = reach._distance(cam, initial)
    assert stats["median_abs_delta_theta_mm"] == 2.0
    assert stats["max_abs_delta_theta_mm"] == 4.0
    assert reach._ratio(stats["max_abs_delta_theta_mm"], 2, 0.005) == 400.0
    assert reach._ratio(0.0, 24, 0.005) == 0.0
    with pytest.raises(ValueError, match="positive"):
        reach._ratio(1.0, 0, 0.005)
    with pytest.raises(ValueError, match="nonempty finite"):
        reach._theta({"scale_mm": 1.0, "bounds": None, "initial_theta": [float("nan")]})


def test_declared_update_count_and_retry_learning_rate_cap():
    config = reach.campaign._load(CONFIG)["template"]
    settings = reach._solver_settings(config)
    assert [(item.method, item.updates, item.lr_max) for item in settings] == [
        ("annealed-adam", 24, 0.005), ("continuation-adam", 24, 0.01)]
    # _budget's additional hard θ0 must not be counted as a 25th Adam update.
    for item in settings:
        budget = reach.runner._budget(config, item.method)
        assert budget.candidate_evaluations == 25
        assert item.updates == budget.candidate_evaluations - 1 == 24


def test_real_dev_initializers_json_table_and_non_overwrite(tmp_path, capsys):
    assert reach.main(("--config", str(CONFIG), "--output-dir", str(tmp_path / "reach"))) == 0
    raw_json, raw_table = (tmp_path / "reach" / reach.RESULT_JSON, tmp_path / "reach" / reach.RESULT_TABLE)
    document = json.loads(raw_json.read_bytes())
    assert document["schema"] == "diffcam-exp1-reach/1"
    assert document["task_count"] == 9 and len(document["tasks"]) == 9
    assert document["arm_count"] == 27
    assert document["source"]["config_sha256"] == C.sha256_bytes(CONFIG.read_bytes())
    assert document["source"]["analysis_script_sha256"] == C.sha256_bytes(
        (reach.ROOT / "scripts/exp1_reach.py").read_bytes())
    assert {task["label"] for task in document["tasks"]} == {
        f"{family}-{index}" for family in ("face", "pocket", "var") for index in ("00", "01", "02")}
    for task in document["tasks"]:
        assert task["task_id"].startswith("procedural-") and "-dev-case-" in task["task_id"]
        assert len(task["arms"]) == 3
        assert [(arm["init"], arm["seed"]) for arm in task["arms"]] == [
            ("straight_line", 0), ("random", 0), ("random", 1)]
        for arm in task["arms"]:
            mean = arm["median_abs_delta_theta_mm"]
            maximum = arm["max_abs_delta_theta_mm"]
            assert np.isfinite([mean, maximum]).all() and mean >= 0 and maximum >= mean
            assert maximum > 0.0
            annealed = arm["reach_ratio_by_method"]["annealed-adam"]
            continuation = arm["reach_ratio_by_method"]["continuation-adam"]
            assert annealed == pytest.approx(maximum / (24 * 0.005), rel=1e-14)
            assert continuation == pytest.approx(maximum / (24 * 0.01), rel=1e-14)
            assert continuation == pytest.approx(annealed / 2.0, rel=1e-14)
            assert arm["theta0_cam_sha256"] != arm["theta0_initializer_sha256"]
            assert arm["initializer_source_motion_hash"]
    summary = reach.json.loads(capsys.readouterr().out)
    assert summary["task_count"] == 9 and summary["arm_count"] == 27
    assert summary["optimizer_or_simulator_launched"] is False
    assert summary["json_sha256"] == C.sha256_bytes(raw_json.read_bytes())
    assert summary["table_sha256"] == C.sha256_bytes(raw_table.read_bytes())
    table = raw_table.read_text()
    assert table.count("| Task | Init | Seed |") == 1
    assert "| Median |Δθ0|" not in table
    assert table.count("\n| face-") == 9
    assert table.count("\n| pocket-") == 9
    assert table.count("\n| var-") == 9
    assert "unreachable within budget/step size" in table
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reach.write_outputs(document, tmp_path / "reach")
