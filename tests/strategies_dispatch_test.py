# SPDX-License-Identifier: MIT
"""Integration tests for the unified clean-room CAM strategy dispatcher (`cam.strategies`)."""

from __future__ import annotations

import dataclasses
import pytest

from benchmarks import contracts as C
from benchmarks.milling_suite import build_milling_suite
from cam.strategies import (
    AdaptiveConfig,
    ContourConfig,
    FacingConfig,
    PocketConfig,
    estimate_radial_engagement_profile,
    generate_cleanroom_toolpath,
)


def test_cleanroom_strategy_dispatcher_all_four_families():
    """Verify unified dispatcher produces contract-valid @2 MotionPlans on every family."""
    bundles = build_milling_suite(voxel_mm=0.5, split="dev")
    seen_families: set[str] = set()
    for bundle in bundles:
        if bundle.task.family in seen_families:
            continue
        seen_families.add(bundle.task.family)
        plan = generate_cleanroom_toolpath(bundle.task)
        plan.validate_for_task(bundle.task)
        assert plan.provenance is not None and plan.provenance.startswith("mit_cleanroom:")
        assert plan.provenance.endswith("@2")
        assert len(plan.segments) >= 4
        if bundle.task.family == "adaptive_channel":
            ae_prof = estimate_radial_engagement_profile(plan, bundle.task, pitch_mm=0.5)
            assert ae_prof.shape == (len(plan.segments),)
    assert seen_families == {
        "facing_open_boss",
        "pocket_island",
        "adaptive_channel",
        "contour_profile",
    }


def test_cleanroom_strategy_dispatcher_rejects_mismatched_config_and_unknown_family():
    bundle = build_milling_suite(voxel_mm=0.5, families=("facing_open_boss",), split="dev")[0]
    with pytest.raises(TypeError, match="Expected FacingConfig"):
        generate_cleanroom_toolpath(bundle.task, config=PocketConfig())

    unknown_task = dataclasses.replace(bundle.task, family="unknown_5axis_swarf")
    with pytest.raises(ValueError, match="Unknown task family"):
        generate_cleanroom_toolpath(unknown_task)

    # Matching explicit configs succeed
    plan = generate_cleanroom_toolpath(bundle.task, config=FacingConfig(raster_angle_deg=30.0))
    plan.validate_for_task(bundle.task)
    assert isinstance(AdaptiveConfig(), AdaptiveConfig)
    assert isinstance(ContourConfig(), ContourConfig)
