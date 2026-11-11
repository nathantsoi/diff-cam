# SPDX-License-Identifier: MIT
"""Unit and integration tests for clean-room tool library, TAMP multi-tool CAM, and test-time overfitting."""

from __future__ import annotations

import pytest

from algorithms.chipload_opt import (
    SubsegmentedCWEModel,
    optimize_chipload_trajectory,
    rollout_chipload_policy,
    train_chipload_shac,
)
from benchmarks.milling_suite import build_milling_suite
from cam.feeds_and_speeds import compute_holemaking_feeds_and_speeds
from cam.tamp_planner import (
    HoleBlueprint,
    PocketBlueprint,
    PrismaticPartBlueprint,
    build_stl_from_blueprint,
    execute_tamp_pipeline,
    load_stl_mesh,
    recognize_part_from_stl,
)
from cam.tool_library import DEFAULT_TOOL_LIBRARY


def test_cleanroom_tool_library_completeness_and_presets() -> None:
    summary = DEFAULT_TOOL_LIBRARY.summary()
    assert summary["license"] == "MIT"
    assert int(summary["library_groups"]) == 15
    assert int(summary["entity_types"]) == 32
    assert int(summary["total_records"]) >= 600
    assert int(summary["holder_records"]) == 72

    # Check metric & inch drill matching and endmill querying
    d5 = DEFAULT_TOOL_LIBRARY.match_drill(5.0, hole_depth_mm=10.0)
    assert d5 is not None
    assert abs(d5.diameter_mm - 5.0) <= 0.05
    preset_d5 = DEFAULT_TOOL_LIBRARY.compute_tool_preset(d5, "al_6061_t6", "drill", doc_mm=15.0)
    assert preset_d5["peck_required"] is True
    assert preset_d5["num_pecks"] >= 2

    em_list = DEFAULT_TOOL_LIBRARY.query_endmills(min_diameter_mm=3.0, max_diameter_mm=10.0)
    assert len(em_list) >= 5
    asm = em_list[0].to_tool_assembly(tool_number=2)
    assert asm.tool_number == 2
    assert asm.cutter_radius_mm == pytest.approx(0.5 * em_list[0].diameter_mm, rel=1e-4)


def test_holemaking_feeds_and_speeds_physics() -> None:
    shallow = compute_holemaking_feeds_and_speeds(
        "al_6061_t6", "carbide_coated", diameter_mm=6.0, hole_depth_mm=8.0, operation_kind="drill"
    )
    deep = compute_holemaking_feeds_and_speeds(
        "al_6061_t6", "carbide_coated", diameter_mm=6.0, hole_depth_mm=24.0, operation_kind="drill"
    )
    ti = compute_holemaking_feeds_and_speeds(
        "ti_6al4v", "carbide_coated", diameter_mm=6.0, hole_depth_mm=18.0, operation_kind="drill"
    )
    assert shallow.peck_required is False
    assert deep.peck_required is True
    assert deep.num_pecks >= 3
    assert ti.plunge_feed_mm_s < deep.plunge_feed_mm_s


def test_stl_loader_rejects_invalid_and_roundtrips_multifeature_part() -> None:
    with pytest.raises(ValueError):
        load_stl_mesh(b"not an stl")

    bp = PrismaticPartBlueprint(
        part_id="test-bracket",
        title="Test Multi-Tool Bracket",
        outer_min_mm=(-18.0, -14.0, -8.0),
        outer_max_mm=(18.0, 14.0, 0.0),
        stock_top_margin_mm=1.0,
        vise_grip_z_mm=2.0,
        pockets=(
            PocketBlueprint(
                min_xy_mm=(-12.0, -9.0),
                max_xy_mm=(12.0, 9.0),
                floor_z_mm=-3.0,
                corner_fillet_mm=2.0,
                islands_xy_mm=(((-2.5, -2.5), (2.5, 2.5)),),
            ),
        ),
        holes=(
            HoleBlueprint(center_xy_mm=(-15.0, -10.5), diameter_mm=4.0, z_top_mm=0.0, z_bottom_mm=-6.0),
            HoleBlueprint(center_xy_mm=(15.0, 10.5), diameter_mm=4.0, z_top_mm=0.0, z_bottom_mm=-6.0),
        ),
    )
    stl_bytes = build_stl_from_blueprint(bp, pitch_mm=0.25)
    mesh = load_stl_mesh(stl_bytes)
    assert mesh.triangle_count > 100
    assert mesh.signed_volume_mm3 > 1000.0

    feats = recognize_part_from_stl(stl_bytes, stock_top_margin_mm=1.0, vise_grip_z_mm=2.0, pitch_mm=0.25)
    assert feats.top_face_depth_mm == pytest.approx(1.0, abs=0.05)
    assert len(feats.pockets) == 1
    assert len(feats.holes) == 2
    assert all(abs(h.diameter_mm - 4.0) <= 0.5 for h in feats.holes)


def test_end_to_end_tamp_multitool_pipeline_and_haas_gcode() -> None:
    bp = PrismaticPartBlueprint(
        part_id="tamp-unit-00",
        title="Unit TAMP Housing",
        outer_min_mm=(-17.0, -13.0, -7.0),
        outer_max_mm=(17.0, 13.0, 0.0),
        stock_top_margin_mm=1.0,
        vise_grip_z_mm=2.0,
        pockets=(
            PocketBlueprint(
                min_xy_mm=(-11.0, -8.5),
                max_xy_mm=(11.0, 8.5),
                floor_z_mm=-2.5,
                corner_fillet_mm=2.0,
            ),
        ),
        holes=(
            HoleBlueprint(center_xy_mm=(-14.0, 0.0), diameter_mm=4.0, z_top_mm=0.0, z_bottom_mm=-6.0),
            HoleBlueprint(center_xy_mm=(14.0, 0.0), diameter_mm=4.0, z_top_mm=0.0, z_bottom_mm=-6.0),
        ),
    )
    stl_bytes = build_stl_from_blueprint(bp, pitch_mm=0.25)
    res = execute_tamp_pipeline(
        stl_bytes,
        part_id="tamp-unit-00",
        title="Unit TAMP Housing",
        program_number=2099,
        stock_material_id="al_6061_t6",
        opt_iters_per_op=20,
    )
    assert res.unique_tools_used >= 3  # Rough/Face Endmill + Rest Endmill (or Endmill) + Spot Drill + Twist Drill
    assert res.safety_pass is True
    assert res.gouge_volume_mm3 <= 0.5
    assert res.vise_collision is False
    assert res.holder_collision is False
    assert res.rapid_collision is False
    assert res.same_speed_total_time_s <= res.naive_total_time_s + 0.05
    assert res.opt_chipload_total_time_s < res.naive_total_time_s
    assert "T1 M6" in res.haas_ngc_program
    assert "T2 M6" in res.haas_ngc_program
    assert "T3 M6" in res.haas_ngc_program
    assert "G187 P2 E0.05" in res.haas_ngc_program


def test_shac_warmstart_fast_inference_trajectory_overfitting() -> None:
    suite = build_milling_suite(voxel_mm=0.50)
    dev_face = [b for b in suite if b.case.split == "dev" and b.case.family == "facing_open_boss"]
    test_face = [b for b in suite if b.case.split == "test" and b.case.family == "facing_open_boss"][0]

    policy = train_chipload_shac(dev_face, epochs=12, seed=0)
    zs = rollout_chipload_policy(test_face, policy, verify_posted=False)
    mdl = SubsegmentedCWEModel(test_face)
    init_theta, _ = policy.map_to_model_theta(mdl)
    overfit = optimize_chipload_trajectory(
        test_face, num_iters=10, verify_posted=True, initial_theta=init_theta
    )
    assert overfit.posted_eval is not None and overfit.posted_eval["geometry_pass"] is True
    assert overfit.telemetry["summary"]["optimal_hex_pct"] > zs.telemetry["summary"]["optimal_hex_pct"]
