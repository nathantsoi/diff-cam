"""D-08 straight-line/random MotionPlan seed tests (no evaluator or campaign)."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from benchmarks import contracts as C
from benchmarks.pilot_task import build_pilot_task, fit_parameterization
from cam import seeds as S


@pytest.fixture(scope="module")
def task():
    return build_pilot_task()


@pytest.fixture(scope="module")
def cam(task):
    return S.seed_for_task(task)


def declaration(*, space="delta", fraction=0.001, reference="stock_xy_extent",
                family="gaussian", detail="iid", root_seed=0):
    return {
        "straight_line_init": {
            "start": "cam_start", "end": "cam_end", "segment_placement": "uniform",
            "feed_and_rapid_entry": "retain limited CAM feed rates and require above-stock rapid entry",
            "same_segment_count_and_k": True, "safe_rapid_entry": True,
            "respect_task_feed_limits": True, "efficiency_not_required": True,
        },
        "random_init": {
            "parameter_space": space,
            "distribution": {"kind": family, "detail": detail},
            "scale": {"fraction": fraction, "reference": {
                "kind": reference,
                "description": "explicit physical reference for deterministic unit tests",
                "units": "mm",
            }},
            "seed_derivation_rule": {
                "algorithm": "sha256_v1", "root_seed": root_seed,
                "input": "task_hash,declaration_hash,root_seed,seed",
                "per_seed_rule": "first_8_bytes_little_endian_uint64",
            },
        },
    }


def source_hash(value):
    payload = json.dumps(value, allow_nan=False, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def positions(plan):
    return np.asarray([segment.start_mm for segment in plan.segments]
                      + [plan.segments[-1].end_mm], dtype=np.float64)


def assert_valid(result, task, cam):
    assert isinstance(result, S.SeedResult)
    assert result.valid, result.invalid_reason
    plan = result.require_plan()
    plan.validate_for_task(task)
    assert len(plan.segments) == len(cam.segments) > 0
    assert plan.initial_state == cam.initial_state == task.initial_state
    assert plan.frame == cam.frame
    assert plan.timing_rule == cam.timing_rule
    actual_fit = fit_parameterization(plan)
    expected_fit = fit_parameterization(cam)
    assert actual_fit.num_steps == expected_fit.num_steps == len(cam.segments)
    assert actual_fit.num_control_points == expected_fit.num_control_points
    assert actual_fit.scale_mm == expected_fit.scale_mm == 1.0
    for actual, source in zip(plan.segments, cam.segments):
        assert actual.kind == source.kind
        assert actual.feed_mm_s == source.feed_mm_s
        assert actual.arc is None
        if actual.feed_mm_s is not None:
            assert 0 < actual.feed_mm_s <= task.limits.max_feed_mm_s
        speed = task.limits.rapid_mm_s if actual.kind == C.SegmentKind.RAPID else actual.feed_mm_s
        assert math.isfinite(speed) and speed > 0
        expect = math.dist(actual.start_mm, actual.end_mm) / speed
        assert actual.duration_s == pytest.approx(expect, abs=1e-12)
    return plan


def test_straight_line_is_structurally_valid_even_when_cam_start_equals_cam_end(task, cam):
    line = S.straight_line_seed(task, declaration())
    plan = assert_valid(line, task, cam)
    # The canonical pilot goes from home to the same home. This is a real
    # collapsed no-init path, not an excuse to invent a CAM-shaped fallback.
    assert plan.initial_state.position_mm == (0.0, 0.0, 10.0)
    assert plan.segments[-1].end_mm == (0.0, 0.0, 10.0)
    assert all(math.dist(s.start_mm, s.end_mm) == 0.0 for s in plan.segments)
    meta = json.loads(line.derivation)
    assert meta["zero_length_segments"] == len(cam.segments)
    assert meta["mode"] == "straight_line"
    assert meta["validation"] == "validate_for_task passed"


def test_explicit_boundary_endpoints_are_not_clipped(task, cam):
    definition = declaration()
    definition["straight_line_init"]["end"] = "machine_mm:(40,30,60)"
    plan = assert_valid(S.straight_line_seed(task, definition), task, cam)
    assert tuple(positions(plan)[0]) == cam.initial_state.position_mm
    assert tuple(positions(plan)[-1]) == (40.0, 30.0, 60.0)
    expected = np.linspace(cam.initial_state.position_mm, (40, 30, 60),
                           len(cam.segments) + 1)
    np.testing.assert_allclose(positions(plan), expected, atol=0.0, rtol=0.0)


def test_random_delta_is_deterministic_and_different_seeds_change_geometry_not_luck(task, cam):
    definition = declaration(fraction=0.0005)
    first = S.random_seed(task, definition, 7)
    other = S.random_seed(task, definition, 8)
    repeated = S.random_seed(task, definition, 7)
    p1 = assert_valid(first, task, cam)
    p2 = assert_valid(other, task, cam)
    same = assert_valid(repeated, task, cam)
    assert p1.to_dict() == same.to_dict()
    assert first.derivation == repeated.derivation
    assert not np.array_equal(positions(p1), positions(p2))


def test_random_theta_zero_reproduces_nominal_cam_displacements(task, cam):
    definition = declaration(space="theta", fraction=0.0, reference="cam_delta_norm")
    result = S.random_seed(task, definition, 11)
    plan = assert_valid(result, task, cam)
    np.testing.assert_allclose(np.asarray(S.seed_deltas_mm(plan)),
                               np.asarray(S.seed_deltas_mm(cam)), atol=1e-12, rtol=0.0)
    assert json.loads(result.derivation)["parameter_map"]["reference_motion"] == "cam"


def test_random_delta_zero_reproduces_declared_straight_line(task, cam):
    definition = declaration(space="delta", fraction=0.0, reference="cam_delta_norm")
    random_plan = assert_valid(S.random_seed(task, definition, 11), task, cam)
    straight = assert_valid(S.straight_line_seed(task, definition), task, cam)
    np.testing.assert_array_equal(positions(random_plan), positions(straight))
    np.testing.assert_array_equal(np.asarray(S.seed_deltas_mm(random_plan)),
                                  np.asarray(S.seed_deltas_mm(straight)))
    assert json.loads(random_plan.provenance)["parameter_map"]["reference_motion"] == "straight_line"


def test_seed_derivation_bytes_and_full_plan_provenance_are_explicit(task, cam):
    definition = declaration(fraction=0.0005, root_seed=19)
    result = S.random_seed(task, definition, 2)
    plan = assert_valid(result, task, cam)
    expected_hash = source_hash(definition)
    assert result.declaration_hash == expected_hash
    meta = json.loads(result.derivation)
    assert meta["declaration_hash"] == expected_hash
    assert meta["task_hash"] == task.task_hash
    assert meta["cam_motion_hash"] == cam.motion_hash
    assert plan.provenance == result.derivation
    derivation = meta["seed_derivation"]
    payload = {
        "algorithm": "sha256_v1", "task_hash": task.task_hash,
        "declaration_hash": expected_hash, "root_seed": 19, "seed": 2,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                        separators=(",", ":")).encode("utf-8")).hexdigest()
    assert derivation["payload"] == payload
    assert derivation["digest_sha256"] == "sha256:" + digest
    assert derivation["derived_seed_u64"] == int.from_bytes(bytes.fromhex(digest)[:8], "little")
    assert result.to_record()["motion"] == plan.to_dict()
    assert result.to_record()["valid"] is True


@pytest.mark.parametrize("space,family", [("theta", "gaussian"), ("theta", "uniform"),
                                          ("delta", "gaussian"), ("delta", "uniform")])
def test_both_registered_distributions_and_parameter_spaces_validate(task, cam, space, family):
    definition = declaration(space=space, family=family,
                             fraction=0.0002 if space == "theta" else 0.001)
    if space == "theta":
        definition["random_init"]["scale"]["reference"]["kind"] = "cam_delta_norm"
    assert_valid(S.random_seed(task, definition, 3), task, cam)


def test_uniform_zero_is_the_same_reference_path_even_when_its_source_is_zero(task, cam):
    definition = declaration(space="delta", family="uniform", fraction=0.0)
    plan = assert_valid(S.random_seed(task, definition, 23), task, cam)
    assert np.all(np.asarray(S.seed_deltas_mm(plan)) == 0.0)
    assert plan.initial_state == cam.initial_state


def test_oversized_scale_is_typed_invalid_without_projection(task, cam):
    definition = declaration(fraction=500.0)
    result = S.random_seed(task, definition, 2)
    assert isinstance(result, S.SeedResult)
    assert not result.valid
    assert result.plan is None
    assert result.invalid_reason
    assert "envelope" in result.invalid_reason or "RAPID" in result.invalid_reason
    with pytest.raises(ValueError, match="envelope|RAPID"):
        result.require_plan()
    record = result.to_record()
    assert record["valid"] is False and record["motion"] is None


def test_explicit_out_of_machine_coordinate_is_rejected_not_clipped(task, cam):
    definition = declaration()
    definition["straight_line_init"]["end"] = "machine_mm:(100,0,20)"
    result = S.straight_line_seed(task, definition)
    assert not result.valid and result.plan is None
    assert result.invalid_reason


@pytest.mark.parametrize("value", [None, 1.0, float("nan"), float("inf"), "TO-BE-FROZEN"])
def test_placeholder_or_nonfinite_scale_is_rejected(task, cam, value):
    definition = declaration()
    definition["random_init"]["scale"]["fraction"] = value
    result = S.random_seed(task, definition, 0)
    assert not result.valid and result.invalid_reason


def test_unregistered_distribution_detail_or_reference_never_means_gaussian_substitution(task, cam):
    definition = declaration(detail="antithetic")
    result = S.random_seed(task, definition, 0)
    assert not result.valid and "detail" in result.invalid_reason
    definition = declaration(family="other_declared")
    assert not S.random_seed(task, definition, 0).valid
    definition = declaration(reference="other_declared")
    assert not S.random_seed(task, definition, 0).valid


def test_missing_fields_and_wrong_units_are_typed_invalid(task, cam):
    definition = declaration()
    del definition["random_init"]["parameter_space"]
    result = S.random_seed(task, definition, 0)
    assert not result.valid and "missing required" in result.invalid_reason
    definition = declaration()
    definition["random_init"]["scale"]["reference"]["units"] = "inches"
    assert not S.random_seed(task, definition, 0).valid


def test_arbitrary_seed_derivation_text_has_no_executable_fallback(task, cam):
    definition = declaration()
    definition["random_init"]["seed_derivation_rule"]["algorithm"] = "sha512"
    assert not S.random_seed(task, definition, 0).valid
    definition = declaration()
    definition["random_init"]["seed_derivation_rule"]["input"] = "random-ish object"
    assert not S.random_seed(task, definition, 0).valid
    definition = declaration()
    assert not S.random_seed(task, definition, -1).valid


def test_frozen_text_in_any_declaration_is_rejected(task, cam):
    definition = declaration()
    definition["straight_line_init"]["start"] = "TO-BE-FROZEN"
    assert not S.random_seed(task, definition, 0).valid
    assert not S.straight_line_seed(task, definition).valid


def test_nonmapping_declaration_returns_a_typed_failure(task, cam):
    result = S.straight_line_seed(task, None)
    assert isinstance(result, S.SeedResult)
    assert not result.valid and result.plan is None and result.invalid_reason
    assert result.declaration_hash == ""


def test_literal_coordinate_parser_does_not_execute_code(task, cam):
    definition = declaration()
    definition["straight_line_init"]["end"] = "machine_mm:__import__('os').system('exit 1')"
    result = S.straight_line_seed(task, definition)
    assert not result.valid and result.plan is None
