"""D-08/D-12 line/random MotionPlan seed tests (no evaluator or campaign)."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from benchmarks import contracts as C
from benchmarks import procedural_suite as PS
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
            "start": "stock_bbox_min_at_cam_min_cutting_z",
            "end": "stock_bbox_max_at_cam_min_cutting_z", "segment_placement": "uniform",
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


def cam_perturbed_declaration(*, fraction=0.0, reference="cam_delta_norm",
                              family="gaussian", exploratory=True, excluded_from_init_axis_claims=True):
    return {"cam_perturbed_init": {
        "parameter_space": "theta",
        "distribution": {"kind": family, "detail": "iid"},
        "scale": {"fraction": fraction, "reference": {
            "kind": reference,
            "description": "separate normalized CAM perturbation reference; not the delta random recipe",
            "units": "mm",
        }},
        "seed_derivation_rule": {
            "algorithm": "sha256_v1", "root_seed": 101,
            "input": "task_hash,declaration_hash,root_seed,seed",
            "per_seed_rule": "first_8_bytes_little_endian_uint64",
        },
        "exploratory": exploratory,
        "excluded_from_init_axis_claims": excluded_from_init_axis_claims,
    }}


def source_hash(value):
    payload = json.dumps(value, allow_nan=False, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def positions(plan):
    return np.asarray([segment.start_mm for segment in plan.segments]
                      + [plan.segments[-1].end_mm], dtype=np.float64)


def assert_valid(result, task, cam, *, diagnostic_only=False):
    assert isinstance(result, S.SeedResult)
    assert result.valid, result.invalid_reason
    plan = result.require_plan(diagnostic_only=diagnostic_only)
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


def expected_floor_corners(task, cam):
    inverse = task.work_to_machine.inverse().apply
    floor_z = min(inverse(point)[2]
                  for segment in cam.segments if segment.kind != C.SegmentKind.RAPID
                  for point in (segment.start_mm, segment.end_mm))
    lower, upper = task.stock.params["min_mm"], task.stock.params["max_mm"]
    return (task.work_to_machine.apply((lower[0], lower[1], floor_z)),
            task.work_to_machine.apply((upper[0], upper[1], floor_z)), floor_z)


def assert_d12_geometry(task, cam, result, *, diagnostic_only=False):
    plan = assert_valid(result, task, cam, diagnostic_only=diagnostic_only)
    point = positions(plan)
    lower, upper = task.stock.params["min_mm"], task.stock.params["max_mm"]
    floor_start, floor_end, floor_z = expected_floor_corners(task, cam)
    safe_work_z = task.work_to_machine.inverse().apply(point[0])[2]
    above_min = task.work_to_machine.apply((lower[0], lower[1], safe_work_z))
    np.testing.assert_array_equal(point[0], cam.initial_state.position_mm)
    np.testing.assert_allclose(point[1], above_min, atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(point[2], floor_start)
    np.testing.assert_array_equal(point[-1], floor_end)
    np.testing.assert_allclose(point[2:], np.linspace(floor_start, floor_end,
                                                      len(cam.segments) - 1),
                               atol=0.0, rtol=0.0)
    work = np.asarray([task.work_to_machine.inverse().apply(p) for p in point[2:]])
    np.testing.assert_allclose(work[:, 2], floor_z, atol=1e-12, rtol=0.0)
    assert min(task.work_to_machine.inverse().apply(point[i])[2] for i in (0, 1)) > upper[2]
    assert plan.segments[0].kind == C.SegmentKind.RAPID
    assert plan.segments[1].kind == C.SegmentKind.FEED
    assert 0 < plan.segments[1].feed_mm_s <= task.limits.max_feed_mm_s
    assert plan.segments[1].feed_mm_s == cam.segments[1].feed_mm_s
    assert plan.segments[1].end_mm == floor_start
    assert math.dist(point[2], point[-1]) > 0.0
    assert sum(segment.duration_s for segment in plan.segments) > 0.0
    meta = json.loads(result.derivation)
    layout = meta["layout"]
    indices = layout["index_layout"]
    assert layout["decision"] == "D-12"
    assert layout["cam_min_cutting_z_work_mm"] == floor_z
    assert layout["floor_min_corner_mm"] == list(floor_start)
    assert layout["floor_max_corner_mm"] == list(floor_end)
    assert layout["floor_diagonal_length_mm"] > 0
    assert layout["minimum_cutting_z_segment_ids"]
    assert layout["same_segment_count_and_k"] is True
    assert layout["segment_count"] == len(cam.segments)
    assert layout["position_count"] == len(cam.segments) + 1
    assert indices["rapid_segment_indices"] == [0]
    assert indices["feed_plunge_segment_indices"] == [1]
    assert indices["floor_diagonal_range_inclusive"] == [2, len(cam.segments)]
    assert indices["free_noise_position_range_inclusive"] == [3, len(cam.segments) - 1]
    assert indices["noise_free_position_indices"] == [0, 1, 2, len(cam.segments)]
    assert indices["pinned_floor_corner_indices"] == [2, len(cam.segments)]
    assert "unchanged downstream" in layout["fitted_k_policy"]
    assert meta["zero_length_segments"] == sum(
        math.dist(segment.start_mm, segment.end_mm) == 0 for segment in plan.segments)
    assert meta["validation"] == "validate_for_task passed"
    return plan, meta


def test_d12_diagonal_is_nondegenerate_on_pilot_and_a_real_suite_task(task, cam):
    assert_d12_geometry(task, cam, S.straight_line_seed(task, declaration()))
    bundle = PS.make_task(PS.generate_scenarios(dev_per_family=1, test_per_family=1)[0])
    suite_cam = S.seed_for_task(bundle.task)
    assert suite_cam.motion_hash == bundle.motion.motion_hash
    assert_d12_geometry(bundle.task, suite_cam,
                        S.straight_line_seed(bundle.task, declaration()))


def test_legacy_cam_endpoints_remain_an_explicit_closed_line_diagnostic(task, cam):
    definition = declaration()
    definition["straight_line_init"]["start"] = "cam_start"
    definition["straight_line_init"]["end"] = "cam_end"
    line = S.straight_line_seed(task, definition)
    plan = assert_valid(line, task, cam)
    assert plan.initial_state.position_mm == (0.0, 0.0, 10.0)
    assert plan.segments[-1].end_mm == (0.0, 0.0, 10.0)
    assert all(math.dist(s.start_mm, s.end_mm) == 0.0 for s in plan.segments)
    meta = json.loads(line.derivation)
    assert meta["zero_length_segments"] == len(cam.segments)
    assert meta["layout"]["geometry"] == "declared_endpoint_line_diagnostic"
    assert meta["layout"]["decision"].startswith("not-D-12")
    assert meta["layout"]["random_delta_supported"] is False
    assert meta["mode"] == "straight_line"


def test_explicit_boundary_endpoints_are_not_clipped(task, cam):
    definition = declaration()
    definition["straight_line_init"]["start"] = "cam_start"
    definition["straight_line_init"]["end"] = "machine_mm:(40,30,60)"
    plan = assert_valid(S.straight_line_seed(task, definition), task, cam)
    assert tuple(positions(plan)[0]) == cam.initial_state.position_mm
    assert tuple(positions(plan)[-1]) == (40.0, 30.0, 60.0)
    expected = np.linspace(cam.initial_state.position_mm, (40, 30, 60),
                           len(cam.segments) + 1)
    np.testing.assert_allclose(positions(plan), expected, atol=0.0, rtol=0.0)


def test_random_delta_is_deterministic_and_different_seeds_change_geometry_not_luck(task, cam):
    definition = declaration(family="uniform", fraction=0.0005)
    first = S.random_seed(task, definition, 7)
    other = S.random_seed(task, definition, 8)
    repeated = S.random_seed(task, definition, 7)
    p1 = assert_valid(first, task, cam)
    p2 = assert_valid(other, task, cam)
    same = assert_valid(repeated, task, cam)
    assert p1.to_dict() == same.to_dict()
    assert first.derivation == repeated.derivation
    assert not np.array_equal(positions(p1), positions(p2))


def test_cam_perturbed_zero_reproduces_cam_but_is_explicitly_exploratory(task, cam):
    other = cam_perturbed_declaration(fraction=0.0)
    result = S.cam_perturbed_seed(task, other, 11)
    plan = assert_valid(result, task, cam, diagnostic_only=True)
    np.testing.assert_allclose(np.asarray(S.seed_deltas_mm(plan)),
                               np.asarray(S.seed_deltas_mm(cam)), atol=1e-12, rtol=0.0)
    meta = json.loads(result.derivation)
    assert meta["parameter_map"]["reference_motion"] == "cam"
    assert meta["scope"] == "exploratory_cam_perturbation"
    assert meta["exploratory"] is True
    assert meta["excluded_from_init_axis_claims"] is True
    assert meta["noise_scale"]["filter_applied"] is False
    assert meta["noise_scale"]["unit"] == "normalized_theta"


def test_raw_theta_in_random_init_is_not_an_exploratory_fallback(task, cam):
    result = S.random_seed(task, declaration(space="theta"), 11)
    assert result.status == "failed_instance"
    assert "delta space only" in result.invalid_reason
    assert result.plan is None
    assert "cam_perturbed_init" in result.invalid_reason


def test_random_delta_rejects_cam_or_literal_endpoints_instead_of_d12(task, cam):
    definition = declaration(family="uniform")
    definition["straight_line_init"]["end"] = (
        "machine_mm:" + repr(cam.segments[len(cam.segments) // 2].end_mm))
    result = S.random_seed(task, definition, 2)
    assert not result.valid and result.plan is None
    assert "D-12 random initialization requires" in result.invalid_reason
    assert result.status == "failed_instance"
    definition = declaration(family="uniform")
    definition["straight_line_init"]["start"] = "cam_start"
    definition["straight_line_init"]["end"] = "cam_end"
    result = S.random_seed(task, definition, 2)
    assert not result.valid and "separate diagnostics" in result.invalid_reason

def test_random_delta_zero_reproduces_declared_straight_line(task, cam):
    definition = declaration(space="delta", fraction=0.0, reference="cam_delta_norm")
    random_plan = assert_valid(S.random_seed(task, definition, 11), task, cam, diagnostic_only=True)
    straight = assert_valid(S.straight_line_seed(task, definition), task, cam)
    np.testing.assert_array_equal(positions(random_plan), positions(straight))
    np.testing.assert_array_equal(np.asarray(S.seed_deltas_mm(random_plan)),
                                  np.asarray(S.seed_deltas_mm(straight)))
    assert json.loads(random_plan.provenance)["parameter_map"]["reference_motion"] == "D-12_stock_bbox_floor_diagonal"


def test_seed_derivation_bytes_and_full_plan_provenance_are_explicit(task, cam):
    definition = declaration(family="uniform", fraction=0.0005, root_seed=19)
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
    assert result.to_record()["status"] == "valid_plan"


@pytest.mark.parametrize("family", ["gaussian", "uniform"])
def test_exploratory_distributions_validate_with_separate_flags_and_baseline(task, cam, family):
    definition = cam_perturbed_declaration(family=family, fraction=0.0002)
    result = S.cam_perturbed_seed(task, definition, 3)
    plan = assert_valid(result, task, cam)
    meta = json.loads(result.derivation)
    assert meta["parameter_space"] == "theta"
    assert meta["exploratory"] is meta["excluded_from_init_axis_claims"] is True
    assert meta["deviations"]["count"] == len(cam.segments) - 1
    assert meta["deviations"]["rms_deviation_mm"] > 0
    # Normalize comparison semantics belongs to this sidecar only; random_init
    # above must reject the same theta space rather than silently reuse it.
    assert result.to_record()["derivation"]["scale"] != declaration()["random_init"]["scale"]


@pytest.mark.parametrize("family", ["gaussian", "uniform"])
def test_delta_distributions_keep_local_deviations_and_pin_entry_and_floor_corners(task, cam, family):
    definition = declaration(family=family, fraction=0.0005)
    result = S.random_seed(task, definition, 4)
    plan = assert_valid(result, task, cam)
    baseline, _ = assert_d12_geometry(task, cam, S.straight_line_seed(task, definition))
    p, base = positions(plan), positions(baseline)
    np.testing.assert_array_equal(p[:3], base[:3])
    np.testing.assert_array_equal(p[-1], base[-1])
    assert not np.array_equal(p[3:-1], base[3:-1])
    assert tuple(p[2]) == expected_floor_corners(task, cam)[0]
    assert tuple(p[-1]) == expected_floor_corners(task, cam)[1]
    meta = result.to_record()["derivation"]
    assert meta["parameter_space"] == "delta"
    assert meta["parameter_map"]["noise_shape"] == [len(cam.segments) - 3, 3]
    assert meta["parameter_map"]["free_position_range_inclusive"] == [3, len(cam.segments) - 1]
    assert meta["line_layout"]["decision"] == "D-12"
    assert meta["entry_plunge_and_floor_pinning_required"] is True
    assert meta["noise_scale"]["sqrt_T_growth"] is False
    assert meta["noise_scale"]["filter_applied"] is False
    assert "bound" not in meta
    assert meta["deviations"]["count"] == len(cam.segments) - 3
    assert math.isfinite(meta["deviations"]["rms_deviation_mm"])
    assert 0 < meta["deviations"]["rms_deviation_mm"]


def test_uniform_zero_is_the_same_d12_reference_path(task, cam):
    definition = declaration(space="delta", family="uniform", fraction=0.0)
    plan = assert_valid(S.random_seed(task, definition, 23), task, cam, diagnostic_only=True)
    straight = assert_valid(S.straight_line_seed(task, definition), task, cam)
    assert np.any(np.asarray(S.seed_deltas_mm(plan)) != 0.0)
    np.testing.assert_array_equal(positions(plan), positions(straight))
    np.testing.assert_array_equal(np.asarray(S.seed_deltas_mm(plan)),
                                  np.asarray(S.seed_deltas_mm(straight)))
    assert plan.initial_state == cam.initial_state


def test_zero_scale_probe_has_explicit_runner_boundary(task, cam):
    result = S.random_seed(task, declaration(space="delta", fraction=0.0), 4)
    assert result.valid
    metadata = json.loads(result.derivation)
    assert metadata["scale"]["fraction_zero_is_explicit_degenerate_probe"] is True
    assert metadata["scale"]["runner_allowed"] is False
    with pytest.raises(ValueError, match="diagnostic/test-only"):
        result.require_plan()
    with pytest.raises(ValueError, match="explicit boolean"):
        result.require_plan(diagnostic_only="yes")
    plan = assert_valid(result, task, cam, diagnostic_only=True)
    assert result.to_record()["derivation"]["scale"]["runner_allowed"] is False
    assert len(plan.segments) == len(cam.segments)


def test_oversized_scale_is_typed_invalid_without_projection(task, cam):
    # Uniform scale itself is a bound; this realization satisfies it but leaves
    # the machine envelope. Refusing the plan must not weaken that geometry check.
    definition = declaration(family="uniform", fraction=1.0)
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
    assert record["status"] == "failed_instance"
    assert record["seed"] == 2


def test_explicit_out_of_machine_coordinate_is_rejected_not_clipped(task, cam):
    definition = declaration()
    definition["straight_line_init"]["start"] = "cam_start"
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
    definition["straight_line_init"]["start"] = "cam_start"
    definition["straight_line_init"]["end"] = "machine_mm:__import__('os').system('exit 1')"
    result = S.straight_line_seed(task, definition)
    assert not result.valid and result.plan is None


def local_cam_subset(task, cam, segment_count):
    """Explicit local-subset construction for T tests; not a changed pilot task."""
    indices = np.linspace(0, len(cam.segments) - 1, segment_count, dtype=int)
    assert len(set(indices)) == len(indices) == segment_count
    chosen = [cam.segments[int(index)] for index in indices]
    points = [cam.initial_state.position_mm] + [source.end_mm for source in chosen]
    segments = []
    for index, source in enumerate(chosen):
        if source.kind == C.SegmentKind.RAPID:
            feed = None
            speed = task.limits.rapid_mm_s
        else:
            feed = source.feed_mm_s
            speed = feed
        start, end = points[index], points[index + 1]
        segments.append(C.Segment(id=source.id, kind=source.kind, start_mm=start,
                                  end_mm=end, duration_s=math.dist(start, end) / speed,
                                  feed_mm_s=feed, tag=source.tag, arc=None))
    return C.MotionPlan(cam.frame, cam.initial_state, tuple(segments),
                        cam.timing_rule, provenance="explicit-local-T-subset: synthetic seed test only")


def test_vertex_scale_and_rms_do_not_grow_with_sqrt_T(task, cam, monkeypatch):
    measured = []
    scales = []
    for segment_count in (10, 75):
        local = local_cam_subset(task, cam, segment_count)
        local.validate_for_task(task)
        monkeypatch.setattr(S, "seed_for_task", lambda _task, value=local: value)
        result = S.random_seed(task, declaration(family="uniform", fraction=0.0001), 9)
        plan = assert_valid(result, task, local)
        points = positions(plan)
        assert len(plan.segments) == segment_count
        assert len(points) == segment_count + 1
        assert tuple(points[0]) == cam.initial_state.position_mm
        assert tuple(points[-1]) == expected_floor_corners(task, local)[1]
        meta = json.loads(result.derivation)
        sigma = meta["scale"]["scale_mm"]
        stats = meta["deviations"]
        assert meta["parameter_map"]["noise_shape"] == [segment_count - 3, 3]
        assert stats["count"] == segment_count - 3
        assert stats["max_abs_coordinate_deviation_mm"] <= sigma
        assert stats["max_deviation_mm"] <= math.sqrt(3.0) * sigma
        # Local RMS per free vertex stays O(sigma), not O(sigma*sqrt(T)).
        assert 0 < stats["rms_deviation_mm"] < 1.5 * sigma
        assert meta["noise_scale"]["sqrt_T_growth"] is False
        assert meta["noise_scale"]["filter_applied"] is False
        measured.append(stats["rms_deviation_mm"] / sigma)
        scales.append(sigma)
    assert all(0 < ratio < 1.5 for ratio in measured)
    assert scales[0] == scales[1]


def test_gaussian_above_sigma_is_retained_when_motion_is_physically_valid(task, cam):
    # This same fixed RNG realization was previously rejected only for |noise|>sigma.
    # Its actual coordinates satisfy the physical task constraints, so it is retained.
    definition = declaration(fraction=0.0005)
    result = S.random_seed(task, definition, 7)
    plan = assert_valid(result, task, cam)
    assert tuple(positions(plan)[0]) == cam.initial_state.position_mm
    assert tuple(positions(plan)[-1]) == expected_floor_corners(task, cam)[1]
    record = result.to_record()
    assert record["schema"] == "diffcam-initialization-seed-result/2"
    assert record["status"] == "valid_plan" and record["seed"] == 7
    meta = record["derivation"]
    assert meta["status"] == "valid_plan"
    scale = meta["noise_scale"]
    assert scale["unit"] == "mm"
    assert scale["filter_applied"] is False
    assert scale["max_abs_coordinate"] > scale["per_coordinate_scale"]
    assert meta["deviations"]["max_abs_coordinate_deviation_mm"] > meta["scale"]["scale_mm"]
    assert 0 < meta["deviations"]["rms_deviation_mm"] < meta["deviations"]["max_deviation_mm"]
    assert meta["seed_derivation"]["requested_seed"] == 7
    assert meta["noise_sha256"].startswith("sha256:")
    assert S.random_seed(task, definition, 7).to_record() == record


def test_T75_gaussian_has_95_plus_valid_of_100_with_only_physical_failures(task, cam):
    assert len(cam.segments) == 75
    definition = declaration(family="gaussian", fraction=0.0005)
    results = [S.random_seed(task, definition, seed) for seed in range(100)]
    valid, failed = [], []
    for seed, result in enumerate(results):
        assert result.seed == seed
        if result.valid:
            valid.append(assert_valid(result, task, cam))
            metadata = json.loads(result.derivation)
            assert metadata["noise_scale"]["filter_applied"] is False
            assert tuple(positions(result.plan)[0]) == cam.initial_state.position_mm
            assert tuple(positions(result.plan)[-1]) == expected_floor_corners(task, cam)[1]
        else:
            failed.append(result)
            assert result.plan is None and result.invalid_reason
            reason = result.invalid_reason.casefold()
            assert "deviation bound" not in reason
            assert "noise exceeded" not in reason
            assert "sigma" not in reason
            assert any(token in reason for token in (
                "envelope", "rapid", "feed", "contigu", "motion", "position", "validate"))
    # Every requested index was evaluated exactly once; there is no retry or
    # discard-and-resample loop that could manufacture either outcome count.
    assert len(valid) + len(failed) == len(results) == 100
    assert len(valid) >= 95


def test_failed_instance_sequence_preserves_order_and_never_redraws(task, cam, monkeypatch):
    generated = []

    class ControlledGaussian:
        def __init__(self, slot):
            self.slot = slot
            self.normal_calls = 0

        def normal(self, mean, sigma, *, size):
            self.normal_calls += 1
            assert mean == 0.0
            if self.slot in (1, 3):
                # The first free vertex is now a floor-diagonal vertex; entry
                # and plunge remain pinned. A huge negative Z creates a real
                # envelope failure, not a dispersion/amplitude rejection.
                noise = np.zeros(size)
                noise[0, 2] = -1000.0
                return noise
            return np.zeros(size)

    def factory(seed_sequence):
        # The actual SHA-derived SeedSequence is still constructed/recorded;
        # only its draw is replaced by this deterministic unit-test oracle.
        assert isinstance(seed_sequence, np.random.SeedSequence)
        generator = ControlledGaussian(len(generated))
        generated.append(generator)
        return generator

    monkeypatch.setattr(S.np.random, "default_rng", factory)
    seeds = [9, 7, 1, 12]
    results = S.random_seed_sequence(task, declaration(fraction=0.0005), seeds)
    assert len(results) == len(generated) == len(seeds)
    assert [result.seed for result in results] == seeds
    assert [result.status for result in results] == [
        "valid_plan", "failed_instance", "valid_plan", "failed_instance"]
    assert results[1].plan is None and results[3].plan is None
    assert [generator.normal_calls for generator in generated] == [1, 1, 1, 1]
    for result in (results[0], results[2]):
        assert_valid(result, task, cam)
    for result in (results[1], results[3]):
        assert "envelope" in result.invalid_reason or "RAPID" in result.invalid_reason
        record = result.to_record()
        assert record["seed"] == result.seed
        assert record["status"] == "failed_instance"
        assert record["derivation"]["evidence"]["noise_scale"]["filter_applied"] is False
        assert record["derivation"]["evidence"]["deviations"]["max_abs_coordinate_deviation_mm"] == 1000.0
    # Repeating a requested slot is explicit, not a hidden post-failure retry.
    duplicate = S.random_seed_sequence(task, declaration(fraction=0.0005), [4, 4, 5])
    assert len(duplicate) == 3 and [item.seed for item in duplicate] == [4, 4, 5]


@pytest.mark.parametrize("flag", ["exploratory", "excluded_from_init_axis_claims"])
def test_cam_perturbed_exclusion_flags_are_mandatory(task, cam, flag):
    definition = cam_perturbed_declaration(fraction=0.0002)
    del definition["cam_perturbed_init"][flag]
    result = S.cam_perturbed_seed(task, definition, 2)
    assert not result.valid and flag in result.invalid_reason
    definition = cam_perturbed_declaration(fraction=0.0001,
                                           **{flag: False})
    result = S.cam_perturbed_seed(task, definition, 2)
    assert not result.valid and flag in result.invalid_reason


def test_cam_perturbed_requires_its_own_registered_recipe(task, cam):
    result = S.cam_perturbed_seed(task, declaration(space="delta"), 2)
    assert not result.valid and "cam_perturbed_init" in result.invalid_reason
    definition = cam_perturbed_declaration(fraction=0.0001, reference="stock_xy_extent")
    definition["random_init"] = declaration()["random_init"]
    result = S.cam_perturbed_seed(task, definition, 2)
    assert_valid(result, task, cam)
    metadata = json.loads(result.derivation)
    assert metadata["scale"]["reference_kind"] == "stock_xy_extent"
    assert metadata["parameter_space"] == "theta"
    assert metadata["mode"] == "cam_perturbed"


def test_failed_numpy_integer_seed_and_overflowing_effective_scale_stay_serializable(task):
    result = S.random_seed(task, declaration(fraction=1e308), np.int64(7))
    assert not result.valid and result.plan is None
    assert result.status == "failed_instance"
    assert result.seed == 7 and isinstance(result.seed, int)
    assert "finite" in result.invalid_reason
    json.dumps(result.to_record(), allow_nan=False)
