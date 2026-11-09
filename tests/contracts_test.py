"""T0.0 contract tests: round trips, hash stability, invalid inputs, coverage and failed runs."""

import dataclasses
import json
import math
import pathlib

import pytest

from benchmarks import contracts as C

DATA = pathlib.Path(__file__).parent / "data" / "contracts"


def _task():
    return C.example_pocket_task()


def _motion():
    return C.example_motion_plan()


def _eval(**over):
    t, m = _task(), _motion()
    kw = dict(
        evaluator_id="reference_carve", evaluator_revision="test", task_hash=t.task_hash,
        tool_setup_hash=t.tool_setup_hash, discretization={"voxel_mm": 0.5, "max_step_mm": 0.1},
        coverage=C.Coverage.COMPLETE, coverage_reasons=(),
        metrics={"removed_volume_mm3": C.Metric(100.0, "mm^3"),
                 "residual_volume_mm3": C.Metric(0.0, "mm^3"),
                 "gouge_volume_mm3": C.Metric(0.0, "mm^3"),
                 "max_surface_error_mm": C.Metric(0.05, "mm")},
        safety_pass=True, geometry_pass=True, motion_hash=m.motion_hash)
    kw.update(over)
    return C.EvaluationResult(**kw)


# ---------------------------------------------------------------- round trips
@pytest.mark.parametrize("make", [_task, _motion, _eval])
def test_round_trip_preserves_record_and_hash(make):
    rec = make()
    back = C.loads(C.dumps(rec))
    assert back == rec
    assert C.content_hash(back) == C.content_hash(rec)


def test_saved_fixtures_load_and_hashes_are_stable():
    """Golden files pin the serialized form; a hash change means a schema change (bump version)."""
    expected = json.loads((DATA / "hashes.json").read_text())
    task = C.loads((DATA / "pocket_task.json").read_text())
    motion = C.loads((DATA / "pocket_motion.json").read_text())
    assert task == _task() and motion == _motion()
    assert task.task_hash == expected["task_hash"]
    assert task.tool_setup_hash == expected["tool_setup_hash"]
    assert motion.motion_hash == expected["motion_hash"]
    for name in ("eval_incomplete.json", "run_failed.json", "run_complete.json"):
        C.loads((DATA / name).read_text())


def test_hash_ignores_key_order_but_not_values():
    t = _task()
    d = t.to_dict()
    shuffled = json.loads(json.dumps(d, sort_keys=False))
    shuffled = dict(reversed(list(shuffled.items())))
    assert C.TaskSpec.from_dict(shuffled).task_hash == t.task_hash
    d2 = json.loads(json.dumps(d))
    d2["tool"]["cutter_radius_mm"] = 3.0
    t2 = C.TaskSpec.from_dict(d2)
    assert t2.task_hash != t.task_hash and t2.tool_setup_hash != t.tool_setup_hash


def test_tool_setup_hash_independent_of_tolerances():
    t = _task()
    t2 = dataclasses.replace(t, tolerances=C.Tolerances(1, 1, 1, 1))
    assert t2.tool_setup_hash == t.tool_setup_hash and t2.task_hash != t.task_hash


def test_nc_hash_is_byte_exact():
    assert C.nc_hash(b"G0 X0\n") != C.nc_hash(b"G0 X0\r\n")


# ------------------------------------------------------- invalid units / frames
def test_inch_units_rejected_with_actionable_message():
    d = _task().to_dict()
    d["length_unit"] = "inch"
    with pytest.raises(C.ContractError, match="inch_to_mm"):
        C.TaskSpec.from_dict(d)


def test_non_rigid_transform_rejected():
    with pytest.raises(C.ContractError, match="orthonormal"):
        C.Transform(((25.4, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    with pytest.raises(C.ContractError, match="reflection"):
        C.Transform(((-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))


def test_transform_inverse_round_trip():
    T = C.Transform(((0, -1, 0, 5), (1, 0, 0, -2), (0, 0, 1, 3), (0, 0, 0, 1)))
    p = (1.0, 2.0, 3.0)
    assert all(math.isclose(a, b) for a, b in zip(T.inverse().apply(T.apply(p)), p))


def test_motion_frame_must_match_initial_state():
    m = _motion()
    with pytest.raises(C.ContractError, match="initial_state.frame"):
        dataclasses.replace(m, frame=C.Frame.WORK)


def test_grid_cell_centre_convention():
    g = C.GridSpec(shape=(2, 2, 2), origin_mm=(0, 0, 0), voxel_mm=1.0)
    assert g.cell_centre_mm(0, 0, 0) == (0.5, 0.5, 0.5)
    assert g.bounds_mm == ((0, 0, 0), (2, 2, 2))
    with pytest.raises(C.ContractError, match="voxel_mm"):
        C.GridSpec(shape=(2, 2, 2), origin_mm=(0, 0, 0), voxel_mm=0)


def test_unknown_fields_and_nonfinite_rejected():
    d = _task().to_dict()
    d["extra"] = 1
    with pytest.raises(C.ContractError, match="unknown fields"):
        C.TaskSpec.from_dict(d)
    with pytest.raises(C.ContractError, match="finite"):
        C.Metric(float("nan"), "mm")


def test_excluded_volume_requires_reason():
    part = _task().original_part
    with pytest.raises(C.ContractError, match="exclusion_reason"):
        C.OperationTarget(geometry=part, excluded_volume_mm3=2.0)


def test_sampled_geometry_requires_grid_and_hash():
    with pytest.raises(C.ContractError, match="sha256"):
        C.GeometrySource(kind="sampled", path="x.npy", sha256="abc")
    with pytest.raises(C.ContractError, match="GridSpec"):
        C.GeometrySource(kind="sampled", path="x.npy", sha256="sha256:" + "0" * 64)


# ------------------------------------------------------------------ motion
def test_motion_discontinuity_rejected():
    m = _motion()
    segs = list(m.segments)
    s = segs[2]
    segs[2] = dataclasses.replace(s, start_mm=(s.start_mm[0] + 0.5, s.start_mm[1], s.start_mm[2]),
                                  duration_s=math.dist((s.start_mm[0] + 0.5,) + s.start_mm[1:], s.end_mm) / s.feed_mm_s)
    with pytest.raises(C.ContractError, match="discontinuous"):
        dataclasses.replace(m, segments=tuple(segs))


def test_rapid_feed_and_arc_field_rules():
    p0, p1 = (0, 0, 0), (1, 0, 0)
    with pytest.raises(C.ContractError, match="rapid"):
        C.Segment(id="r", kind="rapid", start_mm=p0, end_mm=p1, duration_s=0.1, feed_mm_s=5)
    with pytest.raises(C.ContractError, match="feed_mm_s"):
        C.Segment(id="f", kind="feed", start_mm=p0, end_mm=p1, duration_s=0.1)
    with pytest.raises(C.ContractError, match="arc data"):
        C.Segment(id="a", kind="arc", start_mm=p0, end_mm=p1, duration_s=0.1, feed_mm_s=5)


def test_zero_length_feed_is_valid_and_has_zero_duration():
    s = C.Segment(id="z", kind="feed", start_mm=(1, 1, 1), end_mm=(1, 1, 1), duration_s=0.0, feed_mm_s=5)
    assert s.chord_length_mm == 0.0


def test_exact_stop_duration_consistency_and_fixed_dt():
    m = _motion()
    bad = list(m.segments)
    bad[2] = dataclasses.replace(bad[2], duration_s=bad[2].duration_s * 2)
    with pytest.raises(C.ContractError, match="length/feed"):
        dataclasses.replace(m, segments=tuple(bad))
    with pytest.raises(C.ContractError, match="multiple of dt_s"):
        dataclasses.replace(m, timing_rule=C.TimingRule.FIXED_DT, dt_s=0.007)


def test_unknown_initial_state_cannot_carry_a_position():
    with pytest.raises(C.ContractError, match="known=False"):
        C.MachineState(position_mm=(0, 0, 0), known=False)
    st = C.MachineState(position_mm=None, known=False)
    m = _motion()
    m2 = dataclasses.replace(m, initial_state=st)   # no contiguity check against an unknown start
    assert m2.waypoints_mm()[0] == m.segments[0].start_mm


def test_duplicate_segment_ids_rejected():
    m = _motion()
    segs = list(m.segments)
    segs[1] = dataclasses.replace(segs[1], id="s0")
    with pytest.raises(C.ContractError, match="duplicate"):
        dataclasses.replace(m, segments=tuple(segs))


# --------------------------------------------------------------- evaluation
def test_incomplete_coverage_cannot_pass_safety_and_is_not_eligible():
    with pytest.raises(C.ContractError, match="coverage_reasons"):
        _eval(coverage=C.Coverage.INCOMPLETE)
    with pytest.raises(C.ContractError, match="safety_pass"):
        _eval(coverage=C.Coverage.INCOMPLETE, coverage_reasons=("arcs not modeled",))
    r = _eval(coverage=C.Coverage.INCOMPLETE, coverage_reasons=("arcs not modeled",), safety_pass=None)
    assert r.efficiency_eligible is False
    assert _eval().efficiency_eligible is True
    assert _eval(geometry_pass=False).efficiency_eligible is False


def test_unknown_metric_differs_from_zero():
    with pytest.raises(C.ContractError, match="null_reason"):
        C.Metric(None, "mm^3")
    unknown = C.Metric(None, "mm^3", null_reason="holder not modeled")
    zero = C.Metric(0.0, "mm^3")
    assert unknown != zero


def test_metric_units_enforced_for_known_names():
    with pytest.raises(C.ContractError, match="unit"):
        _eval(metrics={"removed_volume_mm3": C.Metric(1.0, "mm")})


def test_posted_eval_requires_hash_and_dialect_together():
    with pytest.raises(C.ContractError, match="nc_hash/dialect"):
        _eval(nc_hash=C.nc_hash(b"G0"))
    r = _eval(nc_hash=C.nc_hash(b"G0"), dialect="haas")
    assert r.dialect == C.Dialect.HAAS


def test_stored_eligibility_must_agree():
    d = _eval(geometry_pass=False).to_dict()
    d["efficiency_eligible"] = True
    with pytest.raises(C.ContractError, match="efficiency_eligible"):
        C.EvaluationResult.from_dict(d)


def test_no_cut_program_fails_geometry():
    t = _task()
    vol = 12 * 12 * 5 - 4 * 4 * 5   # pocket minus island, mm^3
    metrics = {"residual_volume_mm3": C.Metric(float(vol), "mm^3"),
               "gouge_volume_mm3": C.Metric(0.0, "mm^3"),
               "max_surface_error_mm": C.Metric(5.0, "mm")}
    assert C.geometry_pass_from_metrics(metrics, t.tolerances) is False
    metrics["gouge_volume_mm3"] = C.Metric(None, "mm^3", null_reason="not computed")
    assert C.geometry_pass_from_metrics(metrics, t.tolerances) is None


# ------------------------------------------------------------ candidates/runs
def test_candidate_validity_rules():
    h = C.content_hash([0.0, 1.0])
    C.CandidateResult("c0", h, "hard", 0.5, True)
    with pytest.raises(C.ContractError):
        C.CandidateResult("c1", h, "hard", 0.5, False, invalid_reason="nan")
    C.CandidateResult("c1", h, "smooth", None, False, invalid_reason="non-finite gradient")


def _run(**over):
    kw = dict(run_id="r0", task_hash=_task().task_hash, status=C.RunStatus.COMPLETE,
              artifacts=C.REQUIRED_RUN_ARTIFACTS + ("rs274.nc", "eval_posted_rs274.json"),
              posted=(C.PostedProgram("rs274", "supported", nc_hash=C.nc_hash(b"x"), eval_file="eval_posted_rs274.json"),
                      C.PostedProgram("haas", "unsupported", reason="helical arcs not representable")))
    kw.update(over)
    return C.RunRecord(**kw)


def test_complete_run_requirements():
    r = _run()
    assert C.loads(C.dumps(r)) == r
    with pytest.raises(C.ContractError, match="missing"):
        _run(artifacts=("config.json",))
    with pytest.raises(C.ContractError, match="every dialect"):
        _run(posted=_run().posted[:1])
    with pytest.raises(C.ContractError, match="haas.nc present"):
        _run(artifacts=_run().artifacts + ("haas.nc",))


def test_unsupported_dialect_cannot_record_nc():
    with pytest.raises(C.ContractError, match="no NC"):
        C.PostedProgram("haas", "unsupported", nc_hash=C.nc_hash(b"x"), reason="r")


def test_failed_run_needs_reason_and_keeps_partial_artifacts():
    with pytest.raises(C.ContractError, match="failure_reason"):
        _run(status="failed", artifacts=("config.json",), posted=())
    r = _run(status="failed", artifacts=("config.json", "history.json"), posted=(), failure_reason="NaN loss at iter 3")
    assert C.loads(C.dumps(r)) == r


def test_loads_rejects_untagged_records():
    with pytest.raises(C.ContractError, match="record_type"):
        C.loads(json.dumps({"record": {}}))
    with pytest.raises(C.ContractError, match="invalid JSON"):
        C.loads("{")


# ------------------------------------------------- timing (consumer review, worker-2)
def test_arc_length_sweep_and_duration_checked():
    m = _motion()
    arc = m.segments[3]
    g = C.arc_geometry(arc.start_mm, arc.end_mm, arc.arc)
    assert math.isclose(g["radius_mm"], 4.0) and math.isclose(g["sweep_rad"], math.pi)
    assert math.isclose(arc.path_length_mm, 4 * math.pi)
    segs = list(m.segments)
    segs[3] = dataclasses.replace(arc, duration_s=arc.chord_length_mm / arc.feed_mm_s)   # chord timing is wrong
    with pytest.raises(C.ContractError, match="path_length/feed"):
        dataclasses.replace(m, segments=tuple(segs))


def test_arc_direction_full_circle_turns_and_helix():
    A = lambda d, t=0: C.ArcData(centre_mm=(0, 0, 0), plane="XY", direction=d, turns=t)
    s, e = (1, 0, 0), (0, 1, 0)
    assert math.isclose(C.arc_geometry(s, e, A("ccw"))["sweep_rad"], math.pi / 2)
    assert math.isclose(C.arc_geometry(s, e, A("cw"))["sweep_rad"], -3 * math.pi / 2)
    assert math.isclose(C.arc_geometry(s, s, A("ccw"))["sweep_rad"], 2 * math.pi)
    assert math.isclose(C.arc_geometry(s, s, A("cw", 1))["sweep_rad"], -4 * math.pi)
    helix = C.arc_geometry(s, (1, 0, -3), A("ccw"))
    assert math.isclose(helix["length_mm"], math.hypot(2 * math.pi, 3))
    # G18 (XZ plane): axes (Z, X); CCW about +Y goes Z -> X
    g = C.arc_geometry((0, 0, 1), (1, 0, 0), C.ArcData(centre_mm=(0, 0, 0), plane="XZ", direction="ccw"))
    assert math.isclose(g["sweep_rad"], math.pi / 2)


def test_arc_radius_mismatch_rejected():
    with pytest.raises(C.ContractError, match="radii differ"):
        C.arc_geometry((1, 0, 0), (0, 2, 0), C.ArcData(centre_mm=(0, 0, 0), plane="XY", direction="ccw"))


def test_validate_for_task_checks_rapid_duration_and_feed_limit():
    t, m = _task(), _motion()
    m.validate_for_task(t)
    segs = list(m.segments)
    segs[0] = dataclasses.replace(segs[0], duration_s=segs[0].duration_s * 2)
    with pytest.raises(C.ContractError, match="rapid_mm_s"):
        dataclasses.replace(m, segments=tuple(segs)).validate_for_task(t)
    slow = dataclasses.replace(t, limits=dataclasses.replace(t.limits, max_feed_mm_s=1.0))
    with pytest.raises(C.ContractError, match="max_feed_mm_s"):
        m.validate_for_task(slow)
