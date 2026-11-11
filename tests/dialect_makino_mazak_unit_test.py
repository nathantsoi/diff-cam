# SPDX-License-Identifier: MIT
"""Unit tests for clean-room Makino Pro 5/6 and Mazak EIA/ISO dialect specs."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from benchmarks import contracts as C
from cam.config import MachineConfig
from cam.dialects.base import (
    ControllerDialect,
    HardwareSyntaxError,
    build_motion_lines,
    strip_nc_comments,
)
from cam.dialects.makino_mazak import (
    MAKINO_SPEC,
    MAZAK_SPEC,
    MakinoDialectSpec,
    MazakDialectSpec,
)
from cam.gcode_parser import parse_gcode


def _build_task_and_plan(
    *,
    blended: bool,
    include_arc: bool = True,
    length_offset_applied: bool = True,
) -> tuple[C.TaskSpec, C.MotionPlan]:
    """Construct a valid machine-frame TaskSpec + MotionPlan with fractional feeds/coords."""
    base_task = C.example_pocket_task()
    timing_rule = (
        C.TimingRule.BLENDED_ACCEL_BOUNDED
        if blended
        else C.TimingRule.EXACT_STOP_CONSTANT_SPEED
    )
    init_state = replace(
        base_task.initial_state,
        length_offset_applied=length_offset_applied,
    )
    task = replace(
        base_task,
        initial_state=init_state,
        motion_semantics=replace(base_task.motion_semantics, timing_rule=timing_rule),
    )

    tfm = task.work_to_machine
    p_init = init_state.position_mm
    assert p_init is not None
    p_approach = tfm.apply((-4.0, -4.0, 10.0))
    p_plunge = tfm.apply((-4.0, -4.0, -2.0))
    p_feed = tfm.apply((4.0, -4.0, -2.0))
    p_arc_end = tfm.apply((4.0, 4.0, -2.0))
    arc_centre = tfm.apply((4.0, 0.0, -2.0))
    p_retract = tfm.apply((4.0, 4.0, 10.0))

    # Fractional feed rate (12.5055 mm/s -> 750.33 mm/min) exercises shop feed rounding.
    plunge_feed_mm_s = 5.0
    cut_feed_mm_s = 12.5055
    rapid_mm_s = task.limits.rapid_mm_s

    raw_segments: list[C.Segment] = [
        C.Segment(
            id="s_approach",
            kind=C.SegmentKind.RAPID,
            start_mm=p_init,
            end_mm=p_approach,
            duration_s=math.dist(p_init, p_approach) / rapid_mm_s,
            tag="approach",
        ),
        C.Segment(
            id="s_plunge",
            kind=C.SegmentKind.FEED,
            start_mm=p_approach,
            end_mm=p_plunge,
            duration_s=math.dist(p_approach, p_plunge) / plunge_feed_mm_s,
            feed_mm_s=plunge_feed_mm_s,
            tag="plunge",
        ),
        C.Segment(
            id="s_cut",
            kind=C.SegmentKind.FEED,
            start_mm=p_plunge,
            end_mm=p_feed,
            duration_s=math.dist(p_plunge, p_feed) / cut_feed_mm_s,
            feed_mm_s=cut_feed_mm_s,
        ),
    ]

    last_pt = p_feed
    if include_arc:
        arc = C.ArcData(
            centre_mm=arc_centre,
            plane=C.ArcPlane.XY,
            direction=C.ArcDirection.CCW,
        )
        arc_len = C.arc_geometry(p_feed, p_arc_end, arc)["length_mm"]
        raw_segments.append(
            C.Segment(
                id="s_arc",
                kind=C.SegmentKind.ARC,
                start_mm=p_feed,
                end_mm=p_arc_end,
                duration_s=arc_len / cut_feed_mm_s,
                feed_mm_s=cut_feed_mm_s,
                arc=arc,
            )
        )
        last_pt = p_arc_end

    retract_end = (last_pt[0], last_pt[1], p_retract[2])
    raw_segments.append(
        C.Segment(
            id="s_retract",
            kind=C.SegmentKind.RAPID,
            start_mm=last_pt,
            end_mm=retract_end,
            duration_s=math.dist(last_pt, retract_end) / rapid_mm_s,
            tag="retract",
        )
    )

    if blended:
        blended_durs = C.blended_segment_durations(
            raw_segments,
            task.limits.rapid_mm_s,
            task.limits.max_accel_mm_s2,
        )
        segments = tuple(
            replace(seg, duration_s=dur) for seg, dur in zip(raw_segments, blended_durs)
        )
    else:
        segments = tuple(raw_segments)

    plan = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=init_state,
        segments=segments,
        timing_rule=timing_rule,
    )
    plan.validate_for_task(task)
    return task, plan


# ---------------------------------------------------------------------------
# Class metadata & singleton tests
# ---------------------------------------------------------------------------


def test_makino_and_mazak_spec_metadata() -> None:
    assert isinstance(MAKINO_SPEC, MakinoDialectSpec)
    assert MAKINO_SPEC.dialect == ControllerDialect.MAKINO
    assert MAKINO_SPEC.display_name == "Makino Professional 5 / 6 (Super GI)"
    assert MAKINO_SPEC.file_extension == ".nc"
    assert MAKINO_SPEC.force_decimal is True
    assert MAKINO_SPEC.program_digits == 4
    assert MAKINO_SPEC.max_program_number == 8999
    assert MAKINO_SPEC.leading_percent is True
    assert MAKINO_SPEC.trailing_percent is True
    assert MAKINO_SPEC.shop_xyz_decimals("mm") == 3
    assert MAKINO_SPEC.shop_xyz_decimals("inch") == 4
    assert MAKINO_SPEC.shop_feed_decimals("mm") == 1
    assert MAKINO_SPEC.shop_feed_decimals("inch") == 2

    assert isinstance(MAZAK_SPEC, MazakDialectSpec)
    assert MAZAK_SPEC.dialect == ControllerDialect.MAZAK
    assert MAZAK_SPEC.display_name == "Mazak Mazatrol Smooth / Matrix (EIA/ISO)"
    assert MAZAK_SPEC.file_extension == ".eia"
    assert MAZAK_SPEC.force_decimal is True
    assert MAZAK_SPEC.program_digits == 4
    assert MAZAK_SPEC.max_program_number == 9999
    assert MAZAK_SPEC.leading_percent is False
    assert MAZAK_SPEC.trailing_percent is True
    assert MAZAK_SPEC.shop_xyz_decimals("mm") == 3
    assert MAZAK_SPEC.shop_xyz_decimals("inch") == 4
    assert MAZAK_SPEC.shop_feed_decimals("mm") == 2
    assert MAZAK_SPEC.shop_feed_decimals("inch") == 3


# ---------------------------------------------------------------------------
# Makino Professional 5 / 6 (Super GI) output tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("units", ["mm", "inch"])
@pytest.mark.parametrize("coolant", [True, False])
@pytest.mark.parametrize("length_offset_applied", [True, False])
def test_makino_typed_blended_and_exact_stop_output(
    units: str,
    coolant: bool,
    length_offset_applied: bool,
) -> None:
    config = MachineConfig(
        units=units,
        program_number=7,
        program_name="POCKET (SGI;TEST%)",
        tool_number=1,
        spindle_rpm=12000.0,
        coolant=coolant,
    )

    # Blended mode: standalone M252 before motion and standalone M250 after motion.
    task_b, plan_b = _build_task_and_plan(
        blended=True, length_offset_applied=length_offset_applied
    )
    motion_b = build_motion_lines(
        plan_b,
        task_b,
        config,
        xyz_precision=config.precision,
        feed_precision=config.precision,
        force_decimal=True,
    )
    nc_b = MAKINO_SPEC.format_typed_program(motion_b, plan=plan_b, task=task_b, config=config)
    MAKINO_SPEC.verify_hardware_syntax(nc_b, units=units, shop_mode=False)

    lines_b = [ln.strip() for ln in nc_b.splitlines() if ln.strip()]
    assert lines_b[0] == "%"
    assert lines_b[1] == "O0007 (POCKET SGI TEST)"
    assert lines_b[2] == "(Generated by diff-cam typed MotionPlan Makino Pro 5/6 post)"
    assert lines_b[3] == "G90 G94 G17 G49 G40 G80"
    assert lines_b[4] == ("G21" if units == "mm" else "G20")
    assert lines_b[5] == "G54"
    assert "T1 M06" in lines_b
    assert ("G43 H1" if length_offset_applied else "G49") in lines_b
    assert "S12000 M03" in lines_b
    assert lines_b[-3:] == ["M05", "M30", "%"]

    idx_m252 = lines_b.index("M252")
    idx_first_motion = lines_b.index(motion_b[0])
    idx_last_motion = lines_b.index(motion_b[-1])
    idx_m250 = lines_b.index("M250")
    idx_m05 = lines_b.index("M05")
    assert idx_m252 < idx_first_motion <= idx_last_motion < idx_m250 < idx_m05
    if coolant:
        assert lines_b.index("M08") < idx_m252
        assert idx_m250 < lines_b.index("M09") < idx_m05
    else:
        assert "M08" not in lines_b and "M09" not in lines_b
    assert "G61" not in nc_b

    # Exact-stop mode: G54 G61, no M250..M253.
    task_e, plan_e = _build_task_and_plan(
        blended=False, length_offset_applied=length_offset_applied
    )
    motion_e = build_motion_lines(
        plan_e,
        task_e,
        config,
        xyz_precision=config.precision,
        feed_precision=config.precision,
        force_decimal=True,
    )
    nc_e = MAKINO_SPEC.format_typed_program(motion_e, plan=plan_e, task=task_e, config=config)
    MAKINO_SPEC.verify_hardware_syntax(nc_e, units=units, shop_mode=False)

    lines_e = [ln.strip() for ln in nc_e.splitlines() if ln.strip()]
    assert lines_e[0] == "%"
    assert lines_e[5] == "G54 G61"
    for forbidden_sgi in ("M250", "M251", "M252", "M253"):
        assert forbidden_sgi not in lines_e


@pytest.mark.parametrize("units", ["mm", "inch"])
@pytest.mark.parametrize("blended", [True, False])
def test_makino_shop_program_structure_retracts_and_decimals(
    units: str,
    blended: bool,
) -> None:
    task, plan = _build_task_and_plan(blended=blended)
    config = MachineConfig(
        units=units,
        program_number=8999,
        program_name="MAKINO_SHOP",
        tool_number=1,
        spindle_rpm=10000.0,
        coolant=True,
    )
    nc = MAKINO_SPEC.format_shop_program(
        plan, task, config, case_label="SEMI FINISH POCKET"
    )
    MAKINO_SPEC.verify_hardware_syntax(nc, units=units, shop_mode=True)

    lines = [ln.strip() for ln in nc.splitlines() if ln.strip()]
    assert lines[0] == "%"
    assert lines[1] == "O8999 (DIFFCAM MIT MAKINO PRO - SEMI FINISH POCKET)"
    assert lines[2].startswith("(T1 D=")
    assert lines[3] == "G90 G94 G17 G49 G40 G80"
    assert lines[4] == ("G21" if units == "mm" else "G20")
    assert lines[5] == "T1 M06"
    assert lines[6] == "S10000 M03"
    assert lines[7] == "G54"
    assert lines[8] == "G43 H1"
    assert lines[9:11] == ["G28 G91 Z0.0", "G90"]
    assert lines[11].startswith("G0 X") and " Y" in lines[11] and "Z" not in lines[11]
    assert lines[12].startswith("G0 Z")
    assert lines[-5:] == ["M05", "G28 G91 Z0.0", "G90", "M30", "%"]

    if blended:
        assert "M252" in lines and "M250" in lines
        assert "G61" not in lines
        assert lines.index("M08") < lines.index("M252") < lines.index("M250") < lines.index("M09")
    else:
        assert "G61" in lines
        assert "M252" not in lines and "M250" not in lines

    # Verify Makino shop feed rounding: 750.33 mm/min -> F750.3 (1 dec in mm),
    # 29.54055 ipm -> F29.54 (2 dec in inch).
    if units == "mm":
        assert "F750.3" in nc
        assert "F750.33" not in nc
    else:
        assert "F29.54" in nc


def test_makino_legacy_positions_program_and_program_number_bounds() -> None:
    pts = np.array(
        [[0.25, 0.25, 0.8], [0.75, 0.25, 0.8], [0.75, 0.75, 0.8]],
        dtype=np.float64,
    )
    cfg = MachineConfig(
        stock_size_mm=(40.0, 40.0, 20.0),
        stock_origin_in=(1.0, -2.0, -3.0),
        program_number=1,
        program_name="LEGACY",
        precision=3,
    )
    nc = MAKINO_SPEC.format_legacy_positions_program(pts, cfg)
    MAKINO_SPEC.verify_hardware_syntax(nc, units="mm", shop_mode=False)

    lines = [ln.strip() for ln in nc.splitlines() if ln.strip()]
    assert lines[0] == "%"
    assert lines[1] == "O0001 (LEGACY)"
    assert any(ln.startswith("G10 L2 P1 X") for ln in lines)
    assert lines.count("G28 G91 Z0.0") == 2
    assert lines[-1] == "%"

    # Round-trip cutting moves through parse_gcode.
    segs = [s for s in parse_gcode(nc, cfg) if s.kind == "feed"]
    assert len(segs) == len(pts)  # 1 plunge + 2 cutting moves
    np.testing.assert_allclose(segs[0].end, pts[0], atol=1e-4)
    np.testing.assert_allclose(segs[-1].end, pts[-1], atol=1e-4)

    # O9000..O9999 are reserved macro programs on Makino Pro 5/6.
    with pytest.raises(ValueError, match="8999"):
        MAKINO_SPEC.format_legacy_positions_program(
            pts, replace(cfg, program_number=9000)
        )
    with pytest.raises(ValueError, match="positions is empty"):
        MAKINO_SPEC.format_legacy_positions_program(np.zeros((0, 3)), cfg)


# ---------------------------------------------------------------------------
# Mazak Mazatrol Smooth / Matrix (EIA/ISO) output tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("units", ["mm", "inch"])
@pytest.mark.parametrize("coolant", [True, False])
@pytest.mark.parametrize("length_offset_applied", [True, False])
def test_mazak_typed_blended_and_exact_stop_output(
    units: str,
    coolant: bool,
    length_offset_applied: bool,
) -> None:
    config = MachineConfig(
        units=units,
        program_number=1,
        program_name="SMOOTH_EIA",
        tool_number=1,
        spindle_rpm=8000.0,
        coolant=coolant,
    )

    # Blended mode: NO leading %, trailing %, G61.1 P2 + G64 cancel.
    task_b, plan_b = _build_task_and_plan(
        blended=True, length_offset_applied=length_offset_applied
    )
    motion_b = build_motion_lines(
        plan_b,
        task_b,
        config,
        xyz_precision=config.precision,
        feed_precision=config.precision,
        force_decimal=True,
    )
    nc_b = MAZAK_SPEC.format_typed_program(motion_b, plan=plan_b, task=task_b, config=config)
    MAZAK_SPEC.verify_hardware_syntax(nc_b, units=units, shop_mode=False)

    assert not nc_b.lstrip().startswith("%")
    lines_b = [ln.strip() for ln in nc_b.splitlines() if ln.strip()]
    assert lines_b[0] == "O0001 (SMOOTH_EIA)"
    assert lines_b[1] == "(Generated by diff-cam typed MotionPlan Mazak EIA/ISO post)"
    assert lines_b[2] == "G90 G94 G17 G49 G40 G80"
    assert lines_b[3] == ("G21" if units == "mm" else "G20")
    assert lines_b[4] == "G54"
    assert ("G43 H1" if length_offset_applied else "G49") in lines_b
    assert lines_b[-3:] == ["M05", "M30", "%"]

    idx_g61_1 = lines_b.index("G61.1 P2")
    idx_first_motion = lines_b.index(motion_b[0])
    idx_last_motion = lines_b.index(motion_b[-1])
    idx_g64 = lines_b.index("G64")
    idx_m05 = lines_b.index("M05")
    assert idx_g61_1 < idx_first_motion <= idx_last_motion < idx_g64 < idx_m05
    if coolant:
        assert lines_b.index("M08") < idx_g61_1
        assert idx_g64 < lines_b.index("M09") < idx_m05
    else:
        assert "M08" not in lines_b and "M09" not in lines_b

    # Exact-stop mode: G54 G61, NO G61.1 and NO G64.
    task_e, plan_e = _build_task_and_plan(
        blended=False, length_offset_applied=length_offset_applied
    )
    motion_e = build_motion_lines(
        plan_e,
        task_e,
        config,
        xyz_precision=config.precision,
        feed_precision=config.precision,
        force_decimal=True,
    )
    nc_e = MAZAK_SPEC.format_typed_program(motion_e, plan=plan_e, task=task_e, config=config)
    MAZAK_SPEC.verify_hardware_syntax(nc_e, units=units, shop_mode=False)

    lines_e = [ln.strip() for ln in nc_e.splitlines() if ln.strip()]
    assert lines_e[0] == "O0001 (SMOOTH_EIA)"
    assert lines_e[4] == "G54 G61"
    assert lines_e[-2:] == ["M30", "%"]
    assert "G61.1" not in nc_e
    assert "G64" not in nc_e


@pytest.mark.parametrize("units", ["mm", "inch"])
@pytest.mark.parametrize("blended", [True, False])
def test_mazak_shop_program_framing_retract_and_decimals(
    units: str,
    blended: bool,
) -> None:
    task, plan = _build_task_and_plan(blended=blended)
    config = MachineConfig(
        units=units,
        program_number=9999,
        program_name="MAZAK_SHOP",
        tool_number=1,
        spindle_rpm=9000.0,
        coolant=True,
    )
    nc = MAZAK_SPEC.format_shop_program(plan, task, config, case_label="MATRIX CONTOUR")
    MAZAK_SPEC.verify_hardware_syntax(nc, units=units, shop_mode=True)

    assert not nc.lstrip().startswith("%")
    lines = [ln.strip() for ln in nc.splitlines() if ln.strip()]
    assert lines[0] == "O9999 (DIFFCAM MIT MAZAK EIA - MATRIX CONTOUR)"
    assert lines[1].startswith("(T1 D=")
    assert lines[2] == "G90 G94 G17 G49 G40 G80"
    assert lines[3] == ("G21" if units == "mm" else "G20")
    assert lines[4] == "T1 M06"
    assert lines[5] == "S9000 M03"
    assert lines[6] == "G54"
    assert lines[7] == "G43 H1"
    assert lines[8] == "G90 G53 G0 Z0.0"
    assert lines[9].startswith("G0 X") and " Y" in lines[9] and "Z" not in lines[9]
    assert lines[10].startswith("G0 Z")
    assert lines[-4:] == ["M05", "G90 G53 G0 Z0.0", "M30", "%"]
    assert "G28" not in nc

    if blended:
        assert "G61.1 P2" in lines and "G64" in lines
        assert "G61" not in lines
        assert lines.index("M08") < lines.index("G61.1 P2") < lines.index("G64") < lines.index("M09")
    else:
        assert "G61" in lines
        assert "G61.1" not in nc and "G64" not in nc

    # Verify Mazak shop feed rounding: 750.33 mm/min -> F750.33 (2 dec in mm),
    # 29.54055 ipm -> F29.541 (3 dec in inch).
    if units == "mm":
        assert "F750.33" in nc
    else:
        assert "F29.541" in nc


def test_mazak_legacy_positions_program() -> None:
    pts = np.array(
        [[0.2, 0.3, 0.9], [0.8, 0.3, 0.9], [0.8, 0.7, 0.9]],
        dtype=np.float64,
    )
    cfg = MachineConfig(
        stock_size_mm=(50.0, 50.0, 25.0),
        stock_origin_in=(0.5, 1.5, -2.0),
        program_number=42,
        program_name="MAZAK_LEGACY",
        precision=3,
    )
    nc = MAZAK_SPEC.format_legacy_positions_program(pts, cfg)
    MAZAK_SPEC.verify_hardware_syntax(nc, units="mm", shop_mode=False)

    lines = [ln.strip() for ln in nc.splitlines() if ln.strip()]
    assert lines[0] == "O0042 (MAZAK_LEGACY)"
    assert "%" not in lines[:-1]
    assert lines[-1] == "%"
    assert lines.count("G90 G53 G00 Z0.0") == 2
    assert any(ln.startswith("G10 L2 P1 X") for ln in lines)

    segs = [s for s in parse_gcode(nc, cfg) if s.kind == "feed"]
    assert len(segs) == len(pts)
    np.testing.assert_allclose(segs[0].end, pts[0], atol=1e-4)
    np.testing.assert_allclose(segs[-1].end, pts[-1], atol=1e-4)

    with pytest.raises(ValueError, match="9999"):
        MAZAK_SPEC.format_legacy_positions_program(
            pts, replace(cfg, program_number=10000)
        )


# ---------------------------------------------------------------------------
# Mandatory decimal point check across all real words on emitted programs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [MAKINO_SPEC, MAZAK_SPEC])
@pytest.mark.parametrize("blended", [True, False])
def test_emitted_programs_enforce_mandatory_decimals_on_all_real_words(
    spec: MakinoDialectSpec | MazakDialectSpec,
    blended: bool,
) -> None:
    task, plan = _build_task_and_plan(blended=blended, include_arc=True)
    cfg = MachineConfig(units="mm", program_number=1)
    nc = spec.format_shop_program(plan, task, cfg)
    for raw in nc.splitlines():
        code = strip_nc_comments(raw)
        if not code or code == "%":
            continue
        for token in code.split():
            if token[0] in {"X", "Y", "Z", "I", "J", "K", "R", "F"}:
                assert "." in token, f"Missing mandatory decimal in {token!r}"


# ---------------------------------------------------------------------------
# Negative verify_hardware_syntax tests
# ---------------------------------------------------------------------------


def test_mazak_rejects_leading_percent_and_missing_trailing_percent() -> None:
    task, plan = _build_task_and_plan(blended=True)
    valid_mazak = MAZAK_SPEC.format_shop_program(plan, task, MachineConfig())

    with pytest.raises(HardwareSyntaxError, match="must NOT begin with leading '%'"):
        MAZAK_SPEC.verify_hardware_syntax("%\n" + valid_mazak, shop_mode=True)

    without_trailing_pct = valid_mazak.rstrip().removesuffix("%").rstrip() + "\n"
    with pytest.raises(HardwareSyntaxError, match="trailing '%'"):
        MAZAK_SPEC.verify_hardware_syntax(without_trailing_pct, shop_mode=True)

    with pytest.raises(HardwareSyntaxError, match="first non-empty line"):
        MAZAK_SPEC.verify_hardware_syntax(
            "(HEADER COMMENT BEFORE O-NUMBER)\n" + valid_mazak, shop_mode=True
        )


def test_makino_rejects_missing_percent_and_bad_program_numbers() -> None:
    task, plan = _build_task_and_plan(blended=True)
    valid_makino = MAKINO_SPEC.format_shop_program(plan, task, MachineConfig())

    without_leading = "\n".join(valid_makino.splitlines()[1:]) + "\n"
    with pytest.raises(HardwareSyntaxError, match="leading '%'"):
        MAKINO_SPEC.verify_hardware_syntax(without_leading, shop_mode=True)

    without_trailing = valid_makino.rstrip().removesuffix("%").rstrip() + "\n"
    with pytest.raises(HardwareSyntaxError, match="trailing '%'"):
        MAKINO_SPEC.verify_hardware_syntax(without_trailing, shop_mode=True)

    # 5-digit O00001 and reserved O9000 must fail on Makino.
    with pytest.raises(HardwareSyntaxError, match="4-digit"):
        MAKINO_SPEC.verify_hardware_syntax(
            valid_makino.replace("O0001", "O00001", 1), shop_mode=True
        )
    with pytest.raises(HardwareSyntaxError, match="O0001..O8999"):
        MAKINO_SPEC.verify_hardware_syntax(
            valid_makino.replace("O0001", "O9000", 1), shop_mode=True
        )


@pytest.mark.parametrize(
    "bad_word",
    ["X10", "Y-4", "Z0", "I0", "R5", "F600"],
)
def test_missing_decimal_point_rejected_on_both_controllers(bad_word: str) -> None:
    makino_prog = (
        "%\n"
        "O0001 (DEC TEST)\n"
        "G90 G94 G17 G49 G40 G80\n"
        "G21\n"
        "G54 G61\n"
        f"G1 {bad_word}\n"
        "M30\n"
        "%\n"
    )
    with pytest.raises(HardwareSyntaxError, match="missing a mandatory decimal point"):
        MAKINO_SPEC.verify_hardware_syntax(makino_prog)

    mazak_prog = (
        "O0001 (DEC TEST)\n"
        "G90 G94 G17 G49 G40 G80\n"
        "G21\n"
        "G54 G61\n"
        f"G1 {bad_word}\n"
        "M30\n"
        "%\n"
    )
    with pytest.raises(HardwareSyntaxError, match="missing a mandatory decimal point"):
        MAZAK_SPEC.verify_hardware_syntax(mazak_prog)


def test_shop_decimal_caps_enforced_on_makino_and_mazak() -> None:
    task, plan = _build_task_and_plan(blended=False, include_arc=False)
    makino_shop = MAKINO_SPEC.format_shop_program(plan, task, MachineConfig(units="mm"))
    # Makino mm feed max decimals = 1; F300.25 must fail in shop_mode=True.
    with pytest.raises(HardwareSyntaxError, match="max 1"):
        MAKINO_SPEC.verify_hardware_syntax(
            makino_shop.replace("F300.0", "F300.25"), units="mm", shop_mode=True
        )
    # XYZ max decimals in mm = 3; Z10.1234 must fail in shop_mode=True.
    with pytest.raises(HardwareSyntaxError, match="max 3"):
        MAKINO_SPEC.verify_hardware_syntax(
            makino_shop.replace("Z10.0", "Z10.1234", 1), units="mm", shop_mode=True
        )

    mazak_shop = MAZAK_SPEC.format_shop_program(plan, task, MachineConfig(units="mm"))
    # Mazak mm feed max decimals = 2; F300.125 must fail in shop_mode=True.
    with pytest.raises(HardwareSyntaxError, match="max 2"):
        MAZAK_SPEC.verify_hardware_syntax(
            mazak_shop.replace("F300.0", "F300.125"), units="mm", shop_mode=True
        )


def test_makino_sgi_pairing_and_foreign_smoothing_rejection() -> None:
    task, plan = _build_task_and_plan(blended=True, include_arc=False)
    valid = MAKINO_SPEC.format_shop_program(plan, task, MachineConfig())

    # SGI rough (M251) and finish (M253) paired with M250 are valid on Makino.
    MAKINO_SPEC.verify_hardware_syntax(valid.replace("M252", "M251"), shop_mode=True)
    MAKINO_SPEC.verify_hardware_syntax(valid.replace("M252", "M253"), shop_mode=True)

    # Unpaired M252 (missing M250 before M30) must fail.
    unpaired = "\n".join(ln for ln in valid.splitlines() if ln.strip() != "M250") + "\n"
    with pytest.raises(HardwareSyntaxError, match="M250"):
        MAKINO_SPEC.verify_hardware_syntax(unpaired, shop_mode=True)

    # Orphan M250 without prior M251/M252/M253 must fail.
    orphan = "\n".join(ln for ln in valid.splitlines() if ln.strip() != "M252") + "\n"
    with pytest.raises(HardwareSyntaxError, match="without active M251/M252/M253"):
        MAKINO_SPEC.verify_hardware_syntax(orphan, shop_mode=True)

    # Foreign vendor smoothing codes rejected on Makino.
    for bad_code in ("G64", "G187 P2 E0.05", "G05.1 Q1", "G5.1 Q1", "G61.1 P2", "G91.1"):
        injected = valid.replace("M252", f"M252\n{bad_code}", 1)
        with pytest.raises(HardwareSyntaxError):
            MAKINO_SPEC.verify_hardware_syntax(injected, shop_mode=True)


def test_mazak_g61_1_pairing_integer_p_and_foreign_code_rejection() -> None:
    task, plan = _build_task_and_plan(blended=True, include_arc=False)
    valid = MAZAK_SPEC.format_shop_program(plan, task, MachineConfig())

    # Bare G61.1 (no P word) paired with G64 is also valid per spec.
    MAZAK_SPEC.verify_hardware_syntax(valid.replace("G61.1 P2", "G61.1"), shop_mode=True)

    # Non-integer or out-of-range P on G61.1 must fail.
    with pytest.raises(HardwareSyntaxError, match="integer"):
        MAZAK_SPEC.verify_hardware_syntax(
            valid.replace("G61.1 P2", "G61.1 P2.5"), shop_mode=True
        )
    with pytest.raises(HardwareSyntaxError, match="0..7"):
        MAZAK_SPEC.verify_hardware_syntax(
            valid.replace("G61.1 P2", "G61.1 P8"), shop_mode=True
        )

    # Unpaired G61.1 P2 (missing G64 before M30) must fail.
    unpaired = "\n".join(ln for ln in valid.splitlines() if ln.strip() != "G64") + "\n"
    with pytest.raises(HardwareSyntaxError, match="cancelled with G64"):
        MAZAK_SPEC.verify_hardware_syntax(unpaired, shop_mode=True)

    # Cancelling G61.1 with G61 instead of G64 must fail.
    with pytest.raises(HardwareSyntaxError, match="G64"):
        MAZAK_SPEC.verify_hardware_syntax(
            valid.replace("\nG64\n", "\nG61\n"), shop_mode=True
        )

    # Standalone G64 without active G61.1 (e.g. RS274/Haas style) must fail.
    orphan_g64 = "\n".join(ln for ln in valid.splitlines() if ln.strip() != "G61.1 P2") + "\n"
    with pytest.raises(HardwareSyntaxError, match="active G61.1"):
        MAZAK_SPEC.verify_hardware_syntax(orphan_g64, shop_mode=True)

    # Foreign vendor smoothing codes & G28 in shop_mode rejected on Mazak.
    for bad_code in ("G187 P2 E0.05", "G05.1 Q1", "G5.1 Q1", "M252", "M250", "G91.1"):
        injected = valid.replace("G61.1 P2", f"{bad_code}\nG61.1 P2", 1)
        with pytest.raises(HardwareSyntaxError):
            MAZAK_SPEC.verify_hardware_syntax(injected, shop_mode=True)

    with pytest.raises(HardwareSyntaxError, match="G28"):
        MAZAK_SPEC.verify_hardware_syntax(
            valid.replace("G90 G53 G0 Z0.0", "G28 G91 Z0.0\nG90", 1),
            shop_mode=True,
        )


def test_cross_dialect_programs_mutually_rejected() -> None:
    task, plan = _build_task_and_plan(blended=True)
    cfg = MachineConfig()
    makino_nc = MAKINO_SPEC.format_shop_program(plan, task, cfg)
    mazak_eia = MAZAK_SPEC.format_shop_program(plan, task, cfg)

    with pytest.raises(HardwareSyntaxError):
        MAZAK_SPEC.verify_hardware_syntax(makino_nc, shop_mode=True)
    with pytest.raises(HardwareSyntaxError):
        MAKINO_SPEC.verify_hardware_syntax(mazak_eia, shop_mode=True)
