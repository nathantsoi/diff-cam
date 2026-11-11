# SPDX-License-Identifier: MIT
"""Clean-room unit tests for FANUC 0i/30i/31i and FANUC Robodrill Alpha (31i-B5) dialects."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from benchmarks import contracts as C
from cam.config import MachineConfig
from cam.dialects.base import (
    ControllerDialect,
    ControllerDialectSpec,
    HardwareSyntaxError,
    build_motion_lines,
    strip_nc_comments,
)
from cam.dialects.fanuc_robodrill import (
    FANUC_SPEC,
    ROBODRILL_SPEC,
    FanucDialectSpec,
    RobodrillDialectSpec,
)

ALL_SPECS = (FANUC_SPEC, ROBODRILL_SPEC)


def _make_task_and_plan(
    *,
    timing_rule: C.TimingRule = C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
    feed_mm_s: float = 19.9,  # 19.9 mm/s == 1194.0 mm/min == 47.0079 in/min
    length_offset_applied: bool = True,
) -> tuple[C.TaskSpec, C.MotionPlan]:
    """Construct a valid TaskSpec and MotionPlan with integer-valued work coords & arc."""
    base = C.example_pocket_task()
    w2m = base.work_to_machine.apply
    init_state = replace(
        base.initial_state,
        position_mm=w2m((0.0, 0.0, 6.0)),
        length_offset_applied=length_offset_applied,
    )
    task = replace(
        base,
        initial_state=init_state,
        motion_semantics=replace(base.motion_semantics, timing_rule=timing_rule),
    )

    p_init = init_state.position_mm
    p0 = w2m((2.0, -3.0, 6.0))
    p1 = w2m((2.0, -3.0, -2.0))
    p2 = w2m((5.0, -3.0, -2.0))
    p3 = w2m((5.0, 3.0, -2.0))
    p4 = w2m((5.0, 3.0, 6.0))
    arc_data = C.ArcData(
        centre_mm=w2m((5.0, 0.0, -2.0)),
        plane=C.ArcPlane.XY,
        direction=C.ArcDirection.CCW,
    )

    raw_segs = [
        C.Segment(
            id="s0",
            kind=C.SegmentKind.RAPID,
            start_mm=p_init,
            end_mm=p0,
            duration_s=math.dist(p_init, p0) / task.limits.rapid_mm_s,
            tag="approach",
        ),
        C.Segment(
            id="s1",
            kind=C.SegmentKind.FEED,
            start_mm=p0,
            end_mm=p1,
            duration_s=math.dist(p0, p1) / 5.0,
            feed_mm_s=5.0,
            tag="plunge",
        ),
        C.Segment(
            id="s2",
            kind=C.SegmentKind.FEED,
            start_mm=p1,
            end_mm=p2,
            duration_s=math.dist(p1, p2) / feed_mm_s,
            feed_mm_s=feed_mm_s,
        ),
        C.Segment(
            id="s3",
            kind=C.SegmentKind.ARC,
            start_mm=p2,
            end_mm=p3,
            duration_s=(math.pi * 3.0) / feed_mm_s,
            feed_mm_s=feed_mm_s,
            arc=arc_data,
        ),
        C.Segment(
            id="s4",
            kind=C.SegmentKind.RAPID,
            start_mm=p3,
            end_mm=p4,
            duration_s=math.dist(p3, p4) / task.limits.rapid_mm_s,
            tag="retract",
        ),
    ]

    if timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
        durs = C.blended_segment_durations(
            raw_segs, task.limits.rapid_mm_s, task.limits.max_accel_mm_s2
        )
        raw_segs = [replace(seg, duration_s=dur) for seg, dur in zip(raw_segs, durs)]

    plan = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=init_state,
        segments=tuple(raw_segs),
        timing_rule=timing_rule,
        provenance="test:fanuc_robodrill",
    )
    plan.validate_for_task(task)
    return task, plan


def _code_lines(nc_text: str) -> list[str]:
    return [
        stripped
        for raw in nc_text.splitlines()
        if (stripped := strip_nc_comments(raw))
    ]


# ---------------------------------------------------------------------------
# 1. Dialect metadata & singleton invariants
# ---------------------------------------------------------------------------


def test_dialect_metadata_and_singletons():
    assert isinstance(FANUC_SPEC, FanucDialectSpec)
    assert isinstance(FANUC_SPEC, ControllerDialectSpec)
    assert FANUC_SPEC.dialect == ControllerDialect.FANUC
    assert FANUC_SPEC.display_name == "FANUC 0i / 30i / 31i Series"
    assert FANUC_SPEC.file_extension == ".nc"
    assert FANUC_SPEC.force_decimal is True
    assert FANUC_SPEC.program_digits == 4
    assert FANUC_SPEC.max_program_number == 8999
    assert FANUC_SPEC.leading_percent is True
    assert FANUC_SPEC.trailing_percent is True
    assert FANUC_SPEC.shop_xyz_decimals("mm") == 3
    assert FANUC_SPEC.shop_xyz_decimals("inch") == 4
    assert FANUC_SPEC.shop_feed_decimals("mm") == 0
    assert FANUC_SPEC.shop_feed_decimals("inch") == 1

    assert isinstance(ROBODRILL_SPEC, RobodrillDialectSpec)
    assert isinstance(ROBODRILL_SPEC, ControllerDialectSpec)
    assert ROBODRILL_SPEC.dialect == ControllerDialect.ROBODRILL
    assert ROBODRILL_SPEC.display_name == "FANUC Robodrill Alpha (31i-B5)"
    assert ROBODRILL_SPEC.file_extension == ".nc"
    assert ROBODRILL_SPEC.force_decimal is True
    assert ROBODRILL_SPEC.program_digits == 4
    assert ROBODRILL_SPEC.max_program_number == 8999
    assert ROBODRILL_SPEC.leading_percent is True
    assert ROBODRILL_SPEC.trailing_percent is True
    assert ROBODRILL_SPEC.shop_xyz_decimals("mm") == 3
    assert ROBODRILL_SPEC.shop_xyz_decimals("inch") == 4
    assert ROBODRILL_SPEC.shop_feed_decimals("mm") == 0
    assert ROBODRILL_SPEC.shop_feed_decimals("inch") == 1


# ---------------------------------------------------------------------------
# 2. Typed program formatting (mm & inch, exact-stop & AICC II blended)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
@pytest.mark.parametrize("units", ["mm", "inch"])
@pytest.mark.parametrize(
    "timing_rule",
    [C.TimingRule.EXACT_STOP_CONSTANT_SPEED, C.TimingRule.BLENDED_ACCEL_BOUNDED],
    ids=["exact_stop", "blended_aicc"],
)
def test_typed_program_mm_inch_and_smoothing_modes(
    spec: FanucDialectSpec,
    units: str,
    timing_rule: C.TimingRule,
):
    task, plan = _make_task_and_plan(timing_rule=timing_rule)
    cfg = MachineConfig(
        units=units,
        precision=6,
        program_number=7,
        program_name="POCKET_OP1",
        tool_number=1,
        spindle_rpm=8000,
        coolant=True,
    )
    motion_lines = build_motion_lines(
        plan,
        task,
        cfg,
        xyz_precision=cfg.precision,
        feed_precision=cfg.precision,
        force_decimal=spec.force_decimal,
    )
    nc = spec.format_typed_program(motion_lines, plan=plan, task=task, config=cfg)

    # Hardware syntax verifier passes cleanly in typed mode.
    spec.verify_hardware_syntax(nc, units=units, shop_mode=False)

    raw_lines = nc.splitlines()
    assert raw_lines[0] == "%"
    assert raw_lines[-1] == "%"
    assert raw_lines[1] == "O0007 (POCKET_OP1)"
    if spec is FANUC_SPEC:
        assert raw_lines[2] == "(Generated by diff-cam typed MotionPlan FANUC post)"
    else:
        assert raw_lines[2] == "(Generated by diff-cam typed MotionPlan FANUC Robodrill post)"

    codes = _code_lines(nc)
    assert "G90 G94 G17 G49 G40 G80" in codes
    assert ("G21" if units == "mm" else "G20") in codes
    assert "T1 M06" in codes
    assert "G43 H1" in codes
    assert "S8000 M03" in codes
    assert "M08" in codes
    assert "M09" in codes
    assert codes[-3:] == ["M05", "M30", "%"]

    if timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
        assert "G54" in codes
        assert "G54 G61" not in codes
        assert "G61" not in codes
        idx_m08 = codes.index("M08")
        idx_aicc_on = codes.index("G05.1 Q1 R5")
        idx_aicc_off = codes.index("G05.1 Q0")
        idx_m09 = codes.index("M09")
        assert idx_m08 < idx_aicc_on < idx_aicc_off < idx_m09
    else:
        assert "G54 G61" in codes
        assert not any("G05.1" in ln for ln in codes)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
def test_typed_program_g49_and_no_coolant(spec: FanucDialectSpec):
    task, plan = _make_task_and_plan(length_offset_applied=False)
    cfg = MachineConfig(units="mm", coolant=False, program_number=8999)
    motion_lines = build_motion_lines(
        plan, task, cfg, xyz_precision=6, feed_precision=6, force_decimal=True
    )
    nc = spec.format_typed_program(motion_lines, plan=plan, task=task, config=cfg)
    spec.verify_hardware_syntax(nc, units="mm", shop_mode=False)

    codes = _code_lines(nc)
    assert codes[1] == "O8999"
    assert codes.count("G49") == 1  # standalone offset_line (in addition to safety block)
    assert "G43 H1" not in codes
    assert "M08" not in codes
    assert "M09" not in codes


# ---------------------------------------------------------------------------
# 3. Shop program formatting, decimals, and Fanuc vs Robodrill Z retract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
@pytest.mark.parametrize("units", ["mm", "inch"])
@pytest.mark.parametrize(
    "timing_rule",
    [C.TimingRule.EXACT_STOP_CONSTANT_SPEED, C.TimingRule.BLENDED_ACCEL_BOUNDED],
    ids=["exact_stop", "blended_aicc"],
)
def test_shop_program_decimals_feeds_and_smoothing(
    spec: FanucDialectSpec,
    units: str,
    timing_rule: C.TimingRule,
):
    task, plan = _make_task_and_plan(timing_rule=timing_rule, feed_mm_s=19.9)
    cfg = MachineConfig(
        units=units,
        program_number=42,
        program_name="DEFAULT_NAME",
        tool_number=1,
        spindle_rpm=10000,
        coolant=True,
    )
    nc = spec.format_shop_program(plan, task, cfg, case_label="CASE_A")
    spec.verify_hardware_syntax(nc, units=units, shop_mode=True)

    raw_lines = nc.splitlines()
    assert raw_lines[0] == "%"
    assert raw_lines[-1] == "%"
    if spec is FANUC_SPEC:
        assert raw_lines[1] == "O0042 (DIFFCAM MIT FANUC - CASE_A)"
    else:
        assert raw_lines[1] == "O0042 (DIFFCAM MIT FANUC ROBODRILL - CASE_A)"
    assert raw_lines[2].startswith("(T1 D=")

    codes = _code_lines(nc)
    if units == "mm":
        # Mandatory decimal points on integer coordinates and integer-rounded trailing-dot feed.
        assert "G0 X0.0 Y0.0" in codes
        assert "G0 Z6.0" in codes
        assert "G0 X2.0 Y-3.0 Z6.0" in codes
        assert "G1 X2.0 Y-3.0 Z-2.0 F300." in codes
        assert "G1 X5.0 Y-3.0 Z-2.0 F1194." in codes
        assert "G3 X5.0 Y3.0 Z-2.0 I0.0 J3.0 F1194." in codes
    else:
        # 4 XYZ decimals and 1 feed decimal in inch shop mode (1194 mm/min -> F47.0 ipm).
        assert "G0 X0.0 Y0.0" in codes
        assert any(ln.endswith("F47.0") for ln in codes if ln.startswith("G1 "))
        assert any(ln.endswith("F47.0") for ln in codes if ln.startswith("G3 "))

    if timing_rule == C.TimingRule.BLENDED_ACCEL_BOUNDED:
        idx_m08 = codes.index("M08")
        idx_aicc_on = codes.index("G05.1 Q1 R5")
        idx_aicc_off = codes.index("G05.1 Q0")
        idx_m09 = codes.index("M09")
        assert idx_m08 < idx_aicc_on < idx_aicc_off < idx_m09
        assert "G61" not in codes
    else:
        assert "G61" in codes
        assert not any("G05.1" in ln for ln in codes)


def test_fanuc_vs_robodrill_shop_retract_distinction_and_g28_rejection():
    task, plan = _make_task_and_plan(timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED)
    cfg = MachineConfig(units="mm", program_number=1)

    fanuc_nc = FANUC_SPEC.format_shop_program(plan, task, cfg, case_label="RETRACT_CHECK")
    robodrill_nc = ROBODRILL_SPEC.format_shop_program(plan, task, cfg, case_label="RETRACT_CHECK")

    fanuc_codes = _code_lines(fanuc_nc)
    robo_codes = _code_lines(robodrill_nc)

    # Generic Fanuc emits ["G28 G91 Z0.0", "G90"] before approach and after M05.
    assert fanuc_codes.count("G28 G91 Z0.0") == 2
    assert "G90 G53 G0 Z0.0" not in fanuc_codes
    for i, line in enumerate(fanuc_codes):
        if line == "G28 G91 Z0.0":
            assert fanuc_codes[i + 1] == "G90"

    # Robodrill emits ["G90 G53 G0 Z0.0"] before approach and after M05, and NEVER G28.
    assert robo_codes.count("G90 G53 G0 Z0.0") == 2
    assert not any("G28" in ln for ln in robo_codes)

    # Each passes its own hardware syntax check; Fanuc also accepts standard G53,
    # whereas Robodrill strictly rejects G28 in shop_mode=True.
    FANUC_SPEC.verify_hardware_syntax(fanuc_nc, units="mm", shop_mode=True)
    ROBODRILL_SPEC.verify_hardware_syntax(robodrill_nc, units="mm", shop_mode=True)
    FANUC_SPEC.verify_hardware_syntax(robodrill_nc, units="mm", shop_mode=True)

    with pytest.raises(HardwareSyntaxError, match="G28"):
        ROBODRILL_SPEC.verify_hardware_syntax(fanuc_nc, units="mm", shop_mode=True)


# ---------------------------------------------------------------------------
# 4. Legacy positions program formatting & validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
@pytest.mark.parametrize("units", ["mm", "inch"])
def test_legacy_positions_program_and_work_offset_g10(spec: FanucDialectSpec, units: str):
    positions = np.array(
        [
            [0.5, 0.5, 0.8],
            [0.7, 0.5, 0.8],
            [0.7, 0.7, 0.8],
        ],
        dtype=np.float64,
    )
    cfg = MachineConfig(
        units=units,
        stock_size_mm=(20.0, 20.0, 10.0),
        stock_origin_in=(4.0, 3.0, -5.0),
        program_number=15,
        program_name="LEGACY_TEST",
        retract_mm=6.0,
        feed=600.0,
        plunge_feed=200.0,
    )
    nc = spec.format_legacy_positions_program(positions, cfg)
    spec.verify_hardware_syntax(nc, units=units, shop_mode=False)

    codes = _code_lines(nc)
    assert codes[0] == "%"
    assert codes[1] == "O0015"
    assert any(ln.startswith("G10 L2 P1 ") for ln in codes)
    if spec is FANUC_SPEC:
        assert codes.count("G28 G91 Z0.0") == 2
        assert not any("G53" in ln for ln in codes)
    else:
        assert codes.count("G90 G53 G0 Z0.0") == 2
        assert not any("G28" in ln for ln in codes)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
def test_legacy_positions_validation_errors(spec: FanucDialectSpec):
    cfg = MachineConfig(stock_size_mm=(20.0, 20.0, 10.0))
    with pytest.raises(ValueError, match="must be \\(T, 3\\)"):
        spec.format_legacy_positions_program(np.zeros((4, 2)), cfg)
    with pytest.raises(ValueError, match="empty"):
        spec.format_legacy_positions_program(np.zeros((0, 3)), cfg)
    with pytest.raises(ValueError, match="finite"):
        spec.format_legacy_positions_program(
            np.array([[0.1, 0.2, 0.3], [np.nan, 0.2, 0.3]]), cfg
        )


# ---------------------------------------------------------------------------
# 5. Program number range & width checks (O0001..O8999)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
@pytest.mark.parametrize("bad_num", [0, -5, 9000, 9001, 9999, 10000, True])
def test_program_number_bounds_in_formatters(spec: FanucDialectSpec, bad_num):
    task, plan = _make_task_and_plan()
    cfg = MachineConfig(program_number=bad_num, stock_size_mm=(20.0, 20.0, 10.0))
    with pytest.raises(ValueError):
        spec.format_typed_program([], plan=plan, task=task, config=cfg)
    with pytest.raises(ValueError):
        spec.format_shop_program(plan, task, cfg)
    with pytest.raises(ValueError):
        spec.format_legacy_positions_program(np.array([[0.5, 0.5, 0.5]]), cfg)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
@pytest.mark.parametrize(
    "bad_header",
    [
        "O00001 (HAAS FIVE DIGIT)",
        "O01001 (FIVE DIGIT)",
        "O001 (THREE DIGIT)",
        "O1 (ONE DIGIT)",
        "O0000 (ZERO)",
        "O9000 (MACRO RESERVED START)",
        "O9001 (MACRO RESERVED)",
        "O9999 (MACRO RESERVED END)",
        "(MISSING O NUMBER ENTIRELY)",
    ],
)
def test_verify_hardware_syntax_rejects_bad_o_numbers(
    spec: FanucDialectSpec, bad_header: str
):
    nc = "\n".join(
        [
            "%",
            bad_header,
            "G90 G94 G17 G49 G40 G80",
            "G21",
            "G54 G61",
            "G0 X1.0 Y2.0 Z6.0",
            "M30",
            "%",
        ]
    )
    with pytest.raises(HardwareSyntaxError):
        spec.verify_hardware_syntax(nc, units="mm", shop_mode=False)


# ---------------------------------------------------------------------------
# 6. Hardware syntax negative checks: decimals, R5 vs R5.0, AICC pairing, forbidden G/M
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
@pytest.mark.parametrize(
    "bad_motion_line",
    [
        "G0 X2 Y0.0 Z6.0",
        "G0 X2.0 Y0.0 Z6",
        "G1 X2.0 Y0.0 Z-1.0 F1194",
        "G2 X2.0 Y2.0 Z-1.0 I0 J2.0 F600.",
        "G2 X2.0 Y2.0 Z-1.0 R2 F600.",
    ],
)
def test_verify_hardware_syntax_rejects_missing_decimals(
    spec: FanucDialectSpec, bad_motion_line: str
):
    nc = "\n".join(
        [
            "%",
            "O0001 (DECIMAL CHECK)",
            "G90 G21 G54 G61",
            bad_motion_line,
            "M30",
            "%",
        ]
    )
    with pytest.raises(HardwareSyntaxError, match="decimal"):
        spec.verify_hardware_syntax(nc, units="mm", shop_mode=True)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
def test_verify_hardware_syntax_shop_decimal_caps(spec: FanucDialectSpec):
    # Shop mm caps: XYZ <= 3 decimals, F == 0 fractional digits (e.g. F1194., NOT F1194.0).
    too_many_xyz_mm = "%\nO0001 (TEST)\nG90 G21 G61\nG0 X1.1234 Y0.0 Z6.0\nM30\n%\n"
    with pytest.raises(HardwareSyntaxError, match="max 3"):
        spec.verify_hardware_syntax(too_many_xyz_mm, units="mm", shop_mode=True)

    too_many_feed_mm = "%\nO0001 (TEST)\nG90 G21 G61\nG1 X1.0 Y0.0 Z-1.0 F1194.0\nM30\n%\n"
    with pytest.raises(HardwareSyntaxError, match="max 0"):
        spec.verify_hardware_syntax(too_many_feed_mm, units="mm", shop_mode=True)

    # Shop inch caps: XYZ <= 4 decimals, F <= 1 fractional digit.
    too_many_feed_in = "%\nO0001 (TEST)\nG90 G20 G61\nG1 X1.0 Y0.0 Z-0.5 F47.05\nM30\n%\n"
    with pytest.raises(HardwareSyntaxError, match="max 1"):
        spec.verify_hardware_syntax(too_many_feed_in, units="inch", shop_mode=True)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
@pytest.mark.parametrize(
    "aicc_body",
    [
        # R5.0 has a decimal point (must be integer R5)
        ["G05.1 Q1 R5.0", "G1 X2.0 Y0.0 Z-1.0 F600.", "G05.1 Q0"],
        # R out of 1..10 range
        ["G05.1 Q1 R0", "G1 X2.0 Y0.0 Z-1.0 F600.", "G05.1 Q0"],
        ["G05.1 Q1 R11", "G1 X2.0 Y0.0 Z-1.0 F600.", "G05.1 Q0"],
        # Missing G05.1 Q0 before M30
        ["G05.1 Q1 R5", "G1 X2.0 Y0.0 Z-1.0 F600."],
        # Orphan G05.1 Q0 without prior G05.1 Q1
        ["G1 X2.0 Y0.0 Z-1.0 F600.", "G05.1 Q0"],
        # Motion word combined onto G05.1 block
        ["G05.1 Q1 R5 X2.0", "G05.1 Q0"],
        # Shorthand G5.1 missing matching Q0 before M30
        ["G5.1 Q1 R5", "G1 X2.0 Y0.0 Z-1.0 F600."],
    ],
)
def test_verify_hardware_syntax_aicc_g05_rules(
    spec: FanucDialectSpec, aicc_body: list[str]
):
    nc = "\n".join(
        [
            "%",
            "O0001 (AICC CHECK)",
            "G90 G21 G54",
            *aicc_body,
            "M30",
            "%",
        ]
    )
    with pytest.raises(HardwareSyntaxError):
        spec.verify_hardware_syntax(nc, units="mm", shop_mode=True)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
def test_verify_hardware_syntax_accepts_g5_1_shorthand_and_rejects_q0_after_m30(
    spec: FanucDialectSpec,
):
    valid_shorthand = "\n".join(
        [
            "%",
            "O0001 (SHORTHAND G5.1)",
            "G90 G21 G54",
            "G5.1 Q1 R5",
            "G1 X2.0 Y0.0 Z-1.0 F600.",
            "G5.1 Q0",
            "M30",
            "%",
        ]
    )
    spec.verify_hardware_syntax(valid_shorthand, units="mm", shop_mode=True)

    q0_after_m30 = "\n".join(
        [
            "%",
            "O0001 (Q0 AFTER M30)",
            "G90 G21 G54",
            "G05.1 Q1 R5",
            "G1 X2.0 Y0.0 Z-1.0 F600.",
            "M30",
            "G05.1 Q0",
            "%",
        ]
    )
    with pytest.raises(HardwareSyntaxError):
        spec.verify_hardware_syntax(q0_after_m30, units="mm", shop_mode=True)



@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
@pytest.mark.parametrize(
    "forbidden_block",
    [
        "G64",
        "G64 P0.05",
        "G187 P2 E0.05",
        "G91.1",
        "M250",
        "M251",
        "M252",
        "M253",
    ],
)
def test_verify_hardware_syntax_rejects_foreign_smoothing_and_arc_codes(
    spec: FanucDialectSpec, forbidden_block: str
):
    nc = "\n".join(
        [
            "%",
            "O0001 (FOREIGN CODE CHECK)",
            "G90 G21 G54",
            forbidden_block,
            "G0 X2.0 Y0.0 Z6.0",
            "M30",
            "%",
        ]
    )
    with pytest.raises(HardwareSyntaxError):
        spec.verify_hardware_syntax(nc, units="mm", shop_mode=False)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=["fanuc", "robodrill"])
def test_verify_hardware_syntax_requires_leading_and_trailing_percent(
    spec: FanucDialectSpec,
):
    no_leading = "O0001 (TEST)\nG90 G21 G61\nG0 X1.0 Y0.0 Z6.0\nM30\n%\n"
    no_trailing = "%\nO0001 (TEST)\nG90 G21 G61\nG0 X1.0 Y0.0 Z6.0\nM30\n"
    with pytest.raises(HardwareSyntaxError, match="leading '%'"):
        spec.verify_hardware_syntax(no_leading)
    with pytest.raises(HardwareSyntaxError, match="trailing '%'"):
        spec.verify_hardware_syntax(no_trailing)
