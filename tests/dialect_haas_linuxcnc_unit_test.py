# SPDX-License-Identifier: MIT
"""Unit tests for clean-room Haas NGC and LinuxCNC RS274 dialect specifications."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.contracts import (
    ArcData,
    ArcDirection,
    ArcPlane,
    CutterKind,
    Dialect,
    Frame,
    GeometrySource,
    GridSpec,
    MachineLimits,
    MachineState,
    MotionPlan,
    MotionSemantics,
    OperationTarget,
    Segment,
    SegmentKind,
    SourceInfo,
    Split,
    TaskSpec,
    TimingRule,
    Tolerances,
    ToolAssembly,
    Transform,
    blended_segment_durations,
)
from cam.config import MachineConfig
from cam.dialects.base import (
    ControllerDialect,
    HardwareSyntaxError,
    build_motion_lines,
    strip_nc_comments,
)
from cam.dialects.haas_linuxcnc import (
    HAAS_NGC_SPEC,
    LINUXCNC_RS274_SPEC,
    HaasNGCDialectSpec,
    LinuxCNCRS274DialectSpec,
    format_haas_ngc_shop_program,
)
from cam.gcode_parser import parse_gcode, parse_motion_plan
from cam.posts import RS274Post


def _make_task(
    *,
    timing_rule: TimingRule = TimingRule.BLENDED_ACCEL_BOUNDED,
    work_offset_mm: tuple[float, float, float] = (100.0, -50.0, -80.0),
    init_work_mm: tuple[float, float, float] = (2.0, -10.5, 6.0),
    length_offset_applied: bool = True,
    cutter_kind: CutterKind = CutterKind.FLAT,
    tool_number: int = 1,
) -> TaskSpec:
    w2m = Transform.translation(work_offset_mm)
    init_machine = w2m.apply(init_work_mm)
    stock_geom = GeometrySource(
        kind="analytic",
        params={"primitive": "box", "size_mm": [50.0, 50.0, 25.0]},
    )
    return TaskSpec(
        task_id="test_haas_linuxcnc_dialect",
        family="pocket",
        split=Split.DEV,
        source=SourceInfo(origin="unit_test:dialect_haas_linuxcnc@1", license="MIT"),
        length_unit="mm",
        stock=stock_geom,
        original_part=stock_geom,
        operation_target=OperationTarget(geometry=stock_geom),
        grid=GridSpec(shape=(10, 10, 10), origin_mm=(-25.0, -25.0, -25.0), voxel_mm=5.0),
        work_to_machine=w2m,
        initial_state=MachineState(
            position_mm=init_machine,
            frame=Frame.MACHINE,
            known=True,
            length_offset_applied=length_offset_applied,
            work_offset="G54",
        ),
        tool=ToolAssembly(
            cutter_kind=cutter_kind,
            cutter_radius_mm=3.0,
            flute_length_mm=18.0,
            shank_radius_mm=3.0,
            holder_bottom_mm=30.0,
            holder_radius_mm=15.0,
            holder_height_mm=40.0,
            tool_number=tool_number,
        ),
        limits=MachineLimits(
            max_feed_mm_s=60.0,
            rapid_mm_s=100.0,
            max_accel_mm_s2=500.0,
            envelope_min_mm=(-300.0, -300.0, -300.0),
            envelope_max_mm=(300.0, 300.0, 100.0),
        ),
        tolerances=Tolerances(
            completion_residual_mm3=1.0,
            gouge_volume_mm3=0.0,
            local_surface_error_mm=0.1,
            clearance_mm=1.0,
        ),
        motion_semantics=MotionSemantics(
            dialects=(Dialect.RS274, Dialect.HAAS),
            timing_rule=timing_rule,
            arcs_supported=True,
        ),
    )


def _make_plan(
    task: TaskSpec,
    *,
    include_arc: bool = True,
    fractional_coords: bool = False,
) -> MotionPlan:
    w2m = task.work_to_machine
    p0_w = w2m.inverse().apply(task.initial_state.position_mm)
    if fractional_coords:
        p1_w = (15.549331, -10.500442, 6.0)
        p2_w = (15.549331, -10.500442, -2.125678)
        p3_w = (22.0, -4.0, -2.125678)
        feed_mm_s = 19.902083  # 1194.12498 mm/min -> >2 decimals before shop rounding
    else:
        p1_w = (2.0, -10.5, 6.0)
        p2_w = (2.0, -10.5, -3.0)
        p3_w = (12.0, -10.5, -3.0)
        feed_mm_s = 19.9  # 1194.0 mm/min exact integer in mm/min!

    p0 = w2m.apply(p0_w)
    p1 = w2m.apply(p1_w)
    p2 = w2m.apply(p2_w)
    p3 = w2m.apply(p3_w)

    raw_segs: list[Segment] = [
        Segment(
            id="seg-0",
            kind=SegmentKind.RAPID,
            start_mm=p0,
            end_mm=p1,
            duration_s=math.dist(p0, p1) / task.limits.rapid_mm_s,
        ),
        Segment(
            id="seg-1",
            kind=SegmentKind.FEED,
            start_mm=p1,
            end_mm=p2,
            duration_s=math.dist(p1, p2) / feed_mm_s,
            feed_mm_s=feed_mm_s,
            tag="plunge",
        ),
        Segment(
            id="seg-2",
            kind=SegmentKind.FEED,
            start_mm=p2,
            end_mm=p3,
            duration_s=math.dist(p2, p3) / feed_mm_s,
            feed_mm_s=feed_mm_s,
        ),
    ]
    if include_arc:
        # Quarter-circle CCW in XY from (12, -10.5, -3) around (12, -5.5, -3) -> (17, -5.5, -3)
        c_w = (p3_w[0], p3_w[1] + 5.0, p3_w[2])
        p4_w = (p3_w[0] + 5.0, p3_w[1] + 5.0, p3_w[2])
        arc = ArcData(
            centre_mm=w2m.apply(c_w),
            plane=ArcPlane.XY,
            direction=ArcDirection.CCW,
            turns=0,
        )
        p4 = w2m.apply(p4_w)
        arc_len = 0.5 * math.pi * 5.0
        raw_segs.append(
            Segment(
                id="seg-3",
                kind=SegmentKind.ARC,
                start_mm=p3,
                end_mm=p4,
                duration_s=arc_len / feed_mm_s,
                feed_mm_s=feed_mm_s,
                arc=arc,
            )
        )

    if task.motion_semantics.timing_rule == TimingRule.BLENDED_ACCEL_BOUNDED:
        durs = blended_segment_durations(
            raw_segs, task.limits.rapid_mm_s, task.limits.max_accel_mm_s2
        )
        raw_segs = [
            Segment(
                id=s.id,
                kind=s.kind,
                start_mm=s.start_mm,
                end_mm=s.end_mm,
                duration_s=d,
                feed_mm_s=s.feed_mm_s,
                arc=s.arc,
                tag=s.tag,
            )
            for s, d in zip(raw_segs, durs)
        ]

    plan = MotionPlan(
        frame=Frame.MACHINE,
        initial_state=task.initial_state,
        segments=tuple(raw_segs),
        timing_rule=task.motion_semantics.timing_rule,
    )
    plan.validate_for_task(task)
    return plan


def _executable_lines(nc_text: str) -> list[str]:
    return [strip_nc_comments(line) for line in nc_text.splitlines() if strip_nc_comments(line)]


class TestDialectMetadata:
    def test_singleton_types_and_static_attributes(self) -> None:
        assert isinstance(LINUXCNC_RS274_SPEC, LinuxCNCRS274DialectSpec)
        assert LINUXCNC_RS274_SPEC.dialect == ControllerDialect.RS274
        assert LINUXCNC_RS274_SPEC.display_name == "LinuxCNC / NIST RS274NGC"
        assert LINUXCNC_RS274_SPEC.file_extension == ".ngc"
        assert LINUXCNC_RS274_SPEC.force_decimal is False
        assert LINUXCNC_RS274_SPEC.program_digits is None
        assert LINUXCNC_RS274_SPEC.max_program_number == 99999
        assert LINUXCNC_RS274_SPEC.leading_percent is False
        assert LINUXCNC_RS274_SPEC.trailing_percent is False
        assert LINUXCNC_RS274_SPEC.shop_xyz_decimals("mm") == 3
        assert LINUXCNC_RS274_SPEC.shop_xyz_decimals("inch") == 4
        assert LINUXCNC_RS274_SPEC.shop_feed_decimals("mm") == 2
        assert LINUXCNC_RS274_SPEC.shop_feed_decimals("inch") == 3

        assert isinstance(HAAS_NGC_SPEC, HaasNGCDialectSpec)
        assert HAAS_NGC_SPEC.dialect == ControllerDialect.HAAS
        assert HAAS_NGC_SPEC.display_name == "Haas Next Generation Control (NGC)"
        assert HAAS_NGC_SPEC.file_extension == ".nc"
        assert HAAS_NGC_SPEC.force_decimal is True
        assert HAAS_NGC_SPEC.program_digits == 5
        assert HAAS_NGC_SPEC.max_program_number == 99999
        assert HAAS_NGC_SPEC.leading_percent is True
        assert HAAS_NGC_SPEC.trailing_percent is True
        assert HAAS_NGC_SPEC.shop_xyz_decimals("mm") == 3
        assert HAAS_NGC_SPEC.shop_xyz_decimals("inch") == 4
        assert HAAS_NGC_SPEC.shop_feed_decimals("mm") == 2
        assert HAAS_NGC_SPEC.shop_feed_decimals("inch") == 3


class TestHaasNGCDialect:
    @pytest.mark.parametrize(
        ("units", "expected_unit_code", "expected_tol"),
        [
            ("mm", "G21", "E0.05"),
            ("inch", "G20", "E0.002"),
        ],
    )
    def test_typed_blended_mm_and_inch_uses_g187_never_g64(
        self, units: str, expected_unit_code: str, expected_tol: str
    ) -> None:
        task = _make_task(timing_rule=TimingRule.BLENDED_ACCEL_BOUNDED)
        plan = _make_plan(task)
        cfg = MachineConfig(units=units, program_number=7, program_name="POCKET_01")
        motion_lines = build_motion_lines(
            plan,
            task,
            cfg,
            xyz_precision=cfg.precision,
            feed_precision=cfg.precision,
            force_decimal=True,
        )
        nc = HAAS_NGC_SPEC.format_typed_program(
            motion_lines, plan=plan, task=task, config=cfg
        )

        HAAS_NGC_SPEC.verify_hardware_syntax(nc, units=units, shop_mode=False)
        exec_lines = _executable_lines(nc)
        assert exec_lines[0] == "%"
        assert exec_lines[1] == "O00007"
        assert expected_unit_code in exec_lines
        assert f"G187 P2 {expected_tol}" in exec_lines
        # Standalone G187 smoothing reset before M09/M05/M30 shutdown
        reset_idx = exec_lines.index("G187")
        m30_idx = exec_lines.index("M30")
        assert reset_idx < m30_idx
        assert exec_lines[-1] == "%"
        assert all("G64" not in line for line in exec_lines)
        assert all("G91.1" not in line for line in exec_lines)

        # Round-trips through strict MotionPlan parser
        parsed = parse_motion_plan(nc, task, dialect=ControllerDialect.HAAS)
        assert len(parsed.segments) == len(plan.segments)

    @pytest.mark.parametrize("units", ["mm", "inch"])
    def test_typed_exact_stop_uses_g61_zero_g187_zero_g64(self, units: str) -> None:
        task = _make_task(timing_rule=TimingRule.EXACT_STOP_CONSTANT_SPEED)
        plan = _make_plan(task)
        cfg = MachineConfig(units=units, program_number=42)
        motion_lines = build_motion_lines(
            plan,
            task,
            cfg,
            xyz_precision=cfg.precision,
            feed_precision=cfg.precision,
            force_decimal=True,
        )
        nc = HAAS_NGC_SPEC.format_typed_program(
            motion_lines, plan=plan, task=task, config=cfg
        )

        HAAS_NGC_SPEC.verify_hardware_syntax(nc, units=units, shop_mode=False)
        exec_lines = _executable_lines(nc)
        assert "G54 G61" in exec_lines
        assert all("G187" not in line for line in exec_lines)
        assert all("G64" not in line for line in exec_lines)

        parsed = parse_motion_plan(nc, task, dialect=ControllerDialect.HAAS)
        assert len(parsed.segments) == len(plan.segments)

    @pytest.mark.parametrize(
        ("units", "expected_unit_code", "expected_tol"),
        [
            ("mm", "G21", "E0.05"),
            ("inch", "G20", "E0.002"),
        ],
    )
    def test_shop_blended_and_exact_stop_safe_split_positioning_and_decimals(
        self, units: str, expected_unit_code: str, expected_tol: str
    ) -> None:
        for rule in (
            TimingRule.BLENDED_ACCEL_BOUNDED,
            TimingRule.EXACT_STOP_CONSTANT_SPEED,
        ):
            task = _make_task(timing_rule=rule)
            plan = _make_plan(task, fractional_coords=True)
            cfg = MachineConfig(units=units, program_number=123, coolant=True)
            nc = format_haas_ngc_shop_program(
                plan, task, cfg, case_label="MINI_MILL_CASE (A); %1"
            )

            HAAS_NGC_SPEC.verify_hardware_syntax(nc, units=units, shop_mode=True)
            assert "O00123 (DIFFCAM MIT HAAS MINI MILL NGC - MINI_MILL_CASE A 1)" in nc
            exec_lines = _executable_lines(nc)
            assert exec_lines[0] == "%"
            assert exec_lines[-1] == "%"
            assert expected_unit_code in exec_lines
            assert all("G64" not in line for line in exec_lines)

            # Verify safe split XY-then-Z approach (no 3-axis dogleg rapid on approach)
            home_idx = exec_lines.index("G90 G53 G0 Z0.0")
            xy_line = exec_lines[home_idx + 1]
            z_line = exec_lines[home_idx + 2]
            assert xy_line.startswith("G0 X") and "Y" in xy_line and "Z" not in xy_line
            assert z_line.startswith("G0 Z") and "X" not in z_line and "Y" not in z_line

            if rule == TimingRule.BLENDED_ACCEL_BOUNDED:
                assert f"G187 P2 {expected_tol}" in exec_lines
                assert "G187" in exec_lines
                assert "G61" not in exec_lines
            else:
                assert "G61" in exec_lines
                assert all("G187" not in line for line in exec_lines)

            # Verify final retract to machine Z0.0 before M30
            m30_idx = exec_lines.index("M30")
            assert exec_lines[m30_idx - 1] == "G90 G53 G0 Z0.0"

            # Verify shop program also parses cleanly through strict parser
            parsed = parse_motion_plan(nc, task, dialect=ControllerDialect.HAAS)
            assert len(parsed.segments) >= len(plan.segments)

    def test_integer_coordinates_and_feeds_retain_mandatory_decimal_points(self) -> None:
        task = _make_task(
            timing_rule=TimingRule.EXACT_STOP_CONSTANT_SPEED,
            init_work_mm=(2.0, -10.5, 6.0),
        )
        plan = _make_plan(task, include_arc=True, fractional_coords=False)
        nc = format_haas_ngc_shop_program(plan, task, program_number=5)

        assert "G0 X2.0 Y-10.5" in nc
        assert "G0 Z6.0" in nc
        assert "G0 X2.0 Y-10.5 Z6.0" in nc
        assert "G1 X2.0 Y-10.5 Z-3.0 F1194.0" in nc
        assert "G3 X17.0 Y-5.5 Z-3.0 I0.0 J5.0 F1194.0" in nc
        HAAS_NGC_SPEC.verify_hardware_syntax(nc, units="mm", shop_mode=True)

    def test_shop_options_no_coolant_g49_ball_cutter_default_config(self) -> None:
        task = _make_task(
            timing_rule=TimingRule.EXACT_STOP_CONSTANT_SPEED,
            length_offset_applied=False,
            cutter_kind=CutterKind.BALL,
            tool_number=3,
        )
        plan = _make_plan(task, include_arc=False)
        nc_default = format_haas_ngc_shop_program(plan, task, program_number=99)
        assert "O00099" in nc_default
        assert "T3 M06" in nc_default
        assert "BALL END MILL" in nc_default
        exec_default = _executable_lines(nc_default)
        assert "G49" in exec_default
        assert all("G43" not in line for line in exec_default)

        cfg_dry = MachineConfig(
            units="mm", tool_number=3, length_offset=3, coolant=False, program_number=8
        )
        nc_dry = HAAS_NGC_SPEC.format_shop_program(plan, task, cfg_dry)
        HAAS_NGC_SPEC.verify_hardware_syntax(nc_dry, units="mm", shop_mode=True)
        exec_dry = _executable_lines(nc_dry)
        assert "M08" not in exec_dry
        assert "M09" not in exec_dry

    def test_verify_hardware_syntax_rejects_haas_hazards(self) -> None:
        task = _make_task(timing_rule=TimingRule.BLENDED_ACCEL_BOUNDED)
        plan = _make_plan(task, include_arc=False)
        valid_shop = format_haas_ngc_shop_program(plan, task, program_number=1)
        HAAS_NGC_SPEC.verify_hardware_syntax(valid_shop, units="mm", shop_mode=True)

        # Comments mentioning forbidden codes do NOT trigger false positives
        commented = valid_shop.replace(
            "G90 G94 G17 G40 G49 G80",
            "(DO NOT USE G64 OR G91.1 OR Z6)\nG90 G94 G17 G40 G49 G80",
        )
        HAAS_NGC_SPEC.verify_hardware_syntax(commented, units="mm", shop_mode=True)

        # 1. Missing decimal point on Z6 or F1194 -> least-input-increment crash hazard
        with pytest.raises(HardwareSyntaxError, match="Z6"):
            HAAS_NGC_SPEC.verify_hardware_syntax(
                valid_shop.replace("G0 Z6.0", "G0 Z6"), units="mm", shop_mode=True
            )
        with pytest.raises(HardwareSyntaxError, match="F1194"):
            HAAS_NGC_SPEC.verify_hardware_syntax(
                valid_shop.replace("F1194.0", "F1194"), units="mm", shop_mode=False
            )

        # 2. 6-decimal float rejected in shop_mode=True, allowed in shop_mode=False
        six_dec = valid_shop.replace("X2.0", "X15.549331")
        HAAS_NGC_SPEC.verify_hardware_syntax(six_dec, units="mm", shop_mode=False)
        with pytest.raises(HardwareSyntaxError, match="X15.549331"):
            HAAS_NGC_SPEC.verify_hardware_syntax(six_dec, units="mm", shop_mode=True)

        over_feed = valid_shop.replace("F1194.0", "F1194.125")
        with pytest.raises(HardwareSyntaxError, match="F1194.125"):
            HAAS_NGC_SPEC.verify_hardware_syntax(over_feed, units="mm", shop_mode=True)

        # 3. Forbidden G64 (Haas Alarm 302), G91.1, G05.1/G5.1, M250..M253
        for bad_code in ("G64", "G64 P0.05", "G91.1", "G05.1 Q1", "G5.1 Q0", "M250", "M253"):
            corrupted = valid_shop.replace("G187 P2 E0.05", bad_code)
            with pytest.raises(HardwareSyntaxError):
                HAAS_NGC_SPEC.verify_hardware_syntax(corrupted, units="mm", shop_mode=False)

        # 4. Program number width: reject 4-digit O0001, 6-digit O000001, zero O00000
        for bad_o in ("O0001", "O000001", "O00000"):
            with pytest.raises(HardwareSyntaxError):
                HAAS_NGC_SPEC.verify_hardware_syntax(
                    valid_shop.replace("O00001", bad_o), units="mm", shop_mode=False
                )

        # 5. Missing leading or trailing '%'
        with pytest.raises(HardwareSyntaxError):
            HAAS_NGC_SPEC.verify_hardware_syntax(
                valid_shop.lstrip("%\n"), units="mm", shop_mode=False
            )
        with pytest.raises(HardwareSyntaxError):
            HAAS_NGC_SPEC.verify_hardware_syntax(
                valid_shop.rstrip("\n%"), units="mm", shop_mode=False
            )
        with pytest.raises(HardwareSyntaxError):
            HAAS_NGC_SPEC.verify_hardware_syntax(
                valid_shop.replace("M05", "%\nM05"), units="mm", shop_mode=False
            )

        # 6. Non-integer P on G187
        with pytest.raises(HardwareSyntaxError):
            HAAS_NGC_SPEC.verify_hardware_syntax(
                valid_shop.replace("G187 P2 E0.05", "G187 P2.0 E0.05"),
                units="mm",
                shop_mode=False,
            )

    def test_legacy_positions_program_haas(self) -> None:
        cfg = MachineConfig(
            stock_size_mm=(40.0, 40.0, 20.0),
            stock_origin_in=(-8.0, -6.0, -4.0),
            units="mm",
            precision=4,
            program_number=12,
        )
        pts = np.array(
            [
                [0.5, 0.5, 1.0],
                [0.75, 0.5, 0.8],
                [0.75, 0.75, 0.8],
            ],
            dtype=np.float64,
        )
        nc = HAAS_NGC_SPEC.format_legacy_positions_program(pts, cfg)
        HAAS_NGC_SPEC.verify_hardware_syntax(nc, units="mm", shop_mode=False)
        assert "O00012 (DIFFCAM)" in nc
        assert "G10 L2 P1 " in nc
        assert "G90 G53 G00 Z0.0" in nc
        segs = parse_gcode(nc, cfg)
        assert len(segs) >= 3

        with pytest.raises(ValueError, match="empty"):
            HAAS_NGC_SPEC.format_legacy_positions_program(np.zeros((0, 3)), cfg)
        with pytest.raises(ValueError, match="finite"):
            HAAS_NGC_SPEC.format_legacy_positions_program(
                np.array([[0.0, np.nan, 0.0]]), cfg
            )


class TestLinuxCNCRS274Dialect:
    @pytest.mark.parametrize(
        ("units", "expected_unit_code", "expected_path_ctrl"),
        [
            ("mm", "G21", "G54 G64 P0.05"),
            ("inch", "G20", "G54 G64 P0.002"),
        ],
    )
    def test_typed_blended_and_exact_stop_output(
        self, units: str, expected_unit_code: str, expected_path_ctrl: str
    ) -> None:
        task_blend = _make_task(timing_rule=TimingRule.BLENDED_ACCEL_BOUNDED)
        plan_blend = _make_plan(task_blend)
        cfg = MachineConfig(units=units)
        motion_blend = build_motion_lines(
            plan_blend,
            task_blend,
            cfg,
            xyz_precision=cfg.precision,
            feed_precision=cfg.precision,
            force_decimal=False,
        )
        nc_blend = LINUXCNC_RS274_SPEC.format_typed_program(
            motion_blend, plan=plan_blend, task=task_blend, config=cfg
        )

        LINUXCNC_RS274_SPEC.verify_hardware_syntax(nc_blend, units=units, shop_mode=False)
        lines = nc_blend.splitlines()
        assert lines[0] == "(Generated by diff-cam typed MotionPlan RS274 post)"
        assert lines[1] == expected_unit_code
        assert lines[2] == "G17 G40 G49 G80 G90 G94 G91.1"
        assert lines[3] == expected_path_ctrl
        assert lines[4] == "G43 H1"
        assert lines[-1] == "M2"
        assert "%" not in nc_blend

        parsed_blend = parse_motion_plan(
            nc_blend, task_blend, dialect=ControllerDialect.RS274
        )
        assert len(parsed_blend.segments) == len(plan_blend.segments)

        # Exact-stop typed output
        task_exact = _make_task(timing_rule=TimingRule.EXACT_STOP_CONSTANT_SPEED)
        plan_exact = _make_plan(task_exact)
        motion_exact = build_motion_lines(
            plan_exact,
            task_exact,
            cfg,
            xyz_precision=cfg.precision,
            feed_precision=cfg.precision,
            force_decimal=False,
        )
        nc_exact = LINUXCNC_RS274_SPEC.format_typed_program(
            motion_exact, plan=plan_exact, task=task_exact, config=cfg
        )
        LINUXCNC_RS274_SPEC.verify_hardware_syntax(nc_exact, units=units, shop_mode=False)
        assert "G54 G61" in nc_exact.splitlines()
        assert "G64" not in nc_exact

    @pytest.mark.parametrize(
        ("units", "expected_unit_code", "expected_blend_line"),
        [
            ("mm", "G21", "G64 P0.05"),
            ("inch", "G20", "G64 P0.002"),
        ],
    )
    def test_shop_program_framing_split_positioning_and_roundtrip(
        self, units: str, expected_unit_code: str, expected_blend_line: str
    ) -> None:
        task = _make_task(timing_rule=TimingRule.BLENDED_ACCEL_BOUNDED)
        plan = _make_plan(task, fractional_coords=True)
        cfg = MachineConfig(units=units, coolant=True)
        nc = LINUXCNC_RS274_SPEC.format_shop_program(
            plan, task, cfg, case_label="LINUXCNC_POCKET"
        )

        LINUXCNC_RS274_SPEC.verify_hardware_syntax(nc, units=units, shop_mode=True)
        exec_lines = _executable_lines(nc)
        assert exec_lines[0] == "%"
        assert exec_lines[-1] == "%"
        assert "(DIFFCAM MIT LINUXCNC RS274 - LINUXCNC_POCKET)" in nc
        assert "G90 G94 G17 G91.1 G40 G49 G80" in exec_lines
        assert expected_unit_code in exec_lines
        assert expected_blend_line in exec_lines
        assert "M30" in exec_lines
        assert "M2" not in exec_lines

        # Safe split XY-then-Z approach
        home_idx = exec_lines.index("G90 G53 G0 Z0.0")
        assert exec_lines[home_idx + 1].startswith("G0 X")
        assert "Z" not in exec_lines[home_idx + 1]
        assert exec_lines[home_idx + 2].startswith("G0 Z")

        parsed = parse_motion_plan(nc, task, dialect=ControllerDialect.RS274)
        assert len(parsed.segments) >= len(plan.segments)

    def test_verify_hardware_syntax_rejects_proprietary_codes_and_shop_decimal_violations(
        self,
    ) -> None:
        task = _make_task(timing_rule=TimingRule.BLENDED_ACCEL_BOUNDED)
        plan = _make_plan(task, include_arc=False)
        cfg = MachineConfig(units="mm")
        shop_nc = LINUXCNC_RS274_SPEC.format_shop_program(plan, task, cfg)
        LINUXCNC_RS274_SPEC.verify_hardware_syntax(shop_nc, units="mm", shop_mode=True)

        for bad_code in ("G187 P2 E0.05", "G187", "G05.1 Q1", "G5.1 Q0", "M250", "M251", "M252", "M253"):
            with pytest.raises(HardwareSyntaxError):
                LINUXCNC_RS274_SPEC.verify_hardware_syntax(
                    shop_nc.replace("G64 P0.05", bad_code),
                    units="mm",
                    shop_mode=False,
                )

        # Missing decimal allowed in typed RS274 (shop_mode=False), rejected in shop_mode=True
        no_dec = shop_nc.replace("G0 Z6.0", "G0 Z6")
        LINUXCNC_RS274_SPEC.verify_hardware_syntax(no_dec, units="mm", shop_mode=False)
        with pytest.raises(HardwareSyntaxError, match="Z6"):
            LINUXCNC_RS274_SPEC.verify_hardware_syntax(no_dec, units="mm", shop_mode=True)

        with pytest.raises(HardwareSyntaxError, match="X15.549331"):
            LINUXCNC_RS274_SPEC.verify_hardware_syntax(
                shop_nc.replace("X2.0", "X15.549331"), units="mm", shop_mode=True
            )

    def test_legacy_positions_program_matches_rs274_post_byte_for_byte(self) -> None:
        cfg = MachineConfig(stock_size_mm=(50.0, 40.0, 20.0), units="mm", precision=6)
        pts = np.array(
            [
                [0.5, 0.5, 1.0],
                [0.6, 0.4, 0.85],
                [0.8, 0.4, 0.85],
            ],
            dtype=np.float64,
        )
        nc_spec = LINUXCNC_RS274_SPEC.format_legacy_positions_program(pts, cfg)
        nc_legacy = RS274Post().program(pts, cfg)
        assert nc_spec == nc_legacy
        LINUXCNC_RS274_SPEC.verify_hardware_syntax(nc_spec, units="mm", shop_mode=False)
