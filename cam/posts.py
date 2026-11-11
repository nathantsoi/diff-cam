# SPDX-License-Identifier: MIT
"""Configurable clean-room G-code post-processors across six CNC controller dialects.

Supported post-processors (registered in :data:`POSTS`, :func:`post_motion_plan`,
and :func:`post_shop_program`):

* :class:`RS274Post` (``"rs274"``)         -- LinuxCNC / NIST RS274NGC (``.ngc``, ``G91.1``, ``G64 P0.05`` / ``G61``)
* :class:`HaasPost` (``"haas"``)           -- Haas Next Generation Control (``.nc``, 5-digit ``O``, ``G187 P2 E0.05`` / ``G187``)
* :class:`FanucPost` (``"fanuc"``)         -- FANUC 0i / 30i / 31i & Compact (``.nc``, 4-digit ``O``, AICC II ``G05.1 Q1 R5`` / ``Q0``, ``G28``)
* :class:`RobodrillPost` (``"robodrill"``) -- FANUC Robodrill Alpha 31i-B5 (``.nc``, 4-digit ``O``, AICC II ``G05.1 Q1 R5`` / ``Q0``, ``G53``)
* :class:`MakinoPost` (``"makino"``)       -- Makino Professional 5 / 6 Super GI (``.nc``, 4-digit ``O``, SGI ``M252`` / ``M250``, ``G28``)
* :class:`MazakPost` (``"mazak"``)         -- Mazak Mazatrol Smooth / Matrix EIA/ISO (``.eia``, no leading ``%``, ``G61.1 P2`` / ``G64``, ``G53``)
"""

from __future__ import annotations

import numpy as np

from benchmarks.contracts import (
    Dialect,
    Frame,
    MotionPlan,
    SegmentKind,
    TaskSpec,
    TimingRule,
)
from .config import MachineConfig
from .dialects import (
    FANUC_SPEC,
    HAAS_NGC_SPEC,
    LINUXCNC_RS274_SPEC,
    MAKINO_SPEC,
    MAZAK_SPEC,
    ROBODRILL_SPEC,
    ControllerDialect,
    coerce_controller_dialect,
    get_dialect_spec,
    is_dialect_allowed_by_task,
)
from .dialects.base import (
    arc_offset_words,
    build_motion_lines,
    feed_out as _feed_out,
    fmt_real,
    len_out as _len_out,
    posted_work_vector as _posted_work_vector_real,
)


# ---------------------------------------------------------------------------
# Legacy formatting helpers (preserved for backward-compatible callers)
# ---------------------------------------------------------------------------
def _fmt(value: float, precision: int) -> str:
    """Format one finite coordinate value in legacy RS274 compact form."""
    return fmt_real(value, precision, force_decimal=False)


def _axis_words(point_mm, precision: int) -> str:
    x, y, z = point_mm
    return (
        f"X{_fmt(x, precision)} "
        f"Y{_fmt(y, precision)} "
        f"Z{_fmt(z, precision)}"
    )


def _posted_work_vector(point_work, config: MachineConfig, precision: int):
    """Coordinates exactly as a parser receives them for this post precision."""
    return _posted_work_vector_real(point_work, config, precision)


# ---------------------------------------------------------------------------
# Post-processor base + 6 controller implementations
# ---------------------------------------------------------------------------
class PostProcessor:
    """Base class. Subclasses implement :meth:`program`."""

    name = "base"

    def program(self, positions, config: MachineConfig = MachineConfig()) -> str:
        raise NotImplementedError

    @staticmethod
    def _validate(positions):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (T, 3); got {positions.shape}")
        if len(positions) == 0:
            raise ValueError("positions is empty")
        if not np.isfinite(positions).all():
            raise ValueError("positions must contain only finite values")
        return positions


class RS274Post(PostProcessor):
    """LinuxCNC / NIST RS274NGC post-processor."""

    name = "rs274"

    def program(self, positions, config: MachineConfig = MachineConfig()) -> str:
        return LINUXCNC_RS274_SPEC.format_legacy_positions_program(
            self._validate(positions), config
        )


class HaasPost(PostProcessor):
    """Haas Next Generation Control (NGC) post-processor."""

    name = "haas"

    def program(self, positions, config: MachineConfig = MachineConfig()) -> str:
        return HAAS_NGC_SPEC.format_legacy_positions_program(
            self._validate(positions), config
        )


class FanucPost(PostProcessor):
    """FANUC 0i / 30i / 31i & Compact post-processor."""

    name = "fanuc"

    def program(self, positions, config: MachineConfig = MachineConfig()) -> str:
        return FANUC_SPEC.format_legacy_positions_program(
            self._validate(positions), config
        )


class RobodrillPost(PostProcessor):
    """FANUC Robodrill Alpha (31i-B5) post-processor."""

    name = "robodrill"

    def program(self, positions, config: MachineConfig = MachineConfig()) -> str:
        return ROBODRILL_SPEC.format_legacy_positions_program(
            self._validate(positions), config
        )


class MakinoPost(PostProcessor):
    """Makino Professional 5 / 6 (Super GI) post-processor."""

    name = "makino"

    def program(self, positions, config: MachineConfig = MachineConfig()) -> str:
        return MAKINO_SPEC.format_legacy_positions_program(
            self._validate(positions), config
        )


class MazakPost(PostProcessor):
    """Mazak Mazatrol Smooth / Matrix (EIA/ISO) post-processor."""

    name = "mazak"

    def program(self, positions, config: MachineConfig = MachineConfig()) -> str:
        return MAZAK_SPEC.format_legacy_positions_program(
            self._validate(positions), config
        )


# ---------------------------------------------------------------------------
# Typed & shop-floor MotionPlan posting
# ---------------------------------------------------------------------------
class UnsupportedPostError(ValueError):
    """A motion/task feature that the requested NC dialect cannot preserve."""


def _arc_offset_words(start_work, end_work, centre_work, plane, config, precision, segment_id):
    return arc_offset_words(
        start_work,
        end_work,
        centre_work,
        plane,
        config,
        precision,
        segment_id,
        force_decimal=False,
    )


def _validate_typed_post(
    plan: MotionPlan,
    task: TaskSpec,
    ctrl_dialect: ControllerDialect,
    config: MachineConfig,
) -> None:
    if config.h_register != task.tool.tool_number:
        raise UnsupportedPostError(
            f"G43 H{config.h_register} does not match TaskSpec.tool.tool_number={task.tool.tool_number}"
        )
    if ctrl_dialect != ControllerDialect.RS274 and config.tool_number != task.tool.tool_number:
        raise UnsupportedPostError(
            f"T{config.tool_number} does not match TaskSpec.tool.tool_number={task.tool.tool_number}"
        )
    if not is_dialect_allowed_by_task(task, ctrl_dialect):
        raise UnsupportedPostError(f"dialect {ctrl_dialect.value!r} is disabled by this TaskSpec")
    if plan.frame != Frame.MACHINE:
        raise UnsupportedPostError("typed posting requires a machine-frame MotionPlan")
    if not plan.initial_state.known:
        raise UnsupportedPostError("cannot preserve an approach from an unknown initial state")
    if plan.initial_state.work_offset != "G54":
        raise UnsupportedPostError("only G54 is represented by TaskSpec.work_to_machine")
    if plan.timing_rule not in (TimingRule.EXACT_STOP_CONSTANT_SPEED, TimingRule.BLENDED_ACCEL_BOUNDED):
        raise UnsupportedPostError(
            "NC posting preserves exact_stop_constant_speed or blended_accel_bounded, not fixed_dt"
        )
    if task.motion_semantics.timing_rule != plan.timing_rule:
        raise UnsupportedPostError("plan and task timing rules differ")
    if not task.motion_semantics.arcs_supported and any(
        segment.kind == SegmentKind.ARC for segment in plan.segments
    ):
        raise UnsupportedPostError("this TaskSpec disables arc output")
    plan.validate_for_task(task)


def _typed_motion_lines(
    plan: MotionPlan,
    task: TaskSpec,
    config: MachineConfig,
    *,
    force_decimal: bool = False,
) -> list[str]:
    return build_motion_lines(
        plan,
        task,
        config,
        xyz_precision=config.precision,
        feed_precision=config.precision,
        force_decimal=force_decimal,
    )


def post_motion_plan(
    plan: MotionPlan,
    task: TaskSpec,
    dialect: Dialect | ControllerDialect | str = Dialect.RS274,
    config: MachineConfig | None = None,
) -> str:
    """Post machine-frame typed motion without adding synthetic approach/retract moves."""
    ctrl_dialect = coerce_controller_dialect(dialect)
    spec = get_dialect_spec(ctrl_dialect)
    config = config or MachineConfig(units="mm")
    _validate_typed_post(plan, task, ctrl_dialect, config)
    motion_lines = _typed_motion_lines(plan, task, config, force_decimal=spec.force_decimal)
    nc_text = spec.format_typed_program(
        motion_lines,
        plan=plan,
        task=task,
        config=config,
    )
    spec.verify_hardware_syntax(nc_text, units=config.units, shop_mode=False)
    return nc_text


def post_shop_program(
    plan: MotionPlan,
    task: TaskSpec,
    dialect: Dialect | ControllerDialect | str = ControllerDialect.HAAS,
    config: MachineConfig | None = None,
    *,
    case_label: str | None = None,
    blend_tolerance_mm: float = 0.05,
) -> str:
    """Post a standalone physical shop-floor program with safe home retracts and decimal caps."""
    ctrl_dialect = coerce_controller_dialect(dialect)
    spec = get_dialect_spec(ctrl_dialect)
    config = config or MachineConfig(
        units="mm",
        tool_number=task.tool.tool_number,
        length_offset=task.tool.tool_number,
    )
    _validate_typed_post(plan, task, ctrl_dialect, config)
    nc_text = spec.format_shop_program(
        plan,
        task,
        config,
        case_label=case_label,
        blend_tolerance_mm=blend_tolerance_mm,
    )
    spec.verify_hardware_syntax(nc_text, units=config.units, shop_mode=True)
    return nc_text


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
POSTS = {
    RS274Post.name: RS274Post,
    HaasPost.name: HaasPost,
    FanucPost.name: FanucPost,
    RobodrillPost.name: RobodrillPost,
    MakinoPost.name: MakinoPost,
    MazakPost.name: MazakPost,
}


def get_post(name: str = "rs274") -> PostProcessor:
    """Return an instance of the named post-processor."""
    try:
        return POSTS[name.lower()]()
    except KeyError:
        raise ValueError(
            f"unknown post '{name}'; available posts: {sorted(POSTS)}"
        )
