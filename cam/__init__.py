# SPDX-License-Identifier: MIT
"""diff-cam CAM layer: G-code export and G-code -> trajectory path planning.

Pipeline mirrored from LinuxCNC: a trajectory (``(T, 3)`` unit-cube positions)
is exported to RS274/NGC G-code, then re-planned into a time-sampled "executed"
trajectory by an acceleration-limited trapezoidal planner (exact-stop / G61).
Round-trip fidelity is measured with the trajectory-similarity metrics.
"""

from .config import MachineConfig
from . import units
from .units import (
    MM_PER_INCH,
    inch_to_mm,
    mm_to_inch,
    ipm_to_mm_per_min,
    mm_per_min_to_ipm,
    ipm_to_mm_per_s,
    mm_per_s_to_ipm,
)
from .dialects import ControllerDialect, get_dialect_spec, verify_hardware_syntax
from .posts import (
    POSTS,
    FanucPost,
    HaasPost,
    MakinoPost,
    MazakPost,
    PostProcessor,
    RS274Post,
    RobodrillPost,
    get_post,
    post_motion_plan,
    post_shop_program,
)
from .gcode_export import trajectory_to_gcode, save_gcode
from .gcode_parser import parse_gcode, parse_motion_plan, parse_shop_motion_plan, segment_waypoints, MotionSegment
from .trajectory_planner import plan_trajectory, gcode_to_trajectory
from .trajectory_metrics import (
    discrete_frechet,
    dtw_distance,
    resampled_rmse,
    waypoint_roundtrip_error,
    all_metrics,
)

__all__ = [
    "MachineConfig",
    "units",
    "MM_PER_INCH",
    "inch_to_mm",
    "mm_to_inch",
    "ipm_to_mm_per_min",
    "mm_per_min_to_ipm",
    "ipm_to_mm_per_s",
    "mm_per_s_to_ipm",
    "ControllerDialect",
    "get_dialect_spec",
    "verify_hardware_syntax",
    "PostProcessor",
    "RS274Post",
    "HaasPost",
    "FanucPost",
    "RobodrillPost",
    "MakinoPost",
    "MazakPost",
    "get_post",
    "POSTS",
    "post_motion_plan",
    "post_shop_program",
    "trajectory_to_gcode",
    "save_gcode",
    "parse_gcode",
    "parse_motion_plan",
    "parse_shop_motion_plan",
    "segment_waypoints",
    "MotionSegment",
    "plan_trajectory",
    "gcode_to_trajectory",
    "discrete_frechet",
    "dtw_distance",
    "resampled_rmse",
    "waypoint_roundtrip_error",
    "all_metrics",
]
