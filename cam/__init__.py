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
from .posts import PostProcessor, RS274Post, HaasPost, get_post, POSTS
from .gcode_export import trajectory_to_gcode, save_gcode
from .gcode_parser import parse_gcode, segment_waypoints, MotionSegment
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
    "PostProcessor",
    "RS274Post",
    "HaasPost",
    "get_post",
    "POSTS",
    "trajectory_to_gcode",
    "save_gcode",
    "parse_gcode",
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
