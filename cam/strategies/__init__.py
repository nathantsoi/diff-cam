# SPDX-License-Identifier: MIT
"""Modular MIT clean-room 2.5D CAM toolpathing strategies (`@2`).

Exports the four standalone 2.5D CAM strategy engines:
- :mod:`cam.strategies.facing` (`FacingConfig`, `generate_facing_toolpath`)
- :mod:`cam.strategies.pocket` (`PocketConfig`, `generate_pocket_toolpath`)
- :mod:`cam.strategies.adaptive` (`AdaptiveConfig`, `generate_adaptive_toolpath`, `estimate_radial_engagement_profile`)
- :mod:`cam.strategies.contour` (`ContourConfig`, `generate_contour_toolpath`)

along with a unified family dispatcher :func:`generate_cleanroom_toolpath`.
"""

from __future__ import annotations

from benchmarks import contracts as C

from .adaptive import (
    AdaptiveConfig,
    estimate_radial_engagement_profile,
    generate_adaptive_toolpath,
)
from .contour import ContourConfig, generate_contour_toolpath
from .facing import FacingConfig, generate_facing_toolpath
from .pocket import PocketConfig, generate_pocket_toolpath


def generate_cleanroom_toolpath(
    task: C.TaskSpec,
    *,
    config: FacingConfig | PocketConfig | AdaptiveConfig | ContourConfig | None = None,
) -> C.MotionPlan:
    """Dispatch ``task`` by ``task.family`` to its clean-room 2.5D CAM strategy."""
    if task.family == "facing_open_boss":
        if config is not None and not isinstance(config, FacingConfig):
            raise TypeError(f"Expected FacingConfig for {task.family!r}, got {type(config).__name__}")
        return generate_facing_toolpath(task, config=config or FacingConfig())
    if task.family == "pocket_island":
        if config is not None and not isinstance(config, PocketConfig):
            raise TypeError(f"Expected PocketConfig for {task.family!r}, got {type(config).__name__}")
        return generate_pocket_toolpath(task, config=config or PocketConfig())
    if task.family == "adaptive_channel":
        if config is not None and not isinstance(config, AdaptiveConfig):
            raise TypeError(f"Expected AdaptiveConfig for {task.family!r}, got {type(config).__name__}")
        return generate_adaptive_toolpath(task, config=config or AdaptiveConfig())
    if task.family == "contour_profile":
        if config is not None and not isinstance(config, ContourConfig):
            raise TypeError(f"Expected ContourConfig for {task.family!r}, got {type(config).__name__}")
        return generate_contour_toolpath(task, config=config or ContourConfig())
    raise ValueError(
        f"Unknown task family {task.family!r}; expected one of "
        "('facing_open_boss', 'pocket_island', 'adaptive_channel', 'contour_profile')"
    )


__all__ = [
    "AdaptiveConfig",
    "ContourConfig",
    "FacingConfig",
    "PocketConfig",
    "estimate_radial_engagement_profile",
    "generate_adaptive_toolpath",
    "generate_cleanroom_toolpath",
    "generate_contour_toolpath",
    "generate_facing_toolpath",
    "generate_pocket_toolpath",
]
