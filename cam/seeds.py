"""Conservative, in-repository CAM seed heuristics (not adaptive CAM).

``pocket_with_island_seed`` uses a clipped boustrophedon raster, a complete
outside rectangle, and a polygonal square-island finishing contour.
Floor-level connectivity must clear the retained island. There is no claim
of constant engagement, trochoidal motion, or optimality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from benchmarks import contracts as C


Point = tuple[float, float, float]


@dataclass(frozen=True)
class IslandSeedSpec:
    """Physical source and clearance parameters for one correction-free seed."""

    #: Full-radius centres on the 24 x 16 mm original walls: +/-22,+/-14 mm.
    outer_xy: tuple[float, float] = (22.0, 14.0)
    pocket_half_x: float = 24.0
    pocket_half_y: float = 16.0
    island_half_side: float = 3.5
    # Rasters stop on this clearance circle before the square island finish.
    raster_clear_radius: float = 8.0
    floor_z: float = -6.0
    clearance_z: float = 8.0
    home_z: float = 10.0
    y_rows: int = 15
    feed_mm_s: float = 6.0
    plunge_mm_s: float = 4.0
    boundary_offset_margin_mm: float = 0.5
    provenance: str = "cam_seeds:pocket_with_island_seed/1"

    def as_dict(self):
        return {
            "outer_xy": self.outer_xy,
            "pocket_half_x": self.pocket_half_x,
            "pocket_half_y": self.pocket_half_y,
            "island_half_side": self.island_half_side,
            "raster_clear_radius": self.raster_clear_radius,
            "floor_z": self.floor_z,
            "clearance_z": self.clearance_z,
            "home_z": self.home_z,
            "y_rows": self.y_rows,
            "feed_mm_s": self.feed_mm_s,
            "plunge_mm_s": self.plunge_mm_s,
            "boundary_offset_margin_mm": self.boundary_offset_margin_mm,
            "provenance": self.provenance,
        }


def pocket_with_island_seed(
    task: C.TaskSpec,
    spec: IslandSeedSpec = IslandSeedSpec(),
) -> C.MotionPlan:
    """Return one exact-stop, line-only typed nominal pocket program.

    The island is the axis-aligned square with half-side ``island_half_side``.
    The four-line finishing square uses the exact cutter radius, while the
    raster clearance circle leaves additional island margin. A floor-link
    guard lifts paths that would penetrate the retained square. Holder access
    is checked against the actual stock-top height at the cutting floor.
    Feed rates do not assert constant engagement or an adaptive CAM method.
    """
    if task.initial_state.frame != C.Frame.MACHINE or not task.initial_state.known:
        raise ValueError("pocket seed requires a known machine-frame starting position")
    if task.motion_semantics.timing_rule != C.TimingRule.EXACT_STOP_CONSTANT_SPEED:
        raise ValueError("this seed declares exact-stop constant-speed execution")
    if task.initial_state.position_mm is None:
        raise ValueError("known initial position is required for an explicit approach")
    if spec.y_rows < 3 or spec.y_rows % 2 == 0:
        raise ValueError("y_rows must be odd and at least 3 for a centred raster")
    if spec.home_z < spec.clearance_z:
        raise ValueError("home must not be below clearance elevation")
    if spec.pocket_half_x <= 0 or spec.pocket_half_y <= 0:
        raise ValueError("invalid pocket bounds")
    radius = task.tool.cutter_radius_mm
    outer_wall_gap_x = spec.pocket_half_x - abs(spec.outer_xy[0])
    outer_wall_gap_y = spec.pocket_half_y - abs(spec.outer_xy[1])
    island_outer_corner = math.sqrt(2.0) * spec.island_half_side
    inner_gap = spec.raster_clear_radius - island_outer_corner - radius
    if min(outer_wall_gap_x, outer_wall_gap_y) < radius:
        raise ValueError("outer perimeter must keep the finite cutter inside original walls")
    if inner_gap < spec.boundary_offset_margin_mm:
        raise ValueError("inner island clearance leaves less than the cutter safety margin")
    # The finishing square touches only the intended island walls. Bulk links
    # clear the retained square in desired waste. At the cutting floor this
    # assembly's holder is above the stock top with a cell/qualification margin.
    stock_top = task.stock.params["max_mm"][2]
    holder_above_stock = spec.floor_z + task.tool.holder_bottom_mm - stock_top
    required_z_margin = task.tolerances.clearance_mm + math.sqrt(3.0) * task.grid.voxel_mm / 2.0
    if holder_above_stock < required_z_margin:
        raise ValueError("inner bulk clearing requires the holder above stock at the cutting floor")
    if task.tool.holder_radius_mm < radius:
        raise ValueError("holder must be coaxial and at least as wide as the cutter")
    # This seed needs non-cutting access to the top of the stock at clearance z.
    min_guard_to_top = spec.clearance_z
    if min_guard_to_top <= 0 or spec.home_z <= 0:
        raise ValueError("stock-top rapid clearance must be strictly positive")

    to_machine = task.work_to_machine.apply
    to_work = task.work_to_machine.inverse().apply
    points: list[Point] = []
    current_work = to_work(task.initial_state.position_mm)

    def append(kind: C.SegmentKind, end_work: Point, tag: str, feed_mm_s=None) -> None:
        nonlocal current_work
        start_machine = to_machine(current_work)
        end_machine = to_machine(end_work)
        length = math.dist(start_machine, end_machine)
        if length == 0.0:
            # A raster clipped to the current point adds no executed motion.
            # Omitting it avoids a different G-code line count through posts.
            return
        if kind == C.SegmentKind.RAPID:
            speed = task.limits.rapid_mm_s
            feed = None
        else:
            speed = spec.feed_mm_s if feed_mm_s is None else feed_mm_s
            if speed <= 0.0 or speed > task.limits.max_feed_mm_s:
                raise ValueError(f"feed {speed} mm/s exceeds task limits")
            feed = speed
        segment_id = f"{tag}-{len(points):03d}"
        if kind == C.SegmentKind.RAPID and (current_work[2] <= 0.0 or end_work[2] <= 0.0):
            raise ValueError("use feed, not rapid, while any part of the move overlaps stock-top Z=0")
        points.append(
            C.Segment(
                id=segment_id,
                kind=kind,
                start_mm=start_machine,
                end_mm=end_machine,
                duration_s=length / speed,
                feed_mm_s=feed,
                tag=tag,
            )
        )
        current_work = end_work

    ox, oy = map(abs, spec.outer_xy)
    first = (-ox, -oy, spec.clearance_z)
    append(C.SegmentKind.RAPID, first, "approach")
    append(C.SegmentKind.FEED, (first[0], first[1], spec.floor_z), "plunge", spec.plunge_mm_s)

    # Centre the odd y-row family. Its effective pitch does not exceed cutter
    # diameter (2r), but this is a geometric overlap choice, not engagement proof.
    ys = [(-oy + 2.0 * oy * index / (spec.y_rows - 1)) for index in range(spec.y_rows)]
    xmin = -ox

    def raster_clip(y):
        return math.sqrt(max(0.0, spec.raster_clear_radius**2 - y**2))

    def feed_at_floor(end: Point, tag: str) -> None:
        """Cut only outside the retained square, lifting unsafe floor links."""
        if current_work[2] != spec.floor_z or end[2] != spec.floor_z:
            raise ValueError("the floor-link guard requires start/end at floor_z")
        a = current_work[:2]
        b = end[:2]

        def square_clearance(t):
            x = a[0] + t * (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
            return math.hypot(
                max(abs(x) - spec.island_half_side, 0.0),
                max(abs(y) - spec.island_half_side, 0.0),
            )

        # Squared distance to a convex square is convex along a straight line.
        lo, hi = 0.0, 1.0
        for _ in range(80):
            left = (2.0 * lo + hi) / 3.0
            right = (lo + 2.0 * hi) / 3.0
            if square_clearance(left) < square_clearance(right):
                hi = right
            else:
                lo = left
        nearest = square_clearance((lo + hi) / 2.0)
        if min(square_clearance(0.0), square_clearance(1.0)) < radius - 1e-9:
            raise ValueError("floor-link endpoints must not penetrate the retained island")
        if nearest < radius - 1e-9:
            feed_up = (current_work[0], current_work[1], spec.home_z)
            across = (end[0], end[1], spec.home_z)
            append(C.SegmentKind.FEED, feed_up, "cross-disc-retract")
            append(C.SegmentKind.RAPID, across, "cross-disc-reposition")
            append(C.SegmentKind.FEED, end, "plunge")
        else:
            append(C.SegmentKind.FEED, end, tag)

    # Complete the left-side strips first. Outward returns and Y links never
    # cross the island; no right-side point is entered from the opposite wall.
    for row, y in enumerate(ys):
        clipped = raster_clip(y)
        outer_x, inner_x = xmin, -clipped
        if inner_x != outer_x:
            append(C.SegmentKind.FEED, (inner_x, y, spec.floor_z), f"raster-left-{row:02d}-inward")
            append(C.SegmentKind.FEED, (outer_x, y, spec.floor_z), f"raster-left-{row:02d}-outward")
        if row + 1 < len(ys):
            feed_at_floor((-raster_clip(ys[row + 1]), ys[row + 1], spec.floor_z), "raster-link")

    append(C.SegmentKind.FEED, (outer_x, ys[-1], spec.home_z), "panel-retract")
    append(C.SegmentKind.RAPID, (clipped, ys[-1], spec.home_z), "panel-reposition")
    append(C.SegmentKind.FEED, (clipped, ys[-1], spec.floor_z), "plunge")
    for row, y in enumerate(reversed(ys)):
        clipped = raster_clip(y)
        outer_x, inner_x = ox, clipped
        if inner_x != outer_x:
            append(C.SegmentKind.FEED, (inner_x, y, spec.floor_z), f"raster-right-{row:02d}-inward")
            append(C.SegmentKind.FEED, (outer_x, y, spec.floor_z), f"raster-right-{row:02d}-outward")
        if row + 1 < len(ys):
            feed_at_floor((raster_clip(ys[-row - 2]), ys[-row - 2], spec.floor_z), "raster-link")

    # Join the current outside-corner point into a complete outer rectangle.
    current_xy = current_work[:2]
    corners = [
        (ox, oy, spec.floor_z),
        (-ox, oy, spec.floor_z),
        (-ox, -oy, spec.floor_z),
        (ox, -oy, spec.floor_z),
    ]
    index = min(
        range(len(corners)),
        key=lambda corner: math.dist(current_xy, corners[corner][:2]),
    )
    if math.dist(current_xy, corners[index][:2]) > 1e-9:
        raise ValueError("roughing must finish on an outer rectangle corner")

    # Cross-island links lift before a high transfer. Cutter contact on the
    # finishing square itself is intentional, not a free pass for other moves.
    def outer_feed(end: Point) -> None:
        feed_at_floor(end, "outer-finish")

    for offset in (1, 2, 3, 0):
        outer_feed(corners[(index + offset) % 4])

    # The retained island is a square, not an approximation by circles. Its
    # cutter-radius offset is exactly four lines; the corners lie farther
    # from the island and cannot chord through the retained corner material.
    h = spec.island_half_side
    start = (h + radius, -h - radius, spec.floor_z)
    feed_at_floor(start, "island-approach")
    corners = [
        (h + radius, h + radius, spec.floor_z),
        (-h - radius, h + radius, spec.floor_z),
        (-h - radius, -h - radius, spec.floor_z),
        (h + radius, -h - radius, spec.floor_z),
    ]
    for corner in corners:
        append(C.SegmentKind.FEED, corner, "island-finish")

    # First retract vertically above every retained wall/island. Only then make
    # a high non-cutting return to the known home point, avoiding diagonal dives.
    append(C.SegmentKind.FEED, (start[0], start[1], spec.home_z), "retract")
    append(C.SegmentKind.RAPID, (0.0, 0.0, spec.home_z), "home")
    plan = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=task.initial_state,
        segments=tuple(points),
        timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
        provenance=spec.provenance,
    )
    plan.validate_for_task(task)
    return plan


def seed_deltas_mm(plan: C.MotionPlan) -> Sequence[Point]:
    """Return commanded tool-centre increments without resampling/chord replacement."""
    return tuple(
        tuple(segment.end_mm[i] - segment.start_mm[i] for i in range(3))
        for segment in plan.segments
    )


__all__ = ["IslandSeedSpec", "pocket_with_island_seed", "seed_deltas_mm"]
