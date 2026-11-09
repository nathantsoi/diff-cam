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
    # None preserves the pilot's retained square. ``False`` is allowed only
    # when the declared island box is physically disjoint from stock.
    has_retained_island: bool | None = None
    # Local geometry coordinates: offsets from the declared work origin.
    # These are derived from the source bounds, never used to rewrite geometry.
    pocket_centre_xy: tuple[float, float] = (0.0, 0.0)
    island_centre_xy: tuple[float, float] = (0.0, 0.0)
    boundary_offset_margin_mm: float = 0.5
    provenance: str = "cam_seeds:pocket_with_island_seed/1"

    def as_dict(self):
        return {
            "outer_xy": self.outer_xy,
            "pocket_half_x": self.pocket_half_x,
            "pocket_half_y": self.pocket_half_y,
            "island_half_side": self.island_half_side,
            "has_retained_island": self.has_retained_island,
            "pocket_centre_xy": self.pocket_centre_xy,
            "island_centre_xy": self.island_centre_xy,
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


def _bounds_intersect(stock_bounds, island_bounds):
    """True iff two axis-aligned boxes share positive physical volume."""
    return all(
        stock_bounds[0][axis] < island_bounds[1][axis]
        and island_bounds[0][axis] < stock_bounds[1][axis]
        for axis in range(3)
    )


def seed_spec_for_task(task: C.TaskSpec) -> IslandSeedSpec:
    """Derive a safe seed recipe from the declared analytic task geometry.

    The bootstrap seed supports a bounded rectangular pocket, either a square
    retained pad or an explicitly empty retained region. The pocket and pad
    may be offset in work XY; the pad may stop below the stock top. Rectangular
    islands, stepped floors and free-form cases are rejected here, not silently
    re-fit to a simpler shape. Geometry failures are never performance filters.
    """
    geometry = task.original_part
    if task.tool.cutter_kind != C.CutterKind.FLAT:
        raise ValueError("this bootstrap seed requires an upright flat cutter")
    if geometry.kind != "analytic" or geometry.params.get("primitive") != "box_minus_pocket":
        raise ValueError("procedural seed requires analytic box-minus-pocket geometry")
    params = geometry.params
    stock = tuple(tuple(float(v) for v in params[key]) for key in ("box_min_mm", "box_max_mm"))
    pocket = tuple(tuple(float(v) for v in params[key]) for key in ("pocket_min_mm", "pocket_max_mm"))
    island = tuple(tuple(float(v) for v in params[key]) for key in ("island_min_mm", "island_max_mm"))
    for lo, hi in (stock, pocket, island):
        if (
            len(lo) != 3 or len(hi) != 3
            or any(not math.isfinite(value) for value in lo + hi)
            or any(lo[axis] >= hi[axis] for axis in range(3))
        ):
            raise ValueError("seed requires finite ordered three-dimensional bounds")
    if any(stock[0][axis] > pocket[0][axis] or stock[1][axis] < pocket[1][axis]
           for axis in range(3)):
        raise ValueError("pocket must be wholly contained in bounded stock")
    if pocket[1][2] != stock[1][2]:
        raise ValueError("this bootstrap seed requires an open top pocket")
    pocket_centre = tuple(
        (pocket[0][axis] + pocket[1][axis]) / 2.0 for axis in (0, 1)
    )
    radius = task.tool.cutter_radius_mm
    half_x = (pocket[1][0] - pocket[0][0]) / 2.0
    half_y = (pocket[1][1] - pocket[0][1]) / 2.0
    outer_x, outer_y = half_x - radius, half_y - radius
    if outer_x <= 0.0 or outer_y <= 0.0:
        raise ValueError("pocket is too narrow for the finite cutter perimeter")
    retained = _bounds_intersect(stock, island)
    if retained and params.get("empty_island") is True:
        raise ValueError("empty_island flag contradicts retained material overlapping stock")
    if not retained and params.get("empty_island") is not True:
        raise ValueError("declared island does not overlap stock; mark empty_island explicitly")
    half_side = 0.0
    island_centre = (0.0, 0.0)
    if retained:
        hx = (island[1][0] - island[0][0]) / 2.0
        hy = (island[1][1] - island[0][1]) / 2.0
        if hx <= 0.0 or not math.isclose(hx, hy, abs_tol=1e-9):
            raise ValueError("this seed supports square retained pads, not rectangles")
        half_side = hx
        if island[0][2] != pocket[0][2] or island[1][2] > pocket[1][2]:
            raise ValueError("this flat-floor seed requires a pad from the pocket floor, topped at or below stock top")
        if any(island[0][axis] < pocket[0][axis] or island[1][axis] > pocket[1][axis]
               for axis in (0, 1)):
            raise ValueError("retained pad must lie wholly inside the pocket XY")
        island_centre = tuple(
            (island[0][axis] + island[1][axis]) / 2.0 - pocket_centre[axis]
            for axis in (0, 1)
        )
        wall_gap = min(
            outer_x - abs(island_centre[0]), outer_y - abs(island_centre[1])
        )
        if wall_gap < half_side + radius:
            raise ValueError("pocket walls leave insufficient space around retained pad")
        minimum_raster_clip = math.sqrt(2.0) * half_side + radius + 0.5
        if wall_gap < minimum_raster_clip:
            raise ValueError("raster clearance cannot fit the cutter around the retained corner")
        clip = min(8.0, wall_gap)
    else:
        minimum_raster_clip = clip = 0.0
    if clip < minimum_raster_clip:
        raise ValueError("raster clearance would truncate the required retained-island gap")
    span_steps = max(3, math.ceil(2.0 * outer_y / radius))
    if span_steps % 2 == 0:
        span_steps += 1
    return IslandSeedSpec(
        outer_xy=(outer_x, outer_y),
        pocket_half_x=half_x,
        pocket_half_y=half_y,
        island_half_side=half_side,
        raster_clear_radius=clip,
        floor_z=pocket[0][2],
        clearance_z=stock[1][2] + 8.0,
        home_z=stock[1][2] + 10.0,
        y_rows=span_steps,
        has_retained_island=retained,
        pocket_centre_xy=pocket_centre,
        island_centre_xy=island_centre,
    )


def pocket_with_island_seed(
    task: C.TaskSpec,
    spec: IslandSeedSpec | None = None,
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
    derived = seed_spec_for_task(task)
    if spec is None:
        spec = derived
    retained = derived.has_retained_island
    if (
        tuple(spec.pocket_centre_xy) != derived.pocket_centre_xy
        or tuple(spec.island_centre_xy) != derived.island_centre_xy
    ):
        raise ValueError("seed geometry centres disagree with the declared source bounds")
    if spec.has_retained_island is not None and spec.has_retained_island != retained:
        raise ValueError("seed retained-island flag disagrees with the declared geometry")
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
    if retained and inner_gap < spec.boundary_offset_margin_mm:
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
    min_guard_to_top = spec.clearance_z - stock_top
    if min_guard_to_top <= 0 or spec.home_z <= stock_top:
        raise ValueError("stock-top rapid clearance must be strictly positive")

    to_machine = task.work_to_machine.apply
    to_work = task.work_to_machine.inverse().apply
    points: list[Point] = []
    px, py = spec.pocket_centre_xy
    current_source = to_work(task.initial_state.position_mm)
    current_work = (current_source[0] - px, current_source[1] - py, current_source[2])

    def to_machine_local(point: Point) -> Point:
        return to_machine((point[0] + px, point[1] + py, point[2]))

    def append(kind: C.SegmentKind, end_work: Point, tag: str, feed_mm_s=None) -> None:
        nonlocal current_work
        start_machine = to_machine_local(current_work)
        end_machine = to_machine_local(end_work)
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
        if kind == C.SegmentKind.RAPID and (current_work[2] <= stock_top or end_work[2] <= stock_top):
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
        if not retained:
            return 0.0, 0.0
        ix, iy = spec.island_centre_xy
        radial_sq = spec.raster_clear_radius**2 - (y - iy)**2
        radial_sq = spec.raster_clear_radius**2 - (y - iy)**2
        # Beyond the circle the two panel halves join at the pad-centre X,
        # retaining their wall-to-wall swath rather than dropping these rows.
        dx = math.sqrt(max(0.0, radial_sq))
        left, right = ix - dx, ix + dx
        if left < xmin - 1e-9 or right > ox + 1e-9:
            raise ValueError("retained-pad clearance circle escapes the pocket walls")
        return left, right

    def feed_at_floor(end: Point, tag: str) -> None:
        """Cut only outside the retained square, lifting unsafe floor links."""
        if not retained:
            append(C.SegmentKind.FEED, end, tag)
            return
        if current_work[2] != spec.floor_z or end[2] != spec.floor_z:
            raise ValueError("the floor-link guard requires start/end at floor_z")
        a = current_work[:2]
        b = end[:2]
        ix, iy = spec.island_centre_xy

        def square_clearance(t):
            x = a[0] + t * (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
            return math.hypot(
                max(abs(x - ix) - spec.island_half_side, 0.0),
                max(abs(y - iy) - spec.island_half_side, 0.0),
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
        inner_left, _ = raster_clip(y)
        outer_x = xmin
        if inner_left != outer_x:
            append(C.SegmentKind.FEED, (inner_left, y, spec.floor_z), f"raster-left-{row:02d}-inward")
            append(C.SegmentKind.FEED, (outer_x, y, spec.floor_z), f"raster-left-{row:02d}-outward")
        if row + 1 < len(ys):
            feed_at_floor((raster_clip(ys[row + 1])[0], ys[row + 1], spec.floor_z), "raster-link")

    panel_inner_x = raster_clip(ys[-1])[1]
    append(C.SegmentKind.FEED, (outer_x, ys[-1], spec.home_z), "panel-retract")
    append(C.SegmentKind.RAPID, (panel_inner_x, ys[-1], spec.home_z), "panel-reposition")
    append(C.SegmentKind.FEED, (panel_inner_x, ys[-1], spec.floor_z), "plunge")
    for row, y in enumerate(reversed(ys)):
        _, inner_right = raster_clip(y)
        outer_x = ox
        if inner_right != outer_x:
            append(C.SegmentKind.FEED, (inner_right, y, spec.floor_z), f"raster-right-{row:02d}-inward")
            append(C.SegmentKind.FEED, (outer_x, y, spec.floor_z), f"raster-right-{row:02d}-outward")
        if row + 1 < len(ys):
            feed_at_floor((raster_clip(ys[-row - 2])[1], ys[-row - 2], spec.floor_z), "raster-link")

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
    if retained:
        h = spec.island_half_side
        ix, iy = spec.island_centre_xy
        start = (ix + h + radius, iy - h - radius, spec.floor_z)
        feed_at_floor(start, "island-approach")
        corners = [
            (ix + h + radius, iy + h + radius, spec.floor_z),
            (ix - h - radius, iy + h + radius, spec.floor_z),
            (ix - h - radius, iy - h - radius, spec.floor_z),
            (ix + h + radius, iy - h - radius, spec.floor_z),
        ]
        for corner in corners:
            append(C.SegmentKind.FEED, corner, "island-finish")

        # A pad stopping below stock top has a removable cap, not retained
        # material above its top. Clear that cap at its declared physical Z.
        # A floor-level pass inside the pad would gouge it, so this cannot be
        # replaced by extending the lower-floor island-finishing contour.
        pad_top = float(task.original_part.params["island_max_mm"][2])
        if pad_top < stock_top:
            cap_half = h + radius
            cap_rows = max(3, math.ceil(2.0 * cap_half / radius))
            if cap_rows % 2 == 0:
                cap_rows += 1
            cap_ys = [iy - cap_half + 2.0 * cap_half * row / (cap_rows - 1)
                      for row in range(cap_rows)]
            append(C.SegmentKind.FEED, (start[0], start[1], spec.home_z), "cap-retract")
            cap_first = (ix - cap_half, cap_ys[0])
            append(C.SegmentKind.RAPID, (*cap_first, spec.home_z), "cap-reposition")
            append(C.SegmentKind.FEED, (*cap_first, pad_top), "pad-cap-plunge", spec.plunge_mm_s)
            for row, cap_y in enumerate(cap_ys):
                cap_x = ix + cap_half if row % 2 else ix - cap_half
                append(C.SegmentKind.FEED, (cap_x, cap_y, pad_top), f"pad-cap-{row:02d}")
                if row + 1 < len(cap_ys):
                    append(C.SegmentKind.FEED, (cap_x, cap_ys[row + 1], pad_top), "pad-cap-link")
            start = current_work
    else:
        start = current_work

    # First retract vertically above every retained wall/island. Only then make
    # a high non-cutting return to the known home point, avoiding diagonal dives.
    append(C.SegmentKind.FEED, (start[0], start[1], spec.home_z), "retract")
    append(C.SegmentKind.RAPID, (-px, -py, spec.home_z), "home")
    plan = C.MotionPlan(
        frame=C.Frame.MACHINE,
        initial_state=task.initial_state,
        segments=tuple(points),
        timing_rule=C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
        provenance=spec.provenance,
    )
    plan.validate_for_task(task)
    return plan


def seed_for_task(task: C.TaskSpec) -> C.MotionPlan:
    """Return the source-derived nominal seed for one supported task."""
    return pocket_with_island_seed(task, seed_spec_for_task(task))


def seed_deltas_mm(plan: C.MotionPlan) -> Sequence[Point]:
    """Return commanded tool-centre increments without resampling/chord replacement."""
    return tuple(
        tuple(segment.end_mm[i] - segment.start_mm[i] for i in range(3))
        for segment in plan.segments
    )


__all__ = ["IslandSeedSpec", "seed_spec_for_task", "pocket_with_island_seed", "seed_for_task", "seed_deltas_mm"]
