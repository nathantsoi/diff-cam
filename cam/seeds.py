"""Conservative, in-repository CAM seed heuristics (not adaptive CAM).

``pocket_with_island_seed`` uses a clipped boustrophedon raster, a complete
outside rectangle, and a polygonal square-island finishing contour.
Floor-level connectivity must clear the retained island. There is no claim
of constant engagement, trochoidal motion, or optimality.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

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


# D-08 initialization-seed construction. These helpers deliberately use a
# small, explicit grammar rather than interpreting arbitrary protocol prose.
_PLACEHOLDER = re.compile(r"\b(?:TO[-_ ]BE[-_ ]FROZEN|TODO|TBD)\b", re.IGNORECASE)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def _canonical_json(value) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _declaration_hash(declaration) -> str:
    if not isinstance(declaration, Mapping):
        raise ValueError("initialization declaration must be a mapping")
    return "sha256:" + hashlib.sha256(_canonical_json(declaration)).hexdigest()


def _reject_placeholders(value) -> None:
    if isinstance(value, str) and (_PLACEHOLDER.search(value) or
            (value.strip().startswith("<") and value.strip().endswith(">"))):
        raise ValueError("initialization declaration contains an unresolved placeholder")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_placeholders(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_placeholders(child)


def _fields(value, required: Sequence[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    missing = sorted(set(required) - set(value))
    if missing:
        raise ValueError(f"{where}: missing required fields {missing}")
    unknown = sorted(set(value) - set(required))
    if unknown:
        raise ValueError(f"{where}: unknown fields {unknown}; use a versioned declaration amendment")
    return value


def _nonempty_text(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected nonempty text")
    return value.strip()


def _nonnegative_int(value, where: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{where}: expected a nonnegative integer")
    return int(value)


def _finite_number(value, where: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{where}: expected a finite number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{where}: expected a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{where}: expected a finite number")
    if result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"{where}: scale must be {'positive' if not allow_zero else 'nonnegative'}")
    return result


def _truth_flag(mapping: Mapping[str, Any], key: str) -> None:
    if mapping.get(key) is not True:
        raise ValueError(f"{key}: declaration must explicitly require true")


def _straight_fields(declaration) -> Mapping[str, Any]:
    required = ("start", "end", "segment_placement", "feed_and_rapid_entry",
                "same_segment_count_and_k", "safe_rapid_entry",
                "respect_task_feed_limits", "efficiency_not_required")
    statement = declaration.get("straight_line_init", declaration)
    result = _nonempty_mapping_fields(statement, required, "straight_line_init")
    for name in ("start", "end", "segment_placement", "feed_and_rapid_entry"):
        _nonempty_text(result[name], "straight_line_init." + name)
    for name in required[4:]:
        _truth_flag(result, name)
    return result


def _nonempty_mapping_fields(value, required: Sequence[str], where: str) -> Mapping[str, Any]:
    result = _fields(value, required, where)
    if not result:
        raise ValueError(f"{where}: declaration must not be empty")
    return result


def _random_fields(declaration) -> Mapping[str, Any]:
    statement = declaration.get("random_init", declaration)
    return _nonempty_mapping_fields(
        statement, ("parameter_space", "distribution", "scale", "seed_derivation_rule"),
        "random_init")


def _scale_mm(task: C.TaskSpec, cam: C.MotionPlan, scale) -> tuple[float, dict[str, Any]]:
    fields = _fields(scale, ("fraction", "reference"), "random_init.scale")
    fraction = _finite_number(fields["fraction"], "random_init.scale.fraction")
    reference = _fields(fields["reference"], ("kind", "description", "units"),
                        "random_init.scale.reference")
    kind = reference["kind"]
    _nonempty_text(reference["description"], "random_init.scale.reference.description")
    if reference["units"] != "mm":
        raise ValueError("random_init.scale.reference.units must be explicit 'mm' for registered references")
    if kind == "stock_xy_extent":
        lo, hi = task.stock.params["min_mm"][:2], task.stock.params["max_mm"][:2]
        reference_value = float(math.hypot(hi[0] - lo[0], hi[1] - lo[1]))
    elif kind == "cam_delta_norm":
        reference_value = float(np.linalg.norm(np.asarray(seed_deltas_mm(cam), dtype=np.float64)))
    else:
        raise ValueError(f"random_init.scale.reference.kind {kind!r} is not an executable registered reference")
    if not math.isfinite(reference_value) or reference_value <= 0:
        raise ValueError(f"random_init.scale.reference {kind} must resolve to a positive finite mm value")
    return fraction * reference_value, {
        "fraction": fraction, "reference_kind": kind,
        "reference_value_mm": reference_value,
        "scale_mm": fraction * reference_value,
        "fraction_zero_is_explicit_degenerate_probe": fraction == 0.0,
    }


def _seed_index(value) -> int:
    return _nonnegative_int(value, "requested seed")


def _rng(task: C.TaskSpec, declaration, declaration_hash: str,
         rule, seed: int) -> tuple[np.random.Generator, dict[str, Any]]:
    rule = _fields(rule, ("algorithm", "root_seed", "input", "per_seed_rule"),
                   "random_init.seed_derivation_rule")
    if rule["algorithm"] != "sha256_v1":
        raise ValueError("seed_derivation_rule.algorithm must be 'sha256_v1'; no implicit fallback")
    if rule["input"] != "task_hash,declaration_hash,root_seed,seed":
        raise ValueError("seed_derivation_rule.input must be 'task_hash,declaration_hash,root_seed,seed'")
    if rule["per_seed_rule"] != "first_8_bytes_little_endian_uint64":
        raise ValueError("seed_derivation_rule.per_seed_rule must be 'first_8_bytes_little_endian_uint64'")
    root_seed = _nonnegative_int(rule["root_seed"], "random_init.seed_derivation_rule.root_seed")
    payload = {
        "algorithm": "sha256_v1",
        "task_hash": task.task_hash,
        "declaration_hash": declaration_hash,
        "root_seed": root_seed,
        "seed": seed,
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    derived = int.from_bytes(bytes.fromhex(digest)[:8], "little", signed=False)
    return np.random.default_rng(np.random.SeedSequence(derived)), {
        "algorithm": "sha256_v1",
        "input": rule["input"], "per_seed_rule": rule["per_seed_rule"],
        "root_seed": root_seed, "requested_seed": seed,
        "payload": payload, "digest_sha256": "sha256:" + digest,
        "derived_seed_u64": derived,
        "generator": "numpy.default_rng(numpy.random.SeedSequence(derived_seed_u64))",
    }


@dataclass(frozen=True)
class SeedResult:
    """Typed outcome for an initialization plan; invalid never means clipped.

    ``valid`` says the constructor returned a validated MotionPlan. It is not
    safety, completion, efficiency-eligibility or machinability evidence.
    """
    plan: C.MotionPlan | None
    declaration_hash: str
    derivation: str
    seed: int | None = None
    invalid_reason: str | None = None

    def __post_init__(self):
        if self.plan is None and (not isinstance(self.invalid_reason, str)
                                  or not self.invalid_reason.strip()):
            raise ValueError("an invalid SeedResult requires a nonempty invalid_reason")
        if self.plan is not None and self.invalid_reason is not None:
            raise ValueError("a valid SeedResult cannot carry an invalid_reason")
        if self.declaration_hash and _SHA256.fullmatch(self.declaration_hash) is None:
            raise ValueError("declaration_hash must be empty or a sha256:<64 hex> value")
        if self.seed is not None:
            _nonnegative_int(self.seed, "SeedResult.seed")
        if not isinstance(self.derivation, str):
            raise ValueError("derivation must be a canonical JSON document")
        if not isinstance(json.loads(self.derivation), Mapping):
            raise ValueError("derivation must contain a JSON object")

    @property
    def valid(self) -> bool:
        return self.plan is not None

    def require_plan(self) -> C.MotionPlan:
        if self.plan is None:
            raise ValueError(self.invalid_reason)
        return self.plan

    def to_record(self) -> dict[str, Any]:
        return {"schema": "diffcam-noinit-seed-result/1", "valid": self.valid,
                "declaration_hash": self.declaration_hash, "seed": self.seed,
                "derivation": json.loads(self.derivation),
                "motion": None if self.plan is None else self.plan.to_dict(),
                "invalid_reason": self.invalid_reason}


def _invalid(declaration, seed, reason) -> SeedResult:
    try:
        declaration_hash = _declaration_hash(declaration)
    except (TypeError, ValueError):
        declaration_hash = ""
    clean_seed = (seed if isinstance(seed, (int, np.integer))
                  and not isinstance(seed, (bool, np.bool_)) and seed >= 0 else None)
    return SeedResult(None, declaration_hash,
                      _canonical_json({"schema": "diffcam-init-seed/1",
                                       "status": "invalid", "reason": str(reason),
                                       "requested_seed": clean_seed}).decode("utf-8"),
                      clean_seed, str(reason))


def _endpoint(cam: C.MotionPlan, value, where: str) -> Point:
    text = _nonempty_text(value, where)
    selector = text.casefold()
    if selector == "cam_start":
        return tuple(float(v) for v in cam.initial_state.position_mm)
    if selector == "cam_end":
        return tuple(float(v) for v in cam.segments[-1].end_mm)
    if selector.startswith("machine_mm:"):
        try:
            raw = ast.literal_eval(text.split(":", 1)[1].strip())
            if isinstance(raw, (list, tuple)) and len(raw) == 3:
                return tuple(_finite_number(v, where + ".coordinate") for v in raw)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"{where}: malformed finite machine_mm coordinate") from error
    raise ValueError(f"{where}: use 'cam_start', 'cam_end', or 'machine_mm:(x,y,z)'")


def _line_positions(task: C.TaskSpec, cam: C.MotionPlan, declaration) -> np.ndarray:
    statement = _straight_fields(declaration)
    if statement["segment_placement"] != "uniform":
        raise ValueError("straight_line_init.segment_placement must be the registered 'uniform' placement")
    if len(cam.segments) < 4:
        raise ValueError("D-08 line-seed construction requires at least four CAM segments for the native K basis")
    start = _endpoint(cam, statement["start"], "straight_line_init.start")
    end = _endpoint(cam, statement["end"], "straight_line_init.end")
    count = len(cam.segments)
    # This does not omit duplicate positions or zero-length segments: preserving
    # T is more important than manufacturing a less degenerate-looking program.
    return np.linspace(start, end, count + 1, dtype=np.float64)


def _motion_from_positions(task: C.TaskSpec, cam: C.MotionPlan, positions,
                           metadata: Mapping[str, Any]) -> C.MotionPlan:
    # This routine only constructs/validates MotionPlans; it never enters the simulator.
    if not cam.segments:
        raise ValueError("a nonempty CAM anchor is required")
    if any(segment.arc is not None for segment in cam.segments):
        raise ValueError("D-08 line-seed construction does not replace CAM arcs with implicit chords")
    if cam.timing_rule != C.TimingRule.EXACT_STOP_CONSTANT_SPEED:
        raise ValueError("this initializer currently requires exact-stop constant-speed motion semantics")
    points = np.asarray(positions, dtype=np.float64)
    if points.shape != (len(cam.segments) + 1, 3) or not np.isfinite(points).all():
        raise ValueError("generated motion positions must be finite (T+1, 3) coordinates")
    if cam.initial_state.position_mm is None or cam.initial_state.known is False:
        raise ValueError("a known safe CAM initial position is required")
    if math.dist(points[0], cam.initial_state.position_mm) > C.MotionPlan.CONTIGUITY_TOL_MM:
        raise ValueError("generated path does not start at the safe CAM/task initial position")
    lo, hi = task.limits.envelope_min_mm, task.limits.envelope_max_mm
    if np.any(points < lo) or np.any(points > hi):
        raise ValueError("generated motion point is outside the machine envelope; refusing to clip it")
    if cam.segments[0].kind != C.SegmentKind.RAPID:
        raise ValueError("safe entry requires the CAM anchor's first segment to be RAPID")
    inverse = task.work_to_machine.inverse().apply
    stock_top = task.stock.params["max_mm"][2]
    if any(inverse(point)[2] <= stock_top for point in (points[0], points[1])):
        raise ValueError("generated first RAPID entry does not remain above the stock-top plane")
    segments = []
    for index, original in enumerate(cam.segments):
        rapid = original.kind == C.SegmentKind.RAPID
        if rapid:
            feed = None
        else:
            feed = _finite_number(original.feed_mm_s, f"CAM segment {original.id}.feed_mm_s", allow_zero=False)
            if feed > task.limits.max_feed_mm_s:
                raise ValueError(f"feed {feed} mm/s exceeds the task limit in {original.id}")
        start = tuple(float(v) for v in points[index])
        end = tuple(float(v) for v in points[index + 1])
        speed = task.limits.rapid_mm_s if rapid else feed
        tag = original.tag or "motion"
        segments.append(C.Segment(
            id=original.id, kind=original.kind, start_mm=start, end_mm=end,
            duration_s=math.dist(start, end) / speed, feed_mm_s=feed,
            tag=tag + ":noinit", arc=None))
    provenance = json.loads(json.dumps(metadata, allow_nan=False))
    plan = C.MotionPlan(cam.frame, cam.initial_state, tuple(segments),
                        cam.timing_rule, provenance=json.dumps(provenance, sort_keys=True,
                                                               separators=(",", ":"), allow_nan=False))
    plan.validate_for_task(task)
    return plan


def straight_line_seed(task: C.TaskSpec, declaration) -> SeedResult:
    """Build D-08's declared uniform straight-line initialization.

    ``cam_start``/``cam_end`` name CAM's initial/final positions. Between them,
    T equal placements are used, with CAM's original segment kinds and rates.
    Literal absolute MACHINE positions are available as ``machine_mm:(x,y,z)``.
    No implicit endpoint, minimum-motion or efficiency criterion is invented.
    """
    try:
        if not isinstance(task, C.TaskSpec):
            raise ValueError("task must be a TaskSpec")
        declaration_hash = _declaration_hash(declaration)
        _reject_placeholders(declaration)
        cam = seed_for_task(task)
        points = _line_positions(task, cam, declaration)
        metadata = {
            "schema": "diffcam-init-seed/1", "mode": "straight_line",
            "task_hash": task.task_hash, "cam_motion_hash": cam.motion_hash,
            "declaration_hash": declaration_hash,
            "segment_count": len(cam.segments),
            "placement": "uniform", "endpoint_start": list(map(float, points[0])),
            "endpoint_end": list(map(float, points[-1])),
            "feed_rule": "retain each CAM feed_mm_s or task rapid_mm_s; never clamp speeds",
            "safe_entry": "first RAPID endpoint pair strictly above the work-frame stock top",
            "scope": "structural seed validity only; collapse/infeasibility is a valid later outcome",
        }
        plan = _motion_from_positions(task, cam, points, metadata)
        zero_length = sum(math.dist(segment.start_mm, segment.end_mm) == 0.0
                          for segment in plan.segments)
        metadata["zero_length_segments"] = zero_length
        metadata["validation"] = "validate_for_task passed"
        plan = _motion_from_positions(task, cam, points, metadata)
        return SeedResult(plan, declaration_hash,
                          json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                          seed=None)
    except (TypeError, ValueError, C.ContractError) as error:
        return _invalid(declaration, None, str(error))


def random_seed(task: C.TaskSpec, declaration, seed: int) -> SeedResult:
    """Build a deterministic random initialization with recorded derivation.

    ``theta`` samples independent normalized residual controls in the actual
    CAM ``fit_parameterization`` map (zero => CAM). ``delta`` adds physical
    centered displacement noise to the declared straight line (zero => that
    straight line). Gaussian scale is its standard deviation; uniform scale is
    its symmetric half-range. Each scale comes from the explicit mm reference.
    No global-RNG state, arbitrary heuristic seed formula or clipping is used.
    """
    try:
        if not isinstance(task, C.TaskSpec):
            raise ValueError("task must be a TaskSpec")
        requested_seed = _seed_index(seed)
        declaration_hash = _declaration_hash(declaration)
        _reject_placeholders(declaration)
        spec = _random_fields(declaration)
        cam = seed_for_task(task)
        # Validate the shared declaration and obtain its reference-line points
        # before generating random values. Delta's baseline is exactly this line.
        base_points = _line_positions(task, cam, declaration)
        space = spec["parameter_space"]
        if space not in ("theta", "delta"):
            raise ValueError("random_init.parameter_space must be 'theta' or 'delta'")
        distribution = _fields(spec["distribution"], ("kind", "detail"),
                               "random_init.distribution")
        kind = distribution["kind"]
        detail = _nonempty_text(distribution["detail"],
                                "random_init.distribution.detail").casefold()
        if kind not in ("gaussian", "uniform"):
            raise ValueError(f"distribution.kind {kind!r} has no registered executable implementation")
        known_details = {"iid", "independent", "uniform_symmetric",
                         "independent zero-mean normal draws per coordinate"}
        if detail not in known_details:
            raise ValueError("random_init.distribution.detail must declare independent samples ('iid'), not an unregistered estimator")
        scale_mm, scale_info = _scale_mm(task, cam, spec["scale"])
        rng, derivation = _rng(task, declaration, declaration_hash,
                               spec["seed_derivation_rule"], requested_seed)
        shape = (len(cam.segments), 3)
        sigma = scale_mm
        if space == "theta":
            from benchmarks.pilot_task import fit_parameterization
            parameterization = fit_parameterization(cam)
            shape = parameterization.parameter_shape
            sigma = scale_mm / parameterization.scale_mm
            basis_hash = "sha256:" + hashlib.sha256(
                _canonical_json(parameterization.linear_map.tolist())).hexdigest()
        if kind == "gaussian":
            noise = rng.normal(0.0, sigma, size=shape)
        else:
            noise = rng.uniform(-sigma, sigma, size=shape)
        if space == "theta":
            points = parameterization.positions(noise)
            parameter_info = {
                "mode": "delta_spline", "reference_motion": "cam",
                "steps": parameterization.num_steps,
                "control_points": parameterization.num_control_points,
                "control_scale_mm_per_normalized_unit": parameterization.scale_mm,
                "basis_sha256": basis_hash,
            }
        else:
            if scale_mm == 0.0:
                points = base_points.copy()
            else:
                residual = np.diff(base_points, axis=0) + noise
                points = np.vstack((base_points[0], base_points[0] + np.cumsum(residual, axis=0)))
            parameter_info = {
                "mode": "raw_physical_deltas", "reference_motion": "straight_line",
                "steps": len(cam.segments), "raw_displacement_parameters": 3 * len(cam.segments),
                "raw_mm_per_displacement_unit": 1.0,
            }
        metadata = {
            "schema": "diffcam-init-seed/1", "mode": "random",
            "task_hash": task.task_hash, "cam_motion_hash": cam.motion_hash,
            "declaration_hash": declaration_hash, "parameter_space": space,
            "parameter_map": parameter_info, "scale": scale_info,
            "noise_kind": kind, "noise_detail": distribution["detail"],
            "noise_coordinate_semantics": (
                "Gaussian: independent standard-normal draws times sigma; "
                "uniform: independent draws on [-sigma, sigma]; both centered additive residuals"),
            "seed_derivation": derivation,
            "same_segment_count_as_cam": len(cam.segments),
            "feed_rule": "retain each original segment kind and limited feed; recompute duration from generated length/rate",
            "safe_entry": "actual first RAPID point pair must be above stock top; unsafe entries are invalid",
            "scope": "structural seed validity only; generated gouges/collapses are not discarded or converted to passes",
        }
        plan = _motion_from_positions(task, cam, points, metadata)
        zero_length = sum(math.dist(segment.start_mm, segment.end_mm) == 0.0
                          for segment in plan.segments)
        metadata["zero_length_segments"] = zero_length
        metadata["validation"] = "validate_for_task passed"
        plan = _motion_from_positions(task, cam, points, metadata)
        return SeedResult(plan, declaration_hash,
                          json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                          requested_seed)
    except (TypeError, ValueError, C.ContractError) as error:
        return _invalid(declaration, seed, str(error))


__all__ = ["IslandSeedSpec", "seed_spec_for_task", "pocket_with_island_seed", "seed_for_task", "seed_deltas_mm",
           "SeedResult", "straight_line_seed", "random_seed"]
