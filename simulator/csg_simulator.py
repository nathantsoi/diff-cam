import numpy as np

import random
import warnings
import taichi as ti
from pathlib import Path
from typing import Mapping

from benchmarks.contracts import Frame, GeometrySource, content_hash, sha256_bytes
from simulator.simulator_utils import *
from cam.units import inch_to_mm, ipm_to_mm_per_s


# Taichi 1.7.4 lifecycle guard, characterized in holder_chronology_test. On
# x64, repeated CSG construction segfaults on the 13th instance's
# from_numpy launch at both fine and coarse grid sizes. Cached-kernel counts
# differ by case (48 and 60 immediately before index 12), so this is explicitly
# an instance-lifetime guard, NOT a claimed portable kernel-count limit or a
# dense-stock memory bound. A new ti.init() runtime resets the count because it
# invalidates all prior fields; no reset is performed on the user's behalf.
# Keep a conservative configured process-local lifetime budget. The observed
# fatal constructor can move EARLIER when additional sampled/config fields exist;
# reuse one simulator or isolate additional instances rather than tune at edge.
_TAICHI_MAX_CSG_INSTANCES_PER_RUNTIME = 11
_TAICHI_RUNTIME_CSG_STATE = {"identity": None, "instances": 0}


def _guard_taichi_instance_lifecycle() -> None:
    from taichi.lang import impl as taichi_impl

    runtime = taichi_impl.get_runtime()
    if runtime is None:
        raise RuntimeError(
            "CSGSimulatorDelta(init_taichi=False) needs an initialized Taichi "
            "runtime; use init_taichi=True in a fresh process instead"
        )
    identity = id(runtime)
    state = _TAICHI_RUNTIME_CSG_STATE
    if state["identity"] != identity:
        state["identity"] = identity
        state["instances"] = 0
    if state["instances"] >= _TAICHI_MAX_CSG_INSTANCES_PER_RUNTIME:
        backend = taichi_impl.current_cfg().arch
        raise RuntimeError(
            f"Taichi simulator instance lifecycle budget: {state['instances']} / "
            f"{_TAICHI_MAX_CSG_INSTANCES_PER_RUNTIME} instances on {backend}. "
            "Repeated CSG allocation can segfault in from_numpy on Taichi 1.7.4/x64, "
            "with the failing constructor depending on configured field complexity. "
            "Reuse existing fields with reset_candidate_state(), or isolate "
            "additional instances in a fresh process with init_taichi=True. "
            "This is an observed lifecycle guard, not a stock-memory bound. "
            "call ti.init(), which would invalidate existing fields."
        )
    # Registration is deliberately separate: an exception during validation,
    # array allocation or parameter initialization is not a live CSG instance.


def _record_taichi_instance_success() -> None:
    from taichi.lang import impl as taichi_impl

    runtime = taichi_impl.get_runtime()
    if runtime is None:
        raise RuntimeError("Taichi runtime disappeared during simulator construction")
    identity = id(runtime)
    state = _TAICHI_RUNTIME_CSG_STATE
    if identity != state["identity"]:
        raise RuntimeError("Taichi runtime changed during simulator construction")
    state["instances"] += 1


@ti.data_oriented
class CSGSimulatorDelta:
    """
    Differentiable CSG simulator for CNC trajectory optimization.

    Parameter: tool_delta, a sequence of T per-step displacements (T × 3 floats).
    Tool positions are reconstructed by a cumulative sum from a fixed start:

        tool_pos[0]   = tool_start          (fixed, not learnable)
        tool_pos[t+1] = tool_pos[t] + tool_delta[t]

    The simulator carves the stock by subtracting a swept cylinder for each
    segment, then computes a terminal loss against a target shape. Adam
    (external) optimizes tool_delta to minimize the loss.

    """

    def __init__(
        self,
        resolution=32,
        max_steps=512,
        k_init=10.0,
        target_shape=None,
        tool_start=(0.5, 0.5, 1.0),
        init_taichi=True,
        stock_size_in=None,
        stock_size_mm=None,
        voxel_size_mm=None,
        work_volume_in=(16.0, 12.0, 10.0),
        stock_origin_in=None,
        workspace_in=None,
        workspace_mm=None,
        dt=0.01,
        rapid_ipm=500.0,
        feed_ipm=10.0,
        safe_distance_in=0.1,
        enforce_speed_limits=True,
        target_geometry=None,
        stock_geometry=None,
        stock_box_origin_mm=None,
        precision="f32",
    ):
        """Differentiable CSG simulator over a small STOCK box placed inside a
        larger machine work volume.

        The working/geometry coordinate is the normalized box ``[0, 1]^3`` (so
        ``tool_pos`` / ``tool_delta`` / exported trajectories are scale-free).
        That normalized cube is the **stock bounding box** -- only the stock is
        voxelized, so RAM scales with the PART, not the machine. The machine
        **work volume** (toolhead limits; default **Haas Mini Mill** 16 x 12 x 10
        inches, x, y; z up) is separate metadata used for G-code export, the
        holder collision barrier, and reachability validation.

        To keep geometry undistorted, the voxel grid uses per-axis dimensions
        ``(Nx, Ny, Nz)`` chosen so every voxel is a physical CUBE of side ``v``
        mm; all internal SDF distances are measured in **voxels** (isotropic).

        stock_size_in : (x, y, z) stock box in inches (REQUIRED; the normalized
                        cube spans this box). ``stock_size_mm`` is the mm form
                        (scalar accepted as a cube) and takes precedence.
        voxel_size_mm : physical voxel edge (mm) -- the sub-mm precision knob.
                        If omitted, ``resolution`` voxels span the stock's
                        LONGEST axis instead.
        work_volume_in: (x, y, z) machine envelope in inches (toolhead limits).
        stock_origin_in: work origin (G54) = the stock's TOP-CENTRE in machine
                        coords (inches). Used for export/validation only.
        workspace_in / workspace_mm : back-compat aliases for the work volume.
        target_geometry / stock_geometry: optional contracts GeometrySource
                        objects with kind="sampled". Their path+sha256 payload
                        is a finite real SDF in physical millimetres on a work-
                        frame GridSpec. Trilinear/nearest-centre-clamped lookup
                        consumes physical units, never LFS pointers/pickled data.
        stock_box_origin_mm: physical world-coordinate minimum of the allocated
                        normalized stock box (default (0,0,0)). It maps legacy
                        normalized tool positions to sampled fields' physical
                        coordinates. With a custom stock it must match the source
                        grid origin; an omitted custom-stock size uses its GridSpec
                        extents instead of the otherwise required stock_size_*.
        precision: "f32" (default, available for exploration) or "f64" for
                         every float field and adjoint. Use f64 for the
                         comparison protocol. No partial or implicit mixed dtype.
                         default_fp is runtime-wide, one precision per process.
        """
        self._initialization_complete = False
        if precision not in ("f32", "f64"):
            raise ValueError("precision must be 'f32' or 'f64'; partial/mixed precision is unsupported")
        self.precision = precision
        self._precision = {"f32": ti.f32, "f64": ti.f64}[precision]
        # Python's ``self.dtype`` cannot be used in ti.func/kernel annotations at
        # class-definition time. Scalar inputs therefore use ti.template(),
        # scalar casts use this declared dtype, and scalar return annotations
        # intentionally follow the process-wide default_fp instead of hardcoding
        # f32. The runtime check below is made BEFORE allocating any fields.
        from taichi.lang import impl as taichi_impl

        if init_taichi:
            try:
                if ti._lib.core.with_cuda():
                    ti.init(arch=ti.gpu, debug=False, default_fp=self._precision, offline_cache=False)
                else:
                    ti.init(arch=ti.cpu, debug=False, default_fp=self._precision, offline_cache=False)
            except RuntimeError:
                # A CUDA-enabled build does not imply an available device.
                ti.init(arch=ti.cpu, debug=False, default_fp=self._precision, offline_cache=False)
            ti.set_logging_level(ti.WARN)
        runtime = taichi_impl.get_runtime()
        if runtime is None or runtime.default_fp != self._precision:
            actual = None if runtime is None else runtime.default_fp
            raise RuntimeError(
                f"simulator precision={precision} requires Taichi default_fp={self._precision}, "
                f"but the current runtime has {actual}. default_fp is runtime-wide; "
                "one precision per process. Construct the matching runtime in a fresh "
                "process, or pass init_taichi=True in a fresh process: ti.init() "
                "INVALIDATES existing fields and is never used for precision conversion."
            )
        # init_taichi=False attaches fields to the initialized, already-matching
        # runtime. This never converts existing fields or resets their runtime.
        _guard_taichi_instance_lifecycle()
        if max_steps <= 0:
            raise ValueError(
                "max_steps must be positive (at least one segment); "
                "Taichi cannot allocate zero-length tool_delta fields. "
                "Use one zero-length segment for a stationary cut."
            )
        self.max_steps = max_steps

        # ---- Stock box (the normalized cube) & machine work volume -----------
        # ``workspace_in``/``workspace_mm`` are back-compat aliases for the
        # machine work volume.
        if workspace_in is not None:
            work_volume_in = workspace_in
        work_volume_mm = workspace_mm  # None unless an mm alias was passed

        # Machine work volume -> mm (x, y, z up). Pure metadata: toolhead limits,
        # export anchor, holder-barrier height. NOT voxelized.
        if work_volume_mm is None:
            self.work_volume_mm = np.asarray(
                [float(inch_to_mm(c)) for c in work_volume_in], dtype=np.float64
            )
        elif np.isscalar(work_volume_mm):
            self.work_volume_mm = np.asarray([float(work_volume_mm)] * 3, dtype=np.float64)
        else:
            self.work_volume_mm = np.asarray([float(c) for c in work_volume_mm], dtype=np.float64)

        # Stock box -> mm (REQUIRED). The normalized box [0,1]^3 spans this box,
        # and ONLY this box is voxelized.
        if stock_size_mm is not None:
            if np.isscalar(stock_size_mm):
                sx = sy = sz = float(stock_size_mm)
            else:
                sx, sy, sz = (float(c) for c in stock_size_mm)
        elif stock_size_in is not None:
            sx, sy, sz = (float(inch_to_mm(c)) for c in stock_size_in)
        elif stock_geometry is not None:
            source = self._coerce_geometry_source(stock_geometry)
            sx, sy, sz = source.grid.extent_mm
            if stock_box_origin_mm is None:
                stock_box_origin_mm = source.grid.origin_mm
        else:
            raise ValueError(
                "stock_size_in (or stock_size_mm) is required unless a sampled "
                "stock_geometry supplies its extents; the normalized cube "
                "[0,1]^3 is the stock bounding box, not the work volume"
            )
        if not np.all(np.isfinite((sx, sy, sz))) or min(sx, sy, sz) <= 0.0:
            raise ValueError("stock size components must be finite and positive")
        self.Lx, self.Ly, self.Lz = sx, sy, sz          # stock box mm (x, y, z up)

        # Work origin (G54 offset): the stock's TOP-CENTRE in machine coords.
        if stock_origin_in is not None:
            self.stock_origin_mm = np.asarray(
                [float(inch_to_mm(c)) for c in stock_origin_in], dtype=np.float64
            )
        else:
            self.stock_origin_mm = None

        # Cubic voxels over the STOCK box. Prefer an explicit physical voxel
        # size (the sub-mm precision knob); else fall back to ``resolution``
        # voxels along the stock's longest axis. The other axes get round(L/v)
        # voxels so every voxel is a physical cube of side ``v`` mm.
        if voxel_size_mm is not None:
            if not np.isfinite(voxel_size_mm) or float(voxel_size_mm) <= 0.0:
                raise ValueError("voxel_size_mm must be finite and positive")
            self.v = float(voxel_size_mm)               # mm per voxel (cube side)
        else:
            if not np.isfinite(resolution) or float(resolution) <= 0.0:
                raise ValueError("resolution must be finite and positive")
            self.v = max(sx, sy, sz) / float(resolution)
        self.Nx = max(1, int(round(sx / self.v)))
        self.Ny = max(1, int(round(sy / self.v)))
        self.Nz = max(1, int(round(sz / self.v)))
        # Rounded counts are only valid if the ACTUAL physical pitch on every
        # axis remains cubic. Squaring lengths by an unrelated nominal voxel
        # size silently turns a flat cutter into a distorted one. Fail early
        # rather than accepting e.g. (40,24,16) mm at a nominal 3 mm pitch
        # (actual pitches 40/13,24/8,16/5, including 3.2 mm on Z).
        actual_pitch_mm = np.asarray((sx / self.Nx, sy / self.Ny, sz / self.Nz))
        if not np.allclose(actual_pitch_mm, self.v, rtol=2e-6, atol=1e-8):
            raise ValueError(
                "rounded grid counts make physical voxels anisotropic; "
                f"actual per-axis pitches are {actual_pitch_mm.tolist()} mm, "
                f"nominal is {self.v} mm. Choose voxel_size_mm/resolution that "
                "divides every physical stock axis (within 2e-6 relative tolerance) "
                "rather than silently stretching cutter or stock coordinates."
            )
        self.resolution = max(self.Nx, self.Ny, self.Nz)  # voxels on longest axis

        # Reachability: the stock (and, if its origin is known, its placement)
        # must fit inside the machine work volume. Warn rather than fail so
        # exploratory configs still run.
        self._validate_fits(np.asarray([sx, sy, sz], dtype=np.float64))

        # Internal SDF distances are in VOXELS: a normalized [0,1] difference is
        # multiplied by the per-axis grid count to get a voxel-space coordinate
        # (isotropic because voxels are cubes). ``k_ref`` rescales the smooth-CSG
        # ``k`` from the legacy unit-cube convention into voxel space so a cubic
        # envelope at a given resolution reproduces the old behavior exactly.
        self.k_ref = float(self.resolution)

        self.k = ti.field(dtype=self._precision, shape=())  # anneal during training
        self.k[None] = k_init

        # ---- Tool (learnable) ----
        self.tool_delta = ti.Vector.field(
            3, dtype=self._precision, shape=max_steps, needs_grad=True
        )
        self.tool_pos = ti.Vector.field(
            3, dtype=self._precision, shape=max_steps + 1, needs_grad=True
        )

        self.tool_start = ti.Vector.field(3, dtype=self._precision, shape=())
        self._configured_tool_start = tuple(float(value) for value in tool_start)
        self.tool_start[None] = ti.Vector(list(tool_start))

        self.tool_radius = ti.field(dtype=self._precision, shape=())
        self.tool_height = ti.field(dtype=self._precision, shape=())
        # Height of the CUTTING TIP band (mm) used ONLY by the air-time metric
        # / loss (compute_traj_metrics_volumes + compute_traj_diagnostics_hard):
        # the air/engage/swept volumes are integrated over this bottom band of
        # the tool, not the full tool_height shank. This stops the tall shank
        # sitting in already-carved empty space from being counted as "air" on
        # an engaged finishing pass (which otherwise read ~80% air). The carve
        # itself still uses the full cylinder (tool_sdf). 0 -> fall back to the
        # full tool_height (legacy behaviour) inside the tip SDFs. See
        # docs/memory airfrac-shank-volume-bias.
        self.tool_cut_height = ti.field(dtype=self._precision, shape=())

        # ---- Tool holder (collision body, not learnable) ----
        # The holder is the wide spindle/collet shaft that sits coaxially ABOVE
        # the slender cutting flutes. It never removes material, but if it ever
        # touches the remaining stock that is a crash: the spindle would slam
        # into the workpiece. We model it as a cylinder of radius holder_radius
        # whose bottom face is at the top of the tool (tool tip + tool_height)
        # and which extends upward by holder_height.
        #
        # Sizes are in MILLIMETRES now (converted to voxels internally). Default
        # holder is a 2.5 inch diameter spindle/collet; its height spans the full
        # machine Z (not the small stock) so it always clears above the cutter.
        self.holder_radius = ti.field(dtype=self._precision, shape=())
        self.holder_height = ti.field(dtype=self._precision, shape=())
        self.holder_radius[None] = float(inch_to_mm(2.5 / 2.0))   # mm
        self.holder_height[None] = float(self.work_volume_mm[2])  # mm

        # ---- Units & speed limits (constraint, enforced by clipping) ----
        # The geometry lives in the normalized box [0, 1]^3; axis ``a`` spans
        # ``L_a`` mm, so a normalized displacement ``d`` spans ``d (.) L`` mm
        # (component-wise). One step advances the tool over ``dt`` seconds, so
        # the commanded speed of a step is
        #
        #     speed_mm_per_s = |tool_delta[t] (.) L| / dt
        #
        # Two physical max speeds are enforced (clipped) per step, like a real
        # controller's feed/rapid override (cf. LinuxCNC trajectory planner and
        # CAMotics): ``rapid_speed`` when the cutter has clearance from the
        # remaining stock, ``feed_speed`` when it is within ``safe_distance`` of
        # it (i.e. cutting). ALL scale-related math is done in millimetres;
        # inch-valued inputs are converted up front via ``cam.units``. The
        # per-axis envelope ``(Lx, Ly, Lz)`` is fixed at construction and used
        # directly as a kernel constant.
        self.dt = ti.field(dtype=self._precision, shape=())               # seconds / step
        self.rapid_speed = ti.field(dtype=self._precision, shape=())      # mm / s
        self.feed_speed = ti.field(dtype=self._precision, shape=())       # mm / s
        self.safe_distance = ti.field(dtype=self._precision, shape=())    # mm
        self.enforce_speed_limits = ti.field(dtype=ti.i32, shape=())
        self.dt[None] = float(dt)
        self.rapid_speed[None] = float(ipm_to_mm_per_s(rapid_ipm))
        self.feed_speed[None] = float(ipm_to_mm_per_s(feed_ipm))
        self.safe_distance[None] = float(inch_to_mm(safe_distance_in))
        self.enforce_speed_limits[None] = 1 if enforce_speed_limits else 0

        # ---- Loss balancing (constrained-optimization framing) ----
        # The terminal loss is a SOFT OBJECTIVE plus two ONE-SIDED BARRIERS:
        #
        #   objective  : w_residual * residual   (material left outside the part)
        #                -> minimizing this is what *rewards* cutting material away
        #   barrier A  : w_gouge   * gouge       (material removed from inside part)
        #                -> one-sided: only fires when the cutter eats into the part
        #   barrier B  : holder penetration into the stock (see below)
        #
        # Keeping w_gouge >= w_residual puts the stock surface just OUTSIDE the
        # part surface: as close as possible without cutting in. The barriers are
        # heavy because they encode "do not violate", while residual is the light
        # objective optimized up against them.
        self.w_gouge = ti.field(dtype=self._precision, shape=())
        self.w_residual = ti.field(dtype=self._precision, shape=())
        self.w_gouge[None] = 2.0
        self.w_residual[None] = 1.0
        # Annulus-residual emphasis (STRUCTURAL, hole-targeted). The uniform
        # residual under-resolves the thin column/annulus walls of an annular
        # part (sphere_hole): the central-column waste right at the part surface
        # carries the same per-voxel weight as the easy far-exterior waste, so
        # the optimizer clears the exterior and leaves the narrow column walls.
        # This multiplies the residual term by (1 + w_annulus * annulus_weight)
        # where annulus_weight = max(0, 1 - max(0,target_d)/annulus_dref) is HIGH
        # on near-surface waste (just outside the part surface = the thin walls)
        # and 0 in the far exterior. target_d is the fixed baked target SDF (not
        # a differentiable param), so the multiplier is a constant per voxel that
        # merely scales the residual gradient -- autodiff-safe, and w_annulus=0
        # (default) leaves the residual exactly unchanged for every other shape.
        self.w_annulus = ti.field(dtype=self._precision, shape=())
        self.annulus_dref = ti.field(dtype=self._precision, shape=())
        self.w_annulus[None] = 0.0
        self.annulus_dref[None] = 2.0

        # De-biased (hard-carve-aware) loss shift. The soft ``apply_cut``
        # (smooth_max union) over-erodes by ~log(2)/kv per cut vs the hard
        # ``ti.max`` carve, so the soft stock_d is biased NEGATIVE (too carved)
        # and the soft loss is satisfied before the HARD carve actually reaches
        # the target -- the soft/hard gap that fills narrow negative features
        # (holes). Adding ``loss_shift`` to stock_d before the loss sigmoid
        # shifts the loss's view back toward the (less-eroded) hard stock, so
        # the optimizer targets the deployable hard carve. Default 0 = off (no
        # behavior change). A principled value is ~log(2)*k_ref/k_final (the
        # single-cut over-erosion in voxel units at the final sharpness).
        self.loss_shift = ti.field(dtype=self._precision, shape=())
        self.loss_shift[None] = 0.0

        # Air-cut penalty: a per-step term that fires when the swept tool occupies
        # EMPTY stock (tool inside, no remaining material). This is what "cutting
        # air" costs -- the optimizer is charged for every step the cutter spends
        # traversing or hovering in open space instead of removing material. Zero
        # (with zero gradient) when the tool is actually cutting (stock_occ ~ 1).
        self.w_air = ti.field(dtype=self._precision, shape=())
        self.w_air[None] = 0.0
        # Jerk / smoothness penalty: squared difference of consecutive deltas, so
        # the trajectory is penalized for abrupt changes in direction/speed (the
        # "jerky" artefact). Acts directly on tool_delta (needs_grad=True).
        self.w_jerk = ti.field(dtype=self._precision, shape=())
        self.w_jerk[None] = 0.0
        # Speed-regularity (constant-feed) penalty: squared difference of
        # consecutive step LENGTHS (|delta_t| - |delta_{t-1}|)^2. Encourages a
        # uniform feed rate -- the canonical CNC toolpath pattern -- independent
        # of direction. Acts directly on tool_delta (needs_grad=True).
        self.w_step = ti.field(dtype=self._precision, shape=())
        self.w_step[None] = 0.0
        # Distance-weighted AIR-CUT (contour-hug) penalty: like w_air but the
        # charge scales with how far the swept voxel is from the TARGET surface
        # (squared), so re-traversing empty CORNERS far from the part is heavily
        # penalized while surface-hugging and the necessary first-pass carving
        # (in remaining stock, air ~ 0) stay cheap. Directly attacks the
        # "tool moving far from the part surface" failure mode without the
        # blunt collapse that cranking w_air causes. Shares the air loop's
        # tool_sdf eval, so it is nearly free. Gated by w_prox (0 -> disabled).
        self.w_prox = ti.field(dtype=self._precision, shape=())
        self.w_prox[None] = 0.0
        # Trajectory contour-hug penalty on the tool CENTER: a GENTLE per-segment
        # penalty (T terms, not T*N^3) on the segment-midpoint distance from the
        # TARGET surface, with a deadzone of one tool-radius so contact-cutting
        # (including corner-carving, which sits within r_tool of the surface) is
        # FREE and only genuine excursions (deep into empty corners, high
        # retracts beyond r_tool) are charged. Unlike the per-voxel w_prox whose
        # huge corner gradient stalls carving, this is a soft nudge on the
        # trajectory shape. Gated by w_traj_prox (0 -> disabled).
        self.w_traj_prox = ti.field(dtype=self._precision, shape=())
        self.w_traj_prox[None] = 0.0
        # Path-length (minimal-motion) penalty: the squared step length
        # |delta_t|^2 summed over all active segments. Unlike the contour-hug
        # losses (w_prox / w_traj_prox) which pull the tool TOWARD the surface
        # and so oppose the necessary carving excursions, this is agnostic to
        # WHERE the tool is -- it only discourages motion. During carving the
        # strong residual gradient dominates and the tool moves anyway; on the
        # trailing steps (part already carved, no residual, gouge barrier pushes
        # the tool off into open air) the length penalty is the only gradient
        # and shrinks the deltas toward zero, so the tool STOPS instead of
        # wandering away. This is the targeted fix for the trailing-excursion
        # failure mode (tool climbs into air for the last ~25% of the path).
        # Gated by w_len (0 -> disabled).
        self.w_len = ti.field(dtype=self._precision, shape=())
        self.w_len[None] = 0.0
        # TOOL-POSITION gouge barrier (SOFT-UNION-INDEPENDENT surface respect).
        # The stock-based w_gouge charges soft-occupancy target voxels emptied in
        # the SOFT stock -- but the soft union over-erodes, so that barrier is
        # trivially satisfied in soft space while the HARD carve still gouges the
        # part. This term instead charges the TOOL CENTER directly: the tool
        # capsule (radius r_tool) gouges the target when target_sdf(center) <
        # r_tool, so the penalty is relu(r_tool - target_sdf(seg_mid))^2 -- ZERO
        # when the tool is tangent-or-outside the surface (contact-cutting the
        # waste just outside the part is FREE), and grows quadratically as the
        # tool penetrates the target. Differentiable in tool_pos/tool_delta via
        # the segment midpoint and target_sdf_scalar, and crucially INDEPENDENT
        # of the soft stock -- it constrains the trajectory geometry directly,
        # so it transfers to the hard carve. Gated by w_tool_gouge (0 -> disabled).
        self.w_tool_gouge = ti.field(dtype=self._precision, shape=())
        self.w_tool_gouge[None] = 0.0
        # Tool-gouge MARGIN (voxels): inflate the no-penetration radius from
        # r_tool to r_tool + margin so the tool center must stay `margin` voxels
        # FURTHER off the surface than mere tangency. Overlapping tangent tool
        # capsules bite into a CONVEX part at pass seams (the sphere's gouge
        # mechanism -- loss_tool_gouge=0 at midpoints yet the boolean union
        # still over-erodes); a positive margin lifts the tool so the union of
        # capsules stays tangent-only and does not gouge at the seams, at the
        # cost of a little uncut residual. Shape-agnostic (target_sdf only).
        self.tool_gouge_margin = ti.field(dtype=self._precision, shape=())
        self.tool_gouge_margin[None] = 0.0

        # ---- Trajectory-quality measures (time + breakage) ----
        # Three deployable measures reported alongside dice and (when their
        # weight is > 0) incorporated into the soft loss as differentiable
        # surrogates. The hard, non-differentiable final-metric forms are
        # computed by compute_traj_diagnostics_hard on the hard carve.
        #
        # 1. Total toolpath time (seconds): sum of per-segment motion time at
        #    the feed/rapid regime speed. Shorter is better for equal dice.
        # 2. Air-cutting time (seconds): time spent cutting air, weighted by
        #    the per-segment air fraction (swept tool in empty stock). High
        #    retracts clear of the surface contribute ~0 (tool outside the
        #    stock grid); surface-hugging air in empty corners counts. Waste.
        # 3. Tool-breakage probability: heavy engagement is efficient but too
        #    much snaps the tool. Soft surrogate = the docs/algorithms.md §4.1
        #    closed-form stress-strength interference (simplified: constant
        #    strength, single log-variance), aggregated as 1-exp(-sum P_t).
        #    Hard form = the docs/design.md threshold rule (broken iff
        #    F_cut_max > f_max).
        self.w_time = ti.field(dtype=self._precision, shape=())
        self.w_time[None] = 0.0
        self.w_air_time = ti.field(dtype=self._precision, shape=())
        self.w_air_time[None] = 0.0
        # Late-weighted air penalty (Idea 2): a SECOND air-time loss term whose
        # per-segment weight ramps from 0 (early entry/reposition air is cheap)
        # to 1 (late air is expensive), attacking end-of-trajectory air cutting.
        # w_air_late is the overall weight (0 = off, same scale as w_air_time);
        # w_air_ramp_frac is the fraction of the trajectory held at ramp=0 before
        # the linear ramp to 1 begins (0 = ramp over the whole trajectory). The
        # ramp depends only on the step index t (shape-agnostic).
        self.w_air_late = ti.field(dtype=self._precision, shape=())
        self.w_air_late[None] = 0.0
        self.w_air_ramp_frac = ti.field(dtype=self._precision, shape=())
        self.w_air_ramp_frac[None] = 0.0
        self.w_break = ti.field(dtype=self._precision, shape=())
        self.w_break[None] = 0.0
        # Breakage-model constants (see docs/algorithms.md). kc = specific
        # cutting force (Al ~700 N/mm^2); f_ref = nominal force at which
        # P_break=0.5 (effective S_bar/alpha_mean -- calibrate); sigma_risk =
        # combined log-std sqrt(sigma_alpha^2 + pi^2/(6 m^2)); f_max = hard
        # threshold force for the design.md broken flag. D = 2*tool_radius and
        # dt are existing fields, read inside the kernels.
        self.kc = ti.field(dtype=self._precision, shape=())
        self.kc[None] = 700.0
        self.f_ref = ti.field(dtype=self._precision, shape=())
        self.f_ref[None] = 50.0
        self.sigma_risk = ti.field(dtype=self._precision, shape=())
        self.sigma_risk[None] = 0.5
        self.f_max = ti.field(dtype=self._precision, shape=())
        self.f_max[None] = 100.0

        # Holder collision is a PENETRATION BARRIER, not a proximity field: it is
        # exactly zero (zero gradient) while the holder has clearance, and grows
        # with the squared depth the holder pushes into remaining stock. This is
        # what lets the optimizer bring the holder right up to the surface (so the
        # cutter removes the most material) and only resists actual contact.
        # holder_margin (VOXELS) requests a conservative standoff: the barrier
        # starts engaging when stock comes within holder_margin of the holder.
        # Physical clearance standoff = holder_margin * v (physical voxel mm).
        self.holder_penalty_weight = ti.field(dtype=self._precision, shape=())
        self.holder_penalty_weight[None] = 50.0
        self.holder_margin = ti.field(dtype=self._precision, shape=())
        self.holder_margin[None] = 0.0

        # Z-floor: a hard lower bound on the tool BASE z (normalized [0,1]).
        # The zlayer init descends the tool base well below the part (z_bot=-0.95)
        # to carve the below-part slab, but the tool extends UPWARD from its base
        # by tool_height and the wide holder rides above that -- so a deep base
        # plunge drops the holder into the remaining stock (a machine crash). The
        # floor clamps the executed move's base z, exactly like the feed/rapid
        # speed clip below: the init deltas may command a deeper z, but the
        # executed/clipped trajectory (and the saved one) stays at/above the floor.
        # Default -1e9 (disabled); set enforce_z_floor=1 and z_floor to enable.
        self.z_floor = ti.field(dtype=self._precision, shape=())
        self.z_floor[None] = -1e9
        self.enforce_z_floor = ti.field(dtype=ti.i32, shape=())
        self.enforce_z_floor[None] = 0
        # Scratch accumulator for holder_min_clearance_at (a parallel min-reduction
        # needs a global field + ti.atomic_min; a local scalar races under the
        # parallel ti.ndrange loop and returns a partial/garbage value).
        self._clearance_buf = ti.field(dtype=self._precision, shape=())

        # Diagnostics (non-differentiable read-outs of each loss component so the
        # objective/barrier balance is observable during training).
        self.diag_gouge = ti.field(dtype=self._precision, shape=())
        self.diag_residual = ti.field(dtype=self._precision, shape=())
        self.diag_holder = ti.field(dtype=self._precision, shape=())
        self.diag_air = ti.field(dtype=self._precision, shape=())
        self.diag_jerk = ti.field(dtype=self._precision, shape=())
        self.diag_step = ti.field(dtype=self._precision, shape=())
        # Unweighted air-cut fraction: sum of (tool swept volume in empty stock)
        # over the trajectory, normalized per voxel -- measures how much of the
        # tool motion is cutting air (0 = all cutting, 1 = all air). Independent
        # of w_air so it is non-zero even when the air penalty is disabled.
        # diag_air_unweighted is the NUMERATOR (air volume); diag_tool_swept is
        # the DENOMINATOR (total swept tool volume); the ratio air/swept is the
        # true per-step air-cut fraction in [0,1], independent of how much total
        # carving the trajectory did.
        self.diag_air_unweighted = ti.field(dtype=self._precision, shape=())
        self.diag_tool_swept = ti.field(dtype=self._precision, shape=())
        # Distance-weighted air-cut (contour-hug) diagnostic: the w_prox loss
        # component value, independent of w_prox so it is non-zero even when the
        # penalty is disabled. Measures air-cutting concentrated far from the
        # target surface.
        self.diag_prox = ti.field(dtype=self._precision, shape=())
        # Trajectory contour-hug diagnostic (the w_traj_prox loss component,
        # independent of w_traj_prox so it is non-zero even when disabled).
        self.diag_traj_prox = ti.field(dtype=self._precision, shape=())
        # Path-length (minimal-motion) diagnostic: mean squared step length
        # (the w_len loss component, independent of w_len so it is non-zero
        # even when disabled -- measures how much the tool moves).
        self.diag_len = ti.field(dtype=self._precision, shape=())
        # Tool-position gouge diagnostic (the w_tool_gouge loss component,
        # independent of the weight so it is non-zero even when disabled --
        # measures how far the tool center penetrates the target+r_tool).
        self.diag_tool_gouge = ti.field(dtype=self._precision, shape=())
        # Trajectory-quality diagnostics (non-differentiable read-outs of the
        # three new measures, computed on the hard carve by
        # compute_traj_diagnostics_hard). diag_time/diag_air_time are seconds;
        # diag_break_prob_* are in [0,1]; diag_fcut_max is Newtons; diag_broken
        # is 0/1 (the docs/design.md hard threshold rule); diag_engage_* are
        # raw engaged-chip volumes (unit-cube^3) for calibration.
        self.diag_time = ti.field(dtype=self._precision, shape=())
        self.diag_air_time = ti.field(dtype=self._precision, shape=())
        self.diag_break_prob_any = ti.field(dtype=self._precision, shape=())
        self.diag_break_prob_max = ti.field(dtype=self._precision, shape=())
        self.diag_fcut_max = ti.field(dtype=self._precision, shape=())
        self.diag_broken = ti.field(dtype=self._precision, shape=())
        self.diag_engage_max = ti.field(dtype=self._precision, shape=())
        self.diag_engage_mean = ti.field(dtype=self._precision, shape=())
        # Idea 8: air-time split into early/mid/late thirds of the trajectory
        # (seconds). Makes the #1 user complaint -- "air cutting at the END" --
        # measurable: a high late/total ratio is the deployability fault that
        # total air_time cannot distinguish from useful entry/reposition air.
        # Pure diagnostic (computed on the hard carve, never feeds the loss).
        self.diag_air_time_early = ti.field(dtype=self._precision, shape=())
        self.diag_air_time_mid = ti.field(dtype=self._precision, shape=())
        self.diag_air_time_late = ti.field(dtype=self._precision, shape=())

        # Per-segment differentiable intermediates for the three new measures.
        # Written by compute_seg_time / compute_seg_volumes (inside the Tape)
        # and read by compute_traj_metrics, which adds the soft loss terms.
        # needs_grad so gradient flows field->field across these kernels (same
        # pattern as stock[t] -> compute_loss). The Tape clears their grad each
        # iteration. Reused as plain (non-grad) scratch by the hard diagnostic
        # path, which runs outside a Tape.
        self.seg_time = ti.field(dtype=self._precision, shape=max_steps, needs_grad=True)
        self.seg_air = ti.field(dtype=self._precision, shape=max_steps, needs_grad=True)
        self.seg_swept = ti.field(dtype=self._precision, shape=max_steps, needs_grad=True)
        self.seg_engage = ti.field(dtype=self._precision, shape=max_steps, needs_grad=True)
        # Scalar needs_grad accumulator for the trajectory-level breakage
        # probability: P_break_traj = 1 - exp(-sum_t P_break(t)) is a nonlinear
        # function of the WHOLE-trajectory sum, so it cannot be atomic-added
        # per-segment. The per-segment kernel accumulates P_break(t) here, then
        # a separate scalar kernel adds w_break * (1 - exp(-acc_psum)) to loss.
        self.acc_psum = ti.field(dtype=self._precision, shape=(), needs_grad=True)

        # ---- Stock ----
        self.stock = ti.field(
            dtype=self._precision,
            shape=(max_steps + 1, self.Nx, self.Ny, self.Nz),
            needs_grad=True,
        )
        self.stock_volume = ti.field(dtype=self._precision, shape=())
        # Saved mid-cut stock init (for staged training): when use_saved_init
        # is set, ``init_stock`` writes this SDF into stock[0] instead of the
        # full envelope, so a FRESH trajectory can start carving from a partially-
        # carved state left by a previous trajectory. stock[0] is a constant
        # (no grad), so autodiff through apply_cut on stock[1..T] is unaffected.
        self.saved_stock = ti.field(
            dtype=self._precision, shape=(self.Nx, self.Ny, self.Nz)
        )
        self.use_saved_init = ti.field(dtype=ti.i32, shape=())
        self.use_saved_init[None] = 0

        # ---- Target ----
        target_options = ["box", "cylinder", "sphere", "pyramid"]
        if target_shape is None:
            target_shape = random.choice(target_options)
        self.target_shape = target_shape
        self.target_params = {}
        self.target_volume = ti.field(dtype=self._precision, shape=())
        self._init_target_fields()

        self.target = ti.field(
            dtype=self._precision, shape=(self.Nx, self.Ny, self.Nz)
        )  # used for evaluation
        # Scalar (non-Vector) copies of the target params, for use inside
        # differentiable kernels that feed a GRAD-TRACKED position into the
        # target SDF: the Vector-field target_params trigger a Taichi autodiff
        # load-forwarding bug (MatrixPtrStmt assertion) when the SDF input
        # depends on tool_pos. Center in normalized [0,1]; sizes in voxels.
        self.tcx = ti.field(dtype=self._precision, shape=()); self.tcx[None] = 0.5
        self.tcy = ti.field(dtype=self._precision, shape=()); self.tcy[None] = 0.5
        self.tcz = ti.field(dtype=self._precision, shape=()); self.tcz[None] = 0.5
        self.tr_vox = ti.field(dtype=self._precision, shape=()); self.tr_vox[None] = 0.0
        self.th_vox = ti.field(dtype=self._precision, shape=()); self.th_vox[None] = 0.0
        self.thx_vox = ti.field(dtype=self._precision, shape=()); self.thx_vox[None] = 0.0
        self.thy_vox = ti.field(dtype=self._precision, shape=()); self.thy_vox[None] = 0.0
        self.thz_vox = ti.field(dtype=self._precision, shape=()); self.thz_vox[None] = 0.0
        self.tbase_vox = ti.field(dtype=self._precision, shape=()); self.tbase_vox[None] = 0.0
        # Sub-primitive radius (voxels) for combined CSG shapes: the
        # through-hole cylinder in "sphere_hole" and the subtracted sphere in
        # "sphere_bowl". 0 disables (plain shapes ignore it).
        self.tsub_vox = ti.field(dtype=self._precision, shape=()); self.tsub_vox[None] = 0.0

        # ---- Loss ----
        self.loss = ti.field(dtype=self._precision, shape=(), needs_grad=True)

        self._init_sampled_geometry(
            target_geometry, stock_geometry, stock_box_origin_mm
        )
        ti.root.lazy_grad()
        _record_taichi_instance_success()
        self._initialization_complete = True

    def precision_metadata(self):
        """JSON-safe objective/run-setup identity, not an optimizer cost claim.

        Include this entire mapping in setup signatures/run configuration. The
        uniform f32/f64 policy explicitly covers all float fields, kernels and
        adjoints. Comparison decisions belong to the study protocol, not a bare
        variant name.
        """
        from taichi.lang import impl as taichi_impl

        runtime = taichi_impl.get_runtime()
        runtime_fp = str(runtime.default_fp)
        backend = str(taichi_impl.current_cfg().arch)
        policy = {
            "schema_version": "simulator-precision-uniform/1",
            "variant_revision": f"uniform-{self.precision}/1",
            "default_fp": self.precision,
            "field_scope": "all_float_fields",
            "kernel_scope": "all_float_kernels_and_adjoints",
        }
        declared_fp = self.precision
        payload = {
            "schema_version": "simulator-precision-identity/1",
            "requested_precision": self.precision,
            "policy": policy,
            "declared_default_fp": declared_fp,
            "runtime_default_fp": runtime_fp,
            "backend_arch": backend,
            "process_constraint": "Taichi default_fp is runtime-wide: one precision per process",
        }
        payload["precision_identity_sha256"] = content_hash(payload)
        return payload

    def estimated_known_field_bytes(self):
        """Exact declared bytes for the named historical/geometry arrays.

        Not total allocated memory and NOT a runtime peak-RSS measurement:
        trajectories, scalar fields, source buffers and Taichi allocator/compiler
        overhead are excluded explicitly. Only uniform f32/f64 arrays are named.
        """
        bytes_per_float = {"f32": 4, "f64": 8}[self.precision]
        voxels = self.Nx * self.Ny * self.Nz
        history = (self.max_steps + 1) * voxels
        history_and_adjoint = 2 * history * bytes_per_float
        immutable = 2 * voxels * bytes_per_float  # target and saved_stock
        return {
            "bytes_per_float": bytes_per_float,
            "stock_history_and_adjoint_bytes": history_and_adjoint,
            "target_and_saved_stock_bytes": immutable,
            "approximately_scaled_geometry": {
                "stock_npy": "declared source files remain np.float32; no silent source redefinition",
                "target_history_shape_without_grad": [self.Nx, self.Ny, self.Nz],
            },
            "known_fields_total_bytes": history_and_adjoint + immutable,
            "excluded": "all other fields, trajectories, original .npy buffers, allocator and compiler overhead",
        }

    def _validate_fits(self, stock_size_mm):
        """Warn if the stock box can't fit inside the machine work volume.

        Checks the box dimensions, and -- when the work origin (stock top-centre)
        is known -- the placed box's extent against ``[0, work_volume]``.
        """
        wv = self.work_volume_mm
        if np.any(stock_size_mm > wv + 1e-6):
            warnings.warn(
                f"stock box {stock_size_mm.tolist()} mm exceeds machine work "
                f"volume {wv.tolist()} mm",
                stacklevel=3,
            )
        if self.stock_origin_mm is not None:
            ox, oy, oz = self.stock_origin_mm
            sx, sy, sz = stock_size_mm
            # Top-centre origin: box spans [o-sz/2, o+sz/2] in X/Y, [oz-sz, oz] in Z.
            lo = np.asarray([ox - sx / 2.0, oy - sy / 2.0, oz - sz])
            hi = np.asarray([ox + sx / 2.0, oy + sy / 2.0, oz])
            if np.any(lo < -1e-6) or np.any(hi > wv + 1e-6):
                warnings.warn(
                    f"stock placed at top-centre origin {self.stock_origin_mm.tolist()} "
                    f"mm extends outside the work volume {wv.tolist()} mm "
                    f"(box {lo.tolist()}..{hi.tolist()})",
                    stacklevel=3,
                )

    @staticmethod
    def _coerce_geometry_source(source):
        if isinstance(source, Mapping):
            source = GeometrySource(**source)
        if not isinstance(source, GeometrySource):
            raise TypeError("expected a contracts.GeometrySource or mapping")
        if source.kind != "sampled":
            raise ValueError("custom geometry requires GeometrySource(kind='sampled')")
        if source.frame != Frame.WORK or source.grid.frame != Frame.WORK:
            raise ValueError("sampled geometry and its grid must use the work frame")
        return source

    @classmethod
    def _source_payload(cls, source):
        """Validate and return (shape, origin, voxel, float32 payload, metadata)."""
        source = cls._coerce_geometry_source(source)
        path = Path(source.path)
        if not path.is_file():
            raise FileNotFoundError(f"sampled geometry payload is missing: {path}")
        actual_hash = sha256_bytes(path.read_bytes())
        if actual_hash != source.sha256:
            raise ValueError(
                f"sampled geometry checksum mismatch for {path}: "
                f"declared {source.sha256}, actual {actual_hash}"
            )
        values = np.load(path, allow_pickle=False)
        if values.dtype.kind not in "fiu":
            raise ValueError("sampled geometry requires a real numeric array")
        if values.shape != tuple(source.grid.shape):
            raise ValueError(
                f"sampled geometry shape {values.shape} != GridSpec {source.grid.shape}"
            )
        values = np.ascontiguousarray(values, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("sampled geometry includes NaN/Inf or float32 overflow")
        origin = tuple(float(value) for value in source.grid.origin_mm)
        voxel = float(source.grid.voxel_mm)
        metadata = {
            "path": str(path), "sha256": actual_hash,
            "shape": tuple(source.grid.shape), "origin_mm": origin,
            "extent_mm": tuple(source.grid.extent_mm), "voxel_mm": voxel,
            "sign_convention": source.grid.sign_convention,
        }
        return values.shape, origin, voxel, values, metadata

    def _configure_sample(self, which, source, payload=None):
        if getattr(self, "_initialization_complete", False):
            raise RuntimeError(
                f"{which} sampled geometry must be declared during __init__; "
                "adding or replacing sampled fields after Taichi field "
                "materialization is unsafe. Create a fresh simulator instead."
            )
        if payload is None:
            payload = self._source_payload(source)
        shape, origin, voxel, values, metadata = payload
        flag_name = f"use_sampled_{which}"
        shape_name = f"{which}_source_shape"
        previous_shape = getattr(self, shape_name)
        if previous_shape is not None and previous_shape != shape:
            raise ValueError(
                f"{which} source shape is fixed at {previous_shape}. Field buffers "
                "are allocated during __init__; later setters may only bind data "
                "of that same shape, never reallocate a new GridSpec."
            )
        was_unconfigured = getattr(self, f"{which}_source_metadata") is None
        if was_unconfigured and previous_shape is None:
            # Allocation is legal only during __init__, never after lazy_grad().
            # Creating a different sampled configuration requires a new simulator.
            setattr(
                self, f"{which}_samples",
                ti.field(dtype=self._precision, shape=shape),
            )
        if which == "stock":
            expected_size = np.asarray((self.Lx, self.Ly, self.Lz), dtype=np.float64)
            actual_size = np.asarray(metadata["extent_mm"], dtype=np.float64)
            if not np.allclose(actual_size, expected_size, rtol=2e-6, atol=1e-8):
                raise ValueError(
                    "custom stock bounds mismatch: sampled GridSpec extent "
                    f"{actual_size.tolist()} != simulator stock {expected_size.tolist()}"
                )
            box_origin = tuple(self.stock_box_origin.to_numpy().tolist())
            if not np.allclose(box_origin, origin, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"custom stock origin {origin} != simulator stock_box_origin_mm "
                    f"{box_origin}. Bounds/coordinates are never silently stretched."
                )

        getattr(self, f"{which}_samples").from_numpy(values)
        getattr(self, f"{which}_sample_origin")[None] = ti.Vector(origin)
        getattr(self, f"{which}_sample_voxel")[None] = voxel
        setattr(self, shape_name, shape)
        setattr(self, f"{which}_source_metadata", metadata)
        getattr(self, flag_name)[None] = 1
        return metadata

    def set_custom_target(self, source):
        """Install an immutable checked sampled target (see constructor docs).

        Construction must declare sampled geometry. Later addition/replacement
        of sampled fields after Taichi materialization is explicitly forbidden,
        even when a trivial shape appears compatible with an unused placeholder.
        """
        metadata = self._configure_sample("target", source)
        # Switching back to analytic remains an explicit target_params call.
        # Remember which primitive shape produced the current parameters/default.
        if self.target_shape != "sampled":
            self._analytic_shape_before_target_sample = self.target_shape
        self.target_shape = "sampled"
        return metadata

    def set_custom_stock(self, source):
        """Install custom stock with matching physical stock bounds/origin."""
        return self._configure_sample("stock", source)

    def _init_sampled_geometry(self, target_geometry, stock_geometry, stock_box_origin_mm):
        self.use_sampled_target = ti.field(dtype=ti.i32, shape=())
        self.use_sampled_target[None] = 0
        self.use_sampled_stock = ti.field(dtype=ti.i32, shape=())
        self.use_sampled_stock[None] = 0
        self.target_source_shape = None if target_geometry is not None else (1, 1, 1)
        self.stock_source_shape = None if stock_geometry is not None else (1, 1, 1)
        self.target_source_metadata = None
        self.stock_source_metadata = None
        self._analytic_shape_before_target_sample = self.target_shape
        if target_geometry is None:
            self.target_samples = ti.field(dtype=self._precision, shape=(1, 1, 1))
        if stock_geometry is None:
            self.stock_samples = ti.field(dtype=self._precision, shape=(1, 1, 1))
        self.target_sample_origin = ti.Vector.field(3, dtype=self._precision, shape=())
        self.stock_sample_origin = ti.Vector.field(3, dtype=self._precision, shape=())
        self.target_sample_voxel = ti.field(dtype=self._precision, shape=())
        self.stock_sample_voxel = ti.field(dtype=self._precision, shape=())
        self.stock_box_origin = ti.Vector.field(3, dtype=self._precision, shape=())
        if stock_box_origin_mm is None:
            stock_box_origin_mm = (0.0, 0.0, 0.0)
        if len(stock_box_origin_mm) != 3:
            raise ValueError("stock_box_origin_mm must have three physical-mm components")
        origin = ti.Vector([float(value) for value in stock_box_origin_mm])
        if not np.all(np.isfinite(np.asarray(list(stock_box_origin_mm), dtype=np.float64))):
            raise ValueError("stock_box_origin_mm must be finite")
        self.stock_box_origin[None] = origin

        if target_geometry is not None:
            self.set_custom_target(target_geometry)
        if stock_geometry is not None:
            self.set_custom_stock(stock_geometry)

    def _init_target_fields(self):
        if self.target_shape == "box":
            self.target_params["half_size"] = ti.Vector.field(3, dtype=self._precision, shape=())
            self.target_params["center"] = ti.Vector.field(3, dtype=self._precision, shape=())
        elif self.target_shape == "cylinder":
            self.target_params["radius"] = ti.field(dtype=self._precision, shape=())
            self.target_params["height"] = ti.field(dtype=self._precision, shape=())
        elif self.target_shape == "sphere":
            self.target_params["radius"] = ti.field(dtype=self._precision, shape=())
            self.target_params["center"] = ti.Vector.field(3, dtype=self._precision, shape=())
        elif self.target_shape == "pyramid":
            self.target_params["base_half_size"] = ti.Vector.field(
                3, dtype=self._precision, shape=()
            )
            self.target_params["height"] = ti.field(dtype=self._precision, shape=())
            self.target_params["center"] = ti.Vector.field(3, dtype=self._precision, shape=())
        elif self.target_shape in ("sphere_hole", "sphere_bowl"):
            # Combined CSG shapes: a 0.9in stock sphere (radius+center, same
            # fields as the plain sphere) with a 0.75in sub-primitive subtracted
            # (a through-hole cylinder for sphere_hole; a lower hemisphere for
            # sphere_bowl). The sub-primitive radius lives in the scalar
            # tsub_vox field (set by set_target_params), not here.
            self.target_params["radius"] = ti.field(dtype=self._precision, shape=())
            self.target_params["center"] = ti.Vector.field(3, dtype=self._precision, shape=())
        else:
            raise ValueError(f"Unsupported target shape: {self.target_shape}")

    def set_target_params(self, radius_mm=None, height_mm=None, half_size_mm=None,
                          center=(0.5, 0.5, 0.5), sub_radius_mm=None):
        """Set whichever target params exist for the current shape (in mm).

        Shape-agnostic so callers don't need to branch: ``radius_mm`` feeds
        sphere/cylinder radius, ``half_size_mm`` feeds the box half-size / pyramid
        base half-size (scalar broadcast to a cube), and ``center`` is the
        normalized [0,1] centre where the shape defines one. Keys absent for the
        shape are skipped, so e.g. a cylinder (no ``center``) won't raise.

        ``sub_radius_mm`` feeds the SUB-PRIMITIVE radius for the combined CSG
        shapes (the through-hole cylinder in ``sphere_hole``; the subtracted
        sphere in ``sphere_bowl``). Defaults to 0.375 in (9.525 mm) -- half of
        the 0.75 in feature -- when omitted, so callers (eval, env reset) that
        don't pass it still get the spec'd 0.75 in sub-primitive.
        """
        tp = self.target_params
        if self.use_sampled_target[None]:
            self.use_sampled_target[None] = 0
            self.target_shape = self._analytic_shape_before_target_sample
        if "radius" in tp and radius_mm is not None:
            tp["radius"][None] = float(radius_mm)
        if "height" in tp and height_mm is not None:
            tp["height"][None] = float(height_mm)
        if half_size_mm is not None:
            hs = half_size_mm if hasattr(half_size_mm, "__len__") else (half_size_mm,) * 3
            hs = [float(c) for c in hs]
            if "half_size" in tp:
                tp["half_size"][None] = hs
            if "base_half_size" in tp:
                tp["base_half_size"][None] = hs
        if "center" in tp:
            tp["center"][None] = list(center)
        # Mirror into the scalar (autodiff-safe) target-param fields. Center is
        # in normalized [0,1]; sizes are converted to voxels.
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        self.tcx[None] = cx
        self.tcy[None] = cy
        self.tcz[None] = cz
        if radius_mm is not None:
            self.tr_vox[None] = float(radius_mm) / self.v
        if height_mm is not None:
            self.th_vox[None] = float(height_mm) / self.v
        if half_size_mm is not None:
            hs = half_size_mm if hasattr(half_size_mm, "__len__") else (half_size_mm,) * 3
            hs = [float(c) / self.v for c in hs]
            self.thx_vox[None] = hs[0]
            self.thy_vox[None] = hs[1]
            self.thz_vox[None] = hs[2]
            self.tbase_vox[None] = hs[0]
        # Sub-primitive radius for combined CSG shapes (default 0.375 in =
        # 9.525 mm, the 0.75 in feature halved). Stored in voxels for the SDF.
        if sub_radius_mm is None:
            sub_radius_mm = 9.525  # mm (0.75 in / 2)
        self.tsub_vox[None] = float(sub_radius_mm) / self.v

    # ========================================================================
    # SDFs  (all distances in VOXELS; cubic voxels make this isotropic)
    # ========================================================================

    @ti.func
    def _vox(self, q):
        """Normalized [0,1] point/vector -> voxel-space coordinate.

        Multiplying each axis by its grid count maps the (possibly anisotropic)
        normalized box to an isotropic voxel space where 1 unit = 1 cubic voxel
        = ``v`` mm, so every SDF below measures physically-faithful distances.
        """
        return ti.Vector([q.x * self.Nx, q.y * self.Ny, q.z * self.Nz])

    @ti.func
    def _tool_sweep_terms(self, p, t, height_vox: ti.template(), smooth_norm: ti.template()):
        """Signed horizontal/vertical terms for a swept upright cylinder.

        For query height z, vertically feasible segment parameters form one
        interval. Restricting the XY closest-point projection to that interval
        makes the HARD sign ``max(horizontal, vertical)`` exact for membership
        in the continuous sweep. This includes both endpoint cylinders.

        These two terms are NOT a closed-form Euclidean distance to the swept
        solid: capping the vertical interval is intentionally a conservative
        membership certificate, not a proof of Euclidean distance. All values
        are voxel units. Projection/interval boundaries are nonsmooth gates.
        """
        r = self.tool_radius[None] / self.v
        h = height_vox
        pv = self._vox(p)
        a = self._vox(self.tool_pos[t])
        b = self._vox(self.tool_pos[t + 1])
        dz = b.z - a.z

        # b(lambda) <= z <= b(lambda)+h intersected with lambda in [0,1].
        lo = 0.0
        hi = 1.0
        if dz != 0.0:  # no arbitrary "near vertical" epsilon gate
            at_base = (pv.z - a.z) / dz
            at_top = (pv.z - h - a.z) / dz
            lo = ti.max(0.0, ti.min(at_base, at_top))
            hi = ti.min(1.0, ti.max(at_base, at_top))
        else:
            if pv.z < a.z or pv.z > a.z + h:
                lo = 1.0
                hi = 0.0  # empty vertical interval

        # A signed gap to the union of all cutter Z spans. It is positive
        # outside that union and negative inside it (zero only at its caps).
        z_low = ti.min(a.z, b.z)
        z_high = ti.max(a.z, b.z) + h
        vertical = ti.max(z_low - pv.z, pv.z - z_high)

        pa = ti.Vector([pv.x - a.x, pv.y - a.y])
        dxy = ti.Vector([b.x - a.x, b.y - a.y])
        dxy_len2 = dxy.dot(dxy)
        projection = 0.0
        if dxy_len2 > 0.0:  # only exact XY degeneracy bypasses projection
            projection = pa.dot(dxy) / dxy_len2
        if lo <= hi:
            # Restricted nearest centre is exact for fixed-height membership.
            projection = ti.max(lo, ti.min(hi, projection))
        else:
            # No pose covers this height. Vertical is positive and dominates
            # the hard max, so use the unrestricted capsule term outside.
            projection = ti.max(0.0, ti.min(1.0, projection))
        offset = pa - dxy * projection
        radial_squared = offset.dot(offset)
        if smooth_norm:
            # Smooth-only derivative regularization at the cylinder axis;
            # hard closed-membership must not inherit this radial bias.
            radial_squared += 1e-8
        horizontal = ti.sqrt(radial_squared) - r
        return horizontal, vertical

    @ti.func
    def _tool_sweep_smooth_z(self, p, t, height_vox: ti.template()):
        """Smooth Z surrogate, deliberately different from the hard score.

        The hard vertical term selects a closest cap with max(), which creates
        an interior absolute-value kink at equal cap distances. This surrogate
        instead uses a centred norm of the moving Z interval with a smooth norm
        of the segment's |dz| displacement. Coincident poses/cap midpoints no
        longer add an undocumented one-sided derivative branch. This is not
        used by hard membership and need not equal its signed value.
        """
        pv = self._vox(p)
        a = self._vox(self.tool_pos[t])
        b = self._vox(self.tool_pos[t + 1])
        displacement = b.z - a.z
        # min(a,b)+max(a,b)=a+b, so this is the interval centre without abs().
        centre = 0.5 * (a.z + b.z + height_vox)
        extent = height_vox + ti.sqrt(displacement * displacement + 1e-8)
        return ti.sqrt((pv.z - centre) ** 2 + 1e-8) - 0.5 * extent

    @ti.func
    def tool_sdf(self, p, t):
        """SEPARATE differentiable surrogate for the swept flat cutter.

        The hard sign certificate lives in ``tool_sdf_sharp``. This method uses
        the restricted horizontal term but a separately centred, smooth Z norm
        plus smooth-CSG combinations. It is not the hard Euclidean SDF and need
        not preserve its sign/value near boundaries. Interval clamps/endpoints
        in XY remain nonsmooth; the hard max-based vertical cap kink is not
        inherited. Gradient tests stay away from the remaining projection gates. Radius/height in mm become isotropic voxel units.
        """
        kv = self.k[None] / self.k_ref
        d_xy, d_z = self._tool_sweep_terms(p, t, self.tool_height[None] / self.v, True)
        d_z = self._tool_sweep_smooth_z(p, t, self.tool_height[None] / self.v)
        d_xy_pos = smooth_max(d_xy, 0.0, kv)
        d_z_pos = smooth_max(d_z, 0.0, kv)
        outside = ti.sqrt(d_xy_pos * d_xy_pos + d_z_pos * d_z_pos + 1e-8)
        inside = -smooth_max(-smooth_max(d_xy, d_z, kv), 0.0, kv)
        return outside + inside

    @ti.func
    def _tool_cut_h_vox(self) -> float:
        """Cutting-tip band height in voxels: tool_cut_height if set (>0), else
        the full tool_height (legacy). Shared by the smooth/sharp tip SDFs."""
        ch = self.tool_cut_height[None]
        return ti.select(ch > 0.0, ch / self.v, self.tool_height[None] / self.v)

    @ti.func
    def tool_sdf_tip(self, p, t):
        """Smoothed swept-cylinder SDF over the CUTTING TIP band only (voxels).

        Identical to ``tool_sdf`` except the cylinder height is the tip band
        (``tool_cut_height``) instead of the full ``tool_height`` shank, and the
        z-center is anchored to the tool BASE (z_base + 0.5*cut_h) so the band
        is the bottom of the tool -- the part that actually removes material.
        Used by the air-time metric/loss so the swept/air/engage volumes reflect
        whether the cutting tip is in material, not how much shank sits in
        already-carved empty space. Differentiable (same form as tool_sdf).
        """
        kv = self.k[None] / self.k_ref               # smoothness in voxel space
        d_xy, d_z = self._tool_sweep_terms(p, t, self._tool_cut_h_vox(), True)
        d_z = self._tool_sweep_smooth_z(p, t, self._tool_cut_h_vox())

        d_xy_pos = smooth_max(d_xy, 0.0, kv)
        d_z_pos = smooth_max(d_z, 0.0, kv)
        outside = ti.sqrt(d_xy_pos * d_xy_pos + d_z_pos * d_z_pos + 1e-8)
        inside = -smooth_max(-smooth_max(d_xy, d_z, kv), 0.0, kv)
        return outside + inside

    @ti.func
    def holder_sdf(self, p, t):
        """Smoothed conservative holder-surrogate over whole segment ``t``.

        This is not a chronological holder collision field. It uses a holder
        radius with a z-range above the cutting-flute height, then takes the
        conservative union of start/end Z spans. Differentiable with respect
        to tool_pos/tool_delta for the penalty screen. Physical non-cutting
        shank length or a distinct holder_bottom is not modeled.
        """
        r = self.holder_radius[None] / self.v        # voxels
        h = self.holder_height[None] / self.v        # voxels
        tool_h = self.tool_height[None] / self.v     # voxels
        kv = self.k[None] / self.k_ref

        pv = self._vox(p)
        a = self._vox(self.tool_pos[t])
        b = self._vox(self.tool_pos[t + 1])

        pa_xy = ti.Vector([pv.x - a.x, pv.y - a.y])
        ba_xy = ti.Vector([b.x - a.x, b.y - a.y])
        ba_len2 = ba_xy.dot(ba_xy) + 1e-12
        h_param = ti.max(0.0, ti.min(1.0, pa_xy.dot(ba_xy) / ba_len2))
        closest_xy = ti.Vector([a.x, a.y]) + ba_xy * h_param
        d_xy = ti.sqrt((pv.x - closest_xy.x) ** 2 + (pv.y - closest_xy.y) ** 2 + 1e-8) - r

        # Holder Z extent UNIONED over the swept segment (see holder_sdf_sharp
        # for the full rationale). The holder bottom tracks the tool base
        # (z_base + tool_h) from a to b; the body extends up by h. The unioned
        # range [min(bottom_a, bottom_b), max(bottom_a, bottom_b) + h] captures
        # the full swept Z extent -- evaluating at a single h_param misses the
        # sweep on near-vertical segments (deep plunge -> holder drops into
        # material the SDF reads as clear). ti.min/ti.max are subdifferentiable
        # so this stays autodiff-safe. With holder_height = full machine Z, the
        # top is always above the grid and the lowest bottom is binding.
        z_bottom_a = a.z + tool_h
        z_bottom_b = b.z + tool_h
        z_low = ti.min(z_bottom_a, z_bottom_b)
        z_high = ti.max(z_bottom_a, z_bottom_b) + h
        z_center = 0.5 * (z_low + z_high)
        z_half = 0.5 * (z_high - z_low)
        d_z = ti.sqrt((pv.z - z_center) ** 2 + 1e-8) - z_half

        d_xy_pos = smooth_max(d_xy, 0.0, kv)
        d_z_pos = smooth_max(d_z, 0.0, kv)
        outside = ti.sqrt(d_xy_pos * d_xy_pos + d_z_pos * d_z_pos + 1e-8)
        inside = -smooth_max(-smooth_max(d_xy, d_z, kv), 0.0, kv)
        return outside + inside

    @ti.func
    def _sample_physical_point(self, p):
        """Legacy normalized stock coordinate -> physical world mm."""
        p_world = ti.Vector([p.x * self.Lx, p.y * self.Ly, p.z * self.Lz])
        return p_world + self.stock_box_origin[None]

    @ti.func
    def _sample_trilinear(self, field, shape, origin, point_mm, voxel_mm):
        """Bounded trilinear SDF in mm on a cell-centred GridSpec.

        Index-center coordinate is (x-origin)/voxel - .5. Coordinates outside
        the declared grid clamp to the nearest sample-centre plane; there is no
        unconstrained extrapolation and no implied infinite negative stock.
        Indices and weights clamp inside the sampled array, so edge corners are
        repeated rather than indexing one cell beyond an axis.
        """
        u = ti.Vector([
            (point_mm.x - origin[0]) / voxel_mm - 0.5,
            (point_mm.y - origin[1]) / voxel_mm - 0.5,
            (point_mm.z - origin[2]) / voxel_mm - 0.5,
        ])
        i0 = ti.Vector([0, 0, 0], dt=ti.i32)
        i1 = ti.Vector([0, 0, 0], dt=ti.i32)
        weight = ti.Vector([0.0, 0.0, 0.0])
        for axis in ti.static(range(3)):
            coord = ti.max(0.0, ti.min(float(shape[axis] - 1), u[axis]))
            low = ti.max(0, ti.min(shape[axis] - 2, ti.cast(ti.floor(coord), ti.i32)))
            i0[axis] = low
            i1[axis] = low + 1
            weight[axis] = coord - low

        c000 = field[i0[0], i0[1], i0[2]]
        c100 = field[i1[0], i0[1], i0[2]]
        c010 = field[i0[0], i1[1], i0[2]]
        c110 = field[i1[0], i1[1], i0[2]]
        c001 = field[i0[0], i0[1], i1[2]]
        c101 = field[i1[0], i0[1], i1[2]]
        c011 = field[i0[0], i1[1], i1[2]]
        c111 = field[i1[0], i1[1], i1[2]]
        x, y, z = weight.x, weight.y, weight.z
        c00 = c000 * (1 - x) + c100 * x
        c10 = c010 * (1 - x) + c110 * x
        c01 = c001 * (1 - x) + c101 * x
        c11 = c011 * (1 - x) + c111 * x
        c0 = c00 * (1 - y) + c10 * y
        c1 = c01 * (1 - y) + c11 * y
        return c0 * (1 - z) + c1 * z

    @ti.func
    def _sample_physical_target(self, p):
        physical = self._sample_physical_point(p)
        value_mm = self._sample_trilinear(
            self.target_samples,
            self.target_samples.shape,
            self.target_sample_origin[None],
            physical,
            self.target_sample_voxel[None],
        )
        return value_mm / self.v

    @ti.func
    def _sample_physical_stock(self, p):
        physical = self._sample_physical_point(p)
        value_mm = self._sample_trilinear(
            self.stock_samples,
            self.stock_samples.shape,
            self.stock_sample_origin[None],
            physical,
            self.stock_sample_voxel[None],
        )
        return value_mm / self.v

    @ti.func
    def target_sdf(self, p):
        """Target shape SDF (voxels) — branches resolved at compile time.

        Center is normalized [0,1]; radius/height/half-size are millimetres.
        """
        d = 0.0
        pv = self._vox(p)

        if self.use_sampled_target[None]:
            d = self._sample_physical_target(p)
        elif ti.static(self.target_shape == "sphere"):
            d = sphere_sdf(
                pv,
                self._vox(self.target_params["center"][None]),
                self.target_params["radius"][None] / self.v,
            )
        elif ti.static(self.target_shape == "box"):
            d = box_sdf(
                pv,
                self._vox(self.target_params["center"][None]),
                self.target_params["half_size"][None] / self.v,
            )
        elif ti.static(self.target_shape == "cylinder"):
            radius = self.target_params["radius"][None] / self.v
            height = self.target_params["height"][None] / self.v
            cv = self._vox(ti.Vector([0.5, 0.5, 0.5]))
            d = cylinder_sdf(pv, cv.x, cv.y, cv.z - 0.5 * height, radius, height)
        elif ti.static(self.target_shape == "pyramid"):
            height = self.target_params["height"][None] / self.v
            half_base = self.target_params["base_half_size"][None].x / self.v
            cv = self._vox(self.target_params["center"][None])
            d = pyramid_sdf(pv, cv.x, cv.y, cv.z - 0.5 * height, half_base, height)
        elif ti.static(self.target_shape == "sphere_hole"):
            # Stock sphere with a concentric through-hole cylinder subtracted
            # along Z. The cylinder is UNBOUNDED in z (no z-clamp): the smooth
            # max with the sphere SDF bounds the hole to the sphere, so the
            # cylinder only needs to exceed the sphere's z-extent -- an infinite
            # cylinder is the cleanest through-hole. Sub-primitive radius
            # (0.75in/2) from tsub_vox.
            r_vox = self.target_params["radius"][None] / self.v
            cv = self._vox(self.target_params["center"][None])
            d_sphere = (pv - cv).norm() - r_vox
            d_hole = ti.Vector([pv.x - cv.x, pv.y - cv.y]).norm() - self.tsub_vox[None]
            d = ti.max(d_sphere, -d_hole)
        elif ti.static(self.target_shape == "sphere_bowl"):
            # Stock sphere with the LOWER hemisphere of a concentric 0.75in
            # sphere subtracted (z below center) -- a bowl whose cavity opens
            # upward at the equator. Subtraction region R = sub-sphere ∩ {z <=
            # center.z}; the lower half-space SDF is (pv.z - cv.z) (negative
            # below the equator); d_R = max(sub_sphere_sdf, pv.z - cv.z); bowl
            # = max(sphere_sdf, -d_R).
            r_vox = self.target_params["radius"][None] / self.v
            cv = self._vox(self.target_params["center"][None])
            d_sphere = (pv - cv).norm() - r_vox
            d_sub = (pv - cv).norm() - self.tsub_vox[None]
            d_below = pv.z - cv.z  # <0 below the equator (inside lower half)
            d = ti.max(d_sphere, -ti.max(d_sub, d_below))
        return d

    @ti.func
    def target_sdf_scalar(self, p):
        """Target SDF (voxels) from the SCALAR target-param fields -- an
        autodiff-safe mirror of target_sdf for use inside differentiable kernels
        that feed a grad-tracked position p (the Vector-field target_params
        trigger a Taichi autodiff load-forwarding bug there). Center in [0,1],
        sizes pre-converted to voxels.
        """
        d = 0.0
        pv = ti.Vector([p.x * self.Nx, p.y * self.Ny, p.z * self.Nz])
        cv = ti.Vector([self.tcx[None] * self.Nx, self.tcy[None] * self.Ny, self.tcz[None] * self.Nz])
        if self.use_sampled_target[None]:
            d = self._sample_physical_target(p)
        elif ti.static(self.target_shape == "sphere"):
            d = (pv - cv).norm() - self.tr_vox[None]
        elif ti.static(self.target_shape == "box"):
            dd = ti.abs(pv - cv) - ti.Vector([self.thx_vox[None], self.thy_vox[None], self.thz_vox[None]])
            d = ti.max(dd.x, ti.max(dd.y, dd.z))
        elif ti.static(self.target_shape == "cylinder"):
            d_h = ti.Vector([pv.x - cv.x, pv.y - cv.y]).norm() - self.tr_vox[None]
            d_z = ti.max(cv.z - 0.5 * self.th_vox[None] - pv.z, pv.z - (cv.z + 0.5 * self.th_vox[None]))
            d = ti.max(d_h, d_z)
        elif ti.static(self.target_shape == "pyramid"):
            h = self.th_vox[None]
            t = (pv.z - (cv.z - 0.5 * h)) / h
            d_bottom = (cv.z - 0.5 * h) - pv.z
            d_top = pv.z - (cv.z + 0.5 * h)
            allowed = self.tbase_vox[None] * (1.0 - t)
            dx = ti.abs(pv.x - cv.x) - allowed
            dy = ti.abs(pv.y - cv.y) - allowed
            d_sides = ti.max(dx, dy)
            d = ti.max(d_bottom, ti.max(d_top, d_sides))
        elif ti.static(self.target_shape == "sphere_hole"):
            # Autodiff-safe mirror of the sphere_hole branch in target_sdf.
            r_vox = self.tr_vox[None]
            d_sphere = (pv - cv).norm() - r_vox
            d_hole = ti.Vector([pv.x - cv.x, pv.y - cv.y]).norm() - self.tsub_vox[None]
            d = ti.max(d_sphere, -d_hole)
        elif ti.static(self.target_shape == "sphere_bowl"):
            # Autodiff-safe mirror of the sphere_bowl branch in target_sdf.
            r_vox = self.tr_vox[None]
            d_sphere = (pv - cv).norm() - r_vox
            d_sub = (pv - cv).norm() - self.tsub_vox[None]
            d_below = pv.z - cv.z  # <0 below the equator (inside lower half)
            d = ti.max(d_sphere, -ti.max(d_sub, d_below))
        return d

    @ti.kernel
    def set_target_volume(self):
        """Target volume as a fraction of the envelope (occupied voxels / total)."""
        count = 0
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            if self.target_sdf(p) < 0:
                count += 1
        self.target_volume[None] = count / float(self.Nx * self.Ny * self.Nz)

    @ti.kernel
    def init_stock(self):
        """Initial stock: the full envelope block, as a voxel-space SDF in stock[0].

        When ``use_saved_init`` is set (staged training), the saved mid-cut SDF
        in ``saved_stock`` is written to stock[0] instead, so a fresh trajectory
        starts carving from a partially-carved state.
        """
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.use_saved_init[None]:
                # Staged training: start from a saved mid-cut SDF.
                self.stock[0, i, j, k] = self.saved_stock[i, j, k]
            elif self.use_sampled_stock[None]:
                p = ti.Vector(
                    [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
                )
                self.stock[0, i, j, k] = self._sample_physical_stock(p)
            else:
                p = ti.Vector(
                    [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
                )
                self.stock[0, i, j, k] = box_sdf(
                    self._vox(p),
                    self._vox(ti.Vector([0.5, 0.5, 0.5])),
                    self._vox(ti.Vector([0.5, 0.5, 0.5])),
                )

    @ti.kernel
    def bake_target_grid(self):
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            self.target[i, j, k] = self.target_sdf(p)

    # ========================================================================
    # Forward pass: reconstruct positions → init → carving → loss
    # ========================================================================

    @ti.kernel
    def reconstruct_positions(self, T: ti.i32):
        """Cumulative-sum scan: tool_pos[t+1] = tool_pos[t] + tool_delta[t].

        TIt MUST run serially — each iteration depends on the previous one's result —
        so ti.loop_config(serialize=True) is mandatory.
        Without it, Taichi would parallelize the top-level loop
        and the scan would be wrong (and so would its gradient).

        It is gradient-tracked: tool_pos.grad accumulates the indirect-path
        contributions, and Taichi's autodiff propagates those back into
        tool_delta.grad. Run this inside the same ti.ad.Tape as the carving.
        """
        ti.loop_config(serialize=True)
        for _ in range(1):
            self.tool_pos[0] = self.tool_start[None]
            for t in range(T):
                self.tool_pos[t + 1] = self.tool_pos[t] + self.tool_delta[t]

    @ti.kernel
    def reconstruct_positions_from(self, t0: ti.i32, T: ti.i32):
        """Cumulative-sum scan starting from a RESTORED mid-cut state.

        Assumes ``tool_pos[t0]`` has already been populated (e.g. by
        ``restore_state``) and scans ``tool_pos[t+1] = tool_pos[t] +
        tool_delta[t]`` for ``t in [t0, T)``. Used by ``forward_from`` to
        restart a forward pass from a saved state without re-running the
        prefix ``[0, t0)``. Serial for the same reason as
        ``reconstruct_positions``; gradient-tracked for ``t >= t0``.
        """
        ti.loop_config(serialize=True)
        for _ in range(1):
            for t in range(t0, T):
                self.tool_pos[t + 1] = self.tool_pos[t] + self.tool_delta[t]

    @ti.func
    def stock_sdf_at(self, p, t_idx):
        """Trilinear lookup of the stock SDF in slot ``t_idx`` at point ``p``.

        Same interpolation as ``interpolate_stock`` but reads an explicit step
        slot instead of ``current_step`` -- needed by ``advance_position`` to
        measure how close the cutter is to the *remaining* stock at the start of
        a move. Returns a voxel-space signed distance (negative inside material).
        """
        p_grid = self._vox(p)
        x0 = ti.cast(ti.floor(p_grid.x), ti.i32)
        y0 = ti.cast(ti.floor(p_grid.y), ti.i32)
        z0 = ti.cast(ti.floor(p_grid.z), ti.i32)
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        tx, ty, tz = p_grid.x - x0, p_grid.y - y0, p_grid.z - z0

        x0 = ti.max(0, ti.min(self.Nx - 1, x0))
        x1 = ti.max(0, ti.min(self.Nx - 1, x1))
        y0 = ti.max(0, ti.min(self.Ny - 1, y0))
        y1 = ti.max(0, ti.min(self.Ny - 1, y1))
        z0 = ti.max(0, ti.min(self.Nz - 1, z0))
        z1 = ti.max(0, ti.min(self.Nz - 1, z1))

        c000 = self.stock[t_idx, x0, y0, z0]
        c100 = self.stock[t_idx, x1, y0, z0]
        c010 = self.stock[t_idx, x0, y1, z0]
        c110 = self.stock[t_idx, x1, y1, z0]
        c001 = self.stock[t_idx, x0, y0, z1]
        c101 = self.stock[t_idx, x1, y0, z1]
        c011 = self.stock[t_idx, x0, y1, z1]
        c111 = self.stock[t_idx, x1, y1, z1]

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    @ti.kernel
    def advance_position(self, t: ti.i32):
        """Integrate one step with feed/rapid speed clipping (the constraint).

        Reconstructs ``tool_pos[t+1] = tool_pos[t] + clipped(tool_delta[t])``.
        The commanded per-step displacement implies a speed
        ``|delta (.) L| / dt`` (mm/s), where ``L`` is the per-axis envelope; if
        that exceeds the regime's max speed the displacement is scaled down so
        the *actual* move runs exactly at the cap (direction preserved) -- a
        differentiable analogue of a machine controller clamping the feed.

        Regime (all distances in mm): the move is treated as cutting ->
        ``feed_speed`` when the cutter would come within ``safe_distance`` of the
        REMAINING stock; otherwise it is a traverse with clearance ->
        ``rapid_speed``. Engagement is probed at the COMMANDED destination
        (``tool_pos[t] + tool_delta[t]``) against the stock at the start of the
        step -- "am I moving into material?" -- rather than at the current
        position, which sits in the hole the previous cut just made. The regime
        is a comparison (a hard gate with zero gradient), so gradients still flow
        cleanly through the clipped magnitude into ``tool_delta``.
        """
        a = self.tool_pos[t]
        delta = self.tool_delta[t]

        # Commanded step length, in mm (normalized delta scaled per-axis).
        dmm = ti.Vector([delta.x * self.Lx, delta.y * self.Ly, delta.z * self.Lz])
        mag_mm = ti.sqrt(dmm.dot(dmm) + 1e-12)

        # Clearance from the cutter to the remaining stock at the commanded
        # destination, in mm (negative -> the cutter is inside material).
        # The voxel grid only represents the envelope; outside it the trilinear
        # lookup clamps to the boundary layer and is meaningless. Remaining stock
        # is always a subset of the envelope, so the distance to the envelope box
        # is a valid clearance floor -- combine the two so a probe lifted into
        # open air above the stock reads true clearance (rapid) instead of the
        # top surface (feed). Both SDFs are in voxels; convert to mm via ``v``.
        probe = a + delta
        stock_d = self.stock_sdf_at(probe, t)
        cube_d = box_sdf(
            self._vox(probe),
            self._vox(ti.Vector([0.5, 0.5, 0.5])),
            self._vox(ti.Vector([0.5, 0.5, 0.5])),
        )
        clearance = ti.max(stock_d, cube_d)
        clearance_mm = clearance * self.v - self.tool_radius[None]
        is_feed = ti.cast(clearance_mm <= self.safe_distance[None], self._precision)

        # Max distance allowed this step for each regime: speed (mm/s) * dt (s).
        rapid_step = self.rapid_speed[None] * self.dt[None]
        feed_step = self.feed_speed[None] * self.dt[None]
        cap_mm = rapid_step + (feed_step - rapid_step) * is_feed

        # Clip: shrink the step to the cap if it is too fast (else leave it).
        scale = ti.min(1.0, cap_mm / mag_mm)
        enforced = ti.cast(self.enforce_speed_limits[None], self._precision)
        scale = scale * enforced + (1.0 - enforced)  # 1.0 when not enforcing

        next_pos = a + delta * scale

        # Z-floor: clamp the executed tool BASE z to >= z_floor (normalized).
        # Like the speed clip above, this is a subgradient -- gradient flows
        # through above the floor and is zero below it (the requested hard
        # limit). Runtime-gated by enforce_z_floor so it is configurable per run.
        z_new = next_pos.z
        z_floored = ti.max(z_new, self.z_floor[None])
        floor_on = ti.cast(self.enforce_z_floor[None], self._precision)
        z_final = z_floored * floor_on + z_new * (1.0 - floor_on)
        self.tool_pos[t + 1] = ti.Vector([next_pos.x, next_pos.y, z_final])

    @ti.kernel
    def zero_tool_deltas(self):
        for t in range(self.max_steps):
            self.tool_delta[t] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def apply_cut(self, t: ti.i32):
        """stock[t+1] = smooth_max(stock[t], -tool_sdf at segment t)."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # k is scaled by k_ref because SDFs are now in voxels (the carve
            # smooth_max would otherwise overflow exp() and NaN the gradient).
            kv = self.k[None] / self.k_ref  # moved inside loop for autodiff
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            tool_d = self.tool_sdf(p, t)
            self.stock[t + 1, i, j, k] = smooth_max(self.stock[t, i, j, k], -tool_d, kv)

    @ti.kernel
    def apply_cut_hard(self, t: ti.i32):
        """stock[t+1] = max(stock[t], -tool_sdf_sharp at segment t)."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            tool_d = self.tool_sdf_sharp(p, t)
            self.stock[t + 1, i, j, k] = ti.max(self.stock[t, i, j, k], -tool_d)

    @ti.kernel
    def loss_at(self, t: ti.i32) -> float:
        """Exact replica of compute_loss's objective, evaluated on stock[t].

        Same soft-occupancy formulation, same weights, same sigmoid scale.
        Returns the scalar instead of writing to self.loss, and reads no grad —
        safe to call outside a Tape (for RL reward / eval).
        """
        total = 0.0
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            scale = 1.0  # SDFs are already in voxels (1-voxel-wide sigmoid)
            inv_n = 1.0 / (self.Nx * self.Ny * self.Nz)
            p = ti.Vector([(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz])
            stock_d = self.stock[t, i, j, k]
            target_d = self.target_sdf(p)

            sa = ti.max(-50.0, ti.min(50.0, stock_d * scale))
            ta = ti.max(-50.0, ti.min(50.0, target_d * scale))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))
            target_occ = 1.0 / (1.0 + ti.exp(ta))

            gouge = target_occ * (1.0 - stock_occ)
            residual = (1.0 - target_occ) * stock_occ
            # Annulus-residual emphasis (see w_annulus): upweight near-surface
            # waste. No-op when w_annulus=0 (factor is exactly 1).
            td_pos = ti.max(0.0, target_d)
            aw = ti.max(0.0, 1.0 - td_pos / self.annulus_dref[None])
            residual = residual * (1.0 + self.w_annulus[None] * aw)

            w_gouge = self.w_gouge[None]
            w_residual = self.w_residual[None]
            total += inv_n * (w_gouge * gouge * gouge + w_residual * residual * residual)
        return total

    @ti.kernel
    def compute_loss(self, T: ti.i32):
        """Terminal occupancy loss: weighted gouge² + residual²."""
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            # All scalar reads moved inside for autodiff compatibility.
            scale = 1.0  # SDFs are already in voxels (1-voxel-wide sigmoid)
            inv_n = 1.0 / (self.Nx * self.Ny * self.Nz)
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            stock_d = self.stock[T, i, j, k]
            target_d = self.target_sdf(p)

            sa = ti.max(-50.0, ti.min(50.0, (stock_d + self.loss_shift[None]) * scale))
            ta = ti.max(-50.0, ti.min(50.0, target_d * scale))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))
            target_occ = 1.0 / (1.0 + ti.exp(ta))

            gouge = target_occ * (1.0 - stock_occ)
            residual = (1.0 - target_occ) * stock_occ
            # Annulus-residual emphasis (see w_annulus): upweight near-surface
            # waste. No-op when w_annulus=0 (factor is exactly 1).
            td_pos = ti.max(0.0, target_d)
            aw = ti.max(0.0, 1.0 - td_pos / self.annulus_dref[None])
            residual = residual * (1.0 + self.w_annulus[None] * aw)

            w_gouge = self.w_gouge[None]
            w_residual = self.w_residual[None]
            ti.atomic_add(
                self.loss[None],
                inv_n * (w_gouge * gouge * gouge + w_residual * residual * residual),
            )

    @ti.kernel
    def compute_holder_penalty(self, t_start: ti.i32, T: ti.i32):
        """Differentiable holder-collision PENETRATION BARRIER added to ``self.loss``.

        For every cut t the holder rides above the tool along segment t. The
        occupancy source is **stock[t], the pre-segment remaining stock**. The
        holder uses its unioned whole-segment pose interval, so this is a
        CONSERVATIVE WHOLE-SEGMENT training screen, not an exact chronological
        collision claim. Material that the current cutter will remove cannot be
        credited to the holder on the same segment. Temporal refinement reduces
        (does not necessarily eliminate) conservative false positives.

        For each pre-segment material voxel:

            penetration = relu( (holder_margin - holder_sdf) * scale )

        This is zero with zero gradient when conservative clearance holds, and
        quadratic with penetration. holder_sdf and the margin are in voxels;
        physical standoff = margin*v mm. The non-cutting shank interval from
        contract ToolAssembly is not modeled (see handoff geometry limitations).

        This is a one-sided barrier, not a proximity field: a high weight makes
        it a near-hard constraint that stays inactive until violated, instead of
        a force that shoves the tool away from the workpiece pre-emptively.

        ``t_start`` restricts the sum to ``[t_start, T)`` so a restart-from-state
        forward pass only charges for the segments it actually carved (the slots
        below ``t_start`` are stale/restored constants and carry no gradient).
        """
        for t, i, j, k in ti.ndrange((t_start, T), self.Nx, self.Ny, self.Nz):
            scale = 1.0  # SDFs are already in voxels (1-voxel-wide sigmoid)
            inv_n = 1.0 / (self.Nx * self.Ny * self.Nz)
            w = self.holder_penalty_weight[None]
            margin = self.holder_margin[None]
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            stock_d = self.stock[t, i, j, k]  # pre-cut conservative screen
            holder_d = self.holder_sdf(p, t)

            sa = ti.max(-50.0, ti.min(50.0, stock_d * scale))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))    # ~1 inside remaining stock
            penetration = ti.max(0.0, (margin - holder_d) * scale)

            ti.atomic_add(self.loss[None], inv_n * w * stock_occ * penetration * penetration)

    @ti.kernel
    def compute_air_penalty(self, t_start: ti.i32, T: ti.i32):
        """Differentiable AIR-CUT penalty added to ``self.loss``.

        For every cut ``t`` in ``[t_start, T)`` the swept cylinder (``tool_sdf``)
        occupies some volume; the fraction of that volume lying in EMPTY stock
        (no remaining material) is "cutting air". We charge for it:

            air = tool_occ * (1 - stock_occ)

        where ``tool_occ = sigmoid(-tool_sdf)`` (~1 inside the swept tool for
        segment t) and ``stock_occ = sigmoid(stock[t+1])`` (~1 inside remaining
        material after the cut). Their product is ~1 only where the tool swept
        through open space, and ~0 where it actually removed material. Quadratic
        so small engagements are cheap and long air traverses dominate. Gated by
        ``w_air`` (0 -> exactly zero, zero gradient -> disabled).

        In the SAME loop we also add the distance-weighted (contour-hug) air
        penalty gated by ``w_prox``: the air charge is multiplied by
        ``(max(0, target_sdf) / r_tool)^2`` from the precomputed target grid, so
        air-cutting FAR from the target surface (empty corners) is heavily
        penalized while surface-hugging and the necessary first-pass carving
        (in remaining stock, air ~ 0) stay cheap. The target grid is a constant
        so it adds no gradient path -- gradient still flows only through ``air``.

        Differentiable in ``tool_pos``/``tool_delta`` (via ``tool_sdf``) and in
        ``stock`` (via ``stock_occ``); safe under ``ti.ad.Tape``.

        Caveat (same grid-only blind spot as ``compute_traj_diagnostics_hard``
        before its fix): this loop sums only over the [0,1]^3 voxel grid, so a
        tool swept entirely OFF the stock contributes 0 air here too — it will
        not penalize far-from-stock wandering even with w_air>0. ``w_air`` /
        ``w_prox`` are confirmed dead levers (trade off dice), so this is left
        as-is; a shape-blind stock-proximity anchor is the open lever for the
        off-stock collapse (see autoresearch.md).
        """
        for t, i, j, k in ti.ndrange((t_start, T), self.Nx, self.Ny, self.Nz):
            scale = 1.0  # SDFs are already in voxels (1-voxel-wide sigmoid)
            inv_n = 1.0 / (self.Nx * self.Ny * self.Nz)
            w = self.w_air[None]
            w_p = self.w_prox[None]
            r_vox = self.tool_radius[None] / self.v
            r_safe = ti.max(r_vox, 1e-3)
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            stock_d = self.stock[t + 1, i, j, k]
            tool_d = self.tool_sdf(p, t)

            sa = ti.max(-50.0, ti.min(50.0, stock_d * scale))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))            # ~1 inside material
            ta = ti.max(-50.0, ti.min(50.0, -tool_d * scale))
            tool_occ = 1.0 / (1.0 + ti.exp(ta))             # ~1 inside the swept tool
            air = tool_occ * (1.0 - stock_occ)

            # Distance-weighted air: charge air-cutting in proportion to its
            # distance from the target surface (squared, in tool-radii). Zero on
            # the surface, grows into the empty corners. Cheap (target grid is a
            # constant lookup; the expensive tool_sdf is already computed above).
            # Combined with the plain w_air term into ONE atomic_add so the
            # serialized global reduction is not doubled.
            d_t = ti.max(0.0, self.target[i, j, k])
            w_dist = (d_t / r_safe) ** 2
            ti.atomic_add(
                self.loss[None], inv_n * (w + w_p * w_dist) * air * air
            )

    @ti.kernel
    def compute_jerk_penalty(self, t_start: ti.i32, T: ti.i32):
        """Differentiable JERK / smoothness penalty added to ``self.loss``.

        Penalizes abrupt changes in the commanded per-step displacement:

            jerk = |tool_delta[t] - tool_delta[t-1]|^2

        summed over ``t in [max(t_start,1), T-1)`` and normalized by the segment
        count so the magnitude is comparable across restart lengths. Acts
        directly on ``tool_delta`` (``needs_grad=True``), so the gradient is a
        simple finite-difference of the deltas. Gated by ``w_jerk``
        (0 -> disabled).
        """
        for t in range(ti.max(t_start, 1), T - 1):
            w = self.w_jerk[None]
            n = ti.max(1, T - 1 - ti.max(t_start, 1))
            diff = self.tool_delta[t] - self.tool_delta[t - 1]
            ti.atomic_add(self.loss[None], w * diff.dot(diff) / n)

    @ti.kernel
    def compute_step_penalty(self, t_start: ti.i32, T: ti.i32):
        """Differentiable SPEED-REGULARITY (constant-feed) penalty added to ``self.loss``.

        Penalizes changes in the commanded per-step SPEED (squared step length):

            step = (|tool_delta[t]|^2 - |tool_delta[t-1]|^2)^2

        summed over ``t in [max(t_start,1), T-1)`` and normalized by the segment
        count. Unlike ``compute_jerk_penalty`` (which penalizes the full vector
        difference of consecutive deltas, i.e. both speed AND direction changes),
        this acts only on the step LENGTH, so it pushes the feed rate toward a
        constant value without discouraging the legitimate back-and-forth
        direction reversals of a boustrophedon/raster toolpath. Uses squared
        lengths (polynomial, no sqrt) so the gradient stays well-conditioned near
        zero step length. Acts directly on ``tool_delta`` (``needs_grad=True``).
        Gated by ``w_step`` (0 -> disabled).
        """
        for t in range(ti.max(t_start, 1), T - 1):
            w = self.w_step[None]
            n = ti.max(1, T - 1 - ti.max(t_start, 1))
            d2_0 = self.tool_delta[t].dot(self.tool_delta[t])
            d2_1 = self.tool_delta[t - 1].dot(self.tool_delta[t - 1])
            diff = d2_0 - d2_1
            ti.atomic_add(self.loss[None], w * diff * diff / n)

    @ti.kernel
    def compute_length_penalty(self, t_start: ti.i32, T: ti.i32):
        """Differentiable PATH-LENGTH (minimal-motion) penalty on ``self.loss``.

        Penalizes the squared per-step displacement summed over all active
        segments:

            len = |tool_delta[t]|^2

        summed over ``t in [t_start, T)`` and normalized by the segment count.
        Agnostic to WHERE the tool is (unlike w_prox / w_traj_prox, which pull
        toward the surface and oppose carving): it only discourages motion. On
        carving steps the residual gradient dominates so motion is preserved;
        on trailing steps with no residual it shrinks the deltas toward zero so
        the tool stops rather than wandering into air. Acts directly on
        ``tool_delta`` (``needs_grad=True``). Gated by ``w_len`` (0 -> disabled).
        """
        for t in range(ti.max(t_start, 0), T):
            w = self.w_len[None]
            n = ti.max(1, T - ti.max(t_start, 0))
            d2 = self.tool_delta[t].dot(self.tool_delta[t])
            ti.atomic_add(self.loss[None], w * d2 / n)

    @ti.kernel
    def compute_traj_prox_penalty(self, t_start: ti.i32, T: ti.i32):
        """Differentiable TRAJECTORY contour-hug penalty on the tool CENTER.

        A GENTLE per-segment penalty (T terms, not T*N^3) on the segment
        midpoint's distance from the TARGET surface, with a deadzone of one
        tool-radius so contact-cutting is FREE:

            d   = target_sdf(seg_midpoint)        # voxels, >0 outside target
            exc = max(0, d - r_tool)               # only beyond tool reach
            loss += w_traj_prox * exc^2 / n_segments

        Corner-carving sits within r_tool of the surface (the base at r_tool
        from a corner voxel is ~0.4*r_tool from the sphere surface), so it is
        NOT charged; only genuine excursions (deep empty corners beyond r_tool,
        high retracts beyond r_tool) are. Unlike the per-voxel w_prox whose
        huge corner gradient stalls carving, this is a soft nudge on the
        trajectory shape. Differentiable in tool_pos/tool_delta (via the
        midpoint and target_sdf). Gated by w_traj_prox (0 -> disabled).
        """
        for t in range(t_start, T):
            w = self.w_traj_prox[None]
            n = ti.max(1, T - t_start)
            r_vox = self.tool_radius[None] / self.v
            # Component-wise midpoint (avoids a Taichi autodiff load-forwarding
            # issue with Vector-field arithmetic feeding target_sdf).
            mx = 0.5 * (self.tool_pos[t].x + self.tool_pos[t + 1].x)
            my = 0.5 * (self.tool_pos[t].y + self.tool_pos[t + 1].y)
            mz = 0.5 * (self.tool_pos[t].z + self.tool_pos[t + 1].z)
            mid = ti.Vector([mx, my, mz])
            d = self.target_sdf_scalar(mid)
            exc = ti.max(0.0, d - r_vox)
            ti.atomic_add(self.loss[None], w * exc * exc / n)

    @ti.kernel
    def compute_tool_gouge_penalty(self, t_start: ti.i32, T: ti.i32):
        """Differentiable TOOL-POSITION gouge barrier (soft-union-independent).

        Charges the TOOL CENTER directly for penetrating the target expanded by
        the tool radius -- the tool capsule gouges the part exactly when
        ``target_sdf(center) < r_tool``:

            d   = target_sdf(seg_midpoint)        # voxels, >0 outside target
            pen = max(0, r_tool - d)              # >0 only when the tool penetrates
            loss += w_tool_gouge * pen^2 / n_segments

        ZERO (zero gradient) whenever the tool is tangent-or-outside the surface
        (``d >= r_tool``): contact-cutting the waste just outside the part is
        FREE, so this never opposes legitimate carving. Unlike the stock-based
        ``w_gouge`` (which charges soft-occupancy and is trivially satisfied by
        soft-union over-erosion), this constrains the trajectory GEOMETRY
        directly and is independent of the soft stock -- so it transfers to the
        hard carve. Differentiable in tool_pos/tool_delta via the midpoint and
        ``target_sdf_scalar``. Gated by ``w_tool_gouge`` (0 -> disabled).
        """
        for t in range(t_start, T):
            w = self.w_tool_gouge[None]
            n = ti.max(1, T - t_start)
            r_vox = self.tool_radius[None] / self.v
            # Margin inflates the no-penetration radius so the tool center must
            # stay `margin` voxels beyond mere tangency (see tool_gouge_margin).
            r_eff = r_vox + self.tool_gouge_margin[None]
            # Component-wise midpoint (avoids the same Taichi autodiff
            # load-forwarding issue as compute_traj_prox_penalty).
            mx = 0.5 * (self.tool_pos[t].x + self.tool_pos[t + 1].x)
            my = 0.5 * (self.tool_pos[t].y + self.tool_pos[t + 1].y)
            mz = 0.5 * (self.tool_pos[t].z + self.tool_pos[t + 1].z)
            mid = ti.Vector([mx, my, mz])
            d = self.target_sdf_scalar(mid)
            pen = ti.max(0.0, r_eff - d)
            ti.atomic_add(self.loss[None], w * pen * pen / n)

    # ========================================================================
    # Trajectory-quality measures: toolpath time, air-cut time, breakage prob.
    # Soft (differentiable) surrogates added to the loss by compute_traj_metrics;
    # hard (non-differentiable) final metrics computed by
    # compute_traj_diagnostics_hard on the hard carve. See docs/algorithms.md.
    # ========================================================================

    @ti.kernel
    def compute_seg_time(self, t_start: ti.i32, T: ti.i32):
        """Per-segment motion time (seconds), differentiable.

        Mirrors ``advance_position``'s feed/rapid regime selection: the move is
        cutting (feed_speed) when the cutter would come within safe_distance of
        the remaining stock at the commanded destination, else traverse
        (rapid_speed). The segment time is the executed length over the regime
        speed:

            seg_time = min(|delta|.L_mm, cap_mm) / speed

        where cap_mm = speed*dt (the speed clip). Below the cap the time scales
        linearly with the commanded length (gradient 1/speed w.r.t. delta); at
        the cap it saturates to dt (zero gradient, the same subgradient pattern
        as advance_position). is_feed is a hard gate (zero gradient), so grad
        flows only through the magnitude -- exactly the existing speed-clip
        behavior. Reads tool_delta/tool_pos/stock[t]; safe under ti.ad.Tape.
        """
        for t in range(t_start, T):
            a = self.tool_pos[t]
            delta = self.tool_delta[t]
            dmm = ti.Vector([delta.x * self.Lx, delta.y * self.Ly, delta.z * self.Lz])
            mag_mm = ti.sqrt(dmm.dot(dmm) + 1e-12)
            probe = a + delta
            stock_d = self.stock_sdf_at(probe, t)
            cube_d = box_sdf(
                self._vox(probe),
                self._vox(ti.Vector([0.5, 0.5, 0.5])),
                self._vox(ti.Vector([0.5, 0.5, 0.5])),
            )
            clearance = ti.max(stock_d, cube_d)
            clearance_mm = clearance * self.v - self.tool_radius[None]
            is_feed = ti.cast(clearance_mm <= self.safe_distance[None], self._precision)
            rapid_step = self.rapid_speed[None] * self.dt[None]
            feed_step = self.feed_speed[None] * self.dt[None]
            cap_mm = rapid_step + (feed_step - rapid_step) * is_feed
            speed = self.rapid_speed[None] + (
                self.feed_speed[None] - self.rapid_speed[None]
            ) * is_feed
            # min(mag, cap)/speed -- length-limited motion time.
            seg_t = ti.min(mag_mm, cap_mm) / (speed + 1e-12)
            self.seg_time[t] = seg_t

    @ti.kernel
    def zero_seg_volumes(self, t_start: ti.i32, T: ti.i32):
        """Zero the per-segment volume accumulators before a pass.

        Single-loop clear (autodiff requires exactly one top-level loop) so
        compute_seg_volumes' atomic_add never accumulates stale values across
        iterations. acc_psum is cleared separately by zero_acc_psum.
        """
        for t in range(t_start, T):
            self.seg_swept[t] = 0.0
            self.seg_air[t] = 0.0
            self.seg_engage[t] = 0.0

    @ti.kernel
    def zero_acc_psum(self):
        """Clear the breakage-probability accumulator (scalar, no loop)."""
        self.acc_psum[None] = 0.0

    @ti.kernel
    def compute_seg_volumes(self, t_start: ti.i32, T: ti.i32):
        """Per-segment swept/air/engaged volumes (unit-cube^3), differentiable.

        Single top-level for-loop (autodiff requires exactly one). For every
        cut t in [t_start, T) and every voxel:
          tool_occ    = sigmoid(-tool_sdf(p, t))      ~1 inside the swept tool
          stock_pre   = sigmoid(stock[t, p])          ~1 inside material pre-cut
          stock_post  = sigmoid(stock[t+1, p])        ~1 inside material post-cut
          seg_swept  += tool_occ                       (swept tool volume)
          seg_air    += tool_occ * (1 - stock_pre)     (swept in EMPTY stock)
          seg_engage += tool_occ * stock_pre           (chip engaged with material)

        Both air and engage use the PRE-cut stock: engage is the material the
        tooth actually meets, and air is swept volume in ALREADY-EMPTY space
        (re-traversal / flying through a void) -- NOT the material this pass
        carves away. The post-cut form (1 - stock_post) counted every just-cut
        voxel as air, so a descent through solid stock read as 100% air and the
        w_air_time gradient could not distinguish productive carving from real
        air-cut motion. Pre-cut matches the sharp diagnostic in
        compute_traj_diagnostics_hard. Differentiable in tool_pos/tool_delta
        (via tool_sdf) and stock (via stock_occ, which carries grad to earlier
        deltas through the evolving stock); safe under ti.ad.Tape.

        The swept/air/engage volumes are integrated over the CUTTING TIP band
        (tool_sdf_tip), not the full tool_height shank. This makes the air
        fraction reflect whether the cutting tip is in material: an engaged
        finishing pass whose shank sits in already-carved empty space no longer
        reads ~80% air. seg_engage is therefore the tip chip load (the force
        model is now driven by tip chip volume, which is the physically correct
        source of cutting force -- the shank does not cut).
        """
        for t, i, j, k in ti.ndrange((t_start, T), self.Nx, self.Ny, self.Nz):
            inv_n = 1.0 / (self.Nx * self.Ny * self.Nz)
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            tool_d = self.tool_sdf_tip(p, t)
            ta = ti.max(-50.0, ti.min(50.0, -tool_d))
            tool_occ = 1.0 / (1.0 + ti.exp(ta))             # ~1 inside swept tool
            sa_pre = ti.max(-50.0, ti.min(50.0, self.stock[t, i, j, k]))
            stock_occ_pre = 1.0 / (1.0 + ti.exp(sa_pre))    # ~1 inside material
            ti.atomic_add(self.seg_swept[t], inv_n * tool_occ)
            ti.atomic_add(self.seg_air[t], inv_n * tool_occ * (1.0 - stock_occ_pre))
            ti.atomic_add(self.seg_engage[t], inv_n * tool_occ * stock_occ_pre)

    @ti.kernel
    def compute_traj_metrics(self, t_start: ti.i32, T: ti.i32):
        """Per-segment soft loss terms + breakage-probability accumulator.

        Single top-level for-loop (autodiff requires exactly one). Per segment:
          air_time_t = seg_time * is_air_t   (seconds; BINARY per segment)
          mu_F[t]    = kc * (seg_engage[t] * v_mm^3) / (dt * D)     (Newtons)
          P_break[t] = sigmoid((ln(mu_F) - ln(f_ref)) / sigma_risk) (§4.1)

        Air is BINARY: a segment is "air" iff the tool is NOT engaged with
        material AT ALL this segment (it does not matter how much of the tool
        is engaged -- any engagement => cutting, air contribution 0). This
        matches the hard metric in compute_traj_diagnostics_hard. For
        autodiff, is_air_t is a narrow-temperature sigmoid step on seg_engage
        (the tip chip load): is_air -> 1 as engage -> 0, is_air -> 0 once
        engage exceeds ~1 voxel of chip load. Grad flows eng -> seg_engage
        (needs_grad) -> tool_delta/stock.
        Atomic-adds the per-segment time and air-time loss contributions as
        means over segments (each term is divided by n = T - t_start, a
        linear combination -> differentiable per segment) and accumulates
        P_break(t) into acc_psum. The trajectory-level breakage loss
        w_break * (1 - exp(-acc_psum)) is added by the separate scalar kernel
        compute_break_loss, because it is a nonlinear function of the
        WHOLE-trajectory sum. Grad flows through seg_* (needs_grad) ->
        tool_delta/stock.
        """
        for t in range(t_start, T):
            n = ti.max(1, T - t_start)
            inv_n = 1.0 / n
            kc = self.kc[None]
            f_ref = self.f_ref[None]
            sigma_risk = ti.max(self.sigma_risk[None], 1e-6)
            dt = self.dt[None]
            D = 2.0 * self.tool_radius[None]            # tool diameter (mm)
            v_mm3 = self.v ** 3                         # voxel volume (mm^3)
            w_t = self.w_time[None]
            w_at = self.w_air_time[None]
            st = self.seg_time[t]
            eng = self.seg_engage[t]
            # BINARY air, smoothed for autodiff: is_air -> 1 as eng -> 0
            # (tool not engaged at all => air), is_air -> 0 once eng exceeds
            # ~1 voxel of tip chip load (any engagement => cutting). It does
            # not matter HOW MUCH of the tool is engaged. inv_vox is the
            # one-voxel chip-load scale (seg_engage is a normalized volume in
            # [0,1]); temp is narrow so this is a differentiable step. Off-grid
            # segments have eng=0 => is_air=1 automatically (no special case).
            inv_vox = 1.0 / (self.Nx * self.Ny * self.Nz)
            temp = ti.max(1e-5, 0.25 * inv_vox)
            # Clamp the pre-activation to [-50, 50] (as the SDF sigmoids above do):
            # temp is tiny, so unclamped (eng-inv_vox)/temp overflows exp in the
            # backward pass -> NaN grad. Clamped regions saturate (grad 0, intended).
            arg = ti.max(-50.0, ti.min(50.0, (eng - inv_vox) / temp))
            is_air = 1.0 / (1.0 + ti.exp(arg))
            # Per-segment time + air-time loss (means over segments, differentiable).
            ti.atomic_add(self.loss[None], w_t * st * inv_n)
            ti.atomic_add(self.loss[None], w_at * st * is_air * inv_n)
            # Late-weighted air penalty (Idea 2): ramp(t) in [0,1], low early /
            # high late. ramp(t)=0 during the warmup fraction (early entry air is
            # free), then rises linearly to 1 at the final segment. Depends only
            # on t (shape-agnostic); a scalar multiplier on the differentiable
            # is_air term, so autodiff is unaffected. w_air_late=0 => term off.
            w_al = self.w_air_late[None]
            if w_al > 0.0:
                span = ti.max(1, (T - 1) - t_start)
                warm = ti.cast(self.w_air_ramp_frac[None] * ti.cast(span, self._precision), ti.i32)
                ramp = ti.cast(ti.max(0, t - t_start - warm), self._precision) / ti.cast(ti.max(1, span - warm), self._precision)
                ramp = ti.max(0.0, ti.min(1.0, ramp))
                ti.atomic_add(self.loss[None], w_al * st * is_air * ramp * inv_n)
            # Per-step breakage probability; accumulated for the traj-level term.
            eng = self.seg_engage[t]
            mu_F = kc * (eng * v_mm3) / (dt * D + 1e-12)
            ln_mu = ti.log(ti.max(mu_F, 1e-12))
            ln_ref = ti.log(ti.max(f_ref, 1e-12))
            p_t = 1.0 / (1.0 + ti.exp(-(ln_mu - ln_ref) / sigma_risk))
            ti.atomic_add(self.acc_psum[None], p_t)

    @ti.kernel
    def compute_break_loss(self):
        """Scalar differentiable breakage loss: w_break * (1 - exp(-acc_psum)).

        No for-loop (a single scalar op), so it is autodiff-compatible. Reads
        acc_psum (needs_grad, accumulated by compute_traj_metrics) and adds the
        docs/algorithms.md §4.2 trajectory-level probability to self.loss. Grad
        flows through acc_psum -> seg_engage -> tool_delta/stock.
        """
        w_b = self.w_break[None]
        p_traj = 1.0 - ti.exp(-self.acc_psum[None])
        ti.atomic_add(self.loss[None], w_b * p_traj)

    @ti.kernel
    def compute_traj_diagnostics_hard(self, T: ti.i32):
        """Hard (non-differentiable) final-metric form of the three measures.

        Recomputes the per-segment volumes from the SHARP carve (tool_sdf_sharp,
        boolean stock occupancy) and writes the diag_* fields using the same
        formulas as compute_traj_metrics, plus the docs/design.md hard
        threshold rule: broken = 1 iff F_cut_max > f_max. Call outside a Tape
        after forward_hard. Reuses the seg_* fields as plain scratch (no grad
        outside a Tape).
        """
        for t in range(T):
            self.seg_swept[t] = 0.0
            self.seg_air[t] = 0.0
            self.seg_engage[t] = 0.0
            self.seg_time[t] = 0.0
        inv_n = 1.0 / (self.Nx * self.Ny * self.Nz)
        # Per-segment swept/air/engage from the sharp carve, over the CUTTING
        # TIP band (tool_sdf_sharp_tip) so the reported air_time_frac reflects
        # cutting-tip contact with material, not shank volume in carved space.
        for t, i, j, k in ti.ndrange(T, self.Nx, self.Ny, self.Nz):
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            tool_d = self.tool_sdf_sharp_tip(p, t)
            tool_occ = 0.0
            # Feed removal uses a CLOSED swept set (score <= 0), matching
            # apply_cut_hard. Exact-tip-tangent cells are engaged, never air.
            if tool_d <= 0.0:
                tool_occ = 1.0
            pre = 0.0
            if self.stock[t, i, j, k] < 0.0:
                pre = 1.0
            post = 0.0
            if self.stock[t + 1, i, j, k] < 0.0:
                post = 1.0
            ti.atomic_add(self.seg_swept[t], inv_n * tool_occ)
            # Air = tool swept volume that is in ALREADY-EMPTY stock (pre-cut),
            # i.e. the tool re-traversing carved space rather than engaging solid
            # material. Using post-cut stock here was a bug: the tool empties every
            # voxel it touches, so post-cut those voxels read as empty and EVERY
            # on-grid segment reported 100% air. pre-cut gives the true engaged vs
            # air split (air = swept - engage).
            ti.atomic_add(self.seg_air[t], inv_n * tool_occ * (1.0 - pre))
            ti.atomic_add(self.seg_engage[t], inv_n * tool_occ * pre)
        # Per-segment time (sharp regime gate, non-diff).
        for t in range(T):
            a = self.tool_pos[t]
            delta = self.tool_delta[t]
            dmm = ti.Vector([delta.x * self.Lx, delta.y * self.Ly, delta.z * self.Lz])
            mag_mm = ti.sqrt(dmm.dot(dmm) + 1e-12)
            probe = a + delta
            stock_d = self.stock_sdf_at(probe, t)
            cube_d = box_sdf(
                self._vox(probe),
                self._vox(ti.Vector([0.5, 0.5, 0.5])),
                self._vox(ti.Vector([0.5, 0.5, 0.5])),
            )
            clearance = ti.max(stock_d, cube_d)
            clearance_mm = clearance * self.v - self.tool_radius[None]
            is_feed = ti.cast(clearance_mm <= self.safe_distance[None], self._precision)
            rapid_step = self.rapid_speed[None] * self.dt[None]
            feed_step = self.feed_speed[None] * self.dt[None]
            cap_mm = rapid_step + (feed_step - rapid_step) * is_feed
            speed = self.rapid_speed[None] + (
                self.feed_speed[None] - self.rapid_speed[None]
            ) * is_feed
            self.seg_time[t] = ti.min(mag_mm, cap_mm) / (speed + 1e-12)
        # Combine into diag fields.
        total_time = 0.0
        air_time = 0.0
        air_early = 0.0
        air_mid = 0.0
        air_late = 0.0
        t1 = T // 3
        t2 = (2 * T) // 3
        psum = 0.0
        pmax = 0.0
        emax = 0.0
        esum = 0.0
        fmax = 0.0
        kc = self.kc[None]
        f_ref = self.f_ref[None]
        sigma_risk = ti.max(self.sigma_risk[None], 1e-6)
        dt = self.dt[None]
        D = 2.0 * self.tool_radius[None]
        v_mm3 = self.v ** 3
        for t in range(T):
            st = self.seg_time[t]
            # BINARY air: a segment is "air" iff the tool is NOT engaged with
            # material AT ALL this segment -- it does not matter how much of
            # the tool is engaged, any engagement => cutting (air contribution
            # 0). seg_engage is the tip chip load (normalized volume in [0,1]);
            # with the sharp 0/1 occupancy above it is an exact multiple of
            # inv_n, so 0.5*inv_n cleanly separates "engaged at least one
            # voxel" from "engages nothing". Off-grid segments (swept hits no
            # voxel) have seg_engage=0 => air automatically -- no special case
            # needed (the old sw<=eps branch is subsumed).
            eng = self.seg_engage[t]
            is_air = ti.select(eng <= 0.5 * inv_n, 1.0, 0.0)
            total_time += st
            air_time += st * is_air
            # Idea 8: accumulate the same per-segment air into early/mid/late
            # thirds of the trajectory (seconds). t1/t2 are the third boundaries.
            if t < t1:
                air_early += st * is_air
            elif t < t2:
                air_mid += st * is_air
            else:
                air_late += st * is_air
            mu_F = kc * (eng * v_mm3) / (dt * D + 1e-12)
            fmax = ti.max(fmax, mu_F)
            emax = ti.max(emax, eng)
            esum += eng
            ln_mu = ti.log(ti.max(mu_F, 1e-12))
            ln_ref = ti.log(ti.max(f_ref, 1e-12))
            p_t = 1.0 / (1.0 + ti.exp(-(ln_mu - ln_ref) / sigma_risk))
            psum += p_t
            pmax = ti.max(pmax, p_t)
        self.diag_time[None] = total_time
        self.diag_air_time[None] = air_time
        self.diag_air_time_early[None] = air_early
        self.diag_air_time_mid[None] = air_mid
        self.diag_air_time_late[None] = air_late
        self.diag_break_prob_any[None] = 1.0 - ti.exp(-psum)
        self.diag_break_prob_max[None] = pmax
        self.diag_fcut_max[None] = fmax
        self.diag_engage_max[None] = emax
        # segment-mean engagement:
        self.diag_engage_mean[None] = esum / ti.max(1, T)
        # docs/design.md hard threshold rule.
        self.diag_broken[None] = 0.0
        if fmax > self.f_max[None]:
            self.diag_broken[None] = 1.0

    @ti.kernel
    def compute_diagnostics(self, T: ti.i32):
        """Non-differentiable breakdown of the loss into its three components.

        Fills diag_gouge / diag_residual / diag_holder with the SAME weighted
        terms that compute_loss + compute_holder_penalty add to ``self.loss``,
        so training can log how the objective (residual) trades off against the
        two barriers (gouge, holder). Reads no grad; call outside a Tape.
        """
        g = 0.0
        r = 0.0
        h = 0.0
        a = 0.0
        au = 0.0
        ts = 0.0
        px = 0.0
        tpx = 0.0
        jk = 0.0
        st = 0.0
        ln = 0.0
        tg = 0.0
        scale = 1.0  # SDFs are already in voxels (1-voxel-wide sigmoid)
        inv_n = 1.0 / (self.Nx * self.Ny * self.Nz)
        w_g = self.w_gouge[None]
        w_r = self.w_residual[None]
        w_h = self.holder_penalty_weight[None]
        w_a = self.w_air[None]
        w_j = self.w_jerk[None]
        w_s = self.w_step[None]
        w_p = self.w_prox[None]
        r_vox = self.tool_radius[None] / self.v
        r_safe = ti.max(r_vox, 1e-3)
        w_tp = self.w_traj_prox[None]
        w_l = self.w_len[None]
        w_tg = self.w_tool_gouge[None]
        margin = self.holder_margin[None]
        # Geometry terms on the final stock (stock[T]).
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            p = ti.Vector([(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz])
            stock_d = self.stock[T, i, j, k]
            target_d = self.target_sdf(p)
            sa = ti.max(-50.0, ti.min(50.0, stock_d * scale))
            ta = ti.max(-50.0, ti.min(50.0, target_d * scale))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))
            target_occ = 1.0 / (1.0 + ti.exp(ta))
            gouge = target_occ * (1.0 - stock_occ)
            residual = (1.0 - target_occ) * stock_occ
            g += inv_n * w_g * gouge * gouge
            r += inv_n * w_r * residual * residual
        # Holder barrier + air-cut penalty summed over every segment.
        for t, i, j, k in ti.ndrange(T, self.Nx, self.Ny, self.Nz):
            p = ti.Vector([(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz])
            stock_d = self.stock[t, i, j, k]  # pre-cut conservative screen
            holder_d = self.holder_sdf(p, t)
            sa = ti.max(-50.0, ti.min(50.0, stock_d * scale))
            stock_occ = 1.0 / (1.0 + ti.exp(sa))
            penetration = ti.max(0.0, (margin - holder_d) * scale)
            h += inv_n * w_h * stock_occ * penetration * penetration
            # Air-cut: tool swept volume in ALREADY-EMPTY stock. Use pre-cut
            # stock[t]: "empty stock" means empty before this segment cut, i.e.
            # the tool re-traversing carved space. (Post-cut stock[t+1] was a bug:
            # the tool empties what it touches, so post-cut read as empty and
            # over-reported air / distorted the sigmoid-blurred fraction.)
            stock_d_pre = self.stock[t, i, j, k]
            sa_pre = ti.max(-50.0, ti.min(50.0, stock_d_pre * scale))
            stock_occ_pre = 1.0 / (1.0 + ti.exp(sa_pre))
            tool_d = self.tool_sdf(p, t)
            ta = ti.max(-50.0, ti.min(50.0, -tool_d * scale))
            tool_occ = 1.0 / (1.0 + ti.exp(ta))
            air = tool_occ * (1.0 - stock_occ_pre)
            a += inv_n * w_a * air * air
            au += inv_n * air
            ts += inv_n * tool_occ
            # Distance-weighted air (contour-hug): same term as the w_prox
            # loss component, weighted by squared distance from target surface.
            d_t = ti.max(0.0, self.target[i, j, k])
            w_dist = (d_t / r_safe) ** 2
            px += inv_n * w_p * air * air * w_dist
        # Jerk / smoothness over consecutive deltas.
        nj = ti.max(1, T - 2)
        for t in range(1, T - 1):
            diff = self.tool_delta[t] - self.tool_delta[t - 1]
            jk += w_j * diff.dot(diff) / nj
        # Speed regularity over consecutive step lengths.
        ns = ti.max(1, T - 2)
        for t in range(1, T - 1):
            d2_0 = self.tool_delta[t].dot(self.tool_delta[t])
            d2_1 = self.tool_delta[t - 1].dot(self.tool_delta[t - 1])
            sp = d2_0 - d2_1
            st += w_s * sp * sp / ns
        # Trajectory contour-hug (tool-center distance from target surface) +
        # tool-position gouge (tool center penetrating target + r_tool). Both
        # read the same segment midpoint + target_sdf, so they share this loop.
        ntp = ti.max(1, T)
        for t in range(T):
            mx = 0.5 * (self.tool_pos[t].x + self.tool_pos[t + 1].x)
            my = 0.5 * (self.tool_pos[t].y + self.tool_pos[t + 1].y)
            mz = 0.5 * (self.tool_pos[t].z + self.tool_pos[t + 1].z)
            mid = ti.Vector([mx, my, mz])
            d = self.target_sdf_scalar(mid)
            exc = ti.max(0.0, d - r_vox)
            tpx += w_tp * exc * exc / ntp
            pen = ti.max(0.0, r_vox + self.tool_gouge_margin[None] - d)
            tg += w_tg * pen * pen / ntp
        # Path-length (minimal-motion) over all active segments.
        nl = ti.max(1, T)
        for t in range(T):
            d2 = self.tool_delta[t].dot(self.tool_delta[t])
            ln += w_l * d2 / nl
        self.diag_gouge[None] = g
        self.diag_residual[None] = r
        self.diag_holder[None] = h
        self.diag_air[None] = a
        self.diag_jerk[None] = jk
        self.diag_step[None] = st
        self.diag_air_unweighted[None] = au
        self.diag_tool_swept[None] = ts
        self.diag_prox[None] = px
        self.diag_traj_prox[None] = tpx
        self.diag_len[None] = ln
        self.diag_tool_gouge[None] = tg

    @ti.kernel
    def holder_overlap_at(self, t: ti.i32) -> float:
        """Hard (non-diff) holder/stock overlap VOLUME for a single segment t.

        Counts **pre-segment** remaining-material voxels (``stock[t] < 0``)
        inside the whole-segment holder interval (sharp holder score < 0) and
        returns their voxel-centre fraction of the grid (legacy unit-cube^3).
        This is a conservative screen: positive values are not a proof of
        chronological collision and can include legitimate cutter-leading
        motion. For physical mm^3 contact estimates use holder_overlap_mm3_at.
        Exact time-resolved clearance belongs to the independent evaluator.
        """
        vol = 0.0
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            if self.stock[t, i, j, k] < 0.0 and self.holder_sdf_sharp(p, t) < 0.0:
                vol += 1.0 / (self.Nx * self.Ny * self.Nz)
        return vol

    @ti.kernel
    def holder_min_clearance_at(self, t: ti.i32) -> float:
        """Min holder-to-stock CLEARANCE over segment t (voxels; positive = gap).

        For every **pre-segment** remaining-material voxel (stock[t] < 0)
        takes the min of the whole-segment holder score: positive means this
        conservative pre-screen clears that voxel, negative means possible
        penetration. It is not a certified chronological minimum clearance and
        cutter-leading safe paths can be conservatively negative. No-material
        input returns the legacy 1e9 sentinel, not a physical measured clearance.
        Non-differentiable; safe outside a Tape. Use holder_min_clearance_mm_at
        for physical units.

        Uses a global scratch field (``_clearance_buf``) with ``ti.atomic_min``
        because a local-scalar min races under the parallel ``ti.ndrange`` loop.
        """
        self._clearance_buf[None] = 1e9
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.stock[t, i, j, k] < 0.0:
                d = self.holder_sdf_sharp(
                    ti.Vector(
                        [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
                    ),
                    t,
                )
                ti.atomic_min(self._clearance_buf[None], d)
        return self._clearance_buf[None]

    @ti.kernel
    def holder_overlap_total(self, T: ti.i32) -> float:
        """Legacy unit-cube^3 conservative pre-cut overlap summed over cuts.

        This can overestimate collisions because every pre-segment stock is
        evaluated against a whole-segment holder interval. It is not a unique
        chronological collision volume and repeated contact is deliberately
        accumulated by segment. Non-differentiable.
        """
        vol = 0.0
        for t, i, j, k in ti.ndrange(T, self.Nx, self.Ny, self.Nz):
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            if self.stock[t, i, j, k] < 0.0 and self.holder_sdf_sharp(p, t) < 0.0:
                vol += 1.0 / (self.Nx * self.Ny * self.Nz)
        return vol

    @ti.kernel
    def holder_overlap_mm3_at(self, t: ti.i32) -> float:
        """Physical-mm^3 voxel-centre volume of the pre-cut holder screen."""
        volume = 0.0
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            if self.stock[t, i, j, k] < 0.0 and self.holder_sdf_sharp(p, t) < 0.0:
                volume += self.v ** 3
        return volume

    @ti.kernel
    def holder_overlap_total_mm3(self, T: ti.i32) -> float:
        """Per-segment sum of conservative pre-cut holder overlap in mm^3.

        Not unique collision volume and not guaranteed chronological collision.
        """
        volume = 0.0
        for t, i, j, k in ti.ndrange(T, self.Nx, self.Ny, self.Nz):
            p = ti.Vector(
                [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
            )
            if self.stock[t, i, j, k] < 0.0 and self.holder_sdf_sharp(p, t) < 0.0:
                volume += self.v ** 3
        return volume

    @ti.kernel
    def holder_min_clearance_mm_at(self, t: ti.i32) -> float:
        """Conservative pre-cut holder clearance estimate in physical mm.

        Negative = possible penetration (not proof of contact); positive is
        interval-screen clearance, not a time-resolved certificate. If no stock
        remains before this segment, returns 1e9*v as the existing sentinel.
        """
        self._clearance_buf[None] = 1e9
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            if self.stock[t, i, j, k] < 0.0:
                p = ti.Vector(
                    [(i + 0.5) / self.Nx, (j + 0.5) / self.Ny, (k + 0.5) / self.Nz]
                )
                ti.atomic_min(self._clearance_buf[None], self.holder_sdf_sharp(p, t))
        return self._clearance_buf[None] * self.v

    def forward(self, num_active_steps):
        """Pure forward pass. Wrap in ti.ad.Tape externally if you need gradients.

        num_active_steps is the number of tool positions in use; with
        num_active_steps positions there are num_active_steps-1 segments/cuts.

        Position reconstruction is INTERLEAVED with carving: each step is clipped
        to its feed/rapid speed limit by ``advance_position`` using the remaining
        stock at the start of that step, then the cut is applied. This coupling
        (regime depends on the evolving stock) is why we step rather than
        reconstruct the whole path up front. The whole loop must run inside the
        same Tape as the loss for ``tool_delta.grad`` to be correct.
        """
        self.reconstruct_positions(0)  # set tool_pos[0] = tool_start
        self.init_stock()
        for t in range(num_active_steps - 1):
            self.advance_position(t)
            self.apply_cut(t)
        self.compute_loss(num_active_steps - 1)
        self.compute_holder_penalty(0, num_active_steps - 1)
        self.compute_air_penalty(0, num_active_steps - 1)
        self.compute_jerk_penalty(0, num_active_steps - 1)
        self.compute_step_penalty(0, num_active_steps - 1)
        self.compute_traj_prox_penalty(0, num_active_steps - 1)
        self.compute_length_penalty(0, num_active_steps - 1)
        self.compute_tool_gouge_penalty(0, num_active_steps - 1)
        # Trajectory-quality measures (time / air-cut time / breakage). These
        # run inside the Tape so their soft loss terms receive gradient. When
        # all three soft weights are zero the kernels are skipped: they would
        # add 0 to the loss but still build a large autodiff graph (atomic_adds
        # to needs_grad seg_*/acc_psum fields over all voxels x T), which both
        # slows training and corrupts tool_delta.grad. The hard, non-diff
        # final-metric forms (diag_time/diag_air_time/diag_break_prob_*) are
        # computed by compute_traj_diagnostics_hard in eval_metrics, outside
        # the Tape, so the reported metrics are unaffected by this guard.
        w_any = (self.w_time[None] > 0.0
                 or self.w_air_time[None] > 0.0
                 or self.w_air_late[None] > 0.0
                 or self.w_break[None] > 0.0)
        if w_any:
            self.zero_seg_volumes(0, num_active_steps - 1)
            self.zero_acc_psum()
            self.compute_seg_time(0, num_active_steps - 1)
            self.compute_seg_volumes(0, num_active_steps - 1)
            self.compute_traj_metrics(0, num_active_steps - 1)
            self.compute_break_loss()

    def forward_hard(self, num_active_steps, clip_speeds=True):
        """Hard boolean forward pass for evaluation and rendering.

        Position advancement follows forward() when clip_speeds is True, but carving
        uses hard apply_cut_hard (ti.max union with the interval-restricted
        tool_sdf_sharp membership score) instead of smooth_max. The score's
        sign is exact at voxel centres; its outside magnitude is not an
        exact Euclidean distance to the swept cutter.
        """
        self.reconstruct_positions(0)
        self.init_stock()
        if clip_speeds:
            for t in range(num_active_steps - 1):
                self.advance_position(t)
                self.apply_cut_hard(t)
        else:
            self.reconstruct_positions(num_active_steps - 1)
            for t in range(num_active_steps - 1):
                self.apply_cut_hard(t)

    def forward_from(self, t0, num_active_steps):
        """Forward pass RESTARTED from a restored mid-cut state at step ``t0``.

        Unlike ``forward``, this does NOT call ``init_stock`` or
        ``reconstruct_positions(0)``: the caller must have already populated
        ``stock[t0]`` and ``tool_pos[t0]`` (via ``restore_state``). It then
        reconstructs positions from ``t0``, carves ``[t0, num_active_steps-1)``,
        and computes the loss + all barriers/penalties restricted to
        ``[t0, num_active_steps-1)`` so only the segments actually carved this
        pass carry gradient (the restored prefix is a detached constant).

        Wrap in ti.ad.Tape externally. ``num_active_steps`` is the total number
        of tool positions (the loss is still measured on the final stock at
        ``num_active_steps-1``), so the tail segment count is
        ``num_active_steps-1 - t0``.
        """
        self.reconstruct_positions_from(t0, num_active_steps)
        for t in range(t0, num_active_steps - 1):
            self.advance_position(t)
            self.apply_cut(t)
        self.compute_loss(num_active_steps - 1)
        self.compute_holder_penalty(t0, num_active_steps - 1)
        self.compute_air_penalty(t0, num_active_steps - 1)
        self.compute_jerk_penalty(t0, num_active_steps - 1)
        self.compute_step_penalty(t0, num_active_steps - 1)
        self.compute_traj_prox_penalty(t0, num_active_steps - 1)
        self.compute_length_penalty(t0, num_active_steps - 1)
        self.compute_tool_gouge_penalty(t0, num_active_steps - 1)
        # Trajectory-quality measures restricted to the carved tail [t0, T).
        self.zero_seg_volumes(t0, num_active_steps - 1)
        self.zero_acc_psum()
        self.compute_seg_time(t0, num_active_steps - 1)
        self.compute_seg_volumes(t0, num_active_steps - 1)
        self.compute_traj_metrics(t0, num_active_steps - 1)
        self.compute_break_loss()

    # ------------------------------------------------------------------
    # State save / restore (for restart-from-state training)
    # ------------------------------------------------------------------
    # The bank holds snapshots of (stock SDF at step t, tool_pos at step t, t).
    # These are detached numpy copies: restoring them writes CONSTANTS back into
    # the field slots, so a forward_from(t0) pass only accumulates gradient on
    # tool_delta[t0..T-2] (the prefix is fixed). Nondeterministic carve under
    # GPU atomics is irrelevant here -- we only ever READ stock[t]/tool_pos[t]
    # to snapshot, and WRITE them as fixed starts.

    def save_state(self, t):
        """Snapshot the stock SDF + tool position at step ``t`` (numpy copies)."""
        return {
            "stock": self.stock.to_numpy()[t].copy(),
            "tool_pos": self.tool_pos.to_numpy()[t].copy(),
            "t": int(t),
        }

    def restore_state(self, state, t0):
        """Write a saved snapshot back into slots ``t0`` (stock + tool_pos).

        ``t0`` may differ from ``state["t"]``; we place the snapshot at whichever
        slot the caller wants to restart from. Caller then runs
        ``forward_from(t0, T)``.
        """
        stock_np = self.stock.to_numpy()
        pos_np = self.tool_pos.to_numpy()
        stock_np[t0] = state["stock"]
        pos_np[t0] = state["tool_pos"]
        self.stock.from_numpy(stock_np)
        self.tool_pos.from_numpy(pos_np)

    def set_tool_start(self, point):
        """Configure a stable fresh-candidate start (normalized stock frame).

        Use this instead of assigning tool_start directly when changing the
        baseline. load_saved_init may subsequently choose a mid-cut start for
        staged training; reset_candidate_state restores this saved baseline.
        """
        values = tuple(float(value) for value in point)
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError("tool_start must contain three finite normalized coordinates")
        self._configured_tool_start = values
        self.tool_start[None] = ti.Vector(list(values))

    def load_saved_init(self, stock_sdf, tool_pos):
        """Configure staged-training: start each ``init_stock`` from a saved
        mid-cut SDF (the carved stock left by a previous trajectory) and set
        the tool start to the saved tool position. After this, a FRESH
        ``forward(T)`` carves the remaining material (the residual the previous
        trajectory left) starting from the mid-cut state.

        ``stock_sdf``: (Nx, Ny, Nz) float array, the saved stock SDF.
        ``tool_pos``: (3,) float array, the saved tool position in [0,1]^3.
        """
        self.saved_stock.from_numpy(np.asarray(stock_sdf, dtype=np.float32))
        self.use_saved_init[None] = 1
        # Prefix-only start; keep _configured_tool_start untouched so reset
        # cannot leak a staged, mid-cut candidate into a fresh evaluation.
        self.tool_start[None] = ti.Vector(
            [float(tool_pos[0]), float(tool_pos[1]), float(tool_pos[2])]
        )

    @ti.kernel
    def reset_candidate_state(self):
        """Clear all per-candidate mutable evaluation state (T3.0 hook).

        Preserves configured target/tool/holder geometry, k, speeds/limits,
        target grids, and allocated Taichi fields. Restores the saved
        configured tool_start (not a load_saved_init prefix). Clears stock
        history/prefixes and their grads, positions/deltas/grads, loss/grad,
        seg_* values+grads, acc_psum+grad, diagnostic accumulators/scratch, and
        saved_stock/use_saved_init. Does NOT call ti.init or change parameters.

        After this call stock is deliberately zero, NOT a freshly initialized
        box. Supply candidate deltas and call forward()/forward_hard(), which
        reconstructs positions and initializes stock before reading it.
        """
        self.loss[None] = 0.0
        self.loss.grad[None] = 0.0
        self.tool_start[None] = ti.Vector(self._configured_tool_start)
        self.acc_psum[None] = 0.0
        self.acc_psum.grad[None] = 0.0
        self._clearance_buf[None] = 0.0
        self.use_saved_init[None] = 0
        for t in range(self.max_steps):
            self.tool_delta[t] = ti.Vector([0.0, 0.0, 0.0])
            self.tool_delta.grad[t] = ti.Vector([0.0, 0.0, 0.0])
            self.seg_time[t] = 0.0
            self.seg_air[t] = 0.0
            self.seg_swept[t] = 0.0
            self.seg_engage[t] = 0.0
            self.seg_time.grad[t] = 0.0
            self.seg_air.grad[t] = 0.0
            self.seg_swept.grad[t] = 0.0
            self.seg_engage.grad[t] = 0.0
        for t in range(self.max_steps + 1):
            self.tool_pos[t] = ti.Vector([0.0, 0.0, 0.0])
            self.tool_pos.grad[t] = ti.Vector([0.0, 0.0, 0.0])
        for t, i, j, k in ti.ndrange(
            self.max_steps + 1, self.Nx, self.Ny, self.Nz
        ):
            self.stock[t, i, j, k] = 0.0
            self.stock.grad[t, i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.Nx, self.Ny, self.Nz):
            self.saved_stock[i, j, k] = 0.0

        self.diag_gouge[None] = 0.0
        self.diag_residual[None] = 0.0
        self.diag_holder[None] = 0.0
        self.diag_air[None] = 0.0
        self.diag_jerk[None] = 0.0
        self.diag_step[None] = 0.0
        self.diag_air_unweighted[None] = 0.0
        self.diag_tool_swept[None] = 0.0
        self.diag_prox[None] = 0.0
        self.diag_traj_prox[None] = 0.0
        self.diag_len[None] = 0.0
        self.diag_tool_gouge[None] = 0.0
        self.diag_time[None] = 0.0
        self.diag_air_time[None] = 0.0
        self.diag_break_prob_any[None] = 0.0
        self.diag_break_prob_max[None] = 0.0
        self.diag_fcut_max[None] = 0.0
        self.diag_broken[None] = 0.0
        self.diag_engage_max[None] = 0.0
        self.diag_engage_mean[None] = 0.0
        self.diag_air_time_early[None] = 0.0
        self.diag_air_time_mid[None] = 0.0
        self.diag_air_time_late[None] = 0.0

    # ========================================================================
    # Rendering
    # ========================================================================

    @ti.func
    def tool_sdf_sharp(self, p, t):
        """HARD continuous-sweep membership score (voxel units), <= 0 inside.

        At each query Z, ``_tool_sweep_terms`` intersects the cutter's vertical
        admissibility interval with [0,1] before computing the XY closest point.
        ``max(horizontal, vertical) <= 0`` is therefore an exact membership
        test for the swept upright flat cylinder, including pure Z, simultaneous
        XYZ, zero-length motion and both endpoint poses. The returned score is
        NOT an exact Euclidean SDF to the swept solid; do not interpret its
        outside magnitude as that distance. Not differentiable: use tool_sdf
        under Taichi Tape. Float32 boundaries use the tolerances in the tests.
        """
        d_xy, d_z = self._tool_sweep_terms(p, t, self.tool_height[None] / self.v, False)
        return ti.max(d_xy, d_z)

    @ti.func
    def tool_sdf_sharp_tip(self, p, t):
        """HARD continuous-sweep membership score for the CUTTING TIP band.

        Same signed membership test as ``tool_sdf_sharp`` with height equal to
        ``_tool_cut_h_vox`` and the band anchored to the tool base. Used by
        ``compute_traj_diagnostics_hard`` so air-time reflects cutting-tip
        contact rather than shank exposure. This is not a Euclidean swept SDF.
        Not differentiable -- never call under ti.ad.Tape.
        """
        d_xy, d_z = self._tool_sweep_terms(p, t, self._tool_cut_h_vox(), False)
        return ti.max(d_xy, d_z)

    @ti.func
    def holder_sdf_sharp(self, p, t):
        """Hard cell-inflated holder-membership for the pre-cut screen.

        Subtract sqrt(3)/2 voxels (half a cubic-cell diagonal) from the final
        membership/clearance score. Material occupies cells, not centres only;
        this mirrors the reference's lower-bound geometry instead of missing a
        holder that almost fills a cell while its centre is outside it. There
        is no +1e-8 sqrt bias on this hard path. The separate smooth CSG penalty
        retains its own derivative regularization and is intentionally not the
        hard cell-inflated measure. Whole-segment conservatism is not a
        chronological-safety proof: safe cutter-leading paths can still flag.
        Not differentiable -- never call under ti.ad.Tape.
        """
        r = self.holder_radius[None] / self.v        # voxels
        h = self.holder_height[None] / self.v        # voxels
        tool_h = self.tool_height[None] / self.v     # voxels

        pv = self._vox(p)
        a = self._vox(self.tool_pos[t])
        b = self._vox(self.tool_pos[t + 1])

        pa_xy = ti.Vector([pv.x - a.x, pv.y - a.y])
        ba_xy = ti.Vector([b.x - a.x, b.y - a.y])
        ba_len2 = ba_xy.dot(ba_xy) + 1e-12
        h_param = ti.max(0.0, ti.min(1.0, pa_xy.dot(ba_xy) / ba_len2))
        closest_xy = ti.Vector([a.x, a.y]) + ba_xy * h_param
        d_xy = ti.sqrt((pv.x - closest_xy.x) ** 2 + (pv.y - closest_xy.y) ** 2) - r

        # Holder Z extent UNIONED over the swept segment. The holder bottom
        # tracks the tool base (z_base + tool_h) from a to b; the body extends
        # up by h. Evaluating Z at a single h_param (the XY-closest point) misses
        # the sweep on near-vertical segments -- the base can plunge far below
        # the evaluated point, dropping the holder into material the SDF reads
        # as clear. The unioned range [min(bottom_a, bottom_b),
        # max(bottom_a, bottom_b) + h] captures the full swept Z extent (a
        # superset of the true swept volume, used for the conservative screen,
        # not a chronological-safety proof). With the default holder_height =
        # full machine Z (>> stock), the top is always above the grid and the
        # lowest bottom is the binding constraint.
        z_bottom_a = a.z + tool_h
        z_bottom_b = b.z + tool_h
        z_low = ti.min(z_bottom_a, z_bottom_b)
        z_high = ti.max(z_bottom_a, z_bottom_b) + h
        z_center = 0.5 * (z_low + z_high)
        z_half = 0.5 * (z_high - z_low)
        d_z = ti.sqrt((pv.z - z_center) ** 2) - z_half

        d_xy_pos = ti.max(d_xy, 0.0)
        d_z_pos = ti.max(d_z, 0.0)
        outside = ti.sqrt(d_xy_pos * d_xy_pos + d_z_pos * d_z_pos)
        inside = -ti.max(-ti.max(d_xy, d_z), 0.0)
        raw_score = outside + inside
        # Cell half-diagonal, not a free user tuning margin. All hard overlap
        # and clearance consumers use this inflated score, in voxel units.
        return raw_score - ti.sqrt(3.0) * 0.5
