# SPDX-License-Identifier: MIT
"""Clean-room milling feeds, speeds, chip-load, and physical constraint solver.

Mathematical model reconstructed cleanly from standard machining mechanics and
the static mathematical specification in ``HSMAdvisor_chip_load_algorithm.md``:
  1. Clamped piecewise-linear endmill diameter reference chip-load lookup ``Q(D)``.
  2. Workpiece material, tool substrate/coating, helix angle, and slotting
     engagement interpolation ``slot_factor(a_e, D)``.
  3. Radial chip thinning ``R(a_e, D_eff)``, corner-radius axial chip thinning
     ``A(a_p, r_c)``, and DOC/WOC engagement-area load scaling ``C_load(a_p, a_e)``.
  4. Exact closed-form constrained feed/speed clamping across spindle RPM bounds,
     machine maximum feed, spindle power ``K_p * MRR``, machine/tool shaft torque,
     resultant cutting force, and cantilever tool deflection ``F * L^3 / (3 E I)``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

INCH_MM: float = 25.4
HP_TO_KW: float = 0.745699872
LBF_IN_TO_NM: float = 0.112984829


# Reference solid-endmill diameter (in) -> reference IPT (in/tooth) curve.
# Anchored at standard fractional endmill sizes including (0.25", 0.00130)
# and (0.50", 0.00285).
ENDMILL_DIAMETER_IPT_TABLE_IN: Tuple[Tuple[float, float], ...] = (
    (0.03125, 0.00015),
    (0.06250, 0.00032),
    (0.09375, 0.00048),
    (0.12500, 0.00065),
    (0.18750, 0.00097),
    (0.25000, 0.00130),
    (0.31250, 0.00168),
    (0.37500, 0.00206),
    (0.50000, 0.00285),
    (0.62500, 0.00340),
    (0.75000, 0.00390),
    (1.00000, 0.00460),
)


@dataclass(frozen=True)
class StockMaterialSpec:
    """Physical and machinability properties of a workpiece stock alloy/polymer."""

    id: str
    label: str
    category: str
    hardness_hb: float
    kp_hp_min_in3: float
    sfm_carbide: float
    ipt_hss_in: float
    ipt_carbide_in: float
    material_ipt_reduction: float
    max_ipt_in: float
    ideal_woc_frac: float
    ideal_doc_frac: float


@dataclass(frozen=True)
class ToolMaterialSpec:
    """Physical properties and machinability multipliers of a cutting tool material."""

    id: str
    label: str
    is_carbide: bool
    sfm_mult: float
    ipt_mult: float
    coat_ipt_mult: float
    youngs_gpa: float
    yield_ksi: float
    dev_max_ratio: float


@dataclass(frozen=True)
class CutterGeometrySpec:
    """Known geometric parameters of the milling cutter assembly."""

    diameter_mm: float = 5.0
    flutes: int = 3
    flute_length_mm: float = 12.0
    stickout_mm: float = 14.0
    shank_diameter_mm: float = 5.0
    corner_radius_mm: float = 0.0
    helix_deg: float = 35.0
    slot_ipt_factor: float = 0.80
    tool_type_ipt: float = 1.0
    max_rpm: float = 24000.0
    max_deflection_pct: float = 100.0
    max_torque_pct: float = 100.0


@dataclass(frozen=True)
class MachineEnvelopeSpec:
    """Spindle and axis physical limits (default: Haas Mini Mill NGC class)."""

    min_rpm: float = 100.0
    max_rpm: float = 8000.0
    max_feed_mm_s: float = 50.0
    rapid_xy_mm_s: float = 120.0
    rapid_z_mm_s: float = 80.0
    spindle_power_kw: float = 5.6
    max_machine_torque_nm: float = 23.0
    max_cutting_force_n: float = 1200.0


@dataclass(frozen=True)
class FeedsSpeedsResult:
    """Complete solved speeds, feeds, chip-load, and active physical limits."""

    stock_material_id: str
    tool_material_id: str
    aggressiveness: float
    diameter_mm: float
    flutes: int
    doc_mm: float
    woc_mm: float
    spindle_rpm: float
    surface_speed_sfm: float
    surface_speed_m_min: float
    fz_base_mm: float
    fz_programmed_mm: float
    chip_thickness_mm: float
    radial_thinning_factor: float
    axial_thinning_factor: float
    slot_factor: float
    engagement_load_factor: float
    feed_mm_s: float
    feed_mm_min: float
    mrr_mm3_s: float
    spindle_power_kw: float
    power_utilization_pct: float
    torque_nm: float
    cutting_force_n: float
    deflection_mm: float
    max_deflection_mm: float
    deflection_utilization_pct: float
    constraint_feed_scale: float
    active_limit: str


STOCK_MATERIALS: Dict[str, StockMaterialSpec] = {
    "al_6061_t6": StockMaterialSpec(
        id="al_6061_t6",
        label="Aluminum 6061-T6",
        category="Non-Ferrous Alloy",
        hardness_hb=95.0,
        kp_hp_min_in3=0.25,
        sfm_carbide=850.0,
        ipt_hss_in=0.00320,
        ipt_carbide_in=0.00360,
        material_ipt_reduction=0.15,
        max_ipt_in=0.00650,
        ideal_woc_frac=0.35,
        ideal_doc_frac=0.65,
    ),
    "al_7075_t6": StockMaterialSpec(
        id="al_7075_t6",
        label="Aluminum 7075-T6",
        category="Non-Ferrous Alloy",
        hardness_hb=150.0,
        kp_hp_min_in3=0.30,
        sfm_carbide=700.0,
        ipt_hss_in=0.00290,
        ipt_carbide_in=0.00330,
        material_ipt_reduction=0.20,
        max_ipt_in=0.00600,
        ideal_woc_frac=0.30,
        ideal_doc_frac=0.65,
    ),
    "brass_c360": StockMaterialSpec(
        id="brass_c360",
        label="Brass C360 (Free-Machining)",
        category="Copper Alloy",
        hardness_hb=100.0,
        kp_hp_min_in3=0.32,
        sfm_carbide=600.0,
        ipt_hss_in=0.00300,
        ipt_carbide_in=0.00340,
        material_ipt_reduction=0.18,
        max_ipt_in=0.00620,
        ideal_woc_frac=0.35,
        ideal_doc_frac=0.65,
    ),
    "delrin_pom": StockMaterialSpec(
        id="delrin_pom",
        label="Acetal / Delrin (POM)",
        category="Engineering Polymer",
        hardness_hb=40.0,
        kp_hp_min_in3=0.08,
        sfm_carbide=1000.0,
        ipt_hss_in=0.00390,
        ipt_carbide_in=0.00440,
        material_ipt_reduction=0.10,
        max_ipt_in=0.00800,
        ideal_woc_frac=0.45,
        ideal_doc_frac=0.80,
    ),
    "steel_1018": StockMaterialSpec(
        id="steel_1018",
        label="Mild Steel 1018 / A36",
        category="Carbon Steel",
        hardness_hb=135.0,
        kp_hp_min_in3=0.85,
        sfm_carbide=380.0,
        ipt_hss_in=0.00240,
        ipt_carbide_in=0.00276,
        material_ipt_reduction=0.35,
        max_ipt_in=0.00500,
        ideal_woc_frac=0.20,
        ideal_doc_frac=0.60,
    ),
    "steel_4140": StockMaterialSpec(
        id="steel_4140",
        label="Alloy Steel 4140 (28 HRC)",
        category="Alloy Steel",
        hardness_hb=280.0,
        kp_hp_min_in3=1.15,
        sfm_carbide=250.0,
        ipt_hss_in=0.00170,
        ipt_carbide_in=0.00210,
        material_ipt_reduction=0.45,
        max_ipt_in=0.00400,
        ideal_woc_frac=0.12,
        ideal_doc_frac=0.55,
    ),
    "ss_304": StockMaterialSpec(
        id="ss_304",
        label="Stainless Steel 304",
        category="Austenitic Stainless",
        hardness_hb=190.0,
        kp_hp_min_in3=1.25,
        sfm_carbide=220.0,
        ipt_hss_in=0.00180,
        ipt_carbide_in=0.00220,
        material_ipt_reduction=0.50,
        max_ipt_in=0.00400,
        ideal_woc_frac=0.12,
        ideal_doc_frac=0.55,
    ),
    "ti_6al4v": StockMaterialSpec(
        id="ti_6al4v",
        label="Titanium Ti-6Al-4V",
        category="Titanium Alloy",
        hardness_hb=330.0,
        kp_hp_min_in3=1.35,
        sfm_carbide=150.0,
        ipt_hss_in=0.00140,
        ipt_carbide_in=0.00180,
        material_ipt_reduction=0.55,
        max_ipt_in=0.00320,
        ideal_woc_frac=0.10,
        ideal_doc_frac=0.50,
    ),
}


TOOL_MATERIALS: Dict[str, ToolMaterialSpec] = {
    "carbide_coated": ToolMaterialSpec(
        id="carbide_coated",
        label="Coated Carbide (TiAlN / ZrN)",
        is_carbide=True,
        sfm_mult=1.18,
        ipt_mult=1.05,
        coat_ipt_mult=1.05,
        youngs_gpa=600.0,
        yield_ksi=85.0,
        dev_max_ratio=0.0012,
    ),
    "carbide_uncoated": ToolMaterialSpec(
        id="carbide_uncoated",
        label="Solid Carbide (Uncoated)",
        is_carbide=True,
        sfm_mult=1.00,
        ipt_mult=1.00,
        coat_ipt_mult=1.00,
        youngs_gpa=600.0,
        yield_ksi=85.0,
        dev_max_ratio=0.0012,
    ),
    "cobalt_m42": ToolMaterialSpec(
        id="cobalt_m42",
        label="Cobalt HSS (M42 8% Co)",
        is_carbide=False,
        sfm_mult=0.42,
        ipt_mult=0.82,
        coat_ipt_mult=1.00,
        youngs_gpa=215.0,
        yield_ksi=60.0,
        dev_max_ratio=0.0018,
    ),
    "hss_m2": ToolMaterialSpec(
        id="hss_m2",
        label="High-Speed Steel (HSS M2)",
        is_carbide=False,
        sfm_mult=0.30,
        ipt_mult=0.72,
        coat_ipt_mult=1.00,
        youngs_gpa=205.0,
        yield_ksi=52.0,
        dev_max_ratio=0.0020,
    ),
}


def lerp_clamped(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
    """Linearly interpolate between ``(x1, y1)`` and ``(x2, y2)`` with endpoint clamping."""
    if abs(x2 - x1) <= 1e-15:
        return float(y1)
    t = (x - x1) / (x2 - x1)
    if t <= 0.0:
        t = 0.0
    elif t >= 1.0:
        t = 1.0
    return float(y1 + t * (y2 - y1))


def reduce_ipt_factor(tool_ipt_factor: float, material_reduction: float) -> float:
    """Blend ``tool_ipt_factor`` toward ``1.0`` by ``material_reduction``."""
    return float(1.0 + (1.0 - material_reduction) * (tool_ipt_factor - 1.0))


def lookup_endmill_reference_ipt_in(diameter_in: float) -> float:
    """Clamped piecewise-linear lookup ``Q(endmill, D_in)`` in inches/tooth."""
    d = float(diameter_in)
    table = ENDMILL_DIAMETER_IPT_TABLE_IN
    if d <= table[0][0]:
        return float(table[0][1])
    if d >= table[-1][0]:
        return float(table[-1][1])
    for i in range(len(table) - 1):
        d0, q0 = table[i]
        d1, q1 = table[i + 1]
        if d0 <= d <= d1:
            return lerp_clamped(d0, d1, q0, q1, d)
    return float(table[-1][1])


def radial_chip_thinning_multiplier(woc_mm: float, eff_diameter_mm: float) -> float:
    """Programmed feed boost ``R >= 1`` compensating radial chip thinning when ``2 a_e < D``.

    Note: Actual chip thickness satisfies ``h_ex = f_z_prog / R``.
    """
    if woc_mm <= 0.0 or eff_diameter_mm <= 1e-9 or 2.0 * woc_mm >= eff_diameter_mm:
        return 1.0
    u = 1.0 - (2.0 * woc_mm) / eff_diameter_mm
    denom = math.sqrt(max(1.0 - u * u, 1e-9))
    return float(min(3.0, 1.0 / denom))


def axial_chip_thinning_multiplier(doc_mm: float, corner_radius_mm: float) -> float:
    """Programmed feed boost ``A >= 1`` for bull-nose / ball cutters when ``a_p < r_c``."""
    if corner_radius_mm <= 1e-6 or doc_mm <= 0.0 or doc_mm >= corner_radius_mm:
        return 1.0
    u = 1.0 - doc_mm / corner_radius_mm
    denom = math.sqrt(max(1.0 - u * u, 1e-9))
    return float(min(10.0, 1.0 / denom))


def compute_nominal_base_ipt_in(
    stock: StockMaterialSpec,
    tool_mat: ToolMaterialSpec,
    cutter: CutterGeometrySpec,
    woc_mm: float,
) -> Tuple[float, float]:
    """Compute ``(f_z_base_in, slot_factor)`` before chip thinning and load scaling."""
    d_in = max(cutter.diameter_mm / INCH_MM, 1e-5)
    woc_in = max(0.0, woc_mm / INCH_MM)

    q_d = lookup_endmill_reference_ipt_in(d_in)
    q_ref = lookup_endmill_reference_ipt_in(0.50)
    diameter_ratio = q_d / max(q_ref, 1e-12)

    helix_rad = math.radians(cutter.helix_deg)
    helix_factor = 1.0 + 0.5 * (math.sin(helix_rad) - 0.5)

    ideal_side_woc_in = stock.ideal_woc_frac * d_in
    if woc_in <= 0.0:
        slot_factor = 1.0
    else:
        slot_factor = lerp_clamped(
            ideal_side_woc_in, d_in, 1.0, cutter.slot_ipt_factor, woc_in
        )

    if tool_mat.is_carbide and stock.ipt_carbide_in > 0.0:
        material_chip = stock.ipt_carbide_in * tool_mat.coat_ipt_mult
    else:
        material_chip = stock.ipt_hss_in * reduce_ipt_factor(
            tool_mat.ipt_mult, stock.material_ipt_reduction
        )

    fz_base_in = (
        diameter_ratio
        * slot_factor
        * material_chip
        * helix_factor
        * cutter.tool_type_ipt
    )
    if stock.max_ipt_in > 0.0:
        fz_base_in = min(fz_base_in, stock.max_ipt_in)
    return float(fz_base_in), float(slot_factor)


def compute_feeds_and_speeds(
    stock_material_id: str = "al_6061_t6",
    tool_material_id: str = "carbide_coated",
    cutter: Optional[CutterGeometrySpec] = None,
    machine: Optional[MachineEnvelopeSpec] = None,
    doc_mm: float = 3.0,
    woc_mm: float = 1.75,
    aggressiveness: float = 1.0,
    chip_thinning_enabled: bool = True,
    engagement_load_enabled: bool = True,
    hsm_speed_enabled: bool = True,
    fixed_spindle_rpm: Optional[float] = None,
) -> FeedsSpeedsResult:
    """Solve physics-grounded RPM, optimal chip load, feedrate, and active limit."""
    stock = STOCK_MATERIALS.get(stock_material_id, STOCK_MATERIALS["al_6061_t6"])
    tool_mat = TOOL_MATERIALS.get(tool_material_id, TOOL_MATERIALS["carbide_coated"])
    cutter_spec = cutter or CutterGeometrySpec()
    machine_spec = machine or MachineEnvelopeSpec()

    aggr = float(max(0.25, min(2.50, aggressiveness)))
    d_mm = float(max(0.5, cutter_spec.diameter_mm))
    d_in = d_mm / INCH_MM
    doc_clean_mm = float(max(0.0, min(doc_mm, cutter_spec.flute_length_mm * 1.5)))
    woc_clean_mm = float(max(0.0, min(woc_mm, d_mm)))
    doc_in = doc_clean_mm / INCH_MM
    woc_in = woc_clean_mm / INCH_MM

    fz_base_in, slot_factor = compute_nominal_base_ipt_in(
        stock, tool_mat, cutter_spec, woc_clean_mm
    )

    r_thin = radial_chip_thinning_multiplier(woc_clean_mm, d_mm)
    a_thin = axial_chip_thinning_multiplier(doc_clean_mm, cutter_spec.corner_radius_mm)
    c_thin_all = r_thin * a_thin
    c_thin = c_thin_all if chip_thinning_enabled else 1.0

    ideal_woc_in = stock.ideal_woc_frac * d_in
    ideal_doc_in = stock.ideal_doc_frac * d_in
    if engagement_load_enabled and doc_in * woc_in > 1e-8:
        a_target = ideal_doc_in * ideal_woc_in
        c_load_raw = a_target / (doc_in * woc_in)
        k_max = lerp_clamped(0.0, 3.0, 1.375, 1.2, stock.kp_hp_min_in3) * lerp_clamped(
            1.0, 3.0, 1.5, 1.0, c_thin_all
        )
        k_max = max(1.0, min(6.0, k_max))
        c_load = max(0.5, min(k_max, c_load_raw))
    else:
        c_load = 1.0

    # Spindle speed selection (unless locked to a setup-wide fixed RPM)
    rpm_cap = min(machine_spec.max_rpm, cutter_spec.max_rpm)
    speed_limit_tag = ""
    if fixed_spindle_rpm is not None and fixed_spindle_rpm > 0.0:
        rpm = float(max(machine_spec.min_rpm, min(rpm_cap, fixed_spindle_rpm)))
        sfm = (rpm * math.pi * d_in) / 12.0
    else:
        sfm_slot = (
            lerp_clamped(ideal_woc_in, d_in, 1.0, 0.85, woc_in)
            if woc_in > 0.0
            else 1.0
        )
        hsm_boost = (
            min(r_thin * lerp_clamped(1.0, 15.0, 1.0, 1.5, a_thin), 1.5)
            if (hsm_speed_enabled and 0.0 < 2.0 * woc_clean_mm < d_mm)
            else 1.0
        )
        speed_aggr = 0.75 + 0.25 * aggr
        sfm_target = stock.sfm_carbide * tool_mat.sfm_mult * sfm_slot * hsm_boost * speed_aggr
        rpm_raw = (12.0 * sfm_target) / (math.pi * max(d_in, 1e-6))
        if rpm_raw > rpm_cap:
            rpm = float(rpm_cap)
            speed_limit_tag = "Spindle Max RPM"
        elif rpm_raw < machine_spec.min_rpm:
            rpm = float(machine_spec.min_rpm)
            speed_limit_tag = "Spindle Min RPM"
        else:
            rpm = float(round(rpm_raw))
        sfm = (rpm * math.pi * d_in) / 12.0

    # Pre-constraint programmed IPT and feedrate
    fz_prog_in = fz_base_in * c_thin * c_load * aggr
    if stock.max_ipt_in > 0.0:
        max_actual_in = stock.max_ipt_in * max(1.0, 0.65 + 0.35 * aggr)
        if fz_prog_in / max(c_thin_all, 1.0) > max_actual_in:
            fz_prog_in = max_actual_in * c_thin

    feed_in_min = rpm * cutter_spec.flutes * fz_prog_in
    feed_mm_min = feed_in_min * INCH_MM
    feed_mm_s = feed_mm_min / 60.0

    # Physical loads at unconstrained feed (all strictly affine in feedrate)
    mrr_in3_min = doc_in * woc_in * feed_in_min
    power_kw = mrr_in3_min * stock.kp_hp_min_in3 * HP_TO_KW
    omega_rad_s = max((2.0 * math.pi * rpm) / 60.0, 1e-6)
    torque_nm = (power_kw * 1000.0) / omega_rad_s
    r_eff_m = max(0.5 * d_mm * 1e-3, 1e-5)
    tangential_n = torque_nm / r_eff_m
    cutting_force_n = 1.15 * tangential_n

    # Tool shaft yield torque & cantilever deflection limits
    shank_mm = max(0.5, cutter_spec.shank_diameter_mm)
    d_core_in = 0.58 * (min(d_mm, shank_mm) / INCH_MM)
    shaft_yield_lbf_in = (math.pi / 16.0) * (1000.0 * tool_mat.yield_ksi) * (d_core_in ** 3)
    tool_max_torque_nm = (
        shaft_yield_lbf_in * LBF_IN_TO_NM * 0.35 * (cutter_spec.max_torque_pct / 100.0)
    )

    d_stiff_mm = 0.78 * min(d_mm, shank_mm)
    inertia_mm4 = (math.pi / 64.0) * (d_stiff_mm ** 4)
    youngs_n_mm2 = tool_mat.youngs_gpa * 1000.0
    deflection_mm = (cutting_force_n * (cutter_spec.stickout_mm ** 3)) / max(
        3.0 * youngs_n_mm2 * inertia_mm4, 1e-9
    )
    max_deflection_mm = max(
        0.005,
        tool_mat.dev_max_ratio
        * cutter_spec.stickout_mm
        * (cutter_spec.max_deflection_pct / 100.0),
    )

    # Exact closed-form constraint clamping (Section 4 of HSMAdvisor_chip_load_algorithm.md)
    u_scale = 1.0
    feed_limit_tag = ""
    candidates: List[Tuple[float, str]] = []
    if feed_mm_s > 1e-9:
        candidates.append(((machine_spec.max_feed_mm_s * 0.999) / feed_mm_s, "Machine Max Feed"))
    if power_kw > 1e-9:
        candidates.append((machine_spec.spindle_power_kw / power_kw, "Spindle Power"))
    if torque_nm > 1e-9:
        candidates.append(
            ((machine_spec.max_machine_torque_nm * 0.99) / torque_nm, "Spindle Torque")
        )
        candidates.append(((tool_max_torque_nm * 0.99) / torque_nm, "Tool Torque"))
    if cutting_force_n > 1e-9:
        candidates.append(
            ((machine_spec.max_cutting_force_n * 0.99) / cutting_force_n, "Cutting Force")
        )
    if deflection_mm > 1e-9:
        candidates.append(((max_deflection_mm * 0.99) / deflection_mm, "Tool Deflection"))

    for ratio, label in candidates:
        if ratio < u_scale:
            u_scale = ratio
            feed_limit_tag = label

    u_scale = float(max(0.02, min(1.0, u_scale)))
    fz_prog_in *= u_scale
    feed_mm_s *= u_scale
    feed_mm_min *= u_scale
    power_kw *= u_scale
    torque_nm *= u_scale
    cutting_force_n *= u_scale
    deflection_mm *= u_scale
    mrr_mm3_s = doc_clean_mm * woc_clean_mm * feed_mm_s

    chip_thickness_in = fz_prog_in / max(c_thin_all, 1.0)
    active_limit = feed_limit_tag or speed_limit_tag or "Optimal Chipload"

    return FeedsSpeedsResult(
        stock_material_id=stock.id,
        tool_material_id=tool_mat.id,
        aggressiveness=round(aggr, 4),
        diameter_mm=round(d_mm, 4),
        flutes=int(cutter_spec.flutes),
        doc_mm=round(doc_clean_mm, 4),
        woc_mm=round(woc_clean_mm, 4),
        spindle_rpm=round(rpm, 1),
        surface_speed_sfm=round(sfm, 2),
        surface_speed_m_min=round(sfm * 0.3048, 2),
        fz_base_mm=round(fz_base_in * INCH_MM, 6),
        fz_programmed_mm=round(fz_prog_in * INCH_MM, 6),
        chip_thickness_mm=round(chip_thickness_in * INCH_MM, 6),
        radial_thinning_factor=round(r_thin, 5),
        axial_thinning_factor=round(a_thin, 5),
        slot_factor=round(slot_factor, 5),
        engagement_load_factor=round(c_load, 5),
        feed_mm_s=round(feed_mm_s, 4),
        feed_mm_min=round(feed_mm_min, 2),
        mrr_mm3_s=round(mrr_mm3_s, 3),
        spindle_power_kw=round(power_kw, 4),
        power_utilization_pct=round(100.0 * power_kw / max(machine_spec.spindle_power_kw, 1e-6), 2),
        torque_nm=round(torque_nm, 4),
        cutting_force_n=round(cutting_force_n, 2),
        deflection_mm=round(deflection_mm, 5),
        max_deflection_mm=round(max_deflection_mm, 5),
        deflection_utilization_pct=round(
            100.0 * deflection_mm / max(max_deflection_mm, 1e-6), 2
        ),
        constraint_feed_scale=round(u_scale, 5),
        active_limit=active_limit,
    )


def export_feeds_speeds_catalog_json() -> Dict[str, object]:
    """Export material, tool, and diameter lookup tables for exact frontend parity."""
    return {
        "inch_mm": INCH_MM,
        "endmill_diameter_ipt_table_in": [
            [float(d), float(q)] for d, q in ENDMILL_DIAMETER_IPT_TABLE_IN
        ],
        "stock_materials": [asdict(m) for m in STOCK_MATERIALS.values()],
        "tool_materials": [asdict(t) for t in TOOL_MATERIALS.values()],
        "default_machine": asdict(MachineEnvelopeSpec()),
    }


def infer_case_nominal_doc_woc_mm(
    stock_max_z_mm: float,
    pocket_min_z_mm: float,
    cutter_diameter_mm: float,
    family: str,
    segments_doc_mm: Optional[np.ndarray] = None,
    segments_woc_mm: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Determine characteristic axial DOC and radial WOC for a part/operation."""
    if segments_doc_mm is not None and len(segments_doc_mm) > 0:
        pos_doc = segments_doc_mm[segments_doc_mm > 0.05]
        if len(pos_doc) > 0:
            doc = float(np.percentile(pos_doc, 75))
        else:
            doc = max(0.5, float(stock_max_z_mm - pocket_min_z_mm))
    else:
        total_depth = max(0.5, float(stock_max_z_mm - pocket_min_z_mm))
        doc = min(total_depth, cutter_diameter_mm * 0.8)

    if segments_woc_mm is not None and len(segments_woc_mm) > 0:
        pos_woc = segments_woc_mm[segments_woc_mm > 0.05]
        if len(pos_woc) > 0:
            woc = float(np.median(pos_woc))
        else:
            woc = 0.35 * cutter_diameter_mm
    else:
        fam = (family or "").lower()
        if "adapt" in fam:
            woc = 0.22 * cutter_diameter_mm
        elif "face" in fam:
            woc = 0.55 * cutter_diameter_mm
        elif "contour" in fam:
            woc = 0.40 * cutter_diameter_mm
        else:
            woc = 0.45 * cutter_diameter_mm
    return float(doc), float(woc)


# ---------------------------------------------------------------------------
# Multi-tool extensions (Drills, Spot Drills, Reamers, Chamfer & Face Mills)
# ---------------------------------------------------------------------------
DRILL_DIAMETER_IPT_TABLE_IN: Tuple[Tuple[float, float], ...] = (
    (0.03937, 0.00045),  # 1.0 mm
    (0.06250, 0.00075),  # 1/16"
    (0.12500, 0.00145),  # 1/8"
    (0.18750, 0.00210),  # 3/16"
    (0.25000, 0.00275),  # 1/4"
    (0.37500, 0.00385),  # 3/8"
    (0.50000, 0.00480),  # 1/2"
    (0.75000, 0.00640),  # 3/4"
)


@dataclass(frozen=True)
class HoleMakingFeedsSpeedsResult:
    """Solved spindle RPM, axial feed, peck schedule, and thrust/power for hole-making."""

    stock_material_id: str
    tool_material_id: str
    operation_kind: str
    diameter_mm: float
    flutes: int
    hole_depth_mm: float
    spindle_rpm: float
    surface_speed_sfm: float
    fz_mm: float
    feed_per_rev_mm: float
    plunge_feed_mm_s: float
    plunge_feed_mm_min: float
    peck_required: bool
    peck_depth_mm: float
    num_pecks: int
    spindle_power_kw: float
    thrust_force_n: float
    active_limit: str


def lookup_drill_reference_ipt_in(diameter_in: float) -> float:
    """Clamped piecewise-linear lookup ``Q(drill, D_in)`` in inches/flute."""
    d = float(diameter_in)
    table = DRILL_DIAMETER_IPT_TABLE_IN
    if d <= table[0][0]:
        return float(table[0][1])
    if d >= table[-1][0]:
        return float(table[-1][1])
    for i in range(len(table) - 1):
        d0, q0 = table[i]
        d1, q1 = table[i + 1]
        if d0 <= d <= d1:
            return lerp_clamped(d0, d1, q0, q1, d)
    return float(table[-1][1])


def compute_holemaking_feeds_and_speeds(
    stock_material_id: str = "al_6061_t6",
    tool_material_id: str = "carbide_coated",
    diameter_mm: float = 5.0,
    flutes: int = 2,
    hole_depth_mm: float = 8.0,
    operation_kind: str = "drill",
    machine: Optional[MachineEnvelopeSpec] = None,
    aggressiveness: float = 1.0,
) -> HoleMakingFeedsSpeedsResult:
    """Compute physics-grounded RPM, axial plunge feed, and peck depth for hole-making."""
    stock = STOCK_MATERIALS.get(stock_material_id, STOCK_MATERIALS["al_6061_t6"])
    tool_mat = TOOL_MATERIALS.get(tool_material_id, TOOL_MATERIALS["carbide_coated"])
    machine_spec = machine or MachineEnvelopeSpec()
    aggr = float(max(0.35, min(2.0, aggressiveness)))

    d_mm = float(max(0.5, diameter_mm))
    d_in = d_mm / INCH_MM
    depth_mm = float(max(0.2, hole_depth_mm))
    op = (operation_kind or "drill").lower()

    # Drilling / spot-drilling / reaming surface speed derating relative to side milling
    if "ream" in op:
        sfm_op_mult = 0.45
        ipt_op_mult = 1.35
    elif "spot" in op or "center" in op:
        sfm_op_mult = 0.65
        ipt_op_mult = 0.80
    elif "tap" in op:
        sfm_op_mult = 0.22
        ipt_op_mult = 1.0
    else:
        sfm_op_mult = 0.60
        ipt_op_mult = 1.0

    sfm_target = stock.sfm_carbide * tool_mat.sfm_mult * sfm_op_mult * (0.80 + 0.20 * aggr)
    rpm_raw = (12.0 * sfm_target) / (math.pi * max(d_in, 1e-6))
    active_limit = "Optimal Drill Chipload"
    if rpm_raw > machine_spec.max_rpm:
        rpm = float(machine_spec.max_rpm)
        active_limit = "Spindle Max RPM"
    elif rpm_raw < machine_spec.min_rpm:
        rpm = float(machine_spec.min_rpm)
        active_limit = "Spindle Min RPM"
    else:
        rpm = float(round(rpm_raw))
    sfm = (rpm * math.pi * d_in) / 12.0

    q_d = lookup_drill_reference_ipt_in(d_in)
    q_ref = lookup_drill_reference_ipt_in(0.50)
    diam_ratio = q_d / max(q_ref, 1e-12)
    mat_chip = (
        stock.ipt_carbide_in * tool_mat.coat_ipt_mult
        if tool_mat.is_carbide
        else stock.ipt_hss_in * reduce_ipt_factor(tool_mat.ipt_mult, stock.material_ipt_reduction)
    )
    # Deep-hole chip-evacuation derating above 3xD
    aspect = depth_mm / d_mm
    depth_derate = 1.0 if aspect <= 3.0 else max(0.65, 1.0 - 0.08 * (aspect - 3.0))
    fz_in = diam_ratio * mat_chip * ipt_op_mult * depth_derate * aggr
    fz_mm = fz_in * INCH_MM
    f_rev_mm = fz_mm * max(1, int(flutes))
    feed_mm_min = rpm * f_rev_mm
    feed_mm_s = feed_mm_min / 60.0

    # Axial MRR = (pi D^2 / 4) * feed, spindle power & drill thrust bounds
    area_in2 = 0.25 * math.pi * (d_in ** 2)
    mrr_in3_min = area_in2 * (feed_mm_min / INCH_MM)
    power_kw = mrr_in3_min * stock.kp_hp_min_in3 * 1.15 * HP_TO_KW
    # Empirical Shaw/Altintas axial thrust estimation proportional to hardness * D * f_rev
    thrust_n = 1.35 * stock.hardness_hb * d_mm * f_rev_mm

    u_scale = 1.0
    if feed_mm_s > machine_spec.max_feed_mm_s * 0.999:
        u_scale = min(u_scale, (machine_spec.max_feed_mm_s * 0.999) / feed_mm_s)
        active_limit = "Machine Max Feed"
    if power_kw > machine_spec.spindle_power_kw:
        u_scale = min(u_scale, machine_spec.spindle_power_kw / power_kw)
        active_limit = "Spindle Power"
    if thrust_n > machine_spec.max_cutting_force_n * 0.99:
        u_scale = min(u_scale, (machine_spec.max_cutting_force_n * 0.99) / thrust_n)
        active_limit = "Drill Thrust Force"

    u_scale = float(max(0.05, min(1.0, u_scale)))
    fz_mm *= u_scale
    f_rev_mm *= u_scale
    feed_mm_s *= u_scale
    feed_mm_min *= u_scale
    power_kw *= u_scale
    thrust_n *= u_scale

    peck_required = ("drill" in op and "spot" not in op and "center" not in op) and (aspect > 2.2)
    if peck_required:
        peck_depth = max(0.8, min(depth_mm, (1.25 if stock.kp_hp_min_in3 < 0.5 else 0.75) * d_mm))
        num_pecks = int(math.ceil(depth_mm / peck_depth))
    else:
        peck_depth = depth_mm
        num_pecks = 1

    return HoleMakingFeedsSpeedsResult(
        stock_material_id=stock.id,
        tool_material_id=tool_mat.id,
        operation_kind=op,
        diameter_mm=round(d_mm, 4),
        flutes=int(flutes),
        hole_depth_mm=round(depth_mm, 4),
        spindle_rpm=round(rpm, 1),
        surface_speed_sfm=round(sfm, 2),
        fz_mm=round(fz_mm, 5),
        feed_per_rev_mm=round(f_rev_mm, 5),
        plunge_feed_mm_s=round(feed_mm_s, 4),
        plunge_feed_mm_min=round(feed_mm_min, 2),
        peck_required=bool(peck_required),
        peck_depth_mm=round(peck_depth, 4),
        num_pecks=int(num_pecks),
        spindle_power_kw=round(power_kw, 4),
        thrust_force_n=round(thrust_n, 2),
        active_limit=active_limit,
    )
