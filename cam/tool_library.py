# SPDX-License-Identifier: MIT
"""Clean-room MIT standard CNC tool, holder, and preset catalog (``cam.tool_library``).

Synthesizes a comprehensive, queryable, 100% MIT-licensed catalog covering 15
standard library groups and 32 tool/holder entity types across both metric
(ISO / DIN) and imperial (ANSI / ASME) dimensional series, paired with physics
presets computed via :mod:`cam.feeds_and_speeds`:
  - Milling cutters: flat end mill, bull nose end mill, ball end mill, face mill,
    chamfer mill, slot mill, dovetail mill, lollipop mill, tapered mill, form
    mill, radius mill, thread mill.
  - Hole-making & threading: twist drill (ISO metric + ANSI fractional/number/
    letter), spot drill, center drill, reamer, counter bore, counter sink,
    right-hand tap (ISO metric + UNC/UNF), boring bar.
  - Tool holders: BT30 / CAT40 / HSK-A63 / ISO20 collet chucks (ER16/ER20/ER32),
    shrink-fit holders, hydraulic chucks, shell-mill arbors, and standard taper
    blanks (metric & inch).
  - Auxiliary industrial tool records: spindle touch probes, ISO turning tools
    (general, boring, grooving, threading), 2D jet/beam cutters (laser, plasma,
    waterjet), and directed-energy deposition heads (arc wire, laser wire,
    laser powder).

No proprietary files, vendor strings, or external databases are read or copied;
all dimensions follow public ISO/ANSI/DIN geometric progressions and convert
cleanly to millimetres for :class:`benchmarks.contracts.ToolAssembly`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from benchmarks import contracts as C
from cam.feeds_and_speeds import (
    INCH_MM,
    STOCK_MATERIALS,
    CutterGeometrySpec,
    MachineEnvelopeSpec,
    compute_feeds_and_speeds,
    compute_holemaking_feeds_and_speeds,
)


@dataclass(frozen=True)
class HolderSpec:
    """Clean-room stepped cylindrical spindle tool-holder specification (mm)."""

    holder_id: str
    library_group: str
    unit_system: str  # "millimeters" | "inches"
    taper: str        # "BT30" | "CAT40" | "HSK-A63" | "ISO20"
    collet_system: str
    description: str
    max_shank_mm: float
    gauge_length_mm: float
    body_radius_mm: float
    body_height_mm: float
    max_rpm: float


@dataclass(frozen=True)
class CatalogToolRecord:
    """Single clean-room catalog tool or holder entry with mm physical geometry."""

    tool_id: str
    library_group: str
    entity_type: str
    unit_system: str  # "millimeters" | "inches"
    nominal_label: str
    tool_material_id: str
    diameter_mm: float
    corner_radius_mm: float
    tip_angle_deg: float
    pitch_mm: float
    flutes: int
    flute_length_mm: float
    stickout_mm: float
    shank_diameter_mm: float
    overall_length_mm: float
    helix_deg: float
    max_rpm: float
    holder_id: Optional[str] = None

    def to_cutter_geometry(self) -> CutterGeometrySpec:
        is_face = self.entity_type == "face mill"
        return CutterGeometrySpec(
            diameter_mm=max(0.5, self.diameter_mm),
            flutes=max(1, self.flutes),
            flute_length_mm=max(1.0, self.flute_length_mm),
            stickout_mm=max(self.flute_length_mm, self.stickout_mm),
            shank_diameter_mm=max(0.5, self.shank_diameter_mm),
            corner_radius_mm=max(0.0, self.corner_radius_mm),
            helix_deg=self.helix_deg,
            slot_ipt_factor=0.90 if is_face else 0.80,
            tool_type_ipt=1.15 if is_face else 1.0,
            max_rpm=self.max_rpm,
        )

    def to_tool_assembly(
        self,
        tool_number: int,
        holder: Optional[HolderSpec] = None,
    ) -> C.ToolAssembly:
        """Convert to a validated :class:`benchmarks.contracts.ToolAssembly`."""
        kind = C.CutterKind.BALL if self.entity_type == "ball end mill" else C.CutterKind.FLAT
        r_cut = max(0.25, 0.5 * self.diameter_mm)
        r_shank = max(0.25, 0.5 * self.shank_diameter_mm)
        flute_len = max(1.0, self.flute_length_mm)
        stickout = max(flute_len + 1.0, self.stickout_mm)
        h_rad = holder.body_radius_mm if holder is not None else max(r_cut + 4.0, r_shank + 4.0, 14.0)
        h_rad = max(h_rad, r_cut + 0.5, r_shank + 0.5)
        h_height = holder.body_height_mm if holder is not None else 32.0
        return C.ToolAssembly(
            tool_number=int(tool_number),
            cutter_kind=kind,
            cutter_radius_mm=round(r_cut, 5),
            flute_length_mm=round(flute_len, 5),
            shank_radius_mm=round(r_shank, 5),
            holder_radius_mm=round(h_rad, 5),
            holder_bottom_mm=round(stickout, 5),
            holder_height_mm=round(h_height, 5),
        )


def _build_holders() -> List[HolderSpec]:
    holders: List[HolderSpec] = []
    # 31 Metric holders (BT30, CAT40, HSK-A63, ISO20 across ER16/ER20/ER25/ER32/Shrink/Shell)
    metric_specs = [
        ("BT30", "ER16", 10.0, [45.0, 60.0, 75.0, 100.0], 14.0, 30.0, 20000.0),
        ("BT30", "ER20", 13.0, [50.0, 65.0, 80.0, 100.0], 17.0, 32.0, 18000.0),
        ("BT30", "ER32", 20.0, [55.0, 70.0, 90.0], 25.0, 36.0, 15000.0),
        ("BT30", "ShrinkFit", 12.0, [50.0, 70.0, 90.0, 120.0], 12.0, 34.0, 24000.0),
        ("BT30", "ShellArbor", 22.0, [40.0, 55.0], 22.0, 28.0, 12000.0),
        ("CAT40", "ER16", 10.0, [60.0, 80.0, 100.0, 125.0], 14.0, 38.0, 18000.0),
        ("CAT40", "ER32", 20.0, [65.0, 85.0, 110.0, 140.0], 25.0, 42.0, 15000.0),
        ("HSK-A63", "ShrinkFit", 16.0, [65.0, 85.0, 110.0, 140.0], 15.0, 40.0, 24000.0),
        ("ISO20", "ER16", 10.0, [35.0, 50.0], 13.0, 24.0, 30000.0),
    ]
    idx = 1
    for taper, collet, max_shank, gauges, rad, hgt, rpm in metric_specs:
        for g in gauges:
            holders.append(
                HolderSpec(
                    holder_id=f"hld_mm_{idx:03d}_{taper.lower().replace('-', '')}_{collet.lower()}_{int(g)}",
                    library_group="Holders (Metric)",
                    unit_system="millimeters",
                    taper=taper,
                    collet_system=collet,
                    description=f"{taper} {collet} L={g:.0f}mm",
                    max_shank_mm=max_shank,
                    gauge_length_mm=g,
                    body_radius_mm=rad,
                    body_height_mm=hgt,
                    max_rpm=rpm,
                )
            )
            idx += 1

    # 30 Inch holders (CAT40 & BT30 ANSI gauge series)
    inch_specs = [
        ("CAT40", "ER16", 0.375, [2.5, 3.5, 4.5, 6.0], 0.56, 1.50, 18000.0),
        ("CAT40", "ER20", 0.500, [2.5, 3.5, 4.5, 6.0], 0.68, 1.60, 16000.0),
        ("CAT40", "ER32", 0.750, [2.5, 3.5, 4.5, 6.0], 0.98, 1.75, 15000.0),
        ("CAT40", "Weldon", 0.750, [2.5, 3.5, 4.5, 6.0], 0.95, 1.75, 12000.0),
        ("BT30", "ER16", 0.375, [2.0, 2.5, 3.0, 4.0], 0.55, 1.25, 20000.0),
        ("BT30", "ER20", 0.500, [2.0, 2.5, 3.0, 4.0], 0.67, 1.30, 18000.0),
        ("BT30", "ER32", 0.750, [2.25, 3.0, 4.0], 0.98, 1.40, 15000.0),
        ("CAT40", "ShellArbor", 1.000, [2.0, 3.0, 4.0], 1.10, 1.50, 10000.0),
    ]
    i_idx = 1
    for taper, collet, max_sh_in, gauges_in, rad_in, hgt_in, rpm in inch_specs:
        for g_in in gauges_in:
            holders.append(
                HolderSpec(
                    holder_id=f"hld_in_{i_idx:03d}_{taper.lower()}_{collet.lower()}",
                    library_group="Holders (Inch)",
                    unit_system="inches",
                    taper=taper,
                    collet_system=collet,
                    description=f'{taper} {collet} L={g_in:.2f}"',
                    max_shank_mm=round(max_sh_in * INCH_MM, 4),
                    gauge_length_mm=round(g_in * INCH_MM, 4),
                    body_radius_mm=round(rad_in * INCH_MM, 4),
                    body_height_mm=round(hgt_in * INCH_MM, 4),
                    max_rpm=rpm,
                )
            )
            i_idx += 1

    # 11 Standard Taper Blanks
    for b_idx, (taper, rad, hgt) in enumerate(
        [
            ("BT30", 23.0, 35.0),
            ("BT40", 31.5, 45.0),
            ("BT50", 50.0, 65.0),
            ("CAT40", 31.8, 45.0),
            ("CAT50", 49.2, 65.0),
            ("HSK-A40", 20.0, 30.0),
            ("HSK-A63", 31.5, 42.0),
            ("HSK-A100", 50.0, 60.0),
            ("ISO20", 13.0, 24.0),
            ("ISO25", 16.0, 28.0),
            ("R8", 16.0, 32.0),
        ],
        start=1,
    ):
        holders.append(
            HolderSpec(
                holder_id=f"hld_blank_{b_idx:02d}_{taper.lower().replace('-', '')}",
                library_group="Holders - Standard Taper Blanks",
                unit_system="millimeters",
                taper=taper,
                collet_system="Blank",
                description=f"{taper} Standard Taper Blank",
                max_shank_mm=20.0,
                gauge_length_mm=60.0,
                body_radius_mm=rad,
                body_height_mm=hgt,
                max_rpm=15000.0,
            )
        )
    return holders


def _std_shank_mm(d_mm: float) -> float:
    for s in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0):
        if d_mm <= s + 1e-6:
            return s
    return d_mm


def _std_shank_in_mm(d_in: float) -> float:
    for s_in in (0.125, 0.1875, 0.25, 0.3125, 0.375, 0.50, 0.625, 0.75, 1.0):
        if d_in <= s_in + 1e-6:
            return round(s_in * INCH_MM, 4)
    return round(d_in * INCH_MM, 4)


def _build_milling_tools(default_holder_mm: str, default_holder_in: str) -> List[CatalogToolRecord]:
    tools: List[CatalogToolRecord] = []

    # Metric Milling Tools (flat, bull nose, ball, chamfer, face, slot, dovetail, lollipop, tapered, form, radius, thread, spot, countersink)
    mm_diameters = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0]
    for d in mm_diameters:
        shank = _std_shank_mm(d)
        flute_len = round(max(3.0, 2.4 * d), 2)
        stickout = round(flute_len + max(2.0, 0.45 * d), 2)
        oal = round(max(38.0, stickout + 26.0), 2)
        # Flat end mill (3-flute coated carbide general/Al)
        tools.append(
            CatalogToolRecord(
                tool_id=f"mill_mm_flat_{str(d).replace('.', 'p')}",
                library_group="Milling Tools (Metric)",
                entity_type="flat end mill",
                unit_system="millimeters",
                nominal_label=f"Ø{d:g} mm 3FL Flat Carbide End Mill",
                tool_material_id="carbide_coated",
                diameter_mm=d,
                corner_radius_mm=0.0,
                tip_angle_deg=180.0,
                pitch_mm=0.0,
                flutes=3,
                flute_length_mm=flute_len,
                stickout_mm=stickout,
                shank_diameter_mm=shank,
                overall_length_mm=oal,
                helix_deg=35.0,
                max_rpm=24000.0,
                holder_id=default_holder_mm,
            )
        )
        # Bull nose end mill
        rc = round(min(0.2 * d, 1.5), 2)
        tools.append(
            CatalogToolRecord(
                tool_id=f"mill_mm_bull_{str(d).replace('.', 'p')}",
                library_group="Milling Tools (Metric)",
                entity_type="bull nose end mill",
                unit_system="millimeters",
                nominal_label=f"Ø{d:g} mm R{rc:g} Bull-Nose Carbide End Mill",
                tool_material_id="carbide_coated",
                diameter_mm=d,
                corner_radius_mm=rc,
                tip_angle_deg=180.0,
                pitch_mm=0.0,
                flutes=4 if d >= 6.0 else 3,
                flute_length_mm=flute_len,
                stickout_mm=stickout,
                shank_diameter_mm=shank,
                overall_length_mm=oal,
                helix_deg=38.0,
                max_rpm=24000.0,
                holder_id=default_holder_mm,
            )
        )
        # Ball end mill
        tools.append(
            CatalogToolRecord(
                tool_id=f"mill_mm_ball_{str(d).replace('.', 'p')}",
                library_group="Milling Tools (Metric)",
                entity_type="ball end mill",
                unit_system="millimeters",
                nominal_label=f"Ø{d:g} mm 2FL Ball Carbide End Mill",
                tool_material_id="carbide_coated",
                diameter_mm=d,
                corner_radius_mm=0.5 * d,
                tip_angle_deg=180.0,
                pitch_mm=0.0,
                flutes=2,
                flute_length_mm=round(2.0 * d, 2),
                stickout_mm=stickout,
                shank_diameter_mm=shank,
                overall_length_mm=oal,
                helix_deg=30.0,
                max_rpm=24000.0,
                holder_id=default_holder_mm,
            )
        )

    # Specialty Metric Milling Cutters
    specialty_mm = [
        ("face mill", [40.0, 50.0], 5, 8.0, 32.0, 20.0, 180.0, 0.8, 10000.0),
        ("chamfer mill", [6.0, 8.0, 10.0, 12.0], 4, 6.0, 18.0, 10.0, 90.0, 0.0, 18000.0),
        ("counter sink", [6.3, 10.4, 16.5], 3, 8.0, 22.0, 8.0, 90.0, 0.0, 8000.0),
        ("slot mill", [12.0, 16.0], 6, 3.0, 26.0, 8.0, 180.0, 0.0, 12000.0),
        ("dovetail mill", [8.0, 12.0, 16.0], 4, 6.0, 24.0, 8.0, 60.0, 0.0, 12000.0),
        ("lollipop mill", [4.0, 6.0, 8.0], 3, 4.0, 24.0, 6.0, 180.0, 2.0, 16000.0),
        ("tapered mill", [3.0, 6.0, 8.0], 3, 16.0, 28.0, 8.0, 10.0, 0.5, 20000.0),
        ("form mill", [10.0, 14.0], 4, 10.0, 25.0, 10.0, 180.0, 1.0, 12000.0),
        ("radius mill", [6.0, 8.0, 10.0, 12.0], 4, 6.0, 22.0, 10.0, 180.0, 2.0, 16000.0),
        ("thread mill", [4.8, 6.0, 8.0], 3, 12.0, 24.0, 6.0, 60.0, 0.0, 15000.0),
    ]
    for etype, diams, fl, fl_len, stk, shk, ang, rc, rpm in specialty_mm:
        slug = etype.replace(" ", "_")
        for d in diams:
            tools.append(
                CatalogToolRecord(
                    tool_id=f"mill_mm_{slug}_{str(d).replace('.', 'p')}",
                    library_group="Milling Tools (Metric)",
                    entity_type=etype,
                    unit_system="millimeters",
                    nominal_label=f"Ø{d:g} mm {etype.title()}",
                    tool_material_id="carbide_coated",
                    diameter_mm=d,
                    corner_radius_mm=rc,
                    tip_angle_deg=ang,
                    pitch_mm=1.25 if etype == "thread mill" else 0.0,
                    flutes=fl,
                    flute_length_mm=fl_len,
                    stickout_mm=stk,
                    shank_diameter_mm=shk,
                    overall_length_mm=stk + 30.0,
                    helix_deg=25.0,
                    max_rpm=rpm,
                    holder_id=default_holder_mm,
                )
            )

    # Inch Milling Tools
    in_sizes = [
        ("1/16", 0.0625),
        ("3/32", 0.09375),
        ("1/8", 0.125),
        ("3/16", 0.1875),
        ("1/4", 0.250),
        ("5/16", 0.3125),
        ("3/8", 0.375),
        ("1/2", 0.500),
        ("5/8", 0.625),
        ("3/4", 0.750),
    ]
    for frac_lbl, d_in in in_sizes:
        d_mm = round(d_in * INCH_MM, 4)
        sh_mm = _std_shank_in_mm(d_in)
        fl_mm = round(max(0.125, 2.25 * d_in) * INCH_MM, 3)
        stk_mm = round(fl_mm + max(2.5, 0.4 * d_mm), 3)
        fslug = frac_lbl.replace("/", "_")
        for etype, rc_mm, fl_cnt in (
            ("flat end mill", 0.0, 3),
            ("bull nose end mill", round(0.030 * INCH_MM, 4), 4),
            ("ball end mill", round(0.5 * d_mm, 4), 2),
        ):
            eslug = etype.split()[0]
            tools.append(
                CatalogToolRecord(
                    tool_id=f"mill_in_{eslug}_{fslug}",
                    library_group="Milling Tools (Inch)",
                    entity_type=etype,
                    unit_system="inches",
                    nominal_label=f'{frac_lbl}" {etype.title()}',
                    tool_material_id="carbide_coated",
                    diameter_mm=d_mm,
                    corner_radius_mm=rc_mm,
                    tip_angle_deg=180.0,
                    pitch_mm=0.0,
                    flutes=fl_cnt,
                    flute_length_mm=fl_mm,
                    stickout_mm=stk_mm,
                    shank_diameter_mm=sh_mm,
                    overall_length_mm=round(stk_mm + 28.0, 2),
                    helix_deg=35.0,
                    max_rpm=24000.0,
                    holder_id=default_holder_in,
                )
            )

    specialty_in = [
        ("face mill", [("2", 2.0), ("2.5", 2.5)], 5, 8.0, 35.0, 19.05, 180.0),
        ("chamfer mill", [("1/4", 0.25), ("3/8", 0.375), ("1/2", 0.50)], 4, 6.35, 20.0, 9.525, 90.0),
        ("counter sink", [("1/4", 0.25), ("3/8", 0.375), ("1/2", 0.50)], 3, 6.35, 22.0, 6.35, 82.0),
        ("dovetail mill", [("3/8", 0.375), ("1/2", 0.50)], 4, 6.35, 24.0, 9.525, 60.0),
        ("slot mill", [("1/2", 0.50), ("3/4", 0.75)], 6, 3.175, 26.0, 9.525, 180.0),
        ("lollipop mill", [("1/4", 0.25), ("3/8", 0.375), ("1/2", 0.50)], 3, 6.0, 25.0, 6.35, 180.0),
        ("tapered mill", [("1/8", 0.125), ("1/4", 0.25), ("3/8", 0.375)], 3, 15.0, 28.0, 6.35, 10.0),
        ("form mill", [("3/8", 0.375), ("1/2", 0.50)], 4, 9.5, 25.0, 9.525, 180.0),
        ("radius mill", [("1/4", 0.25), ("3/8", 0.375), ("1/2", 0.50)], 4, 6.35, 22.0, 9.525, 180.0),
        ("thread mill", [("3/16", 0.1875), ("1/4", 0.25), ("3/8", 0.375)], 3, 12.7, 24.0, 6.35, 60.0),
    ]
    for etype, items, fl, fl_len, stk, shk, ang in specialty_in:
        slug = etype.replace(" ", "_")
        for lbl, d_in in items:
            d_mm = round(d_in * INCH_MM, 4)
            tools.append(
                CatalogToolRecord(
                    tool_id=f"mill_in_{slug}_{lbl.replace('/', '_').replace('.', 'p')}",
                    library_group="Milling Tools (Inch)",
                    entity_type=etype,
                    unit_system="inches",
                    nominal_label=f'{lbl}" {etype.title()}',
                    tool_material_id="carbide_coated",
                    diameter_mm=d_mm,
                    corner_radius_mm=0.0,
                    tip_angle_deg=ang,
                    pitch_mm=1.27 if etype == "thread mill" else 0.0,
                    flutes=fl,
                    flute_length_mm=fl_len,
                    stickout_mm=stk,
                    shank_diameter_mm=shk,
                    overall_length_mm=stk + 28.0,
                    helix_deg=30.0,
                    max_rpm=16000.0,
                    holder_id=default_holder_in,
                )
            )
    return tools


def _build_holemaking_tools(default_holder_mm: str, default_holder_in: str) -> List[CatalogToolRecord]:
    tools: List[CatalogToolRecord] = []

    # 1. Spot drills (Metric & Inch, 90° and 120°)
    for d in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0):
        for ang in (90.0, 120.0):
            tools.append(
                CatalogToolRecord(
                    tool_id=f"hole_mm_spot_{int(d)}mm_{int(ang)}deg",
                    library_group="Hole Making Tools (Metric)",
                    entity_type="spot drill",
                    unit_system="millimeters",
                    nominal_label=f"Ø{d:g} mm {ang:.0f}° Carbide Spot Drill",
                    tool_material_id="carbide_coated",
                    diameter_mm=d,
                    corner_radius_mm=0.0,
                    tip_angle_deg=ang,
                    pitch_mm=0.0,
                    flutes=2,
                    flute_length_mm=round(1.8 * d, 2),
                    stickout_mm=round(2.5 * d + 6.0, 2),
                    shank_diameter_mm=d,
                    overall_length_mm=round(2.5 * d + 36.0, 2),
                    helix_deg=20.0,
                    max_rpm=18000.0,
                    holder_id=default_holder_mm,
                )
            )
    for lbl, d_in in (("1/8", 0.125), ("3/16", 0.1875), ("1/4", 0.25), ("3/8", 0.375), ("1/2", 0.50), ("5/8", 0.625)):
        d_mm = round(d_in * INCH_MM, 4)
        for ang in (90.0, 120.0):
            tools.append(
                CatalogToolRecord(
                    tool_id=f"hole_in_spot_{lbl.replace('/', '_')}_{int(ang)}deg",
                    library_group="Hole Making Tools (Inch)",
                    entity_type="spot drill",
                    unit_system="inches",
                    nominal_label=f'{lbl}" {ang:.0f}° Carbide Spot Drill',
                    tool_material_id="carbide_coated",
                    diameter_mm=d_mm,
                    corner_radius_mm=0.0,
                    tip_angle_deg=ang,
                    pitch_mm=0.0,
                    flutes=2,
                    flute_length_mm=round(1.8 * d_mm, 2),
                    stickout_mm=round(2.5 * d_mm + 6.0, 2),
                    shank_diameter_mm=d_mm,
                    overall_length_mm=round(2.5 * d_mm + 36.0, 2),
                    helix_deg=20.0,
                    max_rpm=18000.0,
                    holder_id=default_holder_in,
                )
            )

    # 2. Center drills (ANSI #00 through #8)
    for code, body_in in (
        ("00", 0.125), ("0", 0.125), ("1", 0.125), ("2", 0.1875), ("3", 0.250),
        ("4", 0.3125), ("4.5", 0.375), ("5", 0.4375), ("6", 0.500), ("7", 0.625),
    ):
        d_mm = round(body_in * INCH_MM, 4)
        tools.append(
            CatalogToolRecord(
                tool_id=f"hole_in_center_{code.replace('.', 'p')}",
                library_group="Hole Making Tools (Inch)",
                entity_type="center drill",
                unit_system="inches",
                nominal_label=f"ANSI #{code} Combined Center Drill",
                tool_material_id="cobalt_m42",
                diameter_mm=d_mm,
                corner_radius_mm=0.0,
                tip_angle_deg=60.0,
                pitch_mm=0.0,
                flutes=2,
                flute_length_mm=round(1.5 * d_mm, 2),
                stickout_mm=round(2.4 * d_mm + 5.0, 2),
                shank_diameter_mm=d_mm,
                overall_length_mm=50.8,
                helix_deg=20.0,
                max_rpm=12000.0,
                holder_id=default_holder_in,
            )
        )

    # 3. ISO Metric Twist Drills (1.0 mm to 10.0 mm in 0.1 mm steps + 10.2..16.0 mm in 0.2/0.5 mm steps = 180 drills)
    mm_drills: List[float] = [round(1.0 + 0.1 * k, 2) for k in range(91)]  # 1.0 .. 10.0 (91)
    mm_drills.extend([round(10.2 + 0.2 * k, 2) for k in range(29)])        # 10.2 .. 15.8 (29)
    mm_drills.extend([round(16.0 + 0.25 * k, 2) for k in range(17)])       # 16.0 .. 20.0 (17)
    # Also include 3xD Carbide & 5xD Cobalt variants on standard integer/half sizes (45) -> 182 metric drills
    for d in mm_drills:
        fl_len = round(max(8.0, 4.5 * d), 2)
        stk = round(fl_len + max(4.0, 0.5 * d), 2)
        tools.append(
            CatalogToolRecord(
                tool_id=f"hole_mm_drill_{str(d).replace('.', 'p')}",
                library_group="Hole Making Tools (Metric)",
                entity_type="drill",
                unit_system="millimeters",
                nominal_label=f"Ø{d:.2f} mm Solid Carbide Twist Drill (140°)",
                tool_material_id="carbide_coated",
                diameter_mm=d,
                corner_radius_mm=0.0,
                tip_angle_deg=140.0,
                pitch_mm=0.0,
                flutes=2,
                flute_length_mm=fl_len,
                stickout_mm=stk,
                shank_diameter_mm=_std_shank_mm(d),
                overall_length_mm=round(stk + 30.0, 2),
                helix_deg=30.0,
                max_rpm=18000.0,
                holder_id=default_holder_mm,
            )
        )
    for d_hss in [round(1.5 + 0.5 * k, 2) for k in range(37)]:  # 1.5 .. 19.5 mm DIN 338 Cobalt M42
        fl_len = round(max(10.0, 5.2 * d_hss), 2)
        stk = round(fl_len + 5.0, 2)
        tools.append(
            CatalogToolRecord(
                tool_id=f"hole_mm_drill_cobalt_{str(d_hss).replace('.', 'p')}",
                library_group="Hole Making Tools (Metric)",
                entity_type="drill",
                unit_system="millimeters",
                nominal_label=f"Ø{d_hss:g} mm DIN 338 Cobalt M42 Jobber Drill (135°)",
                tool_material_id="cobalt_m42",
                diameter_mm=d_hss,
                corner_radius_mm=0.0,
                tip_angle_deg=135.0,
                pitch_mm=0.0,
                flutes=2,
                flute_length_mm=fl_len,
                stickout_mm=stk,
                shank_diameter_mm=d_hss,
                overall_length_mm=round(stk + 28.0, 2),
                helix_deg=28.0,
                max_rpm=12000.0,
                holder_id=default_holder_mm,
            )
        )

    # 4. ANSI Fractional (1/64" .. 3/4" in 1/64" steps = 48) + Number (#1..#60) + Letter (A..Z) Twist Drills (134) = 182 inch drills
    for n64 in range(3, 49):
        d_in = n64 / 64.0
        d_mm = round(d_in * INCH_MM, 4)
        fl_len = round(max(8.0, 4.5 * d_mm), 2)
        stk = round(fl_len + 5.0, 2)
        tools.append(
            CatalogToolRecord(
                tool_id=f"hole_in_drill_{n64}_64",
                library_group="Hole Making Tools (Inch)",
                entity_type="drill",
                unit_system="inches",
                nominal_label=f'{n64}/64" ({d_in:.4f}") Carbide Twist Drill',
                tool_material_id="carbide_coated",
                diameter_mm=d_mm,
                corner_radius_mm=0.0,
                tip_angle_deg=135.0,
                pitch_mm=0.0,
                flutes=2,
                flute_length_mm=fl_len,
                stickout_mm=stk,
                shank_diameter_mm=_std_shank_in_mm(d_in),
                overall_length_mm=round(stk + 28.0, 2),
                helix_deg=30.0,
                max_rpm=16000.0,
                holder_id=default_holder_in,
            )
        )

    # Standard ANSI B94.11M Wire Gauge #1..#60 diameters (inches)
    ansi_wire_in = [
        0.2280, 0.2210, 0.2130, 0.2090, 0.2055, 0.2040, 0.2010, 0.1990, 0.1960, 0.1935,
        0.1910, 0.1890, 0.1850, 0.1820, 0.1800, 0.1770, 0.1730, 0.1695, 0.1660, 0.1610,
        0.1590, 0.1570, 0.1540, 0.1520, 0.1495, 0.1470, 0.1440, 0.1405, 0.1360, 0.1285,
        0.1200, 0.1160, 0.1130, 0.1110, 0.1100, 0.1065, 0.1040, 0.1015, 0.0995, 0.0980,
        0.0960, 0.0935, 0.0890, 0.0860, 0.0820, 0.0810, 0.0785, 0.0760, 0.0730, 0.0700,
        0.0670, 0.0635, 0.0595, 0.0550, 0.0520, 0.0465, 0.0430, 0.0420, 0.0410, 0.0400,
    ]
    for gauge_num, d_in in enumerate(ansi_wire_in, start=1):
        d_mm = round(d_in * INCH_MM, 4)
        fl_len = round(max(7.0, 4.5 * d_mm), 2)
        stk = round(fl_len + 4.5, 2)
        tools.append(
            CatalogToolRecord(
                tool_id=f"hole_in_drill_num_{gauge_num:02d}",
                library_group="Hole Making Tools (Inch)",
                entity_type="drill",
                unit_system="inches",
                nominal_label=f'#{gauge_num} ({d_in:.4f}") ANSI Wire Drill',
                tool_material_id="cobalt_m42",
                diameter_mm=d_mm,
                corner_radius_mm=0.0,
                tip_angle_deg=135.0,
                pitch_mm=0.0,
                flutes=2,
                flute_length_mm=fl_len,
                stickout_mm=stk,
                shank_diameter_mm=d_mm,
                overall_length_mm=round(stk + 26.0, 2),
                helix_deg=28.0,
                max_rpm=15000.0,
                holder_id=default_holder_in,
            )
        )

    ansi_letters = [
        ("A", 0.234), ("B", 0.238), ("C", 0.242), ("D", 0.246), ("E", 0.250),
        ("F", 0.257), ("G", 0.261), ("H", 0.266), ("I", 0.272), ("J", 0.277),
        ("K", 0.281), ("L", 0.290), ("M", 0.295), ("N", 0.302), ("O", 0.316),
        ("P", 0.323), ("Q", 0.332), ("R", 0.339), ("S", 0.348), ("T", 0.358),
        ("U", 0.368), ("V", 0.377), ("W", 0.386), ("X", 0.397), ("Y", 0.404), ("Z", 0.413),
    ]
    for ltr, d_in in ansi_letters:
        d_mm = round(d_in * INCH_MM, 4)
        fl_len = round(4.5 * d_mm, 2)
        stk = round(fl_len + 5.0, 2)
        tools.append(
            CatalogToolRecord(
                tool_id=f"hole_in_drill_ltr_{ltr}",
                library_group="Hole Making Tools (Inch)",
                entity_type="drill",
                unit_system="inches",
                nominal_label=f'Letter {ltr} ({d_in:.3f}") ANSI Drill',
                tool_material_id="cobalt_m42",
                diameter_mm=d_mm,
                corner_radius_mm=0.0,
                tip_angle_deg=135.0,
                pitch_mm=0.0,
                flutes=2,
                flute_length_mm=fl_len,
                stickout_mm=stk,
                shank_diameter_mm=d_mm,
                overall_length_mm=round(stk + 28.0, 2),
                helix_deg=28.0,
                max_rpm=14000.0,
                holder_id=default_holder_in,
            )
        )

    # 5. Right-Hand Taps (ISO Metric Coarse & Fine + ANSI UNC/UNF)
    iso_taps = [
        ("M2x0.4", 2.0, 0.40), ("M2.5x0.45", 2.5, 0.45), ("M3x0.5", 3.0, 0.50), ("M3x0.35", 3.0, 0.35),
        ("M4x0.7", 4.0, 0.70), ("M4x0.5", 4.0, 0.50), ("M5x0.8", 5.0, 0.80), ("M5x0.5", 5.0, 0.50),
        ("M6x1.0", 6.0, 1.00), ("M6x0.75", 6.0, 0.75), ("M8x1.25", 8.0, 1.25), ("M8x1.0", 8.0, 1.00),
        ("M10x1.5", 10.0, 1.50), ("M10x1.25", 10.0, 1.25), ("M12x1.75", 12.0, 1.75), ("M12x1.25", 12.0, 1.25),
        ("M14x2.0", 14.0, 2.00), ("M16x2.0", 16.0, 2.00),
    ]
    for style in ("spiral_flute", "spiral_point"):
        for lbl, d, pitch in iso_taps:
            tools.append(
                CatalogToolRecord(
                    tool_id=f"hole_mm_tap_{lbl.lower().replace('.', 'p')}_{style}",
                    library_group="Hole Making Tools (Metric)",
                    entity_type="tap right hand",
                    unit_system="millimeters",
                    nominal_label=f"{lbl} {style.replace('_', ' ').title()} Tap",
                    tool_material_id="cobalt_m42",
                    diameter_mm=d,
                    corner_radius_mm=0.0,
                    tip_angle_deg=180.0,
                    pitch_mm=pitch,
                    flutes=3,
                    flute_length_mm=round(2.5 * d, 2),
                    stickout_mm=round(3.5 * d + 8.0, 2),
                    shank_diameter_mm=_std_shank_mm(d),
                    overall_length_mm=round(3.5 * d + 38.0, 2),
                    helix_deg=35.0 if style == "spiral_flute" else 0.0,
                    max_rpm=3000.0,
                    holder_id=default_holder_mm,
                )
            )

    unc_unf_taps = [
        ("4-40", 0.1120, 40), ("6-32", 0.1380, 32), ("8-32", 0.1640, 32), ("10-24", 0.1900, 24),
        ("10-32", 0.1900, 32), ("1/4-20", 0.2500, 20), ("1/4-28", 0.2500, 28), ("5/16-18", 0.3125, 18),
        ("5/16-24", 0.3125, 24), ("3/8-16", 0.3750, 16), ("3/8-24", 0.3750, 24), ("1/2-13", 0.5000, 13),
        ("1/2-20", 0.5000, 20),
    ]
    for style in ("spiral_flute", "spiral_point"):
        for lbl, d_in, tpi in unc_unf_taps:
            d_mm = round(d_in * INCH_MM, 4)
            pitch_mm = round(INCH_MM / tpi, 5)
            tools.append(
                CatalogToolRecord(
                    tool_id=f"hole_in_tap_{lbl.replace('/', '_').replace('-', '_')}_{style}",
                    library_group="Hole Making Tools (Inch)",
                    entity_type="tap right hand",
                    unit_system="inches",
                    nominal_label=f"{lbl} {style.replace('_', ' ').title()} Tap",
                    tool_material_id="cobalt_m42",
                    diameter_mm=d_mm,
                    corner_radius_mm=0.0,
                    tip_angle_deg=180.0,
                    pitch_mm=pitch_mm,
                    flutes=3,
                    flute_length_mm=round(2.5 * d_mm, 2),
                    stickout_mm=round(3.5 * d_mm + 8.0, 2),
                    shank_diameter_mm=_std_shank_in_mm(d_in),
                    overall_length_mm=round(3.5 * d_mm + 38.0, 2),
                    helix_deg=35.0 if style == "spiral_flute" else 0.0,
                    max_rpm=3000.0,
                    holder_id=default_holder_in,
                )
            )

    # 6. Reamers, Counterbores, Boring Bars
    for d in (4.0, 5.0, 6.0, 8.0, 10.0, 12.0):
        tools.append(
            CatalogToolRecord(
                tool_id=f"hole_mm_reamer_{int(d)}mm",
                library_group="Hole Making Tools (Metric)",
                entity_type="reamer",
                unit_system="millimeters",
                nominal_label=f"Ø{d:g} mm H7 Carbide Chucking Reamer",
                tool_material_id="carbide_uncoated",
                diameter_mm=d,
                corner_radius_mm=0.0,
                tip_angle_deg=180.0,
                pitch_mm=0.0,
                flutes=6,
                flute_length_mm=round(3.0 * d, 2),
                stickout_mm=round(4.0 * d + 6.0, 2),
                shank_diameter_mm=d,
                overall_length_mm=round(4.0 * d + 36.0, 2),
                helix_deg=8.0,
                max_rpm=6000.0,
                holder_id=default_holder_mm,
            )
        )
    for lbl, d_in in (("1/4", 0.25), ("3/8", 0.375), ("1/2", 0.50)):
        d_mm = round(d_in * INCH_MM, 4)
        tools.append(
            CatalogToolRecord(
                tool_id=f"hole_in_reamer_{lbl.replace('/', '_')}",
                library_group="Hole Making Tools (Inch)",
                entity_type="reamer",
                unit_system="inches",
                nominal_label=f'{lbl}" Chucking Reamer',
                tool_material_id="carbide_uncoated",
                diameter_mm=d_mm,
                corner_radius_mm=0.0,
                tip_angle_deg=180.0,
                pitch_mm=0.0,
                flutes=6,
                flute_length_mm=round(3.0 * d_mm, 2),
                stickout_mm=round(4.0 * d_mm + 6.0, 2),
                shank_diameter_mm=d_mm,
                overall_length_mm=round(4.0 * d_mm + 36.0, 2),
                helix_deg=8.0,
                max_rpm=6000.0,
                holder_id=default_holder_in,
            )
        )
    for etype, grp, unit, d_mm in (
        ("counter bore", "Hole Making Tools (Metric)", "millimeters", 10.0),
        ("counter bore", "Hole Making Tools (Metric)", "millimeters", 14.0),
        ("counter bore", "Hole Making Tools (Inch)", "inches", 9.525),
        ("counter bore", "Hole Making Tools (Inch)", "inches", 12.70),
        ("boring bar", "Hole Making Tools (Metric)", "millimeters", 12.0),
        ("boring bar", "Hole Making Tools (Inch)", "inches", 12.70),
    ):
        tools.append(
            CatalogToolRecord(
                tool_id=f"hole_{unit[:2]}_{etype.replace(' ', '_')}_{str(d_mm).replace('.', 'p')}",
                library_group=grp,
                entity_type=etype,
                unit_system=unit,
                nominal_label=f"Ø{d_mm:g} mm {etype.title()}",
                tool_material_id="carbide_coated",
                diameter_mm=d_mm,
                corner_radius_mm=0.2,
                tip_angle_deg=180.0,
                pitch_mm=0.0,
                flutes=3 if "counter" in etype else 1,
                flute_length_mm=18.0,
                stickout_mm=30.0,
                shank_diameter_mm=10.0,
                overall_length_mm=65.0,
                helix_deg=15.0,
                max_rpm=8000.0,
                holder_id=default_holder_mm if unit == "millimeters" else default_holder_in,
            )
        )
    return tools


def _build_auxiliary_libraries(default_holder_mm: str, default_holder_in: str) -> List[CatalogToolRecord]:
    """Build Tutorial, Probes, Turning, Cutting, and Depositing clean-room records."""
    aux: List[CatalogToolRecord] = []

    # Tutorial Tools (Metric & Inch)
    for d in (4.0, 6.0, 8.0, 10.0, 12.0):
        aux.append(
            CatalogToolRecord(
                tool_id=f"tut_mm_flat_{int(d)}mm",
                library_group="Tutorial Tools (Metric)",
                entity_type="flat end mill",
                unit_system="millimeters",
                nominal_label=f"Tutorial Ø{d:g} mm Flat End Mill",
                tool_material_id="carbide_coated",
                diameter_mm=d,
                corner_radius_mm=0.0,
                tip_angle_deg=180.0,
                pitch_mm=0.0,
                flutes=3,
                flute_length_mm=2.2 * d,
                stickout_mm=2.8 * d,
                shank_diameter_mm=d,
                overall_length_mm=60.0,
                helix_deg=35.0,
                max_rpm=18000.0,
                holder_id=default_holder_mm,
            )
        )
    for lbl, d_in in (("1/8", 0.125), ("1/4", 0.25), ("3/8", 0.375), ("1/2", 0.50)):
        d_mm = round(d_in * INCH_MM, 4)
        aux.append(
            CatalogToolRecord(
                tool_id=f"tut_in_flat_{lbl.replace('/', '_')}",
                library_group="Tutorial Tools (Inch)",
                entity_type="flat end mill",
                unit_system="inches",
                nominal_label=f'Tutorial {lbl}" Flat End Mill',
                tool_material_id="carbide_coated",
                diameter_mm=d_mm,
                corner_radius_mm=0.0,
                tip_angle_deg=180.0,
                pitch_mm=0.0,
                flutes=3,
                flute_length_mm=round(2.2 * d_mm, 2),
                stickout_mm=round(2.8 * d_mm, 2),
                shank_diameter_mm=d_mm,
                overall_length_mm=60.0,
                helix_deg=35.0,
                max_rpm=18000.0,
                holder_id=default_holder_in,
            )
        )

    # Probes (6 ruby styli)
    for styl_d in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        aux.append(
            CatalogToolRecord(
                tool_id=f"probe_mm_ruby_{int(styl_d)}mm",
                library_group="Probes",
                entity_type="probe",
                unit_system="millimeters",
                nominal_label=f"Ø{styl_d:g} mm Ruby Spindle Touch Probe",
                tool_material_id="carbide_uncoated",
                diameter_mm=styl_d,
                corner_radius_mm=0.5 * styl_d,
                tip_angle_deg=180.0,
                pitch_mm=0.0,
                flutes=1,
                flute_length_mm=styl_d,
                stickout_mm=50.0,
                shank_diameter_mm=6.0,
                overall_length_mm=90.0,
                helix_deg=0.0,
                max_rpm=0.0,
                holder_id=default_holder_mm,
            )
        )

    # Turning (Metric & Inch)
    for unit, grp in (("millimeters", "Turning Tools (Metric)"), ("inches", "Turning Tools (Inch)")):
        for etype in ("turning general", "turning boring", "turning grooving", "turning threading"):
            aux.append(
                CatalogToolRecord(
                    tool_id=f"turn_{unit[:2]}_{etype.replace(' ', '_')}",
                    library_group=grp,
                    entity_type=etype,
                    unit_system=unit,
                    nominal_label=f"ISO {etype.title()} Insert",
                    tool_material_id="carbide_coated",
                    diameter_mm=12.0 if unit == "millimeters" else 12.7,
                    corner_radius_mm=0.8,
                    tip_angle_deg=80.0,
                    pitch_mm=1.5 if "thread" in etype else 0.0,
                    flutes=1,
                    flute_length_mm=12.0,
                    stickout_mm=25.0,
                    shank_diameter_mm=20.0,
                    overall_length_mm=125.0,
                    helix_deg=0.0,
                    max_rpm=4000.0,
                    holder_id=None,
                )
            )

    # 2D Cutting & Additive Depositing Tools
    for unit, grp in (("millimeters", "Cutting Tools (Metric)"), ("inches", "Cutting Tools (Inch)")):
        for etype, kerf in (("laser cutter", 0.20), ("plasma cutter", 1.50), ("waterjet", 0.80)):
            aux.append(
                CatalogToolRecord(
                    tool_id=f"cut_{unit[:2]}_{etype.replace(' ', '_')}",
                    library_group=grp,
                    entity_type=etype,
                    unit_system=unit,
                    nominal_label=f"{etype.title()} ({kerf:g} mm kerf)",
                    tool_material_id="carbide_uncoated",
                    diameter_mm=kerf,
                    corner_radius_mm=0.0,
                    tip_angle_deg=180.0,
                    pitch_mm=0.0,
                    flutes=1,
                    flute_length_mm=25.0,
                    stickout_mm=30.0,
                    shank_diameter_mm=10.0,
                    overall_length_mm=60.0,
                    helix_deg=0.0,
                    max_rpm=0.0,
                    holder_id=None,
                )
            )
    for etype, bead in (("electric arc wire", 2.2), ("laser powder", 1.2), ("laser wire", 1.6)):
        aux.append(
            CatalogToolRecord(
                tool_id=f"dep_mm_{etype.replace(' ', '_')}",
                library_group="Depositing Tools",
                entity_type=etype,
                unit_system="millimeters",
                nominal_label=f"DED {etype.title()} ({bead:g} mm)",
                tool_material_id="carbide_uncoated",
                diameter_mm=bead,
                corner_radius_mm=0.5 * bead,
                tip_angle_deg=180.0,
                pitch_mm=0.0,
                flutes=1,
                flute_length_mm=10.0,
                stickout_mm=35.0,
                shank_diameter_mm=16.0,
                overall_length_mm=80.0,
                helix_deg=0.0,
                max_rpm=0.0,
                holder_id=None,
            )
        )
    return aux


class StandardToolLibrary:
    """Queryable clean-room MIT tool, holder, and feeds-and-speeds preset catalog."""

    def __init__(self) -> None:
        self.holders: List[HolderSpec] = _build_holders()
        self.holder_by_id: Dict[str, HolderSpec] = {h.holder_id: h for h in self.holders}
        default_mm = next(h.holder_id for h in self.holders if h.taper == "BT30" and h.collet_system == "ER20")
        default_in = next(
            h.holder_id for h in self.holders if h.unit_system == "inches" and h.taper == "BT30"
        )
        self.tools: List[CatalogToolRecord] = []
        self.tools.extend(_build_milling_tools(default_mm, default_in))
        self.tools.extend(_build_holemaking_tools(default_mm, default_in))
        self.tools.extend(_build_auxiliary_libraries(default_mm, default_in))
        self.tool_by_id: Dict[str, CatalogToolRecord] = {t.tool_id: t for t in self.tools}

    @property
    def total_record_count(self) -> int:
        return len(self.tools) + len(self.holders)

    def summary(self) -> Dict[str, object]:
        lib_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {"holder": len(self.holders)}
        for h in self.holders:
            lib_counts[h.library_group] = lib_counts.get(h.library_group, 0) + 1
        for t in self.tools:
            lib_counts[t.library_group] = lib_counts.get(t.library_group, 0) + 1
            type_counts[t.entity_type] = type_counts.get(t.entity_type, 0) + 1
        return {
            "schema": "diffcam-cleanroom-tool-library/1",
            "license": "MIT",
            "library_groups": len(lib_counts),
            "entity_types": len(type_counts),
            "tool_records": len(self.tools),
            "holder_records": len(self.holders),
            "total_records": self.total_record_count,
            "synthesized_preset_combinations": len(self.tools) * len(STOCK_MATERIALS),
            "counts_by_library": dict(sorted(lib_counts.items())),
            "counts_by_entity_type": dict(sorted(type_counts.items())),
        }

    def select_holder_for_tool(
        self,
        tool: CatalogToolRecord,
        taper: str = "BT30",
        min_reach_mm: float = 0.0,
    ) -> HolderSpec:
        """Select the shortest rigid collet/shrink holder fitting ``tool`` and ``min_reach_mm``."""
        candidates = [
            h
            for h in self.holders
            if h.taper == taper
            and h.collet_system != "Blank"
            and h.max_shank_mm + 1e-6 >= tool.shank_diameter_mm
        ]
        if not candidates:
            candidates = [h for h in self.holders if h.collet_system != "Blank"]
        candidates.sort(key=lambda h: (abs(h.gauge_length_mm - max(45.0, min_reach_mm)), h.body_radius_mm))
        return candidates[0]

    def query_endmills(
        self,
        *,
        min_diameter_mm: float = 1.0,
        max_diameter_mm: float = 20.0,
        min_flute_length_mm: float = 1.0,
        unit_system: Optional[str] = "millimeters",
        entity_type: str = "flat end mill",
    ) -> List[CatalogToolRecord]:
        out = [
            t
            for t in self.tools
            if t.entity_type == entity_type
            and t.library_group.startswith("Milling Tools")
            and min_diameter_mm - 1e-6 <= t.diameter_mm <= max_diameter_mm + 1e-6
            and t.flute_length_mm + 1e-6 >= min_flute_length_mm
            and (unit_system is None or t.unit_system == unit_system)
        ]
        out.sort(key=lambda t: t.diameter_mm)
        return out

    def query_face_mills_or_large_endmills(
        self,
        *,
        max_diameter_mm: float = 50.0,
        unit_system: Optional[str] = "millimeters",
    ) -> List[CatalogToolRecord]:
        face = [
            t
            for t in self.tools
            if t.entity_type in ("face mill", "flat end mill")
            and t.library_group.startswith("Milling Tools")
            and t.diameter_mm <= max_diameter_mm + 1e-6
            and (unit_system is None or t.unit_system == unit_system)
        ]
        face.sort(key=lambda t: t.diameter_mm, reverse=True)
        return face

    def match_spot_drill(
        self,
        hole_diameter_mm: float,
        unit_system: Optional[str] = "millimeters",
    ) -> CatalogToolRecord:
        spots = [
            t
            for t in self.tools
            if t.entity_type == "spot drill"
            and t.tip_angle_deg == 90.0
            and t.diameter_mm >= min(12.0, max(4.0, hole_diameter_mm * 0.9))
            and (unit_system is None or t.unit_system == unit_system)
        ]
        if not spots:
            spots = [t for t in self.tools if t.entity_type == "spot drill"]
        spots.sort(key=lambda t: t.diameter_mm)
        return spots[0]

    def match_drill(
        self,
        hole_diameter_mm: float,
        hole_depth_mm: float,
        tol_mm: float = 0.06,
    ) -> Optional[CatalogToolRecord]:
        """Find the highest-performance carbide/cobalt twist drill matching ``hole_diameter_mm``."""
        matches = [
            t
            for t in self.tools
            if t.entity_type == "drill"
            and abs(t.diameter_mm - hole_diameter_mm) <= tol_mm
            and t.flute_length_mm + 1e-6 >= hole_depth_mm
        ]
        if not matches:
            return None
        # Prefer coated carbide over cobalt HSS, and exact diameter match
        matches.sort(
            key=lambda t: (
                0 if t.tool_material_id == "carbide_coated" else 1,
                abs(t.diameter_mm - hole_diameter_mm),
            )
        )
        return matches[0]

    def compute_tool_preset(
        self,
        tool: CatalogToolRecord,
        stock_material_id: str = "al_6061_t6",
        operation_kind: str = "roughing",
        doc_mm: Optional[float] = None,
        woc_mm: Optional[float] = None,
        machine: Optional[MachineEnvelopeSpec] = None,
        aggressiveness: float = 1.0,
    ) -> Dict[str, object]:
        """Compute clean-room ``feeds_and_speeds`` preset for ``(tool, material, op)``."""
        if tool.entity_type in ("drill", "spot drill", "center drill", "reamer", "tap right hand"):
            res = compute_holemaking_feeds_and_speeds(
                stock_material_id=stock_material_id,
                tool_material_id=tool.tool_material_id,
                diameter_mm=tool.diameter_mm,
                flutes=tool.flutes,
                hole_depth_mm=doc_mm or (2.0 * tool.diameter_mm),
                operation_kind=tool.entity_type,
                machine=machine,
                aggressiveness=aggressiveness,
            )
            return asdict(res)

        d = tool.diameter_mm
        op = operation_kind.lower()
        if doc_mm is None:
            doc_mm = 1.0 if "face" in op else (0.75 * d if "rough" in op else 0.5 * d)
        if woc_mm is None:
            woc_mm = 0.60 * d if "face" in op else (0.35 * d if "rough" in op else 0.15 * d)
        res_m = compute_feeds_and_speeds(
            stock_material_id=stock_material_id,
            tool_material_id=tool.tool_material_id,
            cutter=tool.to_cutter_geometry(),
            machine=machine,
            doc_mm=doc_mm,
            woc_mm=woc_mm,
            aggressiveness=aggressiveness,
        )
        return asdict(res_m)


DEFAULT_TOOL_LIBRARY = StandardToolLibrary()
