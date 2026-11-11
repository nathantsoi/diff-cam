# SPDX-License-Identifier: MIT
"""Unit tests for the clean-room feeds, speeds, chip-load, and limit solver."""

from __future__ import annotations

import pytest

from cam.feeds_and_speeds import (
    INCH_MM,
    STOCK_MATERIALS,
    TOOL_MATERIALS,
    CutterGeometrySpec,
    MachineEnvelopeSpec,
    compute_feeds_and_speeds,
    compute_nominal_base_ipt_in,
    export_feeds_speeds_catalog_json,
    lookup_endmill_reference_ipt_in,
    radial_chip_thinning_multiplier,
)


def test_worked_baseline_example_parity() -> None:
    """Verify exact numerical match against Section 5 worked baseline example."""
    # Solid uncoated carbide end mill, D = 0.25 in (6.35 mm), 30-deg helix,
    # Mild Steel / A36 (ipt_carbide = 0.00276 in/tooth).
    cutter = CutterGeometrySpec(
        diameter_mm=0.25 * INCH_MM,
        flutes=4,
        helix_deg=30.0,
        slot_ipt_factor=0.80,
        stickout_mm=16.0,
        shank_diameter_mm=0.25 * INCH_MM,
    )
    stock = STOCK_MATERIALS["steel_1018"]
    tool = TOOL_MATERIALS["carbide_uncoated"]

    assert lookup_endmill_reference_ipt_in(0.25) == pytest.approx(0.00130, rel=1e-12)
    assert lookup_endmill_reference_ipt_in(0.50) == pytest.approx(0.00285, rel=1e-12)

    # Non-slotting WOC = 0.1 D -> slot_factor = 1.0
    woc_light_mm = 0.10 * cutter.diameter_mm
    fz_base_in, slot_fac = compute_nominal_base_ipt_in(stock, tool, cutter, woc_light_mm)
    assert slot_fac == pytest.approx(1.0, rel=1e-12)
    assert fz_base_in == pytest.approx(0.001258947368, rel=1e-9)

    r_thin = radial_chip_thinning_multiplier(woc_light_mm, cutter.diameter_mm)
    assert r_thin == pytest.approx(1.6666666667, rel=1e-9)
    assert fz_base_in * r_thin == pytest.approx(0.002098245614, rel=1e-9)

    # Full slotting WOC = 1.0 D -> slot_ipt_factor = 0.80
    fz_slot_in, slot_full = compute_nominal_base_ipt_in(
        stock, tool, cutter, cutter.diameter_mm
    )
    assert slot_full == pytest.approx(0.80, rel=1e-12)
    assert fz_slot_in == pytest.approx(0.001007157895, rel=1e-9)


def test_aggressiveness_material_and_tool_scaling() -> None:
    """Verify cutting speed responds monotonically to aggressiveness, material, and tool."""
    cutter = CutterGeometrySpec(diameter_mm=5.0, flutes=3, stickout_mm=14.0)
    machine = MachineEnvelopeSpec(max_rpm=8000.0, max_feed_mm_s=60.0)

    r_low = compute_feeds_and_speeds(
        "al_6061_t6", "carbide_coated", cutter, machine, doc_mm=3.0, woc_mm=1.75, aggressiveness=0.5
    )
    r_nom = compute_feeds_and_speeds(
        "al_6061_t6", "carbide_coated", cutter, machine, doc_mm=3.0, woc_mm=1.75, aggressiveness=1.0
    )
    r_high = compute_feeds_and_speeds(
        "al_6061_t6", "carbide_coated", cutter, machine, doc_mm=3.0, woc_mm=1.75, aggressiveness=1.5
    )
    assert r_low.feed_mm_s < r_nom.feed_mm_s < r_high.feed_mm_s
    assert 0.025 <= r_nom.chip_thickness_mm <= 0.055

    # Harder / tougher alloys command lower RPM and chip load than 6061-T6
    r_steel = compute_feeds_and_speeds(
        "steel_1018", "carbide_coated", cutter, machine, doc_mm=3.0, woc_mm=1.0, aggressiveness=1.0
    )
    r_ti = compute_feeds_and_speeds(
        "ti_6al4v", "carbide_coated", cutter, machine, doc_mm=3.0, woc_mm=0.8, aggressiveness=1.0
    )
    assert r_ti.spindle_rpm < r_steel.spindle_rpm <= r_nom.spindle_rpm
    assert r_ti.chip_thickness_mm < r_steel.chip_thickness_mm < r_nom.chip_thickness_mm

    # HSS runs slower than Coated Carbide on the same steel workpiece
    r_hss = compute_feeds_and_speeds(
        "steel_1018", "hss_m2", cutter, machine, doc_mm=3.0, woc_mm=1.0, aggressiveness=1.0
    )
    assert r_hss.spindle_rpm < r_steel.spindle_rpm
    assert r_hss.feed_mm_s < r_steel.feed_mm_s


def test_deflection_and_doc_load_constraints_bind_under_heavy_stickout() -> None:
    """Deep axial DOC + long stickout should trigger Tool Deflection clamping."""
    long_cutter = CutterGeometrySpec(diameter_mm=5.0, flutes=3, flute_length_mm=20.0, stickout_mm=32.0)
    res = compute_feeds_and_speeds(
        "steel_4140",
        "carbide_coated",
        long_cutter,
        doc_mm=6.0,
        woc_mm=3.5,
        aggressiveness=1.6,
    )
    assert res.constraint_feed_scale < 1.0
    assert res.active_limit == "Tool Deflection"
    assert res.deflection_mm <= res.max_deflection_mm

    catalog = export_feeds_speeds_catalog_json()
    assert len(catalog["stock_materials"]) == 8
    assert len(catalog["tool_materials"]) == 4


def test_python_and_viewer_js_feeds_speeds_exact_parity() -> None:
    """Verify exact numerical parity between Python cam/feeds_and_speeds.py and viewer/app.js."""
    import json
    import shutil
    import subprocess
    from dataclasses import asdict
    from pathlib import Path

    if shutil.which("node") is None:
        pytest.skip("Node.js not available")

    app_js = Path(__file__).resolve().parents[1] / "viewer" / "app.js"
    probes = [
        ("al_6061_t6", "carbide_coated", 3.0, 0.60, 1.00, 14.0),
        ("al_6061_t6", "carbide_coated", 3.0, 5.00, 1.00, 14.0),
        ("delrin_pom", "carbide_uncoated", 4.0, 2.25, 1.65, 14.0),
        ("steel_1018", "cobalt_m42", 2.5, 1.20, 0.85, 14.0),
        ("ss_304", "hss_m2", 2.0, 0.75, 0.60, 14.0),
        ("ti_6al4v", "carbide_coated", 2.5, 0.50, 1.25, 14.0),
        ("steel_4140", "carbide_coated", 6.0, 3.50, 1.60, 32.0),
    ]

    js_script = """
    const fsMod = require(process.argv[1]);
    const probes = JSON.parse(process.argv[2]);
    const results = probes.map((p) =>
      fsMod.computeFeedsAndSpeeds({
        stock_material_id: p.stock,
        tool_material_id: p.tool,
        cutter: p.cutter,
        doc_mm: p.doc_mm,
        woc_mm: p.woc_mm,
        aggressiveness: p.aggr,
      })
    );
    const samplePlan = {
      segments: [
        { kind: "cut", engaged: false, hex_mm: 0.0, feed_mm_s: 35.0, start: [0, 0, 0], end: [5, 0, 0] },
        { kind: "cut", engaged: true, hex_mm: 0.024, feed_mm_s: 14.0, start: [5, 0, 0], end: [10, 0, 0] },
        { kind: "cut", engaged: true, hex_mm: 0.041, feed_mm_s: 19.0, start: [10, 0, 0], end: [15, 0, 0] },
      ],
    };
    const tight = fsMod.computeTightColorRange([samplePlan], "chipload");
    const tubeThin = fsMod.buildThickPathTubeGeometry(samplePlan.segments, false, 0.12, () => [1, 0, 0]);
    const tubeWide = fsMod.buildThickPathTubeGeometry(samplePlan.segments, false, 0.36, () => [1, 0, 0]);
    console.log(JSON.stringify({
      results,
      tight,
      thinMaxY: Math.max(...Array.from(tubeThin.positions.filter((_, i) => i % 3 === 1))),
      wideMaxY: Math.max(...Array.from(tubeWide.positions.filter((_, i) => i % 3 === 1))),
      totalVerts: tubeWide.totalVerts,
    }));
    """

    req_payload = []
    py_results = []
    for stock_id, tool_id, doc_mm, woc_mm, aggr, stickout_mm in probes:
        cutter = CutterGeometrySpec(
            diameter_mm=5.0,
            flutes=3,
            flute_length_mm=max(12.0, doc_mm + 4.0),
            stickout_mm=stickout_mm,
        )
        py_results.append(
            compute_feeds_and_speeds(
                stock_id, tool_id, cutter=cutter, doc_mm=doc_mm, woc_mm=woc_mm, aggressiveness=aggr
            )
        )
        req_payload.append(
            {
                "stock": stock_id,
                "tool": tool_id,
                "cutter": asdict(cutter),
                "doc_mm": doc_mm,
                "woc_mm": woc_mm,
                "aggr": aggr,
            }
        )

    proc = subprocess.run(
        ["node", "-e", js_script, str(app_js), json.dumps(req_payload)],
        check=True,
        capture_output=True,
        text=True,
    )
    js_out = json.loads(proc.stdout)

    for py_r, js_r in zip(py_results, js_out["results"]):
        assert js_r["spindle_rpm"] == pytest.approx(py_r.spindle_rpm, abs=1e-4)
        assert js_r["feed_mm_s"] == pytest.approx(py_r.feed_mm_s, abs=1e-4)
        assert js_r["chip_thickness_mm"] == pytest.approx(py_r.chip_thickness_mm, abs=1e-6)
        assert js_r["deflection_mm"] == pytest.approx(py_r.deflection_mm, abs=1e-5)
        assert js_r["active_limit"] == py_r.active_limit

    # Tight color range excludes unengaged 0.0 air segments so [0.024, 0.041] spans [0, 1]
    assert js_out["tight"]["min"] == pytest.approx(0.024, abs=1e-6)
    assert js_out["tight"]["max"] == pytest.approx(0.041, abs=1e-6)
    assert js_out["tight"]["constant"] is False
    # Thick 3D path tube scales physical width 3x from 0.12mm to 0.36mm
    assert js_out["totalVerts"] == 3 * 24
    assert js_out["wideMaxY"] == pytest.approx(3.0 * js_out["thinMaxY"], rel=1e-5)

