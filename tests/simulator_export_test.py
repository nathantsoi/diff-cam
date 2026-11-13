# SPDX-License-Identifier: MIT
"""Unit and security verification tests for RustCNC/CAMotics/WebGL simulator exports & server."""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import shutil
import struct
import subprocess
import sys
import threading
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks import contracts as C
from benchmarks.milling_suite import build_milling_suite
from scripts.export_simulator_bundles import export_all_simulator_bundles
from scripts.serve_simulator import SimulatorHTTPServer, validate_loopback_host
from verification.simulator_export import (
    RUSTCNC_CACHE_FORMAT,
    RUSTCNC_CAM_FORMAT,
    RUSTCNC_CONTAINER_FORMAT,
    RUSTCNC_DESIGN_FORMAT,
    RUSTCNC_TOOL_LIB_FORMAT,
    RUSTCNC_TOOLPATH_FORMAT,
    RUSTCNC_VIEW_FORMAT,
    build_target_part_stl_bytes,
    carve_resulting_stock_stl_bytes,
    compute_rustcnc_op_stamp_hex,
    export_camotics_project,
    export_rustcnc_project,
    narrow_uuid_u64,
    reconstruct_motion_plan,
    safe_resolve_under,
    serialize_plan_for_viewer,
)


@pytest.fixture(scope="module")
def sample_test_bundle():
    bundles = build_milling_suite(
        voxel_mm=0.5,
        timing_rule=C.TimingRule.BLENDED_ACCEL_BOUNDED,
    )
    return next(b for b in bundles if b.case.case_id == "pocket-test-00")


def test_safe_resolve_under_blocks_traversal_and_sibling_prefix(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sibling = tmp_path / "sandbox-evil"
    sandbox.mkdir()
    sibling.mkdir()
    (sibling / "secret.txt").write_text("nope", encoding="utf-8")

    ok = safe_resolve_under(sandbox, "nested/part.stl")
    assert str(ok).startswith(str(sandbox.resolve()) + "/")

    for bad in (
        "",
        "../sandbox-evil/secret.txt",
        "nested/../../sandbox-evil/secret.txt",
        "/etc/passwd",
        "nested\\evil.stl",
        "bad\x00name.stl",
        ".",
        "..",
    ):
        with pytest.raises(ValueError):
            safe_resolve_under(sandbox, bad)


def test_export_rustcnc_container_matches_rustcnc_cam_contract(
    tmp_path: Path, sample_test_bundle
) -> None:
    out_rustcnc = tmp_path / "pocket-test-00.rustcnc"
    ste_plan = reconstruct_motion_plan(sample_test_bundle, "STE-Continuation-Adam", None)
    nom_plan = reconstruct_motion_plan(sample_test_bundle, "Nominal-CAM", None)
    shac_plan = reconstruct_motion_plan(sample_test_bundle, "DiffPolicy-SHAC", None)

    meta = export_rustcnc_project(
        out_rustcnc,
        sample_test_bundle,
        [
            ("ste_adam", "DiffCAM STE-Adam", ste_plan),
            ("nominal_cam", "Nominal-CAM (@2)", nom_plan),
            ("shac_policy", "DiffPolicy-SHAC", shac_plan),
        ],
    )
    raw_zip = out_rustcnc.read_bytes()
    assert hashlib.sha256(raw_zip).hexdigest() == meta["sha256"]
    # RustCNC zip.rs rejects ZIP64 locator / EOCD records
    assert b"PK\x06\x06" not in raw_zip
    assert b"PK\x06\x07" not in raw_zip

    with zipfile.ZipFile(out_rustcnc, "r") as zf:
        names = zf.namelist()
        assert len(names) == len(set(names))
        for name in names:
            if name not in ("document.json", "container.json"):
                assert (
                    name == "cam.json"
                    or name.startswith("cam/")
                    or name.startswith("cache/")
                ), f"Disallowed RustCNC extension entry: {name}"

        container = json.loads(zf.read("container.json"))
        assert container == {"format": RUSTCNC_CONTAINER_FORMAT}

        doc_bytes = zf.read("document.json")
        doc = json.loads(doc_bytes)
        assert doc["format"] == RUSTCNC_DESIGN_FORMAT
        assert doc["units"] == "mm"
        # pocket-test-00 has BilletEnvelope + MachinedCavity + RetainedFeature1 + 4x(CornerBlock+CornerFillet)
        feature_types = [f["type"] for f in doc["features"]]
        assert "box" in feature_types
        assert "cylinder" in feature_types

        tools = json.loads(zf.read("cam/tools.json"))
        assert tools["format"] == RUSTCNC_TOOL_LIB_FORMAT
        assert len(tools["tools"]) == 1
        tool_uuid = uuid.UUID(tools["tools"][0]["id"])

        cam_bytes = zf.read("cam.json")
        cam_doc = json.loads(cam_bytes)
        assert cam_doc["format"] == RUSTCNC_CAM_FORMAT
        assert len(cam_doc["setups"]) == 3

        view_doc = json.loads(zf.read("cam/view.json"))
        assert view_doc["format"] == RUSTCNC_VIEW_FORMAT
        assert view_doc["active_setup"] == cam_doc["setups"][0]["id"]

        manifest = json.loads(zf.read("cache/cam/manifest.json"))
        assert manifest["format"] == RUSTCNC_CACHE_FORMAT
        assert len(manifest["entries"]) == 3

        expected_design_sha = hashlib.sha256(doc_bytes).hexdigest()
        expected_cam_sha = hashlib.sha256(cam_bytes).hexdigest()

        for setup, entry in zip(cam_doc["setups"], manifest["entries"]):
            stock_body = setup["stock"]["body"]
            assert stock_body["mode"] == "fixed_box"
            assert stock_body["x_mode"] == "model"
            assert stock_body["y_mode"] == "model"
            assert stock_body["z_mode"] == "model"
            assert stock_body["ground_at_model_origin"] is True
            assert "override_expr" in stock_body["x_offset"]
            assert "override_expr" in stock_body["z_offset"]

            op = setup["operations"][0]
            assert entry["id"] == op["id"]
            assert entry["inputs"]["design"] == expected_design_sha
            assert entry["inputs"]["cam"] == expected_cam_sha
            assert entry["inputs"]["op"] == compute_rustcnc_op_stamp_hex(setup, op)

            tp_part = entry["outputs"][0]
            tp_doc = json.loads(zf.read(tp_part))
            assert tp_doc["format"] == RUSTCNC_TOOLPATH_FORMAT
            assert tp_doc["op"] == op["id"]
            assert tp_doc["tool_id"] == narrow_uuid_u64(tool_uuid)
            move_recs = [r["Move"] for r in tp_doc["records"] if "Move" in r]
            assert len(move_recs) > 20


def test_export_camotics_and_watertight_binary_stl(tmp_path: Path, sample_test_bundle) -> None:
    cam_meta = export_camotics_project(
        tmp_path,
        sample_test_bundle,
        "ste_adam",
        sample_test_bundle.nominal_motion,
        resolution_mm=0.25,
    )
    proj = json.loads((tmp_path / cam_meta["project_file"]).read_text(encoding="utf-8"))
    nc_text = (tmp_path / cam_meta["nc_file"]).read_text(encoding="utf-8")
    assert proj["units"] == "metric"
    assert proj["files"] == [cam_meta["nc_file"]]
    assert "G10 L2 P1" in nc_text

    target_stl = build_target_part_stl_bytes(sample_test_bundle)
    stock_stl, source = carve_resulting_stock_stl_bytes(
        sample_test_bundle,
        sample_test_bundle.nominal_motion,
        use_camotics_docker=False,
    )
    assert source == "reference_voxel_0.25mm"

    for stl_bytes in (target_stl, stock_stl):
        count = struct.unpack_from("<I", stl_bytes, 80)[0]
        assert count > 0
        assert len(stl_bytes) == 84 + 50 * count
        # Verify edge manifold watertightness: every directed edge (u -> v) has matching (v -> u)
        stl_dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
        recs = np.frombuffer(stl_bytes, dtype=stl_dtype, count=count, offset=84)
        verts = np.round(recs["vertices"], 4)
        directed_edges = set()
        for tri in verts:
            for i in range(3):
                u = tuple(tri[i].tolist())
                v = tuple(tri[(i + 1) % 3].tolist())
                directed_edges.add((u, v))
        for u, v in directed_edges:
            assert (v, u) in directed_edges


def test_serialize_plan_for_viewer_includes_cwe_and_chipload_telemetry(sample_test_bundle) -> None:
    plan_json = serialize_plan_for_viewer(sample_test_bundle, sample_test_bundle.nominal_motion)
    eng = plan_json["engagement_summary"]
    assert eng["engaged_time_s"] > 0.0
    assert eng["air_cut_time_s"] > 0.0
    assert eng["rapid_time_s"] > 0.0
    assert eng["removed_volume_mm3"] > 1000.0
    assert eng["engaged_pct"] + eng["air_cut_pct"] + eng["rapid_pct"] == pytest.approx(100.0, abs=0.05)
    assert len(plan_json["segments"]) == len(sample_test_bundle.nominal_motion.segments)
    cut_segs = [s for s in plan_json["segments"] if s["kind"] == "cut" and s["engaged"]]
    assert len(cut_segs) > 10
    assert all(s["hex_mm"] > 0.0 and s["ae_pct"] > 0.0 for s in cut_segs)


def test_viewer_frontend_has_zero_unsafe_dom_or_storage_sinks() -> None:
    viewer_dir = PROJECT_ROOT / "viewer"
    html_src = (viewer_dir / "index.html").read_text(encoding="utf-8")
    js_src = (viewer_dir / "app.js").read_text(encoding="utf-8")

    # Strip JS comments before checking executable code for forbidden sinks
    js_code_only = re.sub(r"//.*", "", js_src)
    forbidden_patterns = (
        r"\binnerHTML\b",
        r"\bouterHTML\b",
        r"\binsertAdjacentHTML\b",
        r"\bdocument\.write\b",
        r"\beval\s*\(",
        r"\blocalStorage\b",
        r"\bsessionStorage\b",
        r"\balert\s*\(",
        r"\bconfirm\s*\(",
        r"\bprompt\s*\(",
        r"\.style\.",
    )
    for pat in forbidden_patterns:
        assert re.search(pat, js_code_only) is None, f"Forbidden sink matched in app.js: {pat}"
    assert "onclick=" not in html_src.lower()
    assert "style=" not in html_src.lower()
    for required_id in (
        'id="speed-policy-select"',
        'value="matched" selected="selected"',
        'id="aggressiveness-slider"',
        'id="stock-mat-select"',
        'id="tool-mat-select"',
        'id="path-width-slider"',
        'id="layout-sbs"',
        'id="chk-vise"',
        'id="op-bar"',
        'id="op-pills-strip"',
        'id="op-timeline-mount"',
        'value="operation"',
    ):
        assert required_id in html_src


def test_kurt_dx6_vise_mesh_and_multi_operation_js_helpers() -> None:
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("node runtime not installed")

    app_js = str((PROJECT_ROOT / "viewer" / "app.js").resolve())
    js_test = """
const assert = require("assert");
const app = require(process.argv[1]);

const dualCase = {
  case_id: "unit-dual-op",
  stock_min_mm: [-50.0, -25.0, -23.0],
  stock_max_mm: [50.0, 25.0, 0.0],
  pocket_min_mm: [-48.0, -23.0, -19.0],
  pocket_max_mm: [48.0, 23.0, -1.5],
  cad_metadata: { has_op2_flip: true, part_dims_mm: [99.0, 49.0, 19.0] },
  operations_table: [
    { op_index: 1, op_id: "s1_op1_face", op_kind: "facing", setup_index: 1, work_offset: "G54", tool_number: 1, tool_label: "Face Mill" },
    { op_index: 2, op_id: "s1_op2_rough", op_kind: "pocket_adaptive", setup_index: 1, work_offset: "G54", tool_number: 2, tool_label: "Rough Endmill" },
    { op_index: 3, op_id: "s2_op3_hat", op_kind: "op2_hat_facing", setup_index: 2, work_offset: "G55", tool_number: 1, tool_label: "Face Mill" },
  ],
};

// 1. Kurt DX6 workholding resolution + Setup 1 (G54 Hard Jaws + Parallels) & Setup 2 (G55 Soft Jaws) 3D meshes
const wh = app.resolveCaseWorkholding(dualCase);
assert.strictEqual(wh.vise_model, "kurt_dx6_6in");
assert.strictEqual(wh.jaw_width_mm, 152.4);
assert.strictEqual(wh.clamping_axis, "Y");

const mesh1 = app.buildKurtDx6ViseMesh(dualCase, 1);
const mesh2 = app.buildKurtDx6ViseMesh(dualCase, 2);
assert(mesh1.vertexCount >= 300, "Setup 1 DX6 vise should have rich 3D geometry");
assert(mesh2.vertexCount >= 300, "Setup 2 DX6 soft-jaw vise should have rich 3D geometry");
assert.strictEqual(mesh1.positions.length, mesh1.vertexCount * 3);
assert.strictEqual(mesh1.normals.length, mesh1.vertexCount * 3);
assert.strictEqual(mesh1.colors.length, mesh1.vertexCount * 3);
for (let i = 0; i < mesh1.positions.length; i++) {
  assert(Number.isFinite(mesh1.positions[i]), "non-finite position in mesh1");
  assert(Number.isFinite(mesh1.normals[i]), "non-finite normal in mesh1");
}
for (let i = 0; i < mesh2.positions.length; i++) {
  assert(Number.isFinite(mesh2.positions[i]), "non-finite position in mesh2");
}

// 2. Multi-operation extraction & selective multi-op ribbon filtering
const variant = {
  plan: {
    segments: [
      { id: 0, kind: "cut", start: [-45, -20, -1.5], end: [45, -20, -1.5], t0_s: 0.0, t1_s: 2.0, setup_index: 1, work_offset: "G54", tool_number: 1, op_id: "s1_op1_face", engaged: true },
      { id: 1, kind: "rapid", start: [45, -20, 5.0], end: [-30, 0, 5.0], t0_s: 2.0, t1_s: 2.5, setup_index: 1, work_offset: "G54", tool_number: 2, op_id: "s1_op2_rough", engaged: false },
      { id: 2, kind: "cut", start: [-30, 0, -8.0], end: [30, 0, -8.0], t0_s: 2.5, t1_s: 6.0, setup_index: 1, work_offset: "G54", tool_number: 2, op_id: "s1_op2_rough", engaged: true },
      { id: 3, kind: "cut", start: [-45, 0, -2.0], end: [45, 0, -2.0], t0_s: 6.0, t1_s: 9.0, setup_index: 2, work_offset: "G55", tool_number: 1, op_id: "s2_op3_hat", engaged: true },
    ],
  },
};
const ops = app.deriveCaseOperations(dualCase, variant);
assert.strictEqual(ops.length, 3);
assert.strictEqual(ops[0].opId, "s1_op1_face");
assert.strictEqual(ops[2].setupIndex, 2);
assert.strictEqual(ops[2].workOffset, "G55");

// Op1 and Op3 share T1, but color-by-operation assigns distinct RGB colors!
const opColorMap = new Map(ops.map((o) => [o.key, o.colorIdx]));
const cOp1 = app.cutSegmentColorRgb(variant.plan.segments[0], "operation", null, opColorMap);
const cOp3 = app.cutSegmentColorRgb(variant.plan.segments[3], "operation", null, opColorMap);
assert.notDeepStrictEqual(cOp1, cOp3);

// Multi-select filter: select Op 1 + Op 3 simultaneously (skipping Op 2)
const selected = new Set(["s1_op1_face", "s2_op3_hat"]);
const allCuts = app.buildThickPathTubeGeometry(variant.plan.segments, false, 0.28, () => [1, 1, 1], -23.0);
const filteredCuts = app.buildThickPathTubeGeometry(
  variant.plan.segments,
  false,
  0.28,
  () => [1, 1, 1],
  -23.0,
  (s) => selected.has(s.op_id)
);
assert.deepStrictEqual(allCuts.segIndices, [0, 2, 3]);
assert.deepStrictEqual(filteredCuts.segIndices, [0, 3]);

// 3. Haas WIPS-R (OMP40-2 + OTS) Setup Sheet & Combined Per-Tool Schedule synthesis
const sheet = app.resolveCaseSetupSheet(dualCase, variant);
assert.strictEqual(sheet.schema, "diffcam-setup-sheet/1");
assert.strictEqual(sheet.probe_package.profile_id, "haas_wips_r");
assert.strictEqual(sheet.setups.length, 2, "Dual-op case must generate G54 and G55 setup sheets");
assert.strictEqual(sheet.setups[0].work_offset, "G54");
assert.strictEqual(sheet.setups[1].work_offset, "G55");
assert(sheet.setups[0].side_touch_z_mm > sheet.setups[0].jaw_top_z_mm);
assert(sheet.setups[1].side_touch_z_mm > sheet.setups[1].jaw_top_z_mm);
assert.strictEqual(sheet.tools_schedule[0].tool_number, 99);
const md = app.formatSetupSheetMarkdown(sheet);
assert(md.includes("OMP40-2") && md.includes("G54") && md.includes("G55") && md.includes("OTS"));
"""
    proc = subprocess.run(
        [node_bin, "-e", js_test, app_js],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node JS helper test failed:\n{proc.stderr}\n{proc.stdout}"


def test_loopback_server_security_headers_method_allowlist_and_traversal(
    tmp_path: Path, sample_test_bundle
) -> None:
    assert validate_loopback_host("127.0.0.1") == "127.0.0.1"
    assert validate_loopback_host("localhost") == "localhost"
    assert validate_loopback_host("::1") == "::1"
    for forbidden_host in ("0.0.0.0", "::", "192.168.1.5", "0.0.0.0:8000", "example.com"):
        with pytest.raises(ValueError):
            validate_loopback_host(forbidden_host)

    # Create a minimal synthetic run directory and export bundles
    run_dir = tmp_path / "mini_run"
    run_dir.mkdir()
    nom_t = float(sum(s.duration_s for s in sample_test_bundle.nominal_motion.segments))
    fake_exp2 = {
        "schema": "diffcam-pub-exp2-main/1",
        "tasks": [
            {
                "case_id": sample_test_bundle.case.case_id,
                "family": sample_test_bundle.case.family,
                "split": "test",
                "methods": {
                    m: [
                        {
                            "method": m,
                            "seed": 0,
                            "cycle_time_s": nom_t,
                            "cycle_time_reduction_pct": 0.0,
                            "peak_mrr_mm3_s": 42.0,
                            "air_fraction": 0.2,
                            "feasible": True,
                            "wall_time_s": 0.5,
                            "best_theta": None,
                        }
                    ]
                    for m in ("Nominal-CAM", "STE-Continuation-Adam", "DiffPolicy-SHAC")
                },
            }
        ],
    }
    (run_dir / "exp2_main_benchmark.json").write_text(json.dumps(fake_exp2), encoding="utf-8")
    export_all_simulator_bundles(run_dir, use_camotics_docker=False)

    srv = SimulatorHTTPServer(
        ("127.0.0.1", 0),
        viewer_root=PROJECT_ROOT / "viewer",
        run_dir=run_dir,
    )
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)

        # 1. Valid GET / checks CSP and security headers
        conn.request("GET", "/")
        resp = conn.getresponse()
        body = resp.read()
        assert resp.status == 200
        assert b"DiffCAM Manufacture Simulator" in body
        csp = resp.getheader("Content-Security-Policy") or ""
        assert "default-src 'self'" in csp
        assert "unsafe-inline" not in csp
        assert "unsafe-eval" not in csp
        assert resp.getheader("X-Content-Type-Options") == "nosniff"
        assert resp.getheader("X-Frame-Options") == "DENY"
        assert resp.getheader("Cache-Control") == "no-store"

        # 2. Download .rustcnc sets attachment Content-Disposition
        cid = sample_test_bundle.case.case_id
        conn.request("GET", f"/downloads/rustcnc/{cid}.rustcnc")
        r_dl = conn.getresponse()
        dl_bytes = r_dl.read()
        assert r_dl.status == 200
        assert len(dl_bytes) > 1024
        assert f'filename="{cid}.rustcnc"' in (r_dl.getheader("Content-Disposition") or "")

        # 3. Non-GET/HEAD verbs rejected with 405 Method Not Allowed
        for verb in ("POST", "PUT", "DELETE", "TRACE"):
            conn.request(verb, "/data/manifest.json")
            r_m = conn.getresponse()
            _ = r_m.read()
            assert r_m.status == 405
            assert r_m.getheader("Allow") == "GET, HEAD"

        # 4. Path traversal probes rejected
        for probe in (
            "/../../etc/passwd",
            "/data/cases/..%2f..%2fexp2_main_benchmark.json",
            "/downloads/rustcnc/..\\secret.rustcnc",
        ):
            conn.request("GET", probe)
            r_p = conn.getresponse()
            _ = r_p.read()
            assert r_p.status in (400, 403, 404)

        conn.close()
    finally:
        srv.shutdown()
        srv.server_close()
