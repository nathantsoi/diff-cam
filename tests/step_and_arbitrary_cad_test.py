# SPDX-License-Identifier: MIT
"""End-to-end unit, physics, and security tests for Arbitrary STEP/STL CAD, Tool Cribs,
6-Dialect Multi-Tool G-code, 3D Tool/Holder Visualization Metadata, and Interactive Upload Server.
"""

from __future__ import annotations

import base64
import http.client
import http.cookies
import json
import os
import stat
import sys
import threading
from pathlib import Path
from typing import Dict, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
EXAMPLE_PARTS_DIR = WORKSPACE_ROOT / "example_parts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cam.multi_dialect_post import SUPPORTED_MULTI_DIALECTS
from cam.step_brep import load_step_brep
from cam.tamp_planner import execute_tamp_pipeline, export_tamp_result_to_viewer_bundle
from cam.tool_library import DEFAULT_TOOL_LIBRARY
from scripts.serve_simulator import SlidingWindowRateLimiter, SimulatorHTTPServer


def test_step_brep_ingests_titan_prismatic_op2_freeform_and_threads() -> None:
    m1 = load_step_brep(EXAMPLE_PARTS_DIR / "Titan-M1.step")
    assert m1.source_unit == "inch"
    assert m1.oriented_triangles_mm.shape[0] > 100
    dx1 = float(m1.oriented_bounds_max_mm[0] - m1.oriented_bounds_min_mm[0])
    dy1 = float(m1.oriented_bounds_max_mm[1] - m1.oriented_bounds_min_mm[1])
    dz1 = float(m1.oriented_bounds_max_mm[2] - m1.oriented_bounds_min_mm[2])
    assert dx1 == pytest.approx(101.6, abs=1.5)
    assert dy1 == pytest.approx(48.26, abs=1.5)
    assert dz1 == pytest.approx(19.05, abs=1.0)
    assert len(m1.op1_holes) >= 5
    assert any(h.thread_spec is not None for h in m1.op1_holes)

    m4 = load_step_brep(EXAMPLE_PARTS_DIR / "Titan-M4.step")
    assert m4.requires_op2_flip is True

    m9 = load_step_brep(EXAMPLE_PARTS_DIR / "Titan-M9.step")
    assert len(m9.op1_freeform) >= 1

    m10 = load_step_brep(EXAMPLE_PARTS_DIR / "Titan-M10.step")
    assert len(m10.op1_freeform) + len(m10.op2_freeform) >= 1
    assert len(m10.op1_holes) + len(m10.op2_holes) >= 1


def test_tamp_us_vs_metric_library_and_restricted_tool_crib() -> None:
    # 1. US Imperial library -> G20 inch G-code + strictly US inch catalog tools
    res_us = execute_tamp_pipeline(
        EXAMPLE_PARTS_DIR / "Titan-M1.step",
        part_id="test-m1-us",
        title="TITAN-1M US",
        unit_system="inches",
        opt_iters_per_op=8,
    )
    assert res_us.safety_pass is True
    assert res_us.opt_chipload_total_time_s < res_us.naive_total_time_s
    for op in res_us.operations:
        if op.tool_record.entity_type != "probe":
            assert op.tool_record.unit_system == "inches"
    assert "G20" in res_us.gcode_by_dialect["haas_ngc"]["text"]

    # 2. Metric library -> G21 mm G-code + strictly Metric catalog tools
    res_mm = execute_tamp_pipeline(
        EXAMPLE_PARTS_DIR / "Titan-M1.step",
        part_id="test-m1-mm",
        title="TITAN-1M Metric",
        unit_system="millimeters",
        opt_iters_per_op=8,
    )
    assert res_mm.safety_pass is True
    for op in res_mm.operations:
        if op.tool_record.entity_type != "probe":
            assert op.tool_record.unit_system == "millimeters"
    assert "G21" in res_mm.gcode_by_dialect["haas_ngc"]["text"]

    # 3. Restricted custom 3-tool shop crib (no twist drills!) -> strictly uses allowed_tool_ids
    mill_10 = next(
        t.tool_id
        for t in DEFAULT_TOOL_LIBRARY.tools
        if t.unit_system == "millimeters" and t.entity_type == "flat end mill" and 9.5 <= t.diameter_mm <= 10.5
    )
    mill_4 = next(
        t.tool_id
        for t in DEFAULT_TOOL_LIBRARY.tools
        if t.unit_system == "millimeters" and t.entity_type == "flat end mill" and 3.8 <= t.diameter_mm <= 4.2
    )
    chamfer_6 = next(
        t.tool_id
        for t in DEFAULT_TOOL_LIBRARY.tools
        if t.unit_system == "millimeters" and t.entity_type in ("chamfer mill", "spot drill")
    )
    allowed_crib = [mill_10, mill_4, chamfer_6]
    res_crib = execute_tamp_pipeline(
        EXAMPLE_PARTS_DIR / "Titan-M1.step",
        part_id="test-m1-crib",
        title="TITAN-1M Restricted Crib",
        unit_system="millimeters",
        allowed_tool_ids=allowed_crib,
        enable_threading=False,
        opt_iters_per_op=8,
    )
    assert res_crib.safety_pass is True
    used_ids = {op.tool_record.tool_id for op in res_crib.operations}
    assert used_ids.issubset(set(allowed_crib))
    assert any(op.op_kind == "helical_bore" for op in res_crib.operations)


def test_probing_dual_setup_6_dialects_and_3d_ball_scallop() -> None:
    # Dual-setup G54 + M00 Flip + G55 + T99 WIPS probing across all 6 CNC controller dialects
    res_op2 = execute_tamp_pipeline(
        EXAMPLE_PARTS_DIR / "Titan-M4.step",
        part_id="test-m4-dual",
        title="TITAN-4M Op1+Op2+Probe",
        unit_system="inches",
        setup_mode="dual_op1_op2",
        enable_probing=True,
        opt_iters_per_op=8,
    )
    assert res_op2.safety_pass is True
    assert any(op.op_kind == "wips_probe" and op.tool_number == 99 for op in res_op2.operations)
    assert any(op.setup_index == 2 and op.work_offset == "G55" for op in res_op2.operations)
    assert "99" in res_op2.tools_table
    assert res_op2.tools_table["99"]["entity_type"] == "probe"

    opt_segs = res_op2.viewer_variants["chipload_ste"]["segments"]
    for dialect in SUPPORTED_MULTI_DIALECTS:
        assert dialect in res_op2.gcode_by_dialect
        gdoc = res_op2.gcode_by_dialect[dialect]
        txt = gdoc["text"]
        line_count = len(txt.splitlines())
        seg_map = gdoc["viewer_segment_to_line"]
        assert "G54" in txt
        assert "G55" in txt
        assert "M00" in txt or "M0" in txt
        assert len(seg_map) == len(opt_segs)
        assert all(1 <= ln <= line_count for ln in seg_map)

    # Haas WIPS probing uses Renishaw Protected Positioning / Z-surface macros without flood coolant
    haas_txt = res_op2.gcode_by_dialect["haas_ngc"]["text"]
    assert "G65 P9832" in haas_txt
    assert "G65 P9995" in haas_txt or "G65 P9811" in haas_txt
    # RS274 uses G38.2 touch probe
    assert "G38.2" in res_op2.gcode_by_dialect["rs274"]["text"]

    # 3D sculptured B-spline ball-endmill scallop surfacing on Titan-M9
    res_3d = execute_tamp_pipeline(
        EXAMPLE_PARTS_DIR / "Titan-M9.step",
        part_id="test-m9-3d",
        title="TITAN-9M 3D Scallop",
        unit_system="inches",
        opt_iters_per_op=8,
    )
    assert res_3d.safety_pass is True
    assert any(op.op_kind == "3d_ball_scallop" for op in res_3d.operations)
    assert any(t["entity_type"] == "ball end mill" for t in res_3d.tools_table.values())


def _start_test_server(tmp_path: Path) -> Tuple[SimulatorHTTPServer, int, threading.Thread]:
    viewer_root = PROJECT_ROOT / "viewer"
    run_dir = tmp_path / "run_srv"
    sim_dir = run_dir / "simulator_viewer"
    (sim_dir / "cases").mkdir(parents=True)
    (sim_dir / "meshes").mkdir(parents=True)
    for sub in ("rustcnc_projects", "camotics_projects", "haas_ngc_packets", "tamp_shop_packets"):
        (run_dir / sub).mkdir(parents=True)
    (sim_dir / "manifest.json").write_text(
        json.dumps({"format": "diffcam_simulator_viewer_v2", "cases": []}),
        encoding="utf-8",
    )
    srv = SimulatorHTTPServer(
        ("127.0.0.1", 0),
        viewer_root=viewer_root,
        run_dir=run_dir,
        rate_limiter=SlidingWindowRateLimiter(max_requests=500, window_s=10.0),
        compute_rate_limiter=SlidingWindowRateLimiter(max_requests=50, window_s=60.0),
        example_parts_dir=EXAMPLE_PARTS_DIR,
    )
    port = int(srv.server_address[1])
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, port, th


def _get_session_and_csrf(port: int, extra_headers: Dict[str, str] | None = None) -> Tuple[str, str, Dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        conn.request("GET", "/api/session", headers=extra_headers or {})
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        set_cookie = resp.getheader("Set-Cookie") or ""
        assert resp.status == 200
        assert "HttpOnly" in set_cookie
        assert "SameSite=Strict" in set_cookie
        jar = http.cookies.SimpleCookie()
        jar.load(set_cookie)
        cookie_pair = next(f"{k}={m.value}" for k, m in jar.items())
        return cookie_pair, body["csrf_token"], body
    finally:
        conn.close()


def test_interactive_upload_replan_and_csrf_security(tmp_path: Path) -> None:
    srv, port, th = _start_test_server(tmp_path)
    try:
        # 1. Session endpoint + HTTPS __Host- cookie upgrade verification
        cookie_hdr, csrf_tok, sess_payload = _get_session_and_csrf(port)
        assert len(csrf_tok) == 64
        assert len(sess_payload["tool_catalog"]["tools"]) >= 600
        assert len(sess_payload["example_parts"]) == 10

        tls_cookie_hdr, _, _ = _get_session_and_csrf(port, {"X-Forwarded-Proto": "https"})
        assert tls_cookie_hdr.startswith("__Host-diffcam_sid=")

        # 2. Method allow-lists: GET /api/machine-part -> 405; POST /data/manifest.json -> 405
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
        conn.request("GET", "/api/machine-part")
        r_m1 = conn.getresponse()
        r_m1.read()
        assert r_m1.status == 405
        assert r_m1.getheader("Allow") == "POST"

        conn.request("POST", "/data/manifest.json", body=b"{}", headers={"Content-Type": "application/json"})
        r_m2 = conn.getresponse()
        r_m2.read()
        assert r_m2.status == 405
        assert r_m2.getheader("Allow") == "GET, HEAD"
        conn.close()

        def _post_machine(payload: Dict, *, headers_override: Dict[str, str] | None = None) -> Tuple[int, Dict]:
            raw = json.dumps(payload).encode("utf-8")
            hdrs = {
                "Content-Type": "application/json",
                "Cookie": cookie_hdr,
                "X-CSRF-Token": csrf_tok,
                "Origin": f"http://127.0.0.1:{port}",
            }
            if headers_override is not None:
                hdrs.update(headers_override)
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            try:
                c.request("POST", "/api/machine-part", body=raw, headers=hdrs)
                resp = c.getresponse()
                data = json.loads(resp.read().decode("utf-8"))
                return resp.status, data
            finally:
                c.close()

        # 3. Missing/wrong CSRF token & Cross-Origin Origin header -> 403 Forbidden
        st_bad_csrf, _ = _post_machine({"source_mode": "example", "example_id": "titan-m1"}, headers_override={"X-CSRF-Token": "0" * 64})
        assert st_bad_csrf == 403

        st_cross_origin, _ = _post_machine(
            {"source_mode": "example", "example_id": "titan-m1"},
            headers_override={"Origin": "https://evil.example"},
        )
        assert st_cross_origin == 403

        # 4. Spoofed STEP magic bytes & disallowed extension -> 400 Bad Request
        spoof_b64 = base64.b64encode(b"<script>alert(1)</script>").decode("ascii")
        st_spoof, _ = _post_machine(
            {"source_mode": "upload", "filename": "evil.step", "cad_base64": spoof_b64}
        )
        assert st_spoof == 400

        st_ext, _ = _post_machine(
            {"source_mode": "upload", "filename": "payload.svg", "cad_base64": spoof_b64}
        )
        assert st_ext == 400

        # 5. Valid CAD Upload (.step) -> quarantined outside web roots with 0o600 permissions + full plan generated
        step_bytes = (EXAMPLE_PARTS_DIR / "Titan-M2.step").read_bytes()
        step_b64 = base64.b64encode(step_bytes).decode("ascii")
        st_up, res_up = _post_machine(
            {
                "source_mode": "upload",
                "filename": "Titan-M2.step",
                "cad_base64": step_b64,
                "part_title": "Uploaded TITAN-2M",
                "unit_system": "inches",
                "crib_mode": "all_unit",
                "stock_material_id": "al_6061_t6",
                "tool_material_id": "carbide_coated",
                "controller_dialect": "fanuc",
                "enable_probing": True,
                "opt_iters": 6,
            }
        )
        assert st_up == 200, res_up
        assert res_up["ok"] is True
        case_entry = res_up["case_entry"]
        new_cid = case_entry["case_id"]

        # Verify quarantine directory & file permissions (0o700 dir, 0o600 UUID file, unroutable over HTTP)
        q_files = list(srv.quarantine_root.iterdir())
        assert len(q_files) == 1
        q_mode = stat.S_IMODE(os.stat(q_files[0]).st_mode)
        assert q_mode == 0o600
        assert "Titan-M2" not in q_files[0].name

        c_check = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c_check.request("GET", f"/uploads_quarantine/{q_files[0].name}")
        r_q = c_check.getresponse()
        r_q.read()
        assert r_q.status == 404

        # Fetch newly generated case JSON via HTTP
        case_url = res_up["case_json_url"] if res_up["case_json_url"].startswith("/") else f"/{res_up['case_json_url']}"
        c_check.request("GET", case_url)
        r_case = c_check.getresponse()
        case_doc = json.loads(r_case.read().decode("utf-8"))
        c_check.close()
        assert r_case.status == 200
        assert "tools_table" in case_doc
        opt_var = next(v for v in case_doc["variants"] if v["id"] == "chipload_ste")
        assert "fanuc" in opt_var["gcode_by_dialect"]

        # 6. Re-plan same case with Metric Starter Shop Crib!
        cribs_val = sess_payload["tool_catalog"]["starter_cribs"]
        cribs_iter = cribs_val.values() if isinstance(cribs_val, dict) else cribs_val
        metric_crib = next(c["tool_ids"] for c in cribs_iter if c["unit_system"] == "millimeters")
        st_re, res_re = _post_machine(
            {
                "source_mode": "replan_case",
                "case_id": new_cid,
                "source_cad_id": case_entry["source_cad_id"],
                "unit_system": "millimeters",
                "crib_mode": "selected_crib",
                "allowed_tool_ids": metric_crib,
                "controller_dialect": "mazak",
                "opt_iters": 6,
            }
        )
        assert st_re == 200, res_re
        assert res_re["ok"] is True
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5.0)
