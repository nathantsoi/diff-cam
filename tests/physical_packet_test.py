"""T6.1 packet tests.

The retained T2.2b nominal supplies the real typed task/motion/simulation and
posted/independent evaluation evidence. The machine, material, fixture-identity
and operator-review setup fields below are explicitly synthetic software-test
records, never a claim about an available physical machine. No NC is executed
by these tests. Final-packet tests exercise refusal paths only; no protocol
freeze or physical trial is claimed here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from benchmarks import contracts as C
from scripts import export_physical_packets as P


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETENTION = Path("runs/retained/pilot-pocket-20261109-t22b-exact-exclusion")


def retained_run() -> Path:
    candidates = [
        Path(os.environ["DIFFCAM_RETAINED_NOMINAL"]) if os.environ.get("DIFFCAM_RETAINED_NOMINAL") else None,
        PROJECT_ROOT / RETENTION,
        PROJECT_ROOT.parent.parent / "diffcam" / RETENTION,
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "task.json").is_file() and (candidate / "source_stock.npy").is_file():
            return candidate
    raise AssertionError("BLOCKED: retained real T2.2b evidence is required, not a synthetic substitute; set DIFFCAM_RETAINED_NOMINAL")


RETAINED = retained_run()


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rehash(root: Path) -> None:
    """Refresh only synthetic mutation fixtures, preserving RunRecord/NC identity."""
    run = C.loads((root / "run.json").read_text(encoding="utf-8"))
    (root / "run.json").write_bytes(C.dumps(run).encode("utf-8"))
    hashes = {}
    for name in sorted(set(run.artifacts) | {"run.json"}):
        hashes[name] = digest((root / name).read_bytes())
    write_json(root / "hashes.json", hashes)
    nominal = load(root / "nominal_result.json")
    nominal["hashes"] = hashes
    write_json(root / "nominal_result.json", nominal)


def clone_run(tmp_path: Path, name: str = "source") -> Path:
    target = tmp_path / name
    shutil.copytree(RETAINED, target)
    return target


def valid_setup() -> dict:
    """Synthetic setup metadata; geometry comes from the real retained task."""
    task = C.loads((RETAINED / "task.json").read_text(encoding="utf-8"))
    plain = lambda value: json.loads(C.canonical_json(value))
    return {
        "schema_version": P.SETUP_SCHEMA,
        "machine": {
            "identifier": "SYNTHETIC-REVIEW-SETUP-NOT-A-PHYSICAL-MACHINE",
            "manufacturer": "synthetic software-review fixture",
            "model": "no actual machine asserted; schema exercise only",
            "record_source": "tests/physical_packet_test.py (synthetic setup only)",
            "controllers": {"rs274": "synthetic-review-controller", "haas": "synthetic-review-controller"},
        },
        "tool": {
            "setup_record_id": "synthetic-tool-identity-from-retained-task",
            "assembly": plain(task.tool),
            "declared_stickout_mm": task.tool.stickout_mm,
            "measurement": {"state": "declared_not_measured", "operator_record_id": None},
        },
        "work_offset": {
            "identifier": task.initial_state.work_offset,
            "transform": plain(task.work_to_machine),
            "initial_machine_position_mm": list(task.initial_state.position_mm),
        },
        "fixtures": {
            "geometry": plain(task.fixtures),
            "identifiers": [],
            "complete_fixture_representation": False,
            "unrepresented_geometry_note": "Retained task has no fixture geometry; actual clamping and its clearance remain an operator-review item, not invented simulation geometry.",
        },
        "floor": {
            "kind": "retained_stock_below_cut",
            "basis": "Retained analytic part pocket_min_mm[2] - stock box_min_mm[2], not a generic through-hole threshold.",
            "stock_bottom_mm": task.original_part.params["box_min_mm"][2],
            "cut_floor_mm": task.original_part.params["pocket_min_mm"][2],
            "remaining_thickness_mm": task.original_part.params["pocket_min_mm"][2] - task.original_part.params["box_min_mm"][2],
        },
        "material": {
            "name": "synthetic stock record, no measured force",
            "alloy_or_grade": "synthetic fixture / not a materials certificate",
            "stock_record": "metadata-only test; retained source geometry is real",
        },
        "process": {
            "spindle_rpm": 5000,
            "spindle_direction": "clockwise",
            "coolant": "synthetic process setting; operator must review",
            "motion_mode": "exact_stop_constant_speed",
            "speed_clipping": False,
            "length_unit": "mm",
        },
    }


def pending_protocol() -> dict:
    return {
        "schema_version": P.PROTOCOL_SCHEMA,
        "protocol_id": "T4.2-pending-physical-packet-software-fixture",
        "freeze_state": "pending",
        "frozen_record": None,
    }


def export(output: Path, *, source: Path = RETAINED, setup=None, protocol=None,
           final: bool = False, refinement_link=None) -> dict:
    return P.export_packet(
        selections={"baseline": source}, setup=valid_setup() if setup is None else setup,
        protocol=pending_protocol() if protocol is None else protocol,
        selected_dialect="haas", output=output, final_for_protocol=final,
        refinement_links={} if refinement_link is None else {"baseline": refinement_link},
    )


def test_known_retained_packet_reproduces_byte_identically_with_explicit_limits(tmp_path):
    before = {name: digest((RETAINED / name).read_bytes()) for name in ("task.json", "haas.nc", "source_stock.npy")}
    first_dir, second_dir = tmp_path / "one", tmp_path / "two"
    first = export(first_dir)
    second = export(second_dir)
    assert first == second
    one = {p.relative_to(first_dir).as_posix(): p.read_bytes() for p in first_dir.rglob("*") if p.is_file()}
    two = {p.relative_to(second_dir).as_posix(): p.read_bytes() for p in second_dir.rglob("*") if p.is_file()}
    assert one == two
    after = {name: digest((RETAINED / name).read_bytes()) for name in before}
    assert after == before
    assert first["status"] == "pending setup review"
    assert first["final_protocol_packet"] is False
    assert first["equipment_execution_authorized"] is False and first["physical_trial_performed"] is False
    assert [entry["item"] for entry in first["manual_setup_verification"]] == [
        "first-article air cut", "Z touch-off", "holder-stickout measurement", "fixture clearance",
    ]
    assert all(entry["operator_record"] is None and entry["required"] for entry in first["manual_setup_verification"])
    program = first["programs"][0]
    assert program["run_id"] == "CAM-Base-dev-pocket-island-000-v2"
    assert program["simulation_qualification"] == "simulation-qualified"
    assert program["selected_dialect"] == "haas"
    assert program["continuous_collision_free_certificate"] is False
    assert program["source_fields"]["stock"]["sha256"].startswith("sha256:")
    assert program["source_fields"]["target"]["sha256"].startswith("sha256:")
    for dialect in P.DIALECTS:
        evaluation = program["evaluations"][dialect]
        assert evaluation["coverage"] == "complete" and evaluation["safety_pass"] is True
        assert evaluation["geometry_pass"] is True
        assert evaluation["refined_recheck"]["coverage"] == "complete"
        assert evaluation["refined_recheck"]["settings"]["subdivisions"] == 8
        assert evaluation["refined_recheck"]["geometry_pass"] is None
        assert evaluation["refined_recheck"]["temporal_refinement_is_nested_vs_production"] is None
        assert evaluation["refined_recheck"]["qualification"]["min_noncutting_clearance_lower_bound_mm"] >= 0.25
        assert evaluation["production_refinement"]["present"] is False
        assert evaluation["production_refinement"]["record"] is None
        assert "final" in evaluation["production_refinement"]["reason"]
        assert evaluation["force_proxy"]["unit"] == "proxy"
        assert evaluation["force_proxy"]["value"] is None
        assert (first_dir / evaluation["nc"]).read_bytes() == (RETAINED / f"{dialect}.nc").read_bytes()
        assert evaluation["qualification"]["max_local_surface_error_mm"] <= 0.5
    assert program["independent_cross_checks"]["haas"]["raw_stock_cell_disagreement"]["contact_boundary_only"] is True
    verified = P.verify_packet(first_dir)
    assert verified["payload_sha256"] == first["payload_sha256"]


def test_malformed_setup_and_inconsistent_tool_holder_are_rejected(tmp_path):
    setup = valid_setup()
    del setup["machine"]["model"]
    with pytest.raises(P.PacketError, match="machine.model"):
        export(tmp_path / "missing-model", setup=setup)
    assert not (tmp_path / "missing-model").exists()
    setup = valid_setup()
    setup["tool"]["assembly"]["holder_radius_mm"] += 1.0
    with pytest.raises(P.PacketError, match="tool/holder assembly"):
        export(tmp_path / "wrong-holder", setup=setup)
    setup = valid_setup()
    setup["tool"]["declared_stickout_mm"] += 0.1
    with pytest.raises(P.PacketError, match="stickout"):
        export(tmp_path / "wrong-stickout", setup=setup)
    setup = valid_setup()
    setup["work_offset"]["identifier"] = "G55"
    with pytest.raises(P.PacketError, match="work-offset identifier"):
        export(tmp_path / "wrong-offset", setup=setup)


def test_fixture_representation_and_part_specific_floor_must_match(tmp_path):
    setup = valid_setup()
    setup["fixtures"]["complete_fixture_representation"] = True
    with pytest.raises(P.PacketError, match="empty task fixtures"):
        export(tmp_path / "fake-fixture-completion", setup=setup)
    setup = valid_setup()
    setup["fixtures"]["identifiers"] = ["invented-clamp"]
    with pytest.raises(P.PacketError, match="fixture identity"):
        export(tmp_path / "invented-fixture", setup=setup)
    setup = valid_setup()
    setup["floor"]["remaining_thickness_mm"] -= 1.0
    with pytest.raises(P.PacketError, match="floor thickness"):
        export(tmp_path / "wrong-floor", setup=setup)


def test_unknown_coverage_is_rejected_after_synthetic_mutation_rehash(tmp_path):
    source = clone_run(tmp_path)
    posted = load(source / "eval_posted_haas.json")
    posted["record"]["coverage"] = "unknown"
    posted["record"]["coverage_reasons"] = ["synthetic unknown coverage for rejection test"]
    posted["record"]["safety_pass"] = None
    posted["record"]["geometry_pass"] = None
    posted["record"]["efficiency_eligible"] = False
    write_json(source / "eval_posted_haas.json", posted)
    rehash(source)
    with pytest.raises(P.PacketError, match="not a complete production hard evaluation"):
        export(tmp_path / "out", source=source)


def test_unknown_independent_reference_coverage_is_not_a_pass(tmp_path):
    source = clone_run(tmp_path)
    record = load(source / "reference_haas.json")
    record["record"]["coverage"] = "unknown"
    record["record"]["coverage_reasons"] = ["synthetic unknown independent coverage for rejection test"]
    record["record"]["safety_pass"] = None
    record["record"]["geometry_pass"] = None
    write_json(source / "reference_haas.json", record)
    rehash(source)
    with pytest.raises(P.PacketError, match="reference simulation coverage is unknown"):
        export(tmp_path / "out", source=source)


def test_immutable_hash_mismatch_is_rejected_before_semantic_relabeling(tmp_path):
    source = clone_run(tmp_path)
    with (source / "haas.nc").open("ab") as stream:
        stream.write(b"( synthetic byte corruption )\n")
    with pytest.raises(P.PacketError, match="immutable baseline artifact hash mismatch: haas.nc"):
        export(tmp_path / "out", source=source)
    nc_only = clone_run(tmp_path, "nc-rehashed")
    with (nc_only / "haas.nc").open("ab") as stream:
        stream.write(b"( independent semantic mismatch )\n")
    rehash(nc_only)
    with pytest.raises(P.PacketError, match="posted NC index/hash mismatch"):
        export(tmp_path / "nc-out", source=nc_only)


@pytest.mark.parametrize(
    ("filename", "key", "bad_value", "message"),
    [
        ("eval_posted_haas.json", "max_surface_error_mm", 1.0, "local surface error"),
        ("eval_posted_haas.json", "min_clearance_lower_bound_mm", 0.1, "clearance lower bound"),
        ("eval_posted_haas.json", "rapid_cutter_penetration_upper_bound_mm", 0.1, "rapid penetration"),
        ("reference_haas.json", "min_clearance_lower_bound_mm", 0.1, "clearance lower bound"),
        ("reference_haas.json", "rapid_cutter_penetration_upper_bound_mm", 0.1, "rapid penetration"),
    ],
)
def test_local_or_conservative_safety_bounds_fail_closed(tmp_path, filename, key, bad_value, message):
    source = clone_run(tmp_path)
    record = load(source / filename)
    record["record"]["metrics"][key]["value"] = bad_value
    record["record"]["metrics"][key]["null_reason"] = None
    write_json(source / filename, record)
    rehash(source)
    with pytest.raises(P.PacketError, match=message):
        export(tmp_path / "out", source=source)


def test_scalar_intersections_or_summary_cannot_replace_independent_check(tmp_path):
    source = clone_run(tmp_path)
    record = load(source / "reference_haas.json")
    record["record"]["coverage"] = "complete"
    record["record"]["metrics"]["min_clearance_lower_bound_mm"] = {
        "unit": "mm", "value": None, "denominator": None, "null_reason": "synthetic missing bound",
    }
    write_json(source / "reference_haas.json", record)
    rehash(source)
    with pytest.raises(P.PacketError, match="min_clearance_lower_bound_mm is unavailable"):
        export(tmp_path / "missing-bound", source=source)
    other = clone_run(tmp_path, "contact")
    metrics = load(other / "metrics.json")
    metrics["raw_stock_cell_disagreement"]["haas"]["contact_boundary_only"] = False
    metrics["raw_stock_cell_disagreement"]["haas"]["status"] = "unresolved_contact_disagreement"
    write_json(other / "metrics.json", metrics)
    rehash(other)
    with pytest.raises(P.PacketError, match="disagreement is unresolved"):
        export(tmp_path / "unresolved-contact", source=other)


def test_development_reproduction_has_pending_physical_review_not_final_freeze(tmp_path):
    with pytest.raises(P.PacketError, match="final packets require"):
        export(tmp_path / "out", final=True)
    assert not (tmp_path / "out").exists()


def test_final_protocol_requires_linked_qualified_production_refinement(tmp_path):
    record = tmp_path / "synthetic-frozen-record-not-a-real-protocol.json"
    write_json(record, {"note": "unit-only file for exercising the final gate; not a protocol freeze"})
    protocol = {
        "schema_version": P.PROTOCOL_SCHEMA,
        "protocol_id": "unit-test-only-final-refusal",
        "freeze_state": "frozen",
        "frozen_record": {"path": str(record), "sha256": digest(record.read_bytes())},
    }
    with pytest.raises(P.PacketError, match="requires a qualified schema-3 refinement link"):
        export(tmp_path / "final", protocol=protocol, final=True)
    assert not (tmp_path / "final").exists()


def auxiliary_fixture(tmp_path):
    """Metadata-only schema fixture; NO refined production evaluation is executed."""
    source = clone_run(tmp_path, "primary")
    primary = C.loads((source / "eval_posted_rs274.json").read_text())
    run_id = C.loads((source / "run.json").read_text()).run_id
    refined_run_id = f"{run_id}-refined"
    auxiliary = tmp_path / "auxiliary" / refined_run_id
    auxiliary.mkdir(parents=True)
    primary_hashes = load(source / "hashes.json")
    refined = load(source / "eval_posted_rs274.json")
    # Synthetic declaration changes exercise transport/audit semantics only.
    # They are NOT a re-execution, measured improvement or physical packet.
    refined["record"]["discretization"]["settings"]["max_step_mm"] = 0.5
    refined["record"]["discretization"]["unit_metadata_only_not_reexecuted"] = True
    refined_path = auxiliary / "eval_posted_rs274_refined.json"
    write_json(refined_path, refined)
    outcome = {
        "safety_pass": True, "geometry_pass": True, "completion_pass": True,
        "efficiency_eligible": True,
        "refinement_disagreement": False,
        "disagreement_fields": [],
        "excluded_from_efficiency": False,
        "rule": "synthetic unit contract only; no actual refinement or physical qualification claimed",
    }
    details = {
        name: {"primary": getattr(primary, name), "refined": getattr(primary, name), "changed": False}
        for name in ("safety_pass", "geometry_pass")
    }
    details["coverage"] = {"primary": primary.coverage.value,
                           "refined": primary.coverage.value, "changed": False}
    details["efficiency_eligible"] = {
        "primary": primary.efficiency_eligible,
        "refined": primary.efficiency_eligible, "changed": False,
    }
    for name in ("residual_volume_mm3", "gouge_volume_mm3", "max_surface_error_mm"):
        metric = primary.metrics[name]
        details[name] = {
            "primary": metric.value, "refined": metric.value, "delta": 0.0, "unit": metric.unit,
        }
    details["completion_pass"] = {"primary": True, "refined": True, "changed": False}
    details["any_outcome_verdict_change"] = False
    comparison = {
        "primary_run_id": run_id,
        "refined_run_id": refined_run_id,
        "comparison": details,
        "conservative_outcome": outcome,
    }
    comparison_path = auxiliary / "comparison.json"
    write_json(comparison_path, comparison)
    auxiliary_path = auxiliary / "run.json"
    write_json(auxiliary_path, {"unit_metadata_fixture": True, "not_a_physical_run": True})
    link = {
        "schema": "diffcam-refinement-link/3",
        "primary_run_id": run_id,
        "primary_run_path": run_id,
        "primary_metrics_sha256": primary_hashes["metrics.json"],
        "primary_rs274_evaluation_sha256": primary_hashes["eval_posted_rs274.json"],
        "primary_haas_evaluation_sha256": primary_hashes["eval_posted_haas.json"],
        "primary_run_json_sha256": primary_hashes["run.json"],
        "refined_run_id": refined_run_id,
        "refined_artifact_path": f"{refined_run_id}/{refined_path.name}",
        "refined_evaluation_sha256": digest(refined_path.read_bytes()),
        "auxiliary_record": {"path": f"{refined_run_id}/{auxiliary_path.name}",
                             "sha256": digest(auxiliary_path.read_bytes())},
        "comparison_path": f"{refined_run_id}/comparison.json",
        "comparison_sha256": digest(comparison_path.read_bytes()),
        "conservative_outcome": outcome,
        "comparison": details,
    }
    link_path = auxiliary.parent / f"{run_id}-refinement-link.json"
    write_json(link_path, link)
    return source, link_path, auxiliary


def test_auxiliary_schema3_refinement_is_verified_and_cited_for_same_motion(tmp_path):
    source, link_path, _ = auxiliary_fixture(tmp_path)
    result = export(tmp_path / "packet", source=source, refinement_link=link_path)
    program = result["programs"][0]
    assert result["status"] == "pending setup review"
    assert result["final_protocol_packet"] is False
    for dialect in P.DIALECTS:
        refinement = program["evaluations"][dialect]["production_refinement"]
        assert refinement["present"] is True
        assert refinement["link_schema_version"] == "diffcam-refinement-link/3"
        assert refinement["refinement_dialect"] == "rs274"
        assert refinement["applies_to_dialect"] == dialect
        assert refinement["coverage"] == "complete" and refinement["safety_pass"] is True
        assert refinement["qualification"]["max_local_surface_error_mm"] <= 0.5
        assert refinement["conservative_outcome"]["excluded_from_efficiency"] is False
        assert refinement["conservative_outcome"]["efficiency_eligible"] is True
        if dialect == "haas":
            assert "raw NC bytes are not conflated" in refinement["motion_relationship"]
        for key in ("link", "record", "auxiliary_record", "comparison"):
            assert (tmp_path / "packet" / refinement[key]).is_file()
    assert (tmp_path / "packet" / "programs/baseline/production_refinement/refinement-link.json").read_bytes() == link_path.read_bytes()
    assert P.verify_packet(tmp_path / "packet") == result


def test_refinement_link_primary_hash_and_conservative_outcome_fail_closed(tmp_path):
    source, link_path, _ = auxiliary_fixture(tmp_path)
    link = load(link_path)
    link["primary_rs274_evaluation_sha256"] = "sha256:" + "0" * 64
    write_json(link_path, link)
    with pytest.raises(P.PacketError, match="primary_rs274_evaluation_sha256 differs"):
        export(tmp_path / "bad-primary", source=source, refinement_link=link_path)
    link = load(link_path)
    link["primary_rs274_evaluation_sha256"] = load(source / "hashes.json")["eval_posted_rs274.json"]
    link["conservative_outcome"] = dict(link["conservative_outcome"], safety_pass=False)
    link["comparison"] = dict(link["comparison"], conservative_outcome=link["conservative_outcome"])
    write_json(link_path, link)
    with pytest.raises(P.PacketError, match="conservative outcome safety_pass is not true"):
        export(tmp_path / "unsafe-outcome", source=source, refinement_link=link_path)


def test_self_consistent_hashes_cannot_substitute_for_actual_comparison_verdict(tmp_path):
    source, link_path, _ = auxiliary_fixture(tmp_path)
    link = load(link_path)
    # Regenerate the linked comparison hash so this reaches semantic validation:
    # a copied immutable file with mutually consistent hashes is still dishonest.
    link["comparison"]["coverage"] = {"primary": "complete", "refined": "complete", "changed": True}
    comparison_path = link_path.parent / link["comparison_path"]
    comparison_file = load(comparison_path)
    comparison_file["comparison"] = link["comparison"]
    write_json(comparison_path, comparison_file)
    link["comparison_sha256"] = digest(comparison_path.read_bytes())
    write_json(link_path, link)
    with pytest.raises(P.PacketError, match="comparison row coverage differs"):
        export(tmp_path / "out", source=source, refinement_link=link_path)
    link = load(link_path)
    link["comparison"]["coverage"] = {"primary": "complete", "refined": "complete", "changed": False}
    # A separate inconsistent eligibility claim is rejected too.
    outcome = dict(link["conservative_outcome"], efficiency_eligible=False,
                   refinement_disagreement=True, disagreement_fields=["coverage"],
                   excluded_from_efficiency=True)
    comparison_file["comparison"] = link["comparison"]
    link["conservative_outcome"] = outcome
    comparison_file["conservative_outcome"] = outcome
    write_json(comparison_path, comparison_file)
    link["comparison_sha256"] = digest(comparison_path.read_bytes())
    write_json(link_path, link)
    with pytest.raises(P.PacketError, match="conservative outcome efficiency_eligible differs"):
        export(tmp_path / "bad-outcome", source=source, refinement_link=link_path)


def test_unknown_refined_coverage_is_rejected_even_with_qualified_comparison_claim(tmp_path):
    source, link_path, auxiliary = auxiliary_fixture(tmp_path)
    refined = load(auxiliary / "eval_posted_rs274_refined.json")
    refined["record"]["coverage"] = "unknown"
    refined["record"]["coverage_reasons"] = ["synthetic unknown refined coverage"]
    refined["record"]["safety_pass"] = None
    refined["record"]["geometry_pass"] = None
    refined["record"]["efficiency_eligible"] = False
    refined_path = auxiliary / "eval_posted_rs274_refined.json"
    write_json(refined_path, refined)
    link = load(link_path)
    link["refined_evaluation_sha256"] = digest(refined_path.read_bytes())
    write_json(link_path, link)
    with pytest.raises(P.PacketError, match="refinement is not complete production evaluation"):
        export(tmp_path / "out", source=source, refinement_link=link_path)


def test_final_rejects_linked_refinement_disagreement_and_efficiency_exclusion(tmp_path):
    source, link_path, _ = auxiliary_fixture(tmp_path)
    outcome = {
        "safety_pass": True, "geometry_pass": True, "completion_pass": True,
        "efficiency_eligible": False, "refinement_disagreement": True,
        "disagreement_fields": ["coverage"], "excluded_from_efficiency": True,
        "rule": "synthetic final-rejection fixture; not an observed disagreement",
    }
    link = load(link_path)
    link["conservative_outcome"] = outcome
    link["comparison"] = dict(link["comparison"], coverage={"changed": True})
    write_json(link_path, link)
    record = tmp_path / "synthetic-not-a-real-freeze.json"
    write_json(record, {"unit_only": True})
    protocol = {
        "schema_version": P.PROTOCOL_SCHEMA, "protocol_id": "unit-only-final-rejection",
        "freeze_state": "frozen",
        "frozen_record": {"path": str(record), "sha256": digest(record.read_bytes())},
    }
    with pytest.raises(P.PacketError, match="final packet cannot use"):
        export(tmp_path / "final", source=source, protocol=protocol,
               final=True, refinement_link=link_path)
    assert not (tmp_path / "final").exists()


def test_cli_writes_same_payload_and_refuses_existing_output(tmp_path):
    setup_path, protocol_path = tmp_path / "setup.json", tmp_path / "protocol.json"
    write_json(setup_path, valid_setup())
    write_json(protocol_path, pending_protocol())
    command = [
        sys.executable, "-m", "scripts.export_physical_packets",
        "--selection", f"baseline={RETAINED}", "--dialect", "haas",
        "--setup", str(setup_path), "--protocol", str(protocol_path),
        "--output", str(tmp_path / "packet"),
    ]
    done = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout)["status"] == "pending setup review"
    assert json.loads(done.stdout)["equipment_execution_authorized"] is False
    repeated = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    packet = P.verify_packet(tmp_path / "packet")
    (tmp_path / "packet" / "setup.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(P.PacketError, match="packet payload hash mismatch"):
        P.verify_packet(tmp_path / "packet")
    (tmp_path / "packet" / "setup.json").write_bytes(setup_path.read_bytes())
    write_json(tmp_path / "packet" / "hashes.json", {})
    with pytest.raises(P.PacketError, match="hashes.json differs from packet.json"):
        P.verify_packet(tmp_path / "packet")


def test_unknown_initial_state_cannot_be_silently_authorized(tmp_path):
    source = clone_run(tmp_path)
    task = C.loads((source / "task.json").read_text(encoding="utf-8"))
    unknown = dataclasses.replace(
        task, initial_state=dataclasses.replace(task.initial_state, known=False, position_mm=None),
    )
    (source / "task.json").write_bytes(C.dumps(unknown).encode("utf-8"))
    rehash(source)
    with pytest.raises(P.PacketError, match="typed motion does not match|initial machine state"):
        export(tmp_path / "out", source=source)
