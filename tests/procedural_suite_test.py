"""T2.3b design variation, exact bootstrap preservation and real cheap gates.

The session fixture generates and qualifies all 39 tasks with fresh bounded
hard-sim workers. No posted G-code evaluation is run here. Geometry masks are
also sampled independently in NumPy. Negative worker tests are explicit unit
checks of rejection/cleanup, not substitutes for the real 39-case evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import contracts as C
from benchmarks import procedural_suite as suite
from benchmarks.pilot_task import build_pilot_task, expected_positions_from_motion, source_fields
from cam.seeds import IslandSeedSpec, pocket_with_island_seed, seed_deltas_mm, seed_for_task


BUNDLES = suite.generate_tasks()
BY_ID = {bundle.task.task_id: bundle for bundle in BUNDLES}
SCRIPT = suite.PROJECT_ROOT / "scripts" / "generate_task_suite.py"
BOOTSTRAP_SOURCE_COMMIT = "6a02b07bfd35442c658309a915325a802fb2cdef"
BOOTSTRAP_MANIFEST_SHA256 = "sha256:4e0dff9a2b34430ad0359bb3f646e50580c84a032881917a31df380b2bc3d0ef"


def _validator():
    name = "t23b_protocol_reader"
    spec = importlib.util.spec_from_file_location(
        name, suite.PROJECT_ROOT / "scripts" / "validate_frozen_protocol.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_cli(*args, timeout=300, expected_code=0):
    process = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        cwd=suite.PROJECT_ROOT, env=dict(os.environ),
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    assert process.returncode == expected_code, (
        f"CLI exit {process.returncode}, expected {expected_code}\n"
        f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
    )
    return process


def _geometry_masks(bundle):
    """Independently sample stock, original P and finite-cutter exclusions."""
    task = bundle.task
    params = task.original_part.params
    grid = task.grid
    axes = [grid.origin_mm[a] + (np.arange(grid.shape[a]) + 0.5) * grid.voxel_mm
            for a in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")

    def box(lo, hi):
        return ((xx >= lo[0]) & (xx <= hi[0])
                & (yy >= lo[1]) & (yy <= hi[1])
                & (zz >= lo[2]) & (zz <= hi[2]))

    stock = box(params["box_min_mm"], params["box_max_mm"])
    pocket = box(params["pocket_min_mm"], params["pocket_max_mm"])
    pad = box(params["island_min_mm"], params["island_max_mm"])
    original = stock & (~pocket | pad)
    radius = task.tool.cutter_radius_mm
    excluded = np.zeros(task.grid.shape, dtype=bool)
    pocket_lo, pocket_hi = params["pocket_min_mm"], params["pocket_max_mm"]
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx = pocket_hi[0] - radius if sx == 1 else pocket_lo[0] + radius
            cy = pocket_hi[1] - radius if sy == 1 else pocket_lo[1] + radius
            corner = box(
                (cx if sx == 1 else pocket_lo[0],
                 cy if sy == 1 else pocket_lo[1], pocket_lo[2]),
                (pocket_hi[0] if sx == 1 else cx,
                 pocket_hi[1] if sy == 1 else cy, pocket_hi[2]),
            )
            excluded |= corner & ((xx - cx)**2 + (yy - cy)**2 >= radius**2)
    return stock, original, excluded


@pytest.fixture(scope="session")
def qualified(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("t23b-qualified")
    output = workspace / "generated"
    manifest = workspace / "task_manifest.json"
    process = _run_cli(
        "--manifest", manifest, "--output-root", output,
        "--seed", suite.GENERATOR_SEED,
        "--dev-count", suite.DEFAULT_DEV_PER_FAMILY,
        "--test-count", suite.DEFAULT_TEST_PER_FAMILY,
        timeout=900,
    )
    report = json.loads((output / "generation_report.json").read_text())
    assert report["status"] == "qualified_cheap", process.stdout + process.stderr
    assert len(report["tasks"]) == len(BUNDLES) == 39
    assert len(report["qualified_task_ids"]) == 39
    assert report["excluded_task_ids"] == []
    return {"output": output, "manifest": manifest, "report": report}


def test_declared_counts_manifest_reproducibility_and_legacy_reader(tmp_path):
    assert Counter((b.task.family, b.task.split) for b in BUNDLES) == {
        (family, split): suite.DEFAULT_DEV_PER_FAMILY if split == C.Split.DEV
        else suite.DEFAULT_TEST_PER_FAMILY
        for family in suite.FAMILIES for split in C.Split if split != C.Split.EXTERNAL
    }
    document = json.loads(suite.manifest_text(BUNDLES))
    assert document["generator_seed"] == suite.GENERATOR_SEED == 133742
    assert document["per_family_counts"] == {
        family: {"dev": 3, "test": 10} for family in suite.FAMILIES
    }
    assert document["generation_version"] == "t2.3/2"
    assert document["design_distance"]["pairs_checked"] == 39 * 38 // 2
    assert document["design_distance"]["observed_minimum"] >= suite.MIN_DESIGN_DISTANCE
    assert len(document["tasks"]) == len(document["task_details"]) == 39
    assert len({b.task.task_hash for b in BUNDLES}) == 39
    for bundle in BUNDLES:
        entry = next(item for item in document["tasks"]
                     if item["task_id"] == bundle.task.task_id)
        assert entry["task_hash"] == bundle.task.task_hash
        assert entry["motion_hash"] == bundle.motion.motion_hash
        assert C.loads(json.dumps(entry["task_record"])) == bundle.task
        detail = document["task_details"][bundle.task.task_id]
        assert detail["task_id"] == bundle.task.task_id
        assert detail["task_hash"] == entry["task_hash"]
        assert detail["motion_hash"] == entry["motion_hash"]
        assert detail["tool_setup_hash"] == bundle.task.tool_setup_hash
        assert detail["posted_qualification"] == "pending"
        assert detail["design_parameters"] == suite.design_parameters_from_task(bundle.task)
        assert detail["accessibility"]["cutter"]["radius_mm"] > 0
        assert detail["accessibility"]["holder"]["radius_mm"] > 0
        assert detail["grid_excluded_volume_mm3"] == bundle.task.operation_target.excluded_volume_mm3
        # Legacy projection stays flat, not hidden inside the nominal seed spec.
        assert entry["seed_spec"] == json.loads(json.dumps(bundle.seed_spec.as_dict()))

    reader = _validator()
    registry, split_hashes = reader.task_registry(document, suite.PROJECT_ROOT)
    assert set(registry) == set(BY_ID)
    assert split_hashes["dev"] and split_hashes["test"]
    assert split_hashes["external"] is None
    for name in ("dev", "test"):
        rows = [bundle for bundle in BUNDLES if bundle.task.split.value == name]
        assert len({b.task.task_id for b in rows}) == len(rows)

    # Independent fresh processes produce byte-identical definition manifests.
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    unused = tmp_path / "must-not-exist"
    _run_cli("--manifest-only", "--manifest", left, "--output-root", unused)
    _run_cli("--manifest-only", "--manifest", right, "--output-root", unused)
    assert left.read_bytes() == right.read_bytes() == suite.manifest_text(BUNDLES).encode()
    assert not unused.exists()
    _run_cli("--verify-only", "--manifest", left, "--output-root", unused)
    assert not unused.exists()


def test_real_design_variation_disjoint_splits_and_count_parameterization():
    for family in suite.FAMILIES:
        rows = [bundle for bundle in BUNDLES if bundle.task.family == family]
        assert len({bundle.topology for bundle in rows}) == 3
        assert {bundle.design_parameters["island_count"] for bundle in rows} == {0, 1}
        assert len({bundle.design_parameters["island_top_z"] for bundle in rows}) >= 3
        assert len({(bundle.design_parameters["pocket_half_x"],
                     bundle.design_parameters["pocket_half_y"],
                     bundle.design_parameters["pocket_depth"]) for bundle in rows}) >= 4
        dev = [b for b in rows if b.task.split == C.Split.DEV]
        held_out = [b for b in rows if b.task.split == C.Split.TEST]
        assert {b.task.task_hash for b in dev}.isdisjoint({b.task.task_hash for b in held_out})
        for left in dev:
            for right in held_out:
                assert suite.design_distance(left, right) >= suite.MIN_DESIGN_DISTANCE

    # Changing counts does not reshuffle or rewrite existing case designs.
    prefix = suite.generate_tasks(1, 2)
    assert len(prefix) == 9
    for bundle in prefix:
        assert bundle.task.task_hash == BY_ID[bundle.task.task_id].task.task_hash
        assert bundle.motion.motion_hash == BY_ID[bundle.task.task_id].motion.motion_hash
    changed = suite.generate_scenarios(1, 1, seed=suite.GENERATOR_SEED + 1)
    assert len(changed) == 6
    assert all(s.generator_seed == suite.GENERATOR_SEED + 1 for s in changed)
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            suite.generate_tasks(value, 1)
    with pytest.raises(ValueError):
        suite.generate_scenarios(seed=True)


def test_near_duplicate_guard_and_source_provenance():
    base = suite.SCENARIOS[0]
    renamed = replace(base, name="renamed", case_index=base.case_index + 1,
                      design_seed=base.design_seed + 1)
    assert suite.design_distance(base, renamed) == 0.0
    with pytest.raises(ValueError, match="near-duplicate"):
        suite.validate_designs((base, renamed))
    shifted = replace(base, case_index=base.case_index + 1, stock=(
        (base.stock[0][0] - 0.5, *base.stock[0][1:]), base.stock[1]))
    assert suite.design_distance(base, shifted) < suite.MIN_DESIGN_DISTANCE
    with pytest.raises(ValueError, match="near-duplicate"):
        suite.validate_designs((base, shifted))

    bundle = BUNDLES[0]
    metadata = dict(bundle.design_parameters)
    digest = hashlib.sha256(
        f"{suite.GENERATION_VERSION}:{suite.GENERATOR_SEED}:{bundle.task.family}:"
        f"{bundle.task.split.value}:{bundle.case_index}".encode("ascii")
    ).digest()
    assert bundle.design_seed == int.from_bytes(digest[:8], "big")
    assert all(isinstance(s.design_seed, int) for s in suite.SCENARIOS)
    assert len({s.design_seed for s in suite.SCENARIOS}) == 39

    metadata["stock_min_x"] += 1.0  # Provenance is bound to the actual task, not a free-floating declaration.
    altered = replace(bundle, design_parameters=metadata)
    with pytest.raises(ValueError, match="design vector does not match task source"):
        suite.build_manifest([
            altered if other.task.task_id == bundle.task.task_id else other for other in BUNDLES
        ])
    if base.island is not None:
        lo, hi = base.island
        rectangular = replace(base, island=(lo, (hi[0] + 0.5, hi[1], hi[2])))
        with pytest.raises(ValueError, match="square"):
            suite._design_parameters(rectangular)


def test_geometry_and_nominal_seed_verification():
    for bundle in BUNDLES:
        task, motion, design = bundle.task, bundle.motion, bundle.design_parameters
        params = task.original_part.params
        assert task.original_part.frame == C.Frame.WORK
        assert task.grid.voxel_mm == 0.25
        np.testing.assert_allclose(task.grid.bounds_mm,
                                   (params["box_min_mm"], params["box_max_mm"]), atol=0)
        assert math.prod(task.grid.shape) <= 1_000_000
        motion.validate_for_task(task)
        assert seed_for_task(task).motion_hash == motion.motion_hash
        assert len(motion.segments) == len(seed_deltas_mm(motion))
        machine = np.asarray(expected_positions_from_motion(motion), dtype=np.float64)
        assert np.isfinite(machine).all()
        np.testing.assert_array_equal(np.diff(machine, axis=0), seed_deltas_mm(motion))
        assert machine[0] == pytest.approx(task.initial_state.position_mm, abs=1e-12)
        assert machine[-1] == pytest.approx(task.initial_state.position_mm, abs=1e-12)
        for segment in motion.segments:
            assert segment.arc is None
            if segment.kind == C.SegmentKind.RAPID:
                assert min(segment.start_mm[2], segment.end_mm[2]) > task.stock.params["max_mm"][2]

        pad = (params["island_min_mm"], params["island_max_mm"])
        retained = all(params["box_min_mm"][a] < pad[1][a] and pad[0][a] < params["box_max_mm"][a]
                       for a in range(3))
        assert retained == (design["island_count"] == 1) == bundle.seed_spec.has_retained_island
        if not retained:
            assert params["empty_island"] is True
            assert pad[0][2] > params["box_max_mm"][2]
        else:
            assert all(params["pocket_min_mm"][a] <= pad[0][a]
                       and pad[1][a] <= params["pocket_max_mm"][a] for a in (0, 1))
            assert (pad[1][0] - pad[0][0]) == pytest.approx(pad[1][1] - pad[0][1])
            assert pad[0][2] == params["pocket_min_mm"][2]
            assert pad[1][2] <= params["box_max_mm"][2]
            actual_relative_centre = tuple((pad[0][a] + pad[1][a]) / 2 - design[f"pocket_c{'x' if a == 0 else 'y'}"]
                                           for a in (0, 1))
            np.testing.assert_allclose(actual_relative_centre, bundle.seed_spec.island_centre_xy,
                                       atol=1e-12, rtol=0)
            assert bundle.seed_spec.island_half_side == pytest.approx(design["island_half_side"])
        if bundle.topology == "offset_square_lower_step_pad":
            assert design["island_top_z"] < 0.0
            assert any(abs(design[f"island_c{axis}"] - design[f"pocket_c{axis}"]) > 0.0
                       for axis in ("x", "y"))
        elif bundle.topology == "centred_square_full_height_pad":
            assert design["island_top_z"] == 0.0

        stock, original, excluded = _geometry_masks(bundle)
        target = original | excluded
        required = stock & ~target
        assert np.all(original <= stock)
        assert np.all(original <= target)
        assert not np.array_equal(original, target)
        assert np.count_nonzero(excluded) > 0
        assert np.count_nonzero(excluded) * task.grid.voxel_mm**3 == pytest.approx(
            task.operation_target.excluded_volume_mm3)
        params = task.operation_target.geometry.params
        assert params["grid_excluded_voxels"] == np.count_nonzero(excluded)
        assert params["grid_exclusion_tolerance_mm3"] == 0.0
        _, target_field, required_source, retained_source = source_fields(task)
        np.testing.assert_array_equal(target_field < 0, target)
        np.testing.assert_array_equal(retained_source, target)
        np.testing.assert_array_equal(required_source, required)
        np.testing.assert_array_equal(source_fields(task)[0] < 0, stock)
        holder = params["holder_clearance"]
        assert holder["above_stock_mm"] >= holder["qualification_and_cell_margin_mm"]
        assert task.tool.holder_radius_mm >= task.tool.cutter_radius_mm


def test_bootstrap_exact_named_subset_preserved(tmp_path):
    original_bytes = subprocess.run(
        ["git", "-C", str(suite.PROJECT_ROOT), "show",
         f"{BOOTSTRAP_SOURCE_COMMIT}:configs/task_manifest.json"],
        capture_output=True, check=True,
    ).stdout
    generated = suite.generate_bootstrap_tasks()
    assert len(generated) == 6
    assert suite.bootstrap_manifest_text().encode() == original_bytes
    assert C.sha256_bytes(original_bytes) == BOOTSTRAP_MANIFEST_SHA256
    assert [(bundle.task.task_hash, bundle.motion.motion_hash) for bundle in generated] == [
        (bundle.task.task_hash, bundle.motion.motion_hash)
        for bundle in suite.generate_bootstrap_tasks()
    ]
    modern = json.loads(suite.manifest_text(BUNDLES))
    assert modern["bootstrap_preservation"]["manifest_sha256"] == BOOTSTRAP_MANIFEST_SHA256
    with pytest.raises(ValueError, match="bootstrap/1"):
        suite.verify_all(BUNDLES[:1], output_root=tmp_path / "must-not-post")
    assert not (tmp_path / "must-not-post").exists()


def test_cheap_artifacts_and_process_evidence(qualified):
    output, report = qualified["output"], qualified["report"]
    actual_manifest = qualified["manifest"].read_bytes()
    assert C.sha256_bytes(actual_manifest) == report["manifest_sha256"]
    assert report["generation_wall_seconds"] > 0
    assert report["definition_seconds"] > 0
    assert report["task_loop_wall_seconds"] > 0
    assert len(report["requested_tasks"]) == 39
    assert all(row["posted_qualification"] == "pending" for row in report["tasks"])
    assert not list(output.rglob("*.nc"))
    assert not list(output.rglob("run.json"))
    for bundle in BUNDLES:
        row = next(item for item in report["tasks"] if item["task_id"] == bundle.task.task_id)
        assert row["status"] == "qualified_cheap_hard_sim"
        assert row["task_hash"] == bundle.task.task_hash
        assert row["motion_hash"] == bundle.motion.motion_hash
        assert row["residual_volume_mm3"] <= bundle.task.tolerances.completion_residual_mm3
        assert row["gouge_volume_mm3"] <= bundle.task.tolerances.gouge_volume_mm3
        assert row["holder_overlap_interval_sum_mm3"] == 0.0
        assert row["fresh_process"] and row["process_parallelism"] == 1
        assert isinstance(row["worker_pid"], int) and row["worker_pid"] > 0
        assert 0 < row["worker_own_hwm_bytes"] <= suite.CHEAP_WORKER_MEMORY_LIMIT_BYTES
        assert (row["controller_observed_hwm_bytes"] is None
                or 0 < row["controller_observed_hwm_bytes"] <= suite.CHEAP_WORKER_MEMORY_LIMIT_BYTES)
        assert row["worker_seconds"] > 0
        assert row["cheap_qualification_seconds"] >= row["worker_seconds"]
        assert row["elapsed_seconds"] >= row["cheap_qualification_seconds"]
        assert row["source_payload_dtype"] == "float32"
        assert row["simulator_precision"]["runtime_default_fp"] == "f64"
        assert row["simulator_precision"]["requested_precision"] == "f64"
        assert len(row["source_hashes"]) == 2
        for name in ("stock", "target"):
            assert row["source_hashes"][name].startswith("sha256:")

        root = output / bundle.task.task_id
        assert C.loads((root / "task.json").read_text()) == bundle.task
        assert C.loads((root / "motion_plan.json").read_text()) == bundle.motion
        saved = json.loads((root / "cheap_qualification.json").read_text())
        assert saved["status"] == "qualified_cheap_hard_sim"
        assert saved["worker_pid"] == row["worker_pid"]
        assert saved["worker_own_hwm_bytes"] == row["worker_own_hwm_bytes"]
        assert saved["source_grid_declaration"]["shape"] == list(bundle.task.grid.shape)
        assert saved["source_grid_declaration"]["voxel_mm"] == 0.25
        generation = json.loads((root / "generation.json").read_text())
        assert generation["posted_qualification"] == "pending"
        assert generation["design_seed"] == bundle.design_seed
        assert generation["cheap_executor"] == suite.CHEAP_GATE_EXECUTOR
        head = subprocess.run(["git", "-C", str(suite.PROJECT_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert generation["source_commit"] == head
        hashes = json.loads((root / "hashes.json").read_text())
        assert {path.name for path in root.iterdir()} == set(hashes) | {"hashes.json"}
        assert not (root / "run.json").exists()
        for name, digest in hashes.items():
            assert C.sha256_bytes((root / name).read_bytes()) == digest
        assert Path(saved["worker_log"]).is_file()


def test_manifest_and_generation_report_overwrite_protection(qualified, tmp_path):
    manifest = qualified["manifest"]
    before_manifest = manifest.read_bytes()
    report = qualified["output"] / "generation_report.json"
    before_report = report.read_bytes()
    _run_cli("--manifest", manifest, "--output-root", qualified["output"], expected_code=1)
    assert manifest.read_bytes() == before_manifest
    assert report.read_bytes() == before_report

    altered = tmp_path / "manifest.json"
    altered.write_bytes(before_manifest + b"\n")
    before_altered = altered.read_bytes()
    with pytest.raises(FileExistsError):
        suite.write_manifest(altered, expected_text=before_manifest.decode())
    assert altered.read_bytes() == before_altered
    _run_cli("--verify-only", "--manifest", altered, expected_code=1)
    assert altered.read_bytes() == before_altered


def test_manifest_only_does_not_execute_cheap_gates(tmp_path):
    manifest, unused = tmp_path / "manifest.json", tmp_path / "must-not-exist"
    result = _run_cli("--manifest-only", "--manifest", manifest,
                      "--dev-count", 1, "--test-count", 1, "--output-root", unused)
    assert "cheap seed qualification NOT run" in result.stdout
    assert not unused.exists()
    _run_cli("--verify-only", "--manifest", manifest,
             "--dev-count", 1, "--test-count", 1, "--output-root", unused)
    assert not unused.exists()


def test_unsupported_source_and_worker_timeout_fail_closed(tmp_path, monkeypatch):
    bundle = BUNDLES[0]
    rejected = replace(bundle.task, original_part=C.GeometrySource(
        kind="mesh", path="synthetic-invalid-marker-never-read",
        sha256=C.sha256_bytes(b"synthetic invalid input only; never a qualified task"),
    ))
    with pytest.raises(RuntimeError, match="analytic|mesh|box") as raised:
        suite.cheap_qualify_seed(replace(bundle, task=rejected), log_directory=tmp_path / "logs")
    assert raised.value.worker_error_type == "ValueError"
    assert "_worker_evaluate" in raised.value.worker_traceback

    stopped = []
    real_stop = suite._stop_owned_worker

    def observe_stop(process):
        real_stop(process)
        stopped.append((process.pid, process.returncode))

    monkeypatch.setattr(suite, "_stop_owned_worker", observe_stop)
    monkeypatch.setattr(suite, "CHEAP_WORKER_TIMEOUT_S", 0.0)
    with pytest.raises(RuntimeError, match="exceeded.*seconds"):
        suite.cheap_qualify_seed(bundle, log_directory=tmp_path / "timeout-logs")
    assert stopped and all(pid > 0 and code is not None for pid, code in stopped)


def test_worker_requires_expected_source_worktree(tmp_path):
    output = tmp_path / "result.json"
    code = suite._cheap_worker_main([
        "cheap-worker", "--job", str(tmp_path / "irrelevant.json"),
        "--output", str(output), "--repository", str(tmp_path / "other-worktree"),
    ])
    assert code == 1
    result = json.loads(output.read_text())
    assert result["status"] == "failed"
    assert "different source worktree" in result["error"]


def test_nonfinite_history_and_nonzero_worker_are_rejected(tmp_path, monkeypatch):
    history = np.zeros((2, 1, 1, 1), dtype=np.float64)
    history[0, 0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        suite._finite_stock_history(history, 1)
    history[0, 0, 0, 0] = np.inf
    with pytest.raises(RuntimeError, match="non-finite"):
        suite._finite_stock_history(history, 1)

    # Exercise the controller's exit-status rejection using an otherwise real,
    # valid child result, while injecting a nonzero final poll result.
    original_popen = suite.subprocess.Popen
    launches = []

    def instrumented_popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        original_poll = child.poll

        def late_nonzero_poll():
            code = original_poll()
            return 1 if code is not None else None

        child.poll = late_nonzero_poll
        launches.append(child.pid)
        return child

    monkeypatch.setattr(suite.subprocess, "Popen", instrumented_popen)
    with pytest.raises(RuntimeError, match="exit 0") as raised:
        suite.cheap_qualify_seed(BUNDLES[0], log_directory=tmp_path / "exit-logs")
    assert launches
    assert raised.value.failure_kind == "worker_process"


def test_artifact_error_is_not_reported_as_a_seed_failure(tmp_path):
    output = tmp_path / "generated"
    output.mkdir()
    first = BUNDLES[0]
    existing = output / first.task.task_id
    existing.mkdir()
    retained = existing / "retained.txt"
    retained.write_bytes(b"existing partial artifact must not be overwritten\n")
    result = _run_cli("--dev-count", 1, "--test-count", 1,
                      "--manifest", tmp_path / "manifest.json", "--output-root", output,
                      expected_code=1)
    report = json.loads((output / "generation_report.json").read_text())
    assert report["status"] == "failed"
    failed = next(row for row in report["tasks"] if row["task_id"] == first.task.task_id)
    assert failed["status"] == "artifact_generation_failure"
    assert failed["failure_class"] == "artifact_generation"
    assert failed["seed_qualification_completed"] is True
    assert failed["error_type"] == "FileExistsError"
    assert failed["worker_pid"] > 0
    assert failed["worker_own_hwm_bytes"] > 0
    assert failed["predeclared_exclusion_reason"] is None
    assert retained.read_bytes() == b"existing partial artifact must not be overwritten\n"
    assert report["excluded_task_ids"] == []
    assert report["infrastructure_failure_task_ids"] == [first.task.task_id]
    assert len(report["qualified_task_ids"]) == 3 * (1 + 1) - 1
    assert "ARTIFACT FAILED" in result.stderr


def test_accepted_pilot_nominal_motion_is_unchanged():
    task = build_pilot_task()
    expected = C.MotionPlan.from_dict(json.loads(C.dumps(
        pocket_with_island_seed(task, IslandSeedSpec())))["record"])
    actual = seed_for_task(task)
    assert actual.segments == expected.segments
    assert actual.motion_hash == expected.motion_hash
    assert actual.motion_hash == (
        "sha256:565fcf62661859ce00392f5fd4673ebbdd88809c3d1bc54e808fef7162a8bb34"
    )
    assert len(actual.segments) == 75
