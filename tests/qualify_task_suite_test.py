"""Focused T2.3c driver tests; no CNC motion or optimizer campaign is launched.

The small real-session fixtures execute the production posted evaluator in
independent RS274/Haas child processes. Tests for concurrency and injected
failure handling are explicitly controller unit tests, not qualification
substitutes for the real 39-task dual-dialect batch.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading

import pytest

from benchmarks import contracts as C
from benchmarks.pilot_task import derive_reachable_target
from cam.seeds import seed_for_task


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("t23c_driver_tests", ROOT / "scripts" / "qualify_task_suite.py")
DRIVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DRIVER
SPEC.loader.exec_module(DRIVER)


def tiny_task(task_id="synthetic-qualification-smoke", *, unsafe_tool=False):
    """A real analytic open pocket, sized for a fast strict posted evaluation."""
    stock_bounds = ((-6.0, -6.0, -6.0), (6.0, 6.0, 0.0))
    pocket_bounds = ((-4.0, -4.0, -3.0), (4.0, 4.0, 0.0))
    grid = C.GridSpec(shape=(24, 24, 12), origin_mm=stock_bounds[0], voxel_mm=0.5)
    stock = C.GeometrySource(
        kind="analytic", frame=C.Frame.WORK,
        params={"primitive": "box", "min_mm": list(stock_bounds[0]), "max_mm": list(stock_bounds[1])},
    )
    part = C.GeometrySource(kind="analytic", frame=C.Frame.WORK, params={
        "primitive": "box_minus_pocket",
        "box_min_mm": list(stock_bounds[0]), "box_max_mm": list(stock_bounds[1]),
        "pocket_min_mm": list(pocket_bounds[0]), "pocket_max_mm": list(pocket_bounds[1]),
        "island_min_mm": [-6.0, -6.0, 1.0], "island_max_mm": [-5.0, -5.0, 2.0],
        "empty_island": True,
    })
    # The negative fixture deliberately brings a large holder into retained
    # stock during actual pocket feeds. This never promotes collision to pass.
    tool = C.ToolAssembly(
        cutter_kind=C.CutterKind.FLAT, cutter_radius_mm=2.0,
        flute_length_mm=1.0 if unsafe_tool else 10.0,
        shank_radius_mm=2.0,
        holder_bottom_mm=1.5 if unsafe_tool else 10.0,
        holder_radius_mm=5.0, holder_height_mm=32.0, tool_number=1,
    )
    tolerances = C.Tolerances(
        completion_residual_mm3=2.0, gouge_volume_mm3=0.0,
        local_surface_error_mm=0.5, clearance_mm=0.25, rapid_penetration_mm=0.0,
    )
    # Both variants have the same cutter. The actual retained-stock collision
    # fixture becomes unsafe only after constructing its cutter-defined target;
    # a standard tall holder is used for that target's reachability margin.
    target_tool = replace(tool, flute_length_mm=10.0, holder_bottom_mm=10.0) if unsafe_tool else tool
    operation = derive_reachable_target(stock, part, target_tool, grid, tolerances.clearance_mm)
    task = C.TaskSpec(
        task_id=task_id, family="pocket_island", split=C.Split.DEV,
        source=C.SourceInfo(origin="t23c-test-fixture/1", license="MIT"),
        length_unit="mm", stock=stock, original_part=part,
        operation_target=C.OperationTarget(
            geometry=operation,
            excluded_volume_mm3=operation.params["grid_excluded_volume_mm3"],
            exclusion_reason="small user-authored unit fixture; not cohort or pilot evidence",
        ),
        grid=grid, work_to_machine=C.Transform.translation((0.0, 0.0, 0.0)),
        initial_state=C.MachineState(position_mm=(0.0, 0.0, 10.0), frame=C.Frame.MACHINE),
        tool=tool,
        limits=C.MachineLimits(
            max_feed_mm_s=8.0, rapid_mm_s=80.0, max_accel_mm_s2=100.0,
            envelope_min_mm=(-10.0, -10.0, -10.0), envelope_max_mm=(10.0, 10.0, 60.0),
        ),
        tolerances=tolerances,
    )
    # Explicit-object API negative test: the geometrically identical tall-tool
    # nominal path is supplied with the unsafe assembly. load_cohort still
    # rejects this as an unsafe production seed; no unsafe cohort is created.
    return task, seed_for_task(replace(task, tool=target_tool) if unsafe_tool else task)


def task_manifest(task, motion, path):
    document = {
        "schema_version": "t2.3-procedural/2",
        "generation_version": "t2.3/2",
        "generator_seed": 133742,
        "scope": "test-only single-task projection through the real driver",
        "sentinel_design_declaration": {"unchanged": [1, 2, 3]},
        "tasks": [{
            "task_id": task.task_id, "task_hash": task.task_hash,
            "motion_hash": motion.motion_hash, "task_record": json.loads(C.dumps(task)),
        }],
        "task_details": {task.task_id: {
            "task_id": task.task_id, "task_hash": task.task_hash,
            "motion_hash": motion.motion_hash, "split": task.split.value,
            "family": task.family, "design_seed": 7, "posted_qualification": "pending",
        }},
        "pending_posted_qualification": {
            "status": "pending", "coordinator_batch": True, "grid_pitch_mm": 0.125,
            "dialects": [dialect.value for dialect in C.Dialect],
        },
    }
    path.write_text(DRIVER._json_bytes(document).decode(), encoding="utf-8")
    return document


def fixture_config(task, **changes):
    values = dict(parallel=2, timeout_s=120.0, memory_limit=1024**3,
                  poll_interval=0.01, task=task)
    values.update(changes)
    return DRIVER._config(**values)


def attempt(task_root, dialect):
    return task_root / dialect.value / "attempt-0001"


def test_provenance_source_guard_and_configuration():
    task, _motion = tiny_task()
    config = fixture_config(task, parallel=8)
    assert config["refinement_check"]["parallel_jobs"] == 2
    assert config["refinement_check"]["verdict_step_mm"] == 0.125
    for invalid in (0, -1, 9, 1.5, True):
        with pytest.raises(DRIVER.QualificationError):
            fixture_config(task, parallel=invalid)
    with pytest.raises(DRIVER.QualificationError):
        fixture_config(task, memory_limit=DRIVER.MEMORY_LIMIT_BYTES + 1)
    provenance = DRIVER._source_snapshot(require_clean=False)
    assert provenance["source_tree_clean"] is False
    assert provenance["dirty_patch_hash"] == "not-applicable-test-fixture"
    assert len(provenance["code_source_sha256"]) == len(DRIVER.CODE_SOURCES)


@pytest.fixture(scope="session")
def real_result(tmp_path_factory):
    task, motion = tiny_task()
    root = tmp_path_factory.mktemp("t23c-real") / "tasks" / task.task_id
    result = DRIVER._run_one(task, motion, root, fixture_config(task))
    assert result["efficiency_eligible"] is True
    return {"task": task, "motion": motion, "root": root, "result": result}


def test_real_dual_dialect_evaluation_and_own_child_resources(real_result):
    result = real_result["result"]
    pids = []
    for dialect in C.Dialect:
        outcome = result["dialects"][dialect.value]
        assert outcome["status"] == "evaluated"
        assert outcome["coverage"] == "complete"
        assert outcome["safety_pass"] is True
        assert outcome["geometry_pass"] is True
        assert outcome["efficiency_eligible"] is True
        assert outcome["qualified"] is True
        assert 0.0 <= outcome["air_fraction"] <= 1.0
        assert outcome["settings"] == dict(
            DRIVER.production_settings(real_result["task"]),
            subdivisions=None, compute_bounds=True, safety_screen=True,
            max_samples_per_segment=4096, record_stations=False,
            exclusion_tolerance_mm3=0.0, max_step_mm=0.125,
            expected_nc_hash=outcome["settings"]["expected_nc_hash"],
        )
        assert outcome["reused"] is False
        assert outcome["worker_pid"] == outcome["original_worker_pid"] > 0
        assert outcome["worker_pid"] != os.getpid()
        pids.append(outcome["worker_pid"])
        assert 0 < outcome["worker_own_hwm_bytes"] <= 1024**3
        assert outcome["original_worker_own_hwm_bytes"] == outcome["worker_own_hwm_bytes"]
        assert outcome["original_worker_self_reported_hwm_bytes"] > 0
        assert outcome["worker_own_hwm_bytes"] >= outcome["original_worker_self_reported_hwm_bytes"]
        assert outcome["worker_wall_seconds"] > 0
        assert outcome["process_wall_seconds"] >= outcome["worker_wall_seconds"]
        assert outcome["motion_comparison"]["max_position_error_mm"] == 0.0
        assert outcome["motion_comparison"]["max_time_error_s"] == 0.0
        assert outcome["motion_comparison"]["semantic_tag_mismatches"] == 0
        workspace = attempt(real_result["root"], dialect)
        raw_worker = json.loads((workspace / "worker-result.json").read_text())
        assert raw_worker["worker_pid"] == outcome["worker_pid"]
        assert raw_worker["worker_peak_rss_kib"] * 1024 == outcome["original_worker_self_reported_hwm_bytes"]
        assert raw_worker["worker_peak_rss_source"].startswith("VmHWM of /proc/self/status")
        evaluation = workspace / "evaluation.json"
        assert C.sha256_bytes(evaluation.read_bytes()) == outcome["evaluation_sha256"]
        typed = C.loads(evaluation.read_text())
        assert typed.task_hash == real_result["task"].task_hash
        assert typed.tool_setup_hash == real_result["task"].tool_setup_hash
        assert typed.nc_hash == outcome["settings"]["expected_nc_hash"]
        assert typed.dialect is dialect
        sidecar = json.loads((workspace / "suite-process.json").read_text())
        assert sidecar["worker_result_sha256"] == C.sha256_bytes(
            (workspace / "worker-result.json").read_bytes())
        assert sidecar["combined_peak_rss_kib"] == max(
            raw_worker["worker_peak_rss_kib"], sidecar["parent_observed_hwm_kib"] or 0)
        assert evaluation.exists() and (workspace / "job.json").exists()
        assert (workspace / C.DIALECT_ARTIFACTS[dialect][0]).exists()
    assert len(set(pids)) == 2
    assert result["phase"]["job_policy"]["parallel_jobs"] == 2
    assert "results" not in result["phase"]
    json.dumps(result, allow_nan=False)


def test_resume_skips_only_valid_work_and_retains_original_identity(real_result, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("valid completed work must not launch a new child")

    monkeypatch.setattr(DRIVER._runner(), "_post_jobs", forbidden)
    resumed = DRIVER.execute_task(
        real_result["task"], real_result["motion"], real_result["root"],
        fixture_config(real_result["task"]),
    )
    assert resumed["efficiency_eligible"] is True
    assert resumed["phase"]["phase_wall_seconds"] == 0.0
    for dialect in C.Dialect:
        previous = real_result["result"]["dialects"][dialect.value]
        current = resumed["dialects"][dialect.value]
        assert current["reused"] is True
        assert current["worker_pid"] is None
        assert current["worker_own_hwm_bytes"] is None
        assert current["original_worker_pid"] == previous["original_worker_pid"]
        assert current["original_worker_own_hwm_bytes"] == previous["original_worker_own_hwm_bytes"]
        assert current["worker_wall_seconds"] == previous["worker_wall_seconds"]
        assert current["process_wall_seconds"] == previous["process_wall_seconds"]
        assert current["evaluation_sha256"] == previous["evaluation_sha256"]
        assert len(DRIVER._attempt_paths(real_result["root"] / dialect.value)) == 1


def test_higher_parent_observed_peak_controls_reuse_cap(real_result):
    dialect = C.Dialect.RS274
    workspace = attempt(real_result["root"], dialect)
    sidecar_path = workspace / "suite-process.json"
    original = json.loads(sidecar_path.read_text())
    child_peak = json.loads((workspace / "worker-result.json").read_text())["worker_peak_rss_kib"]
    changed = dict(original, parent_observed_hwm_kib=child_peak + 1)
    changed["combined_peak_rss_kib"] = child_peak + 1
    task = real_result["task"]
    config = fixture_config(task, memory_limit=(child_peak + 1) * 1024)
    job, _comparison = DRIVER._nc_artifacts(task, real_result["motion"], dialect, workspace, config)
    try:
        sidecar_path.write_text(DRIVER._json_bytes(changed).decode(), encoding="utf-8")
        with pytest.raises(DRIVER.QualificationError, match="memory limit"):
            DRIVER._read_old_result(
                workspace, task, real_result["motion"], job,
                memory_limit=child_peak * 1024,
            )
        accepted = DRIVER._read_old_result(
            workspace, task, real_result["motion"], job,
            memory_limit=config["refinement_check"]["max_rss_bytes_per_job"],
        )
        assert accepted["worker"]["worker_peak_rss_kib"] == child_peak + 1
    finally:
        # This mutable test changes only auxiliary controller evidence, then
        # restores it. Child/evaluator files have stayed byte-for-byte intact.
        sidecar_path.write_text(DRIVER._json_bytes(original).decode(), encoding="utf-8")


@pytest.mark.parametrize("filename", ["evaluation.json", "job.json", "worker-result.json"])
def test_damaged_or_stale_prior_evidence_fails_closed(real_result, tmp_path, filename):
    source = attempt(real_result["root"], C.Dialect.RS274)
    root = tmp_path / real_result["task"].task_id
    shutil.copytree(real_result["root"], root)
    path = root / C.Dialect.RS274.value / "attempt-0001" / filename
    original = path.read_bytes()
    if filename == "evaluation.json":
        path.write_bytes(b"{}\n")
    else:
        record = json.loads(path.read_text())
        record["settings"]["max_step_mm"] = 0.25
        path.write_text(DRIVER._json_bytes(record).decode(), encoding="utf-8")
    damaged = DRIVER.execute_task(real_result["task"], real_result["motion"], root,
                                  fixture_config(real_result["task"]))
    assert damaged["efficiency_eligible"] is False
    assert damaged["dialects"]["rs274"]["status"] == "failed"
    assert "prior" in damaged["dialects"]["rs274"]["reason"]
    assert damaged["dialects"]["haas"]["reused"] is True
    assert damaged["dialects"]["haas"]["efficiency_eligible"] is True
    assert path.read_bytes() != original
    retained = DRIVER._run_one(real_result["task"], real_result["motion"], root,
                               fixture_config(real_result["task"]))
    assert retained["dialects"]["rs274"]["status"] == "failed"


def test_partial_attempt_is_preserved_and_must_not_replace_older_work(real_result, tmp_path):
    task, motion = real_result["task"], real_result["motion"]
    config = fixture_config(task)
    root = tmp_path / task.task_id
    workspace = root / "rs274" / "attempt-0001"
    job, _comparison = DRIVER._nc_artifacts(task, motion, C.Dialect.RS274, workspace, config)
    workspace.mkdir(parents=True)
    DRIVER._runner()._prepare_posted_job(task, job)
    before = {path.name: path.read_bytes() for path in workspace.iterdir()}
    prepared = DRIVER.make_job(task, motion, C.Dialect.RS274, workspace, config)
    assert prepared["kind"] == "new_job"
    assert prepared["job"].workspace == root / "rs274" / "attempt-0002"
    assert {path.name: path.read_bytes() for path in workspace.iterdir()} == before

    # A newer partial attempt cannot displace a verified earlier result. It is
    # inspected and retained rather than deleted or silently completed in place.
    complete_root = tmp_path / "with-history" / task.task_id
    shutil.copytree(real_result["root"], complete_root)
    partial = complete_root / "rs274" / "attempt-0002"
    partial_job, _comparison = DRIVER._nc_artifacts(task, motion, C.Dialect.RS274, partial, config)
    partial.mkdir(parents=True)
    DRIVER._runner()._prepare_posted_job(task, partial_job)
    old_summary = real_result["result"]["dialects"]["rs274"]
    selected = DRIVER.make_job(task, motion, C.Dialect.RS274,
                               complete_root / "rs274" / "attempt-0001", config)
    assert selected["kind"] == "reuse"
    assert selected["evaluation"].task_hash == C.loads(
        (complete_root / "rs274" / "attempt-0001" / "evaluation.json").read_text()).task_hash
    assert (complete_root / "rs274" / "attempt-0002" / "job.json").exists()
    assert not (complete_root / "rs274" / "attempt-0002" / "evaluation.json").exists()
    assert old_summary["evaluation_sha256"] == C.sha256_bytes(
        (complete_root / "rs274" / "attempt-0001" / "evaluation.json").read_bytes())


def test_failure_is_immutable_and_remains_failed_on_resume(tmp_path, monkeypatch):
    task, motion = tiny_task("synthetic-failure-preservation")
    root = tmp_path / task.task_id

    def injected_failure(*_args, **_kwargs):
        raise DRIVER._runner().RunnerError("injected controller-unit failure; not a child outcome")

    monkeypatch.setattr(DRIVER._runner(), "_post_jobs", injected_failure)
    failed = DRIVER.execute_task(task, motion, root, fixture_config(task))
    packets = []
    for dialect in C.Dialect:
        outcome = failed["dialects"][dialect.value]
        assert outcome["status"] == "failed"
        assert outcome.get("efficiency_eligible") is not True
        assert "injected controller-unit failure" in outcome["reason"]
        packet = Path(outcome["prior_artifact"])
        assert packet.name == "suite-failure.json"
        evidence = json.loads(packet.read_text())
        assert evidence["task_hash"] == task.task_hash
        assert evidence["motion_hash"] == motion.motion_hash
        assert evidence["worker_pid"] is None
        packets.append((packet, packet.read_bytes()))
    monkeypatch.setattr(DRIVER._runner(), "_post_jobs",
                        lambda *_args, **_kwargs: pytest.fail("failure must not be silently repaired"))
    resumed = DRIVER.execute_task(task, motion, root, fixture_config(task))
    for dialect in C.Dialect:
        outcome = resumed["dialects"][dialect.value]
        assert outcome["status"] == "failed"
        assert outcome["failure_class"] == "previous_failure"
        assert "injected controller-unit failure" in outcome["reason"]
    assert all(path.read_bytes() == original for path, original in packets)


def test_real_child_timeout_uses_exact_pid_cleanup(tmp_path, monkeypatch):
    task, motion = tiny_task("synthetic-real-timeout")
    config = fixture_config(task, parallel=1, timeout_s=0.01)
    workspace = tmp_path / "timeout"
    job, _comparison = DRIVER._nc_artifacts(task, motion, C.Dialect.RS274, workspace, config)
    helper = DRIVER._runner()
    real_popen = helper.subprocess.Popen
    children = []

    def tracked_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(helper.subprocess, "Popen", tracked_popen)
    with pytest.raises(helper.RunnerError, match="timeout"):
        helper._posted_job_process(task, job, config)
    assert len(children) == 1
    assert children[0].poll() is not None
    assert children[0].returncode != 0
    assert not (workspace / "evaluation.json").exists()
    assert list(workspace.glob("worker.*.log"))


def test_real_child_memory_limit_and_varied_polling(tmp_path, monkeypatch):
    task, motion = tiny_task("synthetic-real-memory-cap")
    config = fixture_config(task, parallel=1, memory_limit=1, timeout_s=120.0)
    workspace = tmp_path / "memory-limit"
    job, _comparison = DRIVER._nc_artifacts(task, motion, C.Dialect.RS274, workspace, config)
    helper = DRIVER._runner()
    real_popen = helper.subprocess.Popen
    children = []

    def tracked_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(helper.subprocess, "Popen", tracked_popen)
    with pytest.raises(helper.RunnerError, match="memory_limit|memory ceiling"):
        helper._posted_job_process(task, job, config)
    assert len(children) == 1
    assert children[0].poll() is not None
    assert children[0].returncode != 0


def test_batch_concurrency_eight_is_four_bounded_pairs(tmp_path, monkeypatch):
    """Scheduler unit test; its synthetic failures are not evaluation evidence."""
    tasks, motions, entries, details = [], [], [], {}
    template, _motion = tiny_task()
    for index in range(12):
        task = replace(template, task_id=f"synthetic-scheduler-{index}")
        motion = seed_for_task(task)
        tasks.append(task)
        motions.append(motion)
        entries.append({"task_id": task.task_id, "task_hash": task.task_hash,
                        "motion_hash": motion.motion_hash, "task_record": json.loads(C.dumps(task))})
        details[task.task_id] = {"task_hash": task.task_hash, "split": task.split.value,
                                 "design_seed": index, "posted_qualification": "pending"}
    document = {
        "schema_version": "t2.3-procedural/2", "generation_version": "t2.3/2",
        "tasks": entries, "task_details": details,
        "pending_posted_qualification": {"status": "pending", "grid_pitch_mm": 0.125,
                                        "dialects": [dialect.value for dialect in C.Dialect]},
    }
    manifest = tmp_path / "task_manifest.json"
    manifest.write_text(DRIVER._json_bytes(document).decode(), encoding="utf-8")
    active = 0
    maximum = 0
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def fake_run_one(task, motion, task_root, config):
        nonlocal active, maximum
        assert config["refinement_check"]["parallel_jobs"] == 2
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            barrier.wait(timeout=10)
            return {
                "task_id": task.task_id, "task_hash": task.task_hash,
                "motion_hash": motion.motion_hash,
                "dialects": {dialect.value: {
                    "status": "failed", "failure_class": "scheduler_unit_fixture",
                    "reason": "deliberate scheduler-unit failure; no evaluator was invoked",
                    "reused": False,
                } for dialect in C.Dialect},
                "efficiency_eligible": False,
            }
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(DRIVER, "_run_one", fake_run_one)
    report = DRIVER.run_batch(
        manifest, tmp_path / "batch", concurrency=8, timeout_s=120.0,
        memory_limit=1024**3, poll_interval=0.01,
        require_full=False, require_clean=False,
    )
    assert maximum == 4  # four pairs times two real helper children = at most 8
    assert report["status"] == "complete_with_failures"
    assert report["provenance"]["source_tree_clean"] is False
    declaration = json.loads((tmp_path / "batch" / "batch-declaration.json").read_text())
    assert declaration["concurrency"] == 8
    assert declaration["pair_concurrency"] == 4
    manifest_result = json.loads(DRIVER.apply_manifest(document, report["tasks"], report["provenance"]))
    assert manifest_result["posted_qualification_batch"]["counts_by_family_split"]["pocket_island/dev"] == {
        "qualified": 0, "not_qualified": 0, "failed": 12, "total": 12,
    }
    with DRIVER._batch_lock(tmp_path / "batch"):
        with pytest.raises(DRIVER.QualificationError, match="owns the batch lock"):
            with DRIVER._batch_lock(tmp_path / "batch"):
                pass


def test_manifest_qualification_rule_and_report_revalidation(real_result, tmp_path):
    task = real_result["task"]
    document = task_manifest(task, real_result["motion"], tmp_path / "manifest.json")
    provenance = dict(DRIVER._source_snapshot(require_clean=False),
                      memory_limit_bytes_per_job=1024**3,
                      timeout_seconds_per_job=120.0, poll_interval_seconds=0.01)
    text = DRIVER.apply_manifest(document, {task.task_id: real_result["result"]}, provenance)
    updated = json.loads(text)
    posted = updated["task_details"][task.task_id]["posted_qualification"]
    assert posted["status"] == "qualified"
    assert posted["efficiency_eligible"] is True
    assert posted["reason"] is None
    assert "pending_posted_qualification" not in updated
    for key, value in document["task_details"][task.task_id].items():
        if key != "posted_qualification":
            assert updated["task_details"][task.task_id][key] == value
    assert updated["sentinel_design_declaration"] == document["sentinel_design_declaration"]
    again = DRIVER.apply_manifest(updated, {task.task_id: real_result["result"]}, provenance)
    assert again == text  # no timing fields were added by this direct manifest call

    tampered = deepcopy(real_result["result"])
    tampered["dialects"]["rs274"]["safety_pass"] = False
    tampered["dialects"]["rs274"]["efficiency_eligible"] = False
    with pytest.raises(DRIVER.QualificationError, match="immutable evidence"):
        DRIVER.apply_manifest(document, {task.task_id: tampered}, provenance)
    tampered = deepcopy(real_result["result"])
    tampered["dialects"]["haas"]["evaluation_path"] = real_result["result"]["dialects"]["rs274"]["evaluation_path"]
    with pytest.raises(Exception):
        DRIVER.apply_manifest(document, {task.task_id: tampered}, provenance)


def test_actual_not_qualified_result_retains_safety_reason(tmp_path):
    task, motion = tiny_task("synthetic-holder-collision", unsafe_tool=True)
    root = tmp_path / "tasks" / task.task_id
    result = DRIVER._run_one(task, motion, root, fixture_config(task))
    assert result["efficiency_eligible"] is False
    for dialect in C.Dialect:
        outcome = result["dialects"][dialect.value]
        assert outcome["status"] == "evaluated"
        assert outcome["qualified"] is False
        assert outcome["coverage"] == "complete"
        assert outcome["safety_pass"] is False
    document = task_manifest(task, motion, tmp_path / "manifest.json")
    # This direct-object API fixture does not relax production canonical-seed
    # reconstruction: it is specifically not a replaceable cohort identity.
    with pytest.raises(ValueError, match="holder above stock"):
        DRIVER.load_cohort(tmp_path / "manifest.json", require_full=False)
    records = {dialect.value: DRIVER._manifest_dialect(result["dialects"][dialect.value])
               for dialect in C.Dialect}
    assert all(record["status"] == "not_qualified" for record in records.values())
    assert all(record["safety_pass"] is False for record in records.values())
    assert all(record["efficiency_eligible"] is False for record in records.values())
    assert all(DRIVER._qualified_of(result["dialects"][name]) is False for name in records)
    for record in records.values():
        assert record["coverage_reasons"] == []
        typed = C.loads(Path(record["evaluation_path"]).read_text())
        assert typed.safety_pass is False
        assert typed.metrics["collision_sample_count"].value > 0


def test_clean_tree_snapshot_rejects_uncommitted_repo(tmp_path, monkeypatch):
    def fake_run(*_args, **_kwargs):
        return type("Result", (), {"stdout": " M configs/task_manifest.json\n"})()

    monkeypatch.setattr(DRIVER.subprocess, "run", fake_run)
    with pytest.raises(DRIVER.QualificationError, match="clean committed tree"):
        DRIVER._source_snapshot(repository=tmp_path)
