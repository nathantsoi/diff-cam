"""Unit seams for RustCNC seed composition, using explicitly synthetic modules.

The fake ports here verify only the coordinator's composition/routing logic.
They are NOT independent CAM algorithms or end-to-end/geometric evidence.
The real four-port construction/posting gates are separate.
"""
from __future__ import annotations

import dataclasses
import json
import math
from types import SimpleNamespace

import pytest

from benchmarks import contracts as C
from benchmarks.procedural_suite import generate_bootstrap_tasks
from cam import rustcnc_seed_adapter as adapter

SHA = "5fc4717a9342313c03461f80a19a79ad8996e757"


def _line(name, start, end, *, rapid=False):
    return C.Segment(id=name, kind=C.SegmentKind.RAPID if rapid else C.SegmentKind.FEED,
                     start_mm=start, end_mm=end,
                     duration_s=math.dist(start, end) / (80.0 if rapid else 6.0),
                     feed_mm_s=None if rapid else 6.0, tag=name)


def _roughing(task, ranges=((2, 3, "outer-wall", 0), (4, 6, "island-wall", 1))):
    points = [
        ((0., 0., 10.), (1., 1., 10.)),
        ((1., 1., 10.), (2., 2., 0.)),
        ((2., 2., 0.), (3., 3., 0.)),
        ((3., 3., 0.), (3., 3., -2.)),
        ((3., 3., -2.), (4., 4., -2.)),
        ((4., 4., -2.), (3., 3., -2.)),
        ((3., 3., -2.), (3., 3., 10.)),
    ]
    segments = tuple(_line(f"native-{index}", *points_, rapid=index == 0)
                     for index, points_ in enumerate(points))
    wall = tuple(SimpleNamespace(role=role, level=level, first_segment=first,
                                  end_segment_exclusive=end)
                 for first, end, role, level in ranges)
    motion = C.MotionPlan(C.Frame.MACHINE, task.initial_state, segments,
                          C.TimingRule.EXACT_STOP_CONSTANT_SPEED,
                          provenance=json.dumps({"strategy_id": "cam.pocket2d"}))
    return SimpleNamespace(motion=motion, wall_finish_passes=wall)


def _fake_ports(monkeypatch, roughing, *, discontinuity=False):
    calls = []
    pocket = SimpleNamespace(emit_pocket_roughing=lambda task, spec, max_segments: roughing)

    def fake_contour(task, points, *, segment_id_prefix, start_machine_mm,
                     contour_z, additional_walls=(), **_kwargs):
        start = tuple(start_machine_mm)
        if discontinuity:
            start = (start[0] + 1.0, start[1], start[2])
        if start[2] < task.stock.params["max_mm"][2]:
            raise RuntimeError("test fake requires a high start")
        first = _line(segment_id_prefix + "-entry", start, (0., 0., 10.), rapid=True)
        next_ = _line(segment_id_prefix + "-finish", first.end_mm, (0., 0., contour_z))
        calls.append({"prefix": segment_id_prefix, "points": points, "options": _kwargs.get("spec"),
                      "additional_walls": additional_walls})
        exit_ = _line(segment_id_prefix + "-exit", next_.end_mm, first.end_mm)
        return (first, next_, exit_)

    contour = SimpleNamespace(RUSTCNC_SHA=SHA, CONTOUR_SHA256="sha256:" + "0" * 64,
                              emit_contour_loop=fake_contour)
    monkeypatch.setitem(__import__("sys").modules, "cam.rustcnc_pocket2d", pocket)
    monkeypatch.setitem(__import__("sys").modules, "cam.rustcnc_face_contour", contour)
    monkeypatch.setattr(__import__("cam"), "rustcnc_pocket2d", pocket, raising=False)
    monkeypatch.setattr(__import__("cam"), "rustcnc_face_contour", contour, raising=False)
    return calls


def test_composition_replaces_only_deepest_wall_ranges_and_records_provenance(monkeypatch):
    task = generate_bootstrap_tasks()[2].task
    native = _roughing(task)
    calls = _fake_ports(monkeypatch, native)
    plan = adapter.pocket_with_contour_seed(task, {"maximum_stepdown": 7.0}, max_segments=72)
    assert [segment.id for segment in plan.segments] == [
        "native-0", "native-1", "native-2", "native-3", "native-6",
        "pocket-contour-outer-entry", "pocket-contour-outer-finish", "pocket-contour-outer-exit",
        "pocket-contour-pad-entry", "pocket-contour-pad-finish", "pocket-contour-pad-exit",
    ]
    plan.validate_for_task(task)
    assert len(calls) == 2
    assert calls[0]["options"]["leadInRadius"] == task.tool.cutter_radius_mm / 2.0
    assert calls[0]["options"]["leadOutVerticalRadius"] == task.tool.cutter_radius_mm / 4.0
    assert calls[0]["additional_walls"]
    assert len(calls[1]["additional_walls"]) == 1
    provenance = json.loads(plan.provenance)
    composition = provenance["composition"]
    assert provenance["strategy_id"] == "cam.pocket2d+cam.contour2d"
    assert composition["deepest_native_wall_level_replaced"] == 1
    assert composition["removed_segment_ids"] == ["native-4", "native-5"]
    assert composition["complete_segment_count"] == len(plan.segments)
    assert composition["contour_source_commit"] == SHA


def test_composition_refuses_broken_contacts_and_over_cap(monkeypatch):
    task = generate_bootstrap_tasks()[2].task
    native = _roughing(task)
    _fake_ports(monkeypatch, native)
    with pytest.raises(ValueError, match="requires 11 segments"):
        adapter.pocket_with_contour_seed(task, max_segments=10)
    _fake_ports(monkeypatch, native, discontinuity=True)
    with pytest.raises(C.ContractError, match="discontinuous with previous end"):
        adapter.pocket_with_contour_seed(task, max_segments=72)


def test_composition_requires_real_finish_ranges_never_drops_approach(monkeypatch):
    task = generate_bootstrap_tasks()[2].task
    _fake_ports(monkeypatch, _roughing(task, ranges=()))
    with pytest.raises(ValueError, match="did not identify native wall"):
        adapter.pocket_with_contour_seed(task)
    _fake_ports(monkeypatch, _roughing(task, ranges=((0, 2, "outer-wall", 0),)))
    with pytest.raises(ValueError, match="may not remove.*initial approach"):
        adapter.pocket_with_contour_seed(task)


def test_dispatch_uses_declared_family_not_a_legacy_raster(monkeypatch):
    sentinel = object()
    for name, target in (("rustcnc_face_contour", "face_seed"),
                         ("rustcnc_adaptive2d", "adaptive2d_seed")):
        module = SimpleNamespace(**{target: lambda task, spec, max_segments: sentinel})
        monkeypatch.setitem(__import__("sys").modules, "cam." + name, module)
        monkeypatch.setattr(__import__("cam"), name, module, raising=False)
    monkeypatch.setattr(adapter, "pocket_with_contour_seed",
                        lambda task, spec, max_segments: sentinel)
    for bundle in generate_bootstrap_tasks():
        assert adapter.cam_seed_for_task(bundle.task) is sentinel
    invalid_types = [True, 0, -1, 1.5]
    for value in invalid_types:
        with pytest.raises(ValueError):
            adapter.cam_seed_for_task(bundle.task, max_segments=value)


def test_reference_qualification_uses_explicit_seed_factory(monkeypatch):
    from benchmarks import procedural_suite as suite

    bundle = generate_bootstrap_tasks()[0]
    called = []
    seeded = adapter.cam_seed_for_task(bundle.task)
    bundle = dataclasses.replace(bundle, motion=seeded)
    monkeypatch.setattr(suite, "run_reference",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("probe-stop")))

    def factory(task):
        called.append(task.task_id)
        return seeded

    with pytest.raises(RuntimeError, match="probe-stop"):
        suite._reference_qualify_seed(bundle, seed_factory=factory)
    assert called == [bundle.task.task_id]
