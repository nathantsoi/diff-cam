"""Per-segment feed scheduling in the G-code posts (feed_mult.npy plumbing).

The physics package's deployable close-out is an OFFLINE feed schedule: at
fixed path geometry, cutting force is linear in feed, so slowing exactly the
violating segments removes tool/part breakage. ``train_csg.py`` saves that
schedule as ``feed_mult.npy`` next to ``trajectory.npy``; these tests pin down
that the posts emit it faithfully (segment ``s`` = move ``P[s] -> P[s+1]``
carries feed ``F_base * mult[s]``), that the schedule survives a parser round
trip, and that runs WITHOUT a schedule produce byte-identical programs to the
pre-scheduling exporter.
"""

import numpy as np
import pytest

from cam import MachineConfig, trajectory_to_gcode, parse_gcode

STOCK_IN = (1.0, 1.0, 1.0)


def _cfg(**kw):
    kw.setdefault("stock_size_in", STOCK_IN)
    return MachineConfig(**kw)


def _path(n=6):
    """n waypoints marching in x at fixed y/z."""
    P = np.tile(np.array([[0.2, 0.5, 0.5]]), (n, 1))
    P[:, 0] = np.linspace(0.2, 0.8, n)
    return P


def _cut_feeds(text, cfg):
    """Per-cutting-segment feed (mm/min) as the parser understands them."""
    segs = parse_gcode(text, cfg)
    return [s.feed for s in segs if s.kind == "feed"]


@pytest.mark.parametrize("post", ["rs274", "haas"])
def test_schedule_reaches_the_parser(post):
    cfg = _cfg(feed=600.0)
    P = _path(6)                                # 5 segments
    mult = np.array([1.0, 1.0, 0.5, 0.5, 0.25])
    text = trajectory_to_gcode(P, cfg, post=post, feed_mults=mult)
    feeds = _cut_feeds(text, cfg)
    # The Haas post prepends a plunge move to the first waypoint (a "feed"
    # segment at plunge_feed); the trajectory's own segments are the last 5.
    feeds = feeds[-5:]
    assert feeds == pytest.approx(list(600.0 * mult), abs=1e-6)


@pytest.mark.parametrize("post", ["rs274", "haas"])
def test_no_schedule_is_byte_identical_to_legacy(post):
    """feed_mults=None and an all-ones schedule both collapse to the single
    modal F word the exporter has always emitted."""
    cfg = _cfg()
    P = _path(6)
    legacy = trajectory_to_gcode(P, cfg, post=post)
    ones = trajectory_to_gcode(P, cfg, post=post, feed_mults=np.ones(5))
    assert legacy == ones
    # Exactly one standalone modal F word in both dialects (haas's plunge feed
    # rides inline on its G01 Z line, not on a standalone F line).
    body = [l for l in legacy.splitlines() if l.startswith("F")]
    assert len(body) == 1


def test_modal_emission_only_on_change():
    cfg = _cfg(feed=600.0)
    P = _path(7)                                # 6 segments
    mult = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0])
    text = trajectory_to_gcode(P, cfg, post="rs274", feed_mults=mult)
    f_words = [l for l in text.splitlines() if l.startswith("F")]
    # 600 -> 300 -> 600: three modal F words, not six.
    assert f_words == ["F600", "F300", "F600"]


@pytest.mark.parametrize("bad", [np.ones(3), np.ones(7),
                                 np.array([1.0, 0.0, 1.0, 1.0, 1.0]),
                                 np.array([1.0, np.nan, 1.0, 1.0, 1.0])])
def test_bad_schedules_are_rejected(bad):
    cfg = _cfg()
    P = _path(6)
    with pytest.raises(ValueError):
        trajectory_to_gcode(P, cfg, post="haas", feed_mults=bad)


def test_inch_units_scale_scheduled_feeds():
    cfg = _cfg(feed=600.0, units="inch")
    P = _path(3)
    mult = np.array([1.0, 0.5])
    text = trajectory_to_gcode(P, cfg, post="rs274", feed_mults=mult)
    f_words = [l for l in text.splitlines() if l.startswith("F")]
    # 600 mm/min = 23.622 ipm; half feed = 11.811 ipm.
    assert f_words[0].startswith("F23.622")
    assert f_words[1].startswith("F11.811")
