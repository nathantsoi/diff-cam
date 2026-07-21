from utils.vram import MiB, VramTracker


def test_vram_tracker_records_device_peak_and_baseline_delta():
    samples = iter([
        (14 * MiB, 16 * MiB),
        (10 * MiB, 16 * MiB),
        (12 * MiB, 16 * MiB),
        (8 * MiB, 16 * MiB),
    ])
    tracker = VramTracker(lambda: next(samples))

    first = tracker.sample()
    second = tracker.sample()
    third = tracker.sample()

    assert first is not None
    assert first.used_mb == 6.0
    assert first.delta_mb == 4.0
    assert second is not None
    assert second.used_mb == 4.0
    assert second.peak_used_mb == 6.0
    assert third is not None
    assert third.used_mb == 8.0
    assert third.peak_used_mb == 8.0

    assert tracker.summary() == {
        "peak_vram_mb": 8.0,
        "peak_vram_delta_mb": 6.0,
        "vram_baseline_mb": 2.0,
        "vram_total_mb": 16.0,
    }


def test_vram_tracker_is_zeroed_when_cuda_is_unavailable():
    tracker = VramTracker(lambda: None)

    assert tracker.sample() is None
    assert tracker.summary() == {
        "peak_vram_mb": 0.0,
        "peak_vram_delta_mb": 0.0,
        "vram_baseline_mb": 0.0,
        "vram_total_mb": 0.0,
    }
