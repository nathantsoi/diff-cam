"""Driver-level GPU-memory sampling.

PyTorch's allocator counters do not include memory allocated by Taichi.  This
module instead uses ``torch.cuda.mem_get_info()``, which queries the CUDA
driver and therefore sees allocations made by both frameworks.  The reported
usage is for the whole visible device, so callers should retain both the
absolute peak and the increase over the pre-run baseline.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


MiB = 1024.0 * 1024.0
MemoryInfo = Tuple[int, int]  # free bytes, total bytes


def _cuda_memory_info() -> Optional[MemoryInfo]:
    """Return CUDA (free, total) bytes, or ``None`` when CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes), int(total_bytes)
    except (RuntimeError, OSError):
        # CPU-only builds and partially configured CUDA nodes should keep
        # training; their metrics retain the historical 0.0 sentinel.
        return None


@dataclass(frozen=True)
class VramSample:
    """One driver-level device-memory observation, measured in MiB."""

    used_mb: float
    delta_mb: float
    total_mb: float
    peak_used_mb: float
    peak_delta_mb: float


class VramTracker:
    """Track sampled CUDA device usage and baseline-adjusted run growth.

    ``used_mb`` includes every process on the visible GPU.  ``delta_mb`` is the
    increase over usage immediately before the simulator is constructed and is
    the more useful quantity on a dedicated compute node when comparing with an
    analytical allocation model.  Sampling cannot observe a short-lived
    allocation that is created and freed entirely between calls, so training
    code should sample after each forward/backward iteration.
    """

    def __init__(self, sampler: Callable[[], Optional[MemoryInfo]] = _cuda_memory_info):
        self._sampler = sampler
        self.available = False
        self.baseline_used_mb = 0.0
        self.total_mb = 0.0
        self.peak_used_mb = 0.0
        self.peak_delta_mb = 0.0
        self.current: Optional[VramSample] = None

        info = self._sampler()
        if info is not None:
            free_bytes, total_bytes = info
            self.available = True
            self.total_mb = total_bytes / MiB
            self.baseline_used_mb = (total_bytes - free_bytes) / MiB
            self.peak_used_mb = self.baseline_used_mb
            self.current = VramSample(
                used_mb=self.baseline_used_mb,
                delta_mb=0.0,
                total_mb=self.total_mb,
                peak_used_mb=self.peak_used_mb,
                peak_delta_mb=0.0,
            )

    def sample(self) -> Optional[VramSample]:
        """Record and return the current device-memory observation."""
        if not self.available:
            return None
        info = self._sampler()
        if info is None:
            return self.current

        free_bytes, total_bytes = info
        used_mb = (total_bytes - free_bytes) / MiB
        delta_mb = max(0.0, used_mb - self.baseline_used_mb)
        self.total_mb = total_bytes / MiB
        self.peak_used_mb = max(self.peak_used_mb, used_mb)
        self.peak_delta_mb = max(self.peak_delta_mb, delta_mb)
        self.current = VramSample(
            used_mb=used_mb,
            delta_mb=delta_mb,
            total_mb=self.total_mb,
            peak_used_mb=self.peak_used_mb,
            peak_delta_mb=self.peak_delta_mb,
        )
        return self.current

    def summary(self) -> dict:
        """Return JSON-ready metrics, preserving the legacy peak field name."""
        return {
            "peak_vram_mb": self.peak_used_mb if self.available else 0.0,
            "peak_vram_delta_mb": self.peak_delta_mb if self.available else 0.0,
            "vram_baseline_mb": self.baseline_used_mb if self.available else 0.0,
            "vram_total_mb": self.total_mb if self.available else 0.0,
        }
