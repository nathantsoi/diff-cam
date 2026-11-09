"""Stateless candidate run artifacts, source capture and compute accounting.

The writer never posts a partial artifact set as completed evidence: every supplied
artifact is replaced atomically and the validated ``run.json`` is written last.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import resource
import subprocess
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from benchmarks import contracts as C


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def record_bytes(record: Any) -> bytes:
    return C.dumps(record).encode("utf-8")


def numpy_bytes(array: Any) -> bytes:
    values = np.asarray(array)
    if values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
        raise ValueError("trajectory arrays must contain finite real numeric values")
    output = io.BytesIO()
    np.save(output, values, allow_pickle=False)
    return output.getvalue()


def source_metadata(repository: str | Path) -> Dict[str, Any]:
    """Capture HEAD, tracked+untracked working-tree content and the core lock hash."""
    root = Path(repository)
    def git(*arguments: str) -> bytes:
        return subprocess.run(["git", "-C", str(root), *arguments], check=True,
                              capture_output=True).stdout

    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    patch = bytearray(git("diff", "--binary", "HEAD", "--"))
    untracked = [line for line in git("ls-files", "--others", "--exclude-standard", "-z")
                 .decode("utf-8").split("\0") if line]
    for name in sorted(untracked):
        patch.extend(name.encode("utf-8") + b"\0" + (root / name).read_bytes() + b"\0")
    lock = root / "uv.lock"
    return {
        "source_commit": commit,
        "dirty_patch_hash": C.sha256_bytes(patch) if patch else None,
        "uv_lock_hash": C.sha256_bytes(lock.read_bytes()) if lock.is_file() else None,
    }


def parameterization_metadata(parameterization: Any) -> Dict[str, Any]:
    if parameterization is None:
        return {"kind": "raw_deltas_mm"}
    result = {
        "class": type(parameterization).__name__,
        "mode": getattr(parameterization, "mode", "unknown"),
        "num_params": int(parameterization.num_params),
        "num_steps": int(parameterization.num_steps),
        "scale_mm": float(getattr(parameterization, "scale_mm", 1.0)),
    }
    bounds = getattr(parameterization, "bounds", None)
    if bounds is not None:
        result["bounds"] = [np.asarray(bound, dtype=float).tolist() for bound in bounds]
    return result


def build_run_config(run_id: str, task: C.TaskSpec, motion: C.MotionPlan, *, seed: int,
                     parameterization: Any, objective: Any, budgets: Mapping[str, Any],
                     repository: str | Path) -> Dict[str, Any]:
    """Required configuration evidence; no runtime-only random hidden state."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if dataclasses.is_dataclass(objective) and not isinstance(objective, type):
        objective = dataclasses.asdict(objective)
    config = {
        "run_id": run_id, "schema_version": C.SCHEMA_VERSION,
        "task": task.to_dict(), "task_hash": task.task_hash, "task_id": task.task_id,
        "task_split": task.split.value, "motion_hash": motion.motion_hash,
        "seed": seed, "parameterization": parameterization_metadata(parameterization),
        "objective": objective, "budgets": dict(budgets),
    }
    config.update(source_metadata(repository))
    json_bytes(config)  # finite/serializable boundary before artifacts are committed.
    return config


@dataclass(frozen=True)
class _Stamp:
    candidate_id: str
    objective_kind: str
    started_s: float
    warmup: bool


class ComputeLedger:
    """Sequential candidate-call accounting; validation calls are a distinct budget.

    Warm-up is declared as the first candidate call, regardless of whether a backend
    had cached kernels. It is not mislabelled as measured JIT-only time. RSS is the
    process high-water mark, not a claim of device/GPU allocation by this caller.
    """

    def __init__(self) -> None:
        self.candidate_evaluations = 0
        self.forward_calls = 0
        self.backward_calls = 0
        self.validation_calls = 0
        self.segment_steps = 0
        self.voxel_steps = 0
        self.entries: list[dict[str, Any]] = []
        self._active: Optional[_Stamp] = None
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def start_candidate(self, candidate_id: str, objective_kind: str) -> _Stamp:
        if self._active is not None:
            raise RuntimeError("ComputeLedger is sequential; finish the active candidate first")
        self._active = _Stamp(candidate_id, objective_kind, time.perf_counter(),
                              warmup=self.candidate_evaluations == 0)
        return self._active

    def finish_candidate(self, *, valid: bool, forward_calls: int, backward_calls: int,
                         segment_steps: int, voxel_steps: int) -> Dict[str, Any]:
        if self._active is None:
            raise RuntimeError("no active candidate")
        stamp, self._active = self._active, None
        elapsed = max(0.0, time.perf_counter() - stamp.started_s)
        self.candidate_evaluations += 1
        self.forward_calls += int(forward_calls)
        self.backward_calls += int(backward_calls)
        self.segment_steps += int(segment_steps)
        self.voxel_steps += int(voxel_steps)
        entry = {
            "ordinal": self.candidate_evaluations,
            "candidate_id": stamp.candidate_id, "objective_kind": stamp.objective_kind,
            "valid": bool(valid), "wall_time_s": elapsed, "warmup": stamp.warmup,
            "forward_calls": int(forward_calls), "backward_calls": int(backward_calls),
            "segment_steps": int(segment_steps), "voxel_steps": int(voxel_steps),
        }
        self.entries.append(entry)
        return entry

    def count_validation(self, count: int = 1) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("validation count must be a positive integer")
        self.validation_calls += count

    def snapshot(self) -> Dict[str, Any]:
        current, python_peak = tracemalloc.get_traced_memory()
        wall = sum(entry["wall_time_s"] for entry in self.entries)
        warm = sum(entry["wall_time_s"] for entry in self.entries if entry["warmup"])
        return {
            "candidate_evaluations": self.candidate_evaluations,
            "forward_calls": self.forward_calls, "backward_calls": self.backward_calls,
            "validation_calls": self.validation_calls,
            "segment_steps": self.segment_steps, "voxel_steps": self.voxel_steps,
            "warmup_wall_time_s": warm, "steady_state_wall_time_s": wall - warm,
            "end_to_end_candidate_wall_time_s": wall,
            "warmup_definition": "first candidate call, including any uncached backend compilation",
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "python_traced_memory_current_bytes": int(current),
            "python_traced_memory_peak_bytes": int(python_peak),
            "memory_scope": "process high-water and Python traced allocations; not per-candidate device memory",
            "entries": self.entries.copy(),
        }


@dataclass
class PostingResult:
    files: Dict[str, bytes]
    posted: tuple[C.PostedProgram, ...]


def post_motion(task: C.TaskSpec, motion: C.MotionPlan) -> PostingResult:
    """Post each declared dialect; unsupported motion produces no NC/eval bytes."""
    from cam.posts import UnsupportedPostError, post_motion_plan

    files: Dict[str, bytes] = {}
    posted = []
    index: Dict[str, Any] = {}
    for dialect in C.Dialect:
        nc_name, eval_name = C.DIALECT_ARTIFACTS[dialect]
        try:
            nc = post_motion_plan(motion, task, dialect).encode("utf-8")
        except UnsupportedPostError as error:
            program = C.PostedProgram(dialect, C.PostStatus.UNSUPPORTED, reason=str(error))
        else:
            files[nc_name] = nc
            program = C.PostedProgram(dialect, C.PostStatus.SUPPORTED,
                                      nc_hash=C.nc_hash(nc), eval_file=eval_name)
        posted.append(program)
        index[dialect.value] = {
            "dialect": dialect.value, "status": program.status.value,
            "nc_hash": program.nc_hash, "eval_file": program.eval_file,
        }
        if program.status != C.PostStatus.SUPPORTED:
            index[dialect.value]["reason"] = program.reason
    files["eval_posted.json"] = json_bytes({"schema_version": 1, "programs": index})
    return PostingResult(files=files, posted=tuple(posted))


class ArtifactWriter:
    """T0.0 writer. ``run_dir`` must be new; ``run.json`` is committed last."""

    @staticmethod
    def _path(root: Path, name: str) -> Path:
        relative = PurePosixPath(name)
        if relative.is_absolute() or not relative.parts or any(part in ("..", ".") for part in relative.parts):
            raise ValueError(f"unsafe artifact relative path: {name!r}")
        candidate = root.joinpath(*relative.parts)
        try:
            candidate.resolve(strict=False).relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"artifact escapes run directory: {name!r}") from error
        return candidate

    @staticmethod
    def _replace_bytes(destination: Path, data: bytes) -> None:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def write(self, run_dir: str, record: C.RunRecord, files: Mapping[str, bytes]) -> None:
        root = Path(run_dir)
        if root.name != record.run_id:
            raise ValueError(f"run directory name {root.name!r} must equal run_id {record.run_id!r}")
        if len(record.artifacts) != len(set(record.artifacts)):
            raise ValueError("duplicate artifact names")
        if set(record.artifacts) != set(files):
            raise ValueError("file payload names must exactly match record.artifacts")
        if "run.json" in files:
            raise ValueError("run.json is generated from the validated RunRecord, not caller bytes")
        root.mkdir(parents=True, exist_ok=False)  # refuse to overwrite/mix existing runs.
        for name in sorted(files):
            payload = files[name]
            if not isinstance(payload, bytes):
                raise TypeError(f"artifact {name} must be bytes")
            destination = self._path(root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._replace_bytes(destination, payload)
        # The only JSON advertised in run.json is generated here, always last.
        self._replace_bytes(self._path(root, "run.json"), record_bytes(record))
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def assemble_record(run_id: str, task: C.TaskSpec, status: C.RunStatus, *,
                    posting: PostingResult, artifact_names: Sequence[str],
                    source_commit: Optional[str] = None, dirty_patch_hash: Optional[str] = None,
                    failure_reason: Optional[str] = None) -> C.RunRecord:
    if status != C.RunStatus.COMPLETE and failure_reason is None:
        failure_reason = "candidate execution or posted evaluation is incomplete"
    return C.RunRecord(
        run_id=run_id, task_hash=task.task_hash, status=status,
        artifacts=tuple(artifact_names), posted=posting.posted,
        source_commit=source_commit, dirty_patch_hash=dirty_patch_hash,
        failure_reason=failure_reason,
    )
