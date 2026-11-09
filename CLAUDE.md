# Model Restrictions
- When using a text-only local model (GLM 5.2), do NOT attempt to read, parse, or invoke visual/image tools.
- Treat all media paths (.png, .jpg, .pdf) strictly as plain-text filenames.

# Project authority
- Task registry (acceptance criteria are authoritative): `tasks.md`.
- Agent cards, file leases, waves, handoff format: `detailed_plan.md`.
- Scientific rationale and source links: `reports/experimental_plan.md`.
- Live task status ledger (coordinator-only edits): `reports/task_status.json`.
- Older background: `reports/status-and-rss-plan.tex` (implementation audit), `reports/validation.txt`
  (2026-11-08 cleanup validation: 20 CPU tests + optimizer smoke passed in an offline venv; no uv.lock yet).

# Multi-agent workflow
- Coordinator session: `diffcam-orchestrator`. Workers: `diffcam-worker-1/2/3` (max three concurrent).
- Integration branch is `core`, checked out at `/home/ntsoi/src/cnc/diffcam` and owned by the coordinator.
  Workers never switch branches or commit there; each card runs in its own worktree under
  `~/src/cnc/diffcam-wt/<task-id>-<desc>` on branch `task/<task-id>-<desc>`. Only the coordinator merges to `core`.
- Cards and replies: `~/src/cnc/dispatch/diffcam/` (`<task>-<worker>.md` / `.reply.md`; see its README.md).
  Write the reply file, then SendMessage `diffcam-orchestrator` a one-line pointer.
- Coordinator alone edits `pyproject.toml`, `uv.lock`, `.gitignore`, `tasks.md`, `detailed_plan.md`,
  the ledger and `benchmarks/contracts.py`/`docs/contracts.md` (contracts). Request changes in your reply.
- A lease covers a whole file (see detailed_plan.md §1 table). Don't edit files outside your card's ownership.

# Running code
- Always `uv run ...` inside your own worktree. Per-agent caches:
  `TI_OFFLINE_CACHE_FILE_PATH=/tmp/diffcam-ti-<agent> UV_CACHE_DIR=/tmp/diffcam-uv-<agent>`.
- Taichi: `ti.init()` invalidates existing fields — init once per process (`init_taichi=False` for extra simulators)
  or use separate processes. Use a fresh output directory per run.
- Hardware: one RTX 5070 Ti (16 GB), shared. Serialize GPU timing runs; default tests to CPU.
- No large downloads or sweeps as unit-test side effects. External clones live under `external/`,
  bulk data under `data/external/` (both git-ignored); manifest in `configs/references.json`.
- Shared box: kill processes only by exact pid you started; never pkill/kill by pattern.

# Evidence rules
- Never claim a gate passed without running it; report commands + exit codes.
- Record pass / fail / blocked / not-applicable / incomplete-coverage distinctly. Missing tools are "blocked", not passing.
- Mocks are fine for unit development, never as end-to-end evidence. Negative results are valid outcomes.
- Slow opt-in tests: `DIFFCAM_REAL_NOMINAL=1` (real pilot nominal, ~7 min), `DIFFCAM_REAL_PILOT_SMOKE=1` (runner real-task smoke); run alone, required for pilot-affecting changes.
- Units: internal physical dimensions are millimetres; G54 at stock top-centre (see README.md and docs/contracts.md once T0.0 lands).
