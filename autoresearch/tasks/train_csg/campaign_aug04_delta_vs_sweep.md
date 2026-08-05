# Delta vs Sweep — local campaign, 2026-08-04/05 (RTX 4070 Laptop, seed 1)

Sweep = Aditya's waypoint/B-spline method at its tuned operating point (2000 iters, T=832, n_ctrl=139, raster_arc init, amin refresh 4). Delta = Nathan's champion configuration (multidepth_cavity init, T=128, dt=0.45; 5000 iters on analytic shapes, 1500 time-boxed on STEP grids — flatline-vs-carve resolves in <100 iters; rrph_hi carries the full 3x5000-iter record in delta_vs_sweep_step.md). Every run exports Haas G-code; 'rtrip' is the G-code-vs-sim carve dice from the eval stage (1.000 = the .nc reproduces the simulated carve exactly). Safety = no tool break, no part break, no holder-stock overlap under the scheduled feed. VRAM column: sweep runs carry commit c65fe2d (dead-adjoint fix).

| target | method | hard dice | baseline | improv | soft | train s | air % | Fmax N | roundtrip | safety |
|---|---|---|---|---|---|---|---|---|---|---|
| sphere | delta | **0.6196** | 0.5480 | +0.159 | 0.752 | 1415 | 86% | 0 | 1.0000 | OK |
| sphere | sweep | **0.8404** | 0.5480 | +0.647 | 0.000 | 459 | 27% | 64 | 1.0000 | OK |
| cylinder | delta | **0.7752** | 0.7175 | +0.204 | 0.936 | 1408 | 80% | 0 | 1.0000 | OK |
| cylinder | sweep | **0.9209** | 0.7175 | +0.720 | 0.000 | 409 | 41% | 35 | 1.0000 | OK |
| box | delta | **0.8144** | 0.8144 | +0.000 | 0.876 | 1409 | 100% | 0 | 1.0000 | OK |
| box | sweep | **0.9111** | 0.8144 | +0.521 | 0.000 | 406 | 54% | 24 | 1.0000 | OK |
| pyramid | delta | **0.4120** | 0.3729 | +0.062 | 0.849 | 1405 | 42% | 0 | 1.0000 | OK |
| pyramid | sweep | **0.8369** | 0.3729 | +0.740 | 0.000 | 410 | 22% | 73 | 1.0000 | OK |
| grid:rrph_hi | delta | **0.6870** | 0.6870 | +0.000 | 0.754 | 376 | 100% | 0 | — | OK |
| grid:rrph_hi | sweep | **0.9742** | 0.6870 | +0.918 | 0.000 | 2060 | 50% | 90 | — | OK |
| grid:extrusion_hi | delta | **0.6441** | 0.6441 | +0.000 | 0.614 | 1074 | 100% | 0 | — | OK |
| grid:extrusion_hi | sweep | **0.6722** | 0.6441 | +0.079 | 0.000 | 1132 | 57% | 90 | — | OK |
| grid:titan_hi | delta | **0.6080** | 0.6060 | +0.005 | 0.611 | 3446 | 94% | 0 | — | OK |
| grid:bowl_hi | delta | **0.0591** | 0.0590 | +0.000 | 0.052 | 4698 | 4% | 0 | — | OK |

## TACC A100 cells (Aug 5, ls6 gpu-a100, commit c65fe2d fix + taichi 1.7.2 pin)

| target | method | hard dice | baseline | improv | train s | peak VRAM | note |
|---|---|---|---|---|---|---|---|
| grid:rrph_hi | sweep | **0.9739** | 0.6870 | +0.917 | 844 | 2.6 GB | validation: reproduces local 0.9742; VRAM halved by the dead-adjoint fix (was 5.1 GB) |
| grid:titan_hi | sweep | **0.8616** | 0.6060 | +0.649 | 2778 | 3.9 GB | full T=832 (local 8 GB card OOM'd pre-fix); delta cell sits at baseline |
| grid:bowl_hi | sweep | **0.0584** | 0.0590 | -0.001 | 4292 | 5.4 GB | **both methods fail bowl**: ~94% of a 260x260x76 mm block must go — orders beyond a T=832 program; the scaling-matrix motivation in one row |

A100 vs 4070 training time (same config, rrph sweep): 844 s vs 2060 s (2.4x).

## Resolution/T scaling matrix redo (array 3341627, A100-40GB, sha cd96daa)

July's attempt died twice over: the mis-tagged taichi 1.7.4 wheel (claims
manylinux_2_27, needs GLIBC_2.32 — instant ImportError on ls6's glibc 2.28)
killed fresh envs, and the (T+1)*N^3 stock adjoint OOM'd the rest. Redo with
the dead-adjoint fix (c65fe2d) + taichi==1.7.2 pin, vs July per cell:

| method | N | T | July vram | redo vram | outcome change |
|---|---|---|---|---|---|
| sweep | 96 | 128 | 1142 MB | **694 MB** | ok, -adjoint exactly |
| sweep | 128 | 256 | 4406 MB | **2358 MB** | ok, -2048 MB = T=256 adjoint |
| sweep | 192 | 128 | 7350 MB | **3862 MB** | ok — sweep now ~half of delta (7254 MB) |
| delta | any | any | unchanged | unchanged | fix is sweep-only, as designed |
| sweep | 128 | 1024 | fail (OOM) | fail (**int32**) | alloc now SUCCEEDS (8.5 GB recorded), then Taichi asserts `total_n <= int32max` |
| sweep | 256 | 128 | fail (OOM) | fail (**int32**) | same: 129*256^3 = 2.16e9 > 2^31 |
| sweep | 128/320 | 2048/128 | oom | cuda_illegal_address | int32 overflow wrapping indices in another kernel path |
| both | 128 | 5120 | oom | oom | genuine bytes > 40 GB |

**Finding: memory is no longer sweep's binding wall — indexing is.** The
stock history field's ELEMENT COUNT (T+1)*N^3 crosses Taichi's int32
dense-field limit (2.147e9) at exactly T=1024/N=128 and N=256/T=128; past it,
kernels either assert (demote_dense_struct_fors) or wrap into illegal
addresses. The tier-2 fix (2-slot rolling stock for sweep, needs rolling
per-step diagnostics) caps elements at 2*N^3 — int32-safe to N~1000 — and
makes sweep memory T-independent. It is not an optimization; it is the
enabling change for T>=1024 or N>=256.

## Limits campaign (rolling stock, commit cc59940; A100 fleet 3341748 + matrix 3341749, Aug 5)

The 2-slot rolling stock (slot 0 = pristine, slot 1 = in-place working stock;
history diagnostics recomputed by one cached replay) removes both walls and,
by retiring the bulk eval kernels, makes sweep ~3x faster wall-clock at equal
iters. Validated: delta byte-identical (rrph champion config, exact hard
metrics + identical peak VRAM); sweep sphere reproduces to best-checkpoint
noise; titan T=832 runs on the 8 GB 4070 at 1.25 GB.

**Probe matrix (iters=2):** every sweep cell passes — T-axis at N=128 up to
**T=20,480 at a flat 310 MB** (memory provably T-independent; 156x delta's
numeric T ceiling), N-axis at T=128 up to **N=896 at 22.2 GB** (bytes match
2*N^3 + aux fields exactly; the 40 GB frontier extrapolates to N~1024 =
0.1 mm voxels on a 100 mm part), corners N=256/T=2048 (758 MB) and
N=512/T=832 (4.3 GB) both fine. Delta control rows fail with exit 134
exactly as in July — the fix touched nothing on that path.

**Quality at scale (2000 iters, seed 1):**

| target | T=832 | T=1536 | T=2048 | T=2560 | T=4096 | reading |
|---|---|---|---|---|---|---|
| titan_hi | 0.8616 | **0.9462** | — | 0.9342 | — | path-starved until T~1536, then iteration-bound: more ctrl points dilute a fixed 2000-iter budget |
| rrph_hi | 0.9742 | — | **0.9742** | — | — | control: at the reachability ceiling, 2.5x more program changes nothing |
| bowl_hi | (OOM pre-fix) | — | 0.0644 | — | 0.0712 | T=8192 (1000 it): **0.0800**. +~0.008 per T-doubling at 0.7 GB flat: removal-volume arithmetic, not simulator limits — reaching ~0.9 would need T in the hundreds of thousands. Structural next step: multi-pass roughing |

**The limits of the algorithm, stated:** with the rolling stock, sweep's
scaling is bounded by (1) compute time, linear in T*N^3 per argmin refresh;
(2) ~N=1024 on 40 GB from the auxiliary N^3 fields; (3) optimization budget —
past coverage sufficiency, quality declines unless iters scale with n_ctrl;
and (4) single-continuous-path removal volume on bulk-removal parts (bowl),
which no feasible T fixes — the structural next step is CAM-style multi-pass
roughing. Force scheduling held <=90 N with zero tool/part breakage across
every run at every scale.

Historical anchors: delta-on-rrph 3x5000 iters = 0.6870 flatline (delta_vs_sweep_step.md); delta analytic bests from the July W&B evals = sphere 0.9306 (physically invalid), cylinder 0.9162, box 0.8921, pyramid 0.8565 (hard, executed); Nathan's feedback-loop champion sphere 0.741.
