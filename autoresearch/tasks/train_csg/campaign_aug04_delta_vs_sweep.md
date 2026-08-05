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

Pending cells (TACC, awaiting tunnel): sweep on titan_hi / bowl_hi at full T=832 (local 8 GB OOM pre-fix), the N/T resolution-scaling matrix redo, and A100 training-time columns.

Historical anchors: delta-on-rrph 3x5000 iters = 0.6870 flatline (delta_vs_sweep_step.md); delta analytic bests from the July W&B evals = sphere 0.9306 (physically invalid), cylinder 0.9162, box 0.8921, pyramid 0.8565 (hard, executed); Nathan's feedback-loop champion sphere 0.741.
