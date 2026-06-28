## Size and Scale

Simulation envelope is configurable, default = Haas Mini Mill (16×12×10 in).

The simulator was refactored from an isotropic unit cube + single workspace_mm scalar to a faithful, anisotropic, cubic-voxel model.

- Geometry stays in normalized [0,1]³ (trajectories/actions/camera unchanged), but the voxel grid now uses per-axis dims (Nx,Ny,Nz) so every voxel is a physical cube of side v mm. Default envelope = Mini Mill (406.4 × 304.8 × 254.0 mm).
- Internal SDF distances are now measured in voxels (isotropic, undistorted) by scaling [0,1] differences by (Nx,Ny,Nz); tool/target sizes are now millimetres.
- resolution = voxels along the longest axis. Speed clipping uses the per-axis envelope.
- Verified: physical voxel sizes equal on all axes (cubic), forward+backward gradients finite, speed clipping intact, training + Mini Mill G-code export produce correct physical mm coordinates.

I propagated this through every consumer: cam_env (obs/action dims now from the real grid), train_csg, eval_csg, sim_exec, and the CAM layer (MachineConfig.workspace_in/workspace_vec, posts, parser, planner, export script) so anisotropic G-code round-trips exactly (verified <1e-16). All 28 tests pass; cube round-trips preserved via scalar back-compat.

One important bug I caught and fixed mid-way: because SDFs are now in voxels (range ±28 vs ±0.87), the carve smooth_max's k had to be rescaled (k/k_ref) — otherwise exp(k·voxel_sdf) overflowed and NaN'd the backward pass while the forward looked fine.

### Resolution

For non-sub-voxel cuts (computed against the Mini Mill volume; the stock[max_steps+1, Nx,Ny,Nz] buffer is the memory bottleneck, shown at the default 64 steps):

┌────────┬─────────────┬──────────────┬───────────┬───────────┐
│ voxel  │    grid     │ stock buffer │ 1/4″ tool │ 1/2″ tool │
├────────┼─────────────┼──────────────┼───────────┼───────────┤
│ 4 mm   │ 102×76×64   │ 0.13 GB      │ 1.6 vox   │ 3.2 vox   │
├────────┼─────────────┼──────────────┼───────────┼───────────┤
│ 3 mm   │ 135×102×85  │ 0.30 GB      │ 2.1 vox   │ 4.2 vox   │
├────────┼─────────────┼──────────────┼───────────┼───────────┤
│ 2 mm   │ 203×152×127 │ 1.02 GB      │ 3.2 vox   │ 6.3 vox   │
├────────┼─────────────┼──────────────┼───────────┼───────────┤
│ 1.5 mm │ 271×203×169 │ 2.42 GB      │ 4.2 vox   │ 8.5 vox   │
├────────┼─────────────┼──────────────┼───────────┼───────────┤
│ 1 mm   │ 406×305×254 │ 8.2 GB       │ 6.3 vox   │ 12.7 vox  │
└────────┴─────────────┴──────────────┴───────────┴───────────┘

Recommendation: ~2 mm voxels (--resolution 200) — a 1/4″ cutter spans ~3 voxels, ~1 GB stock buffer (halve --max_steps for ~0.5 GB). Below 1.5 mm memory explodes; above 3 mm a 1/4″ cutter is barely 2 voxels wide.

There's also a temporal constraint: at dt=0.01 the feed step is ~0.042 mm — sub-voxel in time — so to advance ~1 voxel per feed step, pair ~2 mm voxels with --dt ≈ 0.3–0.5 (≈ voxel/feed). This is documented in the README alongside the table.