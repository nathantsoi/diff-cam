"""Part-feature fragility from the target SDF grid (shape-agnostic).

Slender features of the TARGET (a pin, raised lettering, a thin wall) have
finite bending strength: a side force F applied at height h above the feature's
root puts the root in bending, and the feature snaps when the root stress
exceeds the material strength. Machining practice bounds cutting force near
such features ("light passes on thin walls"); this module computes, from the
target SDF alone, the per-voxel allowable side force that encodes that rule.

Method (Telea & Jalba, ISMM 2011 voxel-printability recipe, adapted):

1. thin mask  = part \\ opening(part, ball r_thin) — computed with two
   Euclidean distance transforms (the SDF grid IS an EDT, but we recompute
   binary EDTs so the radii are exact in voxels). A feature survives the
   opening iff its local half-thickness exceeds r_thin; the residue is the
   thin/detail set. Default r_thin = tool radius: features narrower than the
   cutter are "detail" the tool can push on but never support.
2. per-feature decomposition = connected components of the thin mask.
3. lever arm h_feat = max geodesic height of the component above its
   attachment interface to the rump (BFS through the component from the
   interface voxels) — the cantilever length.
4. thickness t_feat = 2 * max interior EDT over the component — the true
   (inscribed-ball) thickness of the feature's core.
5. allowable side force per feature (cantilever root bending). The section
   WIDTH matters: a pin is attached over ~t x t, but a thin rim/edge strip is
   attached along its whole length and is far stronger than a free-standing
   post of the same thickness (the supported- vs unsupported-wire distinction
   in Telea/Shapeways printability rules — treating rims as cantilevers made
   machining their neighborhood look impossible). We take the bending width
   from the attachment footprint, b_eff = A_interface / t (>= t):

       F_allow = sigma_y * t^2 * b_eff / (6 * h)   [N, sigma_y in MPa]

   For a pin b_eff ~ (pi/4) t recovers the t^3 law; for a strip attached
   along L, b_eff ~ L. Components attached at both ends (bridges) or
   unreached voxels keep the conservative single-cantilever value.
6. contact splat: every voxel (waste or part) within contact_mm of a fragile
   feature's surface gets that feature's (F_allow, feature id); everywhere
   else is "safe" (F_ALLOW_SAFE). Cutting a voxel in a feature's contact band
   transmits force into the feature, so per-segment cutting force is capped by
   the weakest feature the segment touches.

Everything is geometry derived from the SDF grid — no shape names, no task
metadata. Pure numpy/scipy, run once at startup (sub-second at campaign grids).
"""

import numpy as np
from scipy import ndimage

# "No cap" sentinel: far above any physical cutting force (N).
F_ALLOW_SAFE = 1e9


def _ball(r_vox):
    """Boolean ball footprint of radius r_vox (voxels)."""
    n = max(1, int(np.floor(r_vox)))
    xx, yy, zz = np.mgrid[-n:n + 1, -n:n + 1, -n:n + 1]
    return (xx * xx + yy * yy + zz * zz) <= r_vox * r_vox


def _geodesic_height(component, seeds):
    """Max BFS depth (voxels) from ``seeds`` through ``component`` (6-conn).

    Returns (height array, max height). Voxels of the component never reached
    from the seeds get the max height (worst case: fully cantilevered).
    """
    h = np.full(component.shape, -1, dtype=np.int32)
    frontier = seeds & component
    h[frontier] = 0
    remaining = component & ~frontier
    level = 0
    st = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    while frontier.any() and remaining.any():
        level += 1
        frontier = ndimage.binary_dilation(frontier, structure=st) & remaining
        h[frontier] = level
        remaining &= ~frontier
    if remaining.any():
        h[remaining] = max(level, 1)
    return h, max(level, 1)


def compute_fragility(target_sdf_vox, voxel_mm, sigma_y_mpa=276.0,
                      r_thin_mm=None, contact_mm=1.0, tool_radius_mm=3.175):
    """Per-voxel allowable-force field + per-feature table from the target SDF.

    Args:
        target_sdf_vox: (Nx,Ny,Nz) signed distance grid, <=0 inside the part
            (values in voxels; only the sign is used).
        voxel_mm: physical voxel edge (mm).
        sigma_y_mpa: part material bending strength (MPa). 276 = Al 6061 yield;
            ~10 = machining wax; ~70 = acrylic.
        r_thin_mm: thinness threshold radius (mm). Features whose local
            half-thickness is below this are fragility candidates. Default
            (None) = tool radius.
        contact_mm: force-transmission distance beyond the feature surface (mm):
            cutting within this band pushes on the feature.
        tool_radius_mm: cutter radius (mm), used for the default r_thin and for
            the tool-center lookup field.

    Returns dict:
        f_allow_vox:  (Nx,Ny,Nz) float32 — allowable force (N) for cutting AT
                      that voxel (feature bands splatted with per-feature
                      F_allow; F_ALLOW_SAFE elsewhere). For the per-voxel
                      (chip-attribution) soft penalty.
        f_allow_tool: (Nx,Ny,Nz) float32 — min of f_allow_vox over a
                      tool-radius ball: allowable force for a tool CENTERED at
                      that voxel. For host-side per-segment hard diagnostics.
        feat_id_vox:  (Nx,Ny,Nz) int16 — 1-based feature id owning each band
                      voxel, 0 = safe. (Nearest fragile feature.)
        features:     list of dicts (id, f_allow_n, t_mm, h_mm, n_voxels).
        thin_mask:    boolean thin-feature mask (part-space).
    """
    part = target_sdf_vox <= 0.0
    shape = part.shape
    f_allow = np.full(shape, F_ALLOW_SAFE, dtype=np.float32)
    feat_id = np.zeros(shape, dtype=np.int16)
    out = {"f_allow_vox": f_allow, "f_allow_tool": f_allow.copy(),
           "feat_id_vox": feat_id, "features": [],
           "thin_mask": np.zeros(shape, dtype=bool)}
    if not part.any():
        return out

    r_thin_vox = (tool_radius_mm if r_thin_mm is None else r_thin_mm) / voxel_mm

    # Opening by a ball of radius r_thin via two EDTs: erosion keeps voxels
    # deeper than r_thin; dilation of the erosion reconstructs the "rump".
    # The dilation radius carries a +1 voxel tolerance: with exactly matched
    # radii, grid quantization leaves sub-voxel sliver shells on smooth thick
    # surfaces (a sphere read back ~144 phantom "features") — a genuinely
    # thin feature clears the tolerance because its whole cross-section, not
    # a fractional-voxel rind, lies outside the eroded core.
    edt_in = ndimage.distance_transform_edt(part)          # voxels, 0 outside
    rump_core = edt_in > r_thin_vox
    if not rump_core.any():
        # The whole part is thinner than the threshold (e.g. tiny target):
        # there is no rigid rump to cantilever from — treat as safe rather
        # than flagging everything (no root, no lever arm).
        return out
    d_from_core = ndimage.distance_transform_edt(~rump_core)
    rump = d_from_core <= r_thin_vox + 1.0                  # opening + tolerance
    thin = part & ~rump
    out["thin_mask"] = thin
    if not thin.any():
        return out

    labels, n_feat = ndimage.label(thin, structure=np.ones((3, 3, 3), bool))
    st6 = ndimage.generate_binary_structure(3, 1)
    interface = thin & ndimage.binary_dilation(rump & part, structure=st6)

    feats = []
    slices = ndimage.find_objects(labels)
    for fid in range(1, n_feat + 1):
        # Crop to the component's bbox (+1 halo) so the BFS scales with the
        # feature, not the grid.
        sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in slices[fid - 1])
        comp = labels[sl] == fid
        n_vox = int(comp.sum())
        iface = interface[sl] & comp
        _, h_max = _geodesic_height(comp, interface[sl])
        t_mm = 2.0 * float(edt_in[sl][comp].max()) * voxel_mm  # inscribed diam
        h_mm = max(float(h_max) * voxel_mm, voxel_mm)
        # Attachment footprint -> effective bending width (>= t): interface
        # voxel count is a surface-area proxy in voxel^2.
        a_iface_mm2 = float(iface.sum()) * voxel_mm ** 2
        b_eff_mm = max(a_iface_mm2 / max(t_mm, voxel_mm), t_mm)
        f_n = sigma_y_mpa * t_mm ** 2 * b_eff_mm / (6.0 * h_mm)
        feats.append({"id": fid, "f_allow_n": float(f_n), "t_mm": t_mm,
                      "h_mm": h_mm, "b_eff_mm": b_eff_mm, "n_voxels": n_vox})
    out["features"] = feats

    # Contact splat: nearest fragile voxel within (contact_mm) of any voxel
    # donates its feature's F_allow. EDT with return_indices gives the nearest
    # fragile voxel per grid voxel in one pass.
    d_frag, (ix, iy, iz) = ndimage.distance_transform_edt(
        ~thin, return_indices=True)
    band = d_frag <= (contact_mm / voxel_mm)
    nearest_fid = labels[ix[band], iy[band], iz[band]]
    f_table = np.array([F_ALLOW_SAFE] + [f["f_allow_n"] for f in feats],
                       dtype=np.float32)
    f_allow[band] = f_table[nearest_fid]
    feat_id[band] = nearest_fid.astype(np.int16)

    # Tool-center lookup: min over a tool-radius ball (grey erosion), so a
    # segment's cap can be read at its sample positions directly.
    r_tool_vox = tool_radius_mm / voxel_mm
    out["f_allow_tool"] = ndimage.grey_erosion(
        f_allow, footprint=_ball(r_tool_vox)).astype(np.float32)
    out["f_allow_vox"] = f_allow
    out["feat_id_vox"] = feat_id
    return out
