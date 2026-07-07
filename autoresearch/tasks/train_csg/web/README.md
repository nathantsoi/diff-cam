# Interactive results visualization — serving instructions

This directory holds a self-contained dashboard for the train_csg autoresearch
results: a D3.js scatter/bars overview plus a 3D viewer that renders the
**voxelized carved stock** (WebGPU) or a rotatable tool-trajectory plot (SVG
fallback). Open it in a browser from a remote computer via one of the methods
below.

## Files

| file | purpose |
|---|---|
| `index.html` | the dashboard (scatter + per-shape bars + 3D viewer + Idea/Report tabs + Downloads card with on-demand video) |
| `d3.v7.min.js` | vendored D3 v7 (no CDN, no internet needed by the page) |
| `voxel.js` | WebGPU voxelizer + lit-solid renderer (ported from aim3d) |
| `marchingCubesTables.js` | vendored marching-cubes lookup tables used by `voxel.js` |
| `data.json` | all experiment metadata + trajectories + artifact paths. **Gitignored** — regenerated from `results.tsv` + `runs/` by `scripts/build_results_web.py` (see below); it is *not* in the repo, so run the build script once before serving. |

The dashboard also depends on two repo-root scripts (not in this dir):
`scripts/serve_web_https.py` (serves the page over HTTPS + the `__api/video`
on-demand render endpoint) and `scripts/render_run_video.py` (replays a run's
trajectory through the Taichi CSG simulator to make the mp4).

`data.json` is generated from `results.tsv` + the per-run artifacts under
`runs/<run>/`. STL meshes and G-code are served directly from the `runs/` tree
(paths in `data.json` are relative to the repo root), so the page must be served
**from the repo root**.

## Build / refresh the data

`data.json` is **not committed** — it is a generated artifact (gitignored) and
must be built locally from the experiment log + per-run artifacts. From the repo
root, run once before first serving, and again whenever new experiments land:

```bash
uv run python scripts/build_results_web.py
```

This joins `results.tsv` to `runs/`, (re)generates missing Haas G-code for each
run, extracts the tool trajectories, and writes
`web/data.json`. (Takes ~1–2 min, mostly G-code
generation for ~510 runs.) It requires a local `results.tsv` (the autoresearch
harness's experiment log) and the matched run dirs under `runs/` — both are
machine-local and untracked, so a fresh clone has neither until you run
experiments on that host.

## Serve it

**Serve from the repo root** so the `runs/...` download links resolve. There are
two server options; the 3D viewer's behavior depends on which you pick (and on
the browser's WebGPU support).

### Option A — HTTPS (recommended; enables the WebGPU voxel viewer)

WebGPU is gated to "secure contexts" (HTTPS, `localhost`, or `file://`). Over
plain `http://<lan-host>:<port>` Chrome leaves `navigator.gpu` undefined, so the
carved-stock viewer falls back to the SVG trajectory plot. Use the bundled
self-signed HTTPS server so WebGPU activates on any LAN host:

```bash
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
uv run python3 scripts/serve_web_https.py            # default: 0.0.0.0:8443
# overrides: --port 8443 --host 0.0.0.0 --root . --web web
```

This generates a self-signed cert once into `web/.cert/` (gitignored) and serves
the repo root over HTTPS. Open in Chrome:

```
https://robolidar:8443/web/index.html
https://128.83.141.126:8443/web/index.html
```

Chrome will warn about the untrusted cert on first visit — click
**Advanced → Proceed**. The badge in the top-left of the viewport should then
read `WebGPU · 96³ · N tris` and the voxelized carved stock will render.

### Option B — plain HTTP (SVG trajectory viewer only)

If you don't need the WebGPU voxelizer, plain HTTP works and the SVG trajectory
plot (orbit/pan/zoom + playback) is fully functional:

```bash
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
python3 -m http.server 8000 --bind 0.0.0.0
```

Then open in a browser:

```
http://robolidar:8000/web/index.html
```
(or `http://128.83.141.126:8000/web/index.html`)

### Access from a remote computer

If the host's port is firewalled (or you want encryption without self-signed
certs), use an **SSH tunnel** from the remote computer. This works for either
server option — just tunnel the matching port:

```bash
# HTTPS server (Option A) → then open https://localhost:8443/...
ssh -L 8443:localhost:8443 ntsoi@128.83.141.126

# plain HTTP server (Option B) → then open http://localhost:8000/...
ssh -L 8000:localhost:8000 ntsoi@128.83.141.126
```

Keep that session open, then in the remote computer's browser open the
corresponding `localhost:<port>/web/index.html` URL.
`localhost` is a secure context, so WebGPU works through the HTTPS tunnel.

To bind only locally (single-user on the host; safest if you don't want other
LAN hosts to reach it), pass `--host 127.0.0.1` to either server, then tunnel in
as above.

## Using the dashboard

- **Tabs** (top of the page): **Dashboard** (scatter/bars/3D viewer), **Idea**
  (renders `idea.md`), and **Report** (renders `report.md` + the `results_plot.png`
  summary plot). The markdown is fetched live from the task dir (`../idea.md`,
  `../report.md`) and rendered with a small built-in markdown renderer, so it
  stays in sync with the committed files — no rebuild needed.
- **Scatter plot**: each point is one experiment (x = chronological order,
  y = dice), colored by target shape, dimmed for discard/crash. The dashed red
  line is the running best.
- **3D objective scatter** (the "Quality vs air-cut time vs tool-breakage
  probability" card): a rotatable 3D plot of the three trajectory-quality
  objectives — x = dice (higher better), y = air-cut time in seconds (lower
  better), z = tool-breakage probability (lower better; sqrt-scaled so the tiny
  values spread out). Drag to orbit, wheel to zoom. Points are colored by shape,
  the yellow ring marks the Pareto-best run (high dice, low air, low break), and
  clicking a point opens the detail panel just like the 2D scatter. Only runs
  whose metrics include `air_time` / `break_prob_any` appear (29 of the 32 in
  the jul6-traj-quality batch).
- **Hover** any point for its description + dice + status + air/break summary.
- **Click** a point to open the detail panel: metadata, full metrics (including
  `hard_dice`, `air_time`, `total_time`, `break_prob_any`/`break_prob_max`, and
  `air_cut_fraction`), the **3D viewer**, and a **Downloads** card at the bottom
  of the dashboard with links to that run's Haas G-code, the three STL meshes
  (`stock_initial`, `stock_carved`, `target`), and the run directory.
- **Generate run video** (button in the Downloads card, run selected): renders
  that run's carve as an mp4 by replaying its saved trajectory through the
  Taichi CSG simulator (`scripts/render_run_video.py`, the same raymarch+ffmpeg
  path training uses) and shows it inline in a player. The first click runs the
  simulator (~30–60s, auto-picks the GPU with the most free memory); the mp4 is
  cached at `runs/<run>/videos/run.mp4` so later views are instant. Click the
  button again to force a re-render. **Requires the HTTPS server** — the button
  calls a `GET /__api/video?run=...` endpoint implemented in
  `scripts/serve_web_https.py`; plain `python3 -m http.server` has no such
  endpoint and the button will report that.
- **3D viewer** (WebGPU path, badge reads `WebGPU · 96³ · N tris`): renders the
  voxelized carved stock with lit marching-cubes mesh extraction, the stock
  wireframe, an XYZ axis triad, the reached (bright) and not-yet-reached (dim)
  tool path, and a colored tool-tip marker at the current step. Drag to orbit,
  shift-drag to pan, wheel to zoom. The **playback bar** animates the carve
  (▶/⏸ play, ◀ ▶ step, speed slider, scrubber); the **nav-cube** snaps to
  standard views (FRONT/BACK/LEFT/RIGHT/TOP/BOT/ISO); the **⚙ settings menu**
  toggles the stock cube, axes, and commanded (pre-clip) path.
- **3D viewer** (SVG fallback, badge reads `SVG fallback`): the same orbit/pan/
  zoom, nav-cube, playback bar, and settings, rendered as a painter's-algorithm
  SVG trajectory plot (solid = actual speed-clipped path, dashed = commanded
  pre-clip, green ● start, red ● end).
- The **per-shape bar chart** is clickable: click a bar to jump to that shape's
  best experiment.
- Filter by shape / status using the dropdowns in the header.

## Notes

- **Browser support for WebGPU**: Chrome/Edge ≥113 on a machine with a working
  GPU driver. Check `chrome://gpu` — "WebGPU: Hardware accelerated" must be
  listed. Firefox does not enable WebGPU by default. On macOS Safari, use a
  recent Safari Technology Preview. Without WebGPU the page automatically falls
  back to the SVG trajectory viewer.
- G-code is Haas-format (`gcode_haas.nc`). Re-run the build script after new
  training runs to refresh `data.json` and generate their G-code.
- `data.json` is gitignored and ~4 MB; the browser loads it once on page open.
  Trajectories are embedded so click-through is instant. If the page shows no
  experiments, run `scripts/build_results_web.py` from the repo root first.
