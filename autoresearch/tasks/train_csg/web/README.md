# Interactive results visualization — serving instructions

This directory holds a self-contained D3.js dashboard for the train_csg
autoresearch results. Open it in a browser from a remote computer via one of the
methods below.

## Files

| file | purpose |
|---|---|
| `index.html` | the dashboard (scatter + per-shape bars + 3D trajectory + download links) |
| `d3.v7.min.js` | vendored D3 v7 (no CDN, no internet needed by the page) |
| `data.json` | all experiment metadata + trajectories + artifact paths (regenerate with `scripts/build_results_web.py`) |

`data.json` is generated from `results.tsv` + the per-run artifacts under
`runs/<run>/`. STL meshes and G-code are served directly from the `runs/` tree
(paths in `data.json` are relative to the repo root), so the page must be served
**from the repo root**.

## Build / refresh the data

From the repo root (run once, or whenever new experiments land):

```bash
uv run python scripts/build_results_web.py
```

This joins `results.tsv` to `runs/`, (re)generates missing Haas G-code for each
run, extracts the tool trajectories, and writes
`autoresearch/tasks/train_csg/web/data.json`. (Takes ~1–2 min, mostly G-code
generation for ~510 runs.)

## Serve it

**Serve from the repo root** so the `runs/...` download links resolve. Pick a
port (e.g. `8000`):

```bash
cd /home/ntsoi/papers/icra26-diffcam/diff-cam
python3 -m http.server 8000 --bind 0.0.0.0
```

Then open in a browser:

```
http://robolidar:8000/autoresearch/tasks/train_csg/web/index.html
```
(or `http://128.83.141.126:8000/autoresearch/tasks/train_csg/web/index.html`)

### Access from a remote computer

**Option A — direct over LAN** (if the remote machine can reach the host and
port `8000` is open in the firewall):
```
http://128.83.141.126:8000/autoresearch/tasks/train_csg/web/index.html
```
If the port is firewalled, use Option B.

**Option B — SSH tunnel** (recommended; no firewall changes, encrypted):
from the **remote computer**, run:
```bash
ssh -L 8000:localhost:8000 ntsoi@128.83.141.126
```
keep that session open, then in the remote computer's browser open:
```
http://localhost:8000/autoresearch/tasks/train_csg/web/index.html
```

**Option C — bind only locally** (single-user on the host; safest if you don't
want other LAN hosts to reach it):
```bash
python3 -m http.server 8000 --bind 127.0.0.1
```
then tunnel in per Option B, or open `http://localhost:8000/...` on the host.

## Using the dashboard

- **Scatter plot**: each point is one experiment (x = chronological order,
  y = dice), colored by target shape, dimmed for discard/crash. The dashed red
  line is the running best.
- **Hover** any point for its description + dice + status.
- **Click** a point to open the detail panel: metadata, full metrics, a
  **rotatable 3D tool-trajectory plot** (drag to orbit; solid = actual
  speed-clipped path, dashed = commanded pre-clip, green ● start, red ● end),
  and **download links** for the Haas G-code and the three STL meshes
  (`stock_initial`, `stock_carved`, `target`).
- The **per-shape bar chart** is clickable: click a bar to jump to that shape's
  best experiment.
- Filter by shape / status using the dropdowns in the header.

## Notes

- G-code is Haas-format (`gcode_haas.nc`). Re-run the build script after new
  training runs to refresh `data.json` and generate their G-code.
- `data.json` is ~4 MB; the browser loads it once on page open. Trajectories are
  embedded so click-through is instant.
