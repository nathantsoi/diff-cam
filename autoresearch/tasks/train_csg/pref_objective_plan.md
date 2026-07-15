# Preference-based objective learning — research plan

> Companion to `idea_list.md` "Idea 9 (PRIORITY): Active pairwise preference
> learning". This file is the working plan the autoresearch agent follows to
> *steer* its objective formulation from human A/B preferences. It is read at
> the top of every loop iteration alongside `scripts/pref_digest.py`.

## 1. Goal & hard constraint

**Goal.** Learn what trajectory qualities the human actually cares about — end-
of-path air cutting, surface following, tool-break safety, finish — which the
blunt `hard_dice` deployability gate cannot see, by presenting pairs of
trajectories that vary one objective knob at two magnitudes and asking the
human to pick the better one with a short text reason.

**Constraint (updated; see `autoresearch.md` "What you CAN/CANNOT do").** The
training loss, optimizer, init, simulator differentiable functions, and even
the metric/eval-harness code are **all editable**. So preferences can be used
two ways, and you should use both as appropriate:

1. **Steering** (lowest-risk, always valid): let the digest choose which
   experiment / objective reformulation to try next. The agent's *understanding*
   lives as text in §6 of this file; the *objective formulation* it proposes
   next is a concrete loss-weight/structural change swept in a later
   experiment, chosen because the preference digest points that way.
2. **Direct encoding** (now permitted): encode a recurring preference directly
   as a loss term (e.g. a position-weighted air-cutting penalty when the notes
   consistently say "less air at the end"). Per-shape branching in the
   loss/init/optimizer is also allowed if it aids generalization.

The one guardrail: generalization across *unseen* shapes is still the research
goal — don't let preference encoding or per-shape branching overfit to the few
shapes you can elicit preferences on. And (per the autoresearch.md integrity
warning) never edit the metric/eval-harness to inflate `hard_dice`; edit it only
to make the measurement more honest, and re-baseline + record any metric change.

## 2. Dimensions to explore

Each dimension is one objective knob. Each **pair** holds shape / radius /
seed / iters / all-other-flags fixed and varies only that one knob at two
magnitudes (A vs B). Magnitudes below are starting suggestions — adjust from
the digest. Pick 2–3 magnitudes per dimension across pairs.

| dimension | flag(s) | example magnitudes (A vs B) | what the human is judging |
|---|---|---|---|
| `w_air_time` | `--w-air-time` | `1e-3` vs `1e-2` (vs `5e-2`) | end-of-path air cutting |
| `w_gouge` | `--w-gouge` | `4.0` vs `12.0` | cutting into the part vs staying just outside |
| `w_residual` | `--w-residual` | `1.0` vs `3.0` | stock left / over-erosion |
| `w_break` | `--w-break` | `1e-3` vs `1e-1` | tool-break safety |
| `w_time` | `--w-time` | `1e-3` vs `1e-2` | path length / cycle time |
| `k-anneal` | `--k-init` / `--k-final` | `k_init=2/kf=70` vs `k_init=20/kf=120` | sharpness ramp, surface finish |
| `init_mode` | `--init-mode` | `random` vs `multidepth_cavity` | starting trajectory quality |
| `multidepth_revs` | `--multidepth-revs` | `3.0` vs `6.0` | stepover / contour following |
| `loss_shift` | `--loss-shift` | `0.0` vs `0.5` | where the loss focuses along the path |

> `best_w_airtime` / `best_w_time` / `best_w_break` are *selection* weights
> (which of the training iterations to keep), not loss weights — still fair
> preference dimensions, but note the distinction when writing the pair's
> `--prompt`.

## 3. Per-pair procedure

1. Pick a **dimension** + **two magnitudes** (informed by the current digest in
   §6 — chase the dimension with the weakest signal or a recurring note theme).
2. Launch the pair with `sweep_pref_pair.sh` (or by hand):
   ```bash
   DIM=w_air_time FLAG_A=--w-air-time MAG_A=1e-3 \
   FLAG_B=--w-air-time MAG_B=1e-2 \
   SHAPE=sphere RADIUS=9.0 SEED=1 ITERS=5000 \
   TAG=pref_wairtime_sph \
   PROMPT="Which trajectory air-cuts less at the end?" \
   SCENARIO="sphere s1 iters5000" \
   bash autoresearch/tasks/train_csg/sweep_pref_pair.sh
   ```
   The driver runs both experiments on free GPUs, lands them in `runs/<TAG>/`
   with `--save-model` (so trajectories render in the viewer), and enqueues the
   pair via `scripts/enqueue_pair.py`.
3. **Cap the pending queue at ~8 pairs** — do not flood the human. If ≥8 are
   pending, wait for answers before enqueuing more.
4. Vary shape/seed across pairs so the learned preference generalizes
   (shape-agnostic); never make the *pair's* dimension about a shape name.

## 4. User side

Open `web/compare.html` (served by `scripts/serve_web_https.py`). For each
pair the bar shows `Dimension: <d> · A=<mag_a> vs B=<mag_b> · <scenario>` plus
the prompt. Watch both trajectories side-by-side (synced scrubber), pick
**A / B / tie**, and leave a short text **reason** ("Why? — the agent reads
this"). The collapsible *Preference digest* panel shows what's been learned so
far, updating after each answer.

## 5. Feedback → next loop

At the top of each loop iteration (per the step added to `autoresearch.md`):

1. Run `uv run python scripts/pref_digest.py` and read the per-dimension
   preferred direction + the notes.
2. Update §6 "Current preference understanding" below with any new/changed
   direction or recurring note theme (with a date).
3. **Translate recurring note themes into the next objective reformulation to
   try**, swept in a later experiment. Examples:
   - notes saying "less air at the end" → sweep `w_air_time` up, or propose a
     position-weighted air term, or promote `init_mode=multidepth_cavity`.
   - notes saying "cutter digs in" → sweep `w_gouge` up.
   - notes saying "leaves stock" → sweep `w_residual` up / `loss_shift`.
   The proposed reformulation is a concrete loss-weight/structural change run
   as a normal `hard_dice` experiment afterwards — *preference steering picks
   what to try, the metric confirms whether it worked.*

## 6. Current preference understanding

*(Empty until the first pairs are answered. Update here each loop. Format:)*
- `w_air_time` — preferred: ? (n=?). Notes: …
- `init_mode` — preferred: ? (n=?). Notes: …

Recurring themes → next reformulation to try:
- *(none yet)*

## 7. Gate (Idea 9)

Preference ranking **concordance with existing star ratings** on overlapping
runs ≥ 80%: when a pair's A/B runs both have star ratings in the feedback
store, the preference direction should agree with the star ordering ≥ 80% of
the time. If it diverges, investigate whether the pair's framing (dimension
choice / prompt) is testing the wrong thing before trusting that dimension's
signal.

## Infra reference

- `scripts/enqueue_pair.py` — enqueue a pair headless (no server needed).
- `scripts/pref_digest.py` — per-dimension digest of answered pairs (`--json`
  for machine-readable).
- `scripts/pref_lib.py` — shared `digest`/`pending`/`summary_counts` used by
  the digest CLI and the `GET /__api/pref-digest` endpoint.
- `scripts/serve_web_https.py` — `GET/POST /__api/pairs`, `GET /__api/pref-digest`.
- `autoresearch/tasks/train_csg/web/compare.html` — the A/B viewer + digest panel.
- `algorithms/train_csg.py` `load_pairwise_preferences` — logs the per-dimension
  breakdown + recent notes to stderr and `metrics.json["pairwise"]` on every
  run (selection layer only).
