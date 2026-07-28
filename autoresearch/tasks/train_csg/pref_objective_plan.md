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
3. **Keep generating comparisons while waiting for the user — do NOT cap or
   block.** The human answers asynchronously and may be away for hours, so a
   full or growing pending queue is the *intended* state, not a signal to pause.
   Whenever GPUs are free, enqueue the next pair; never "wait for answers before
   enqueuing more." The only things that gate enqueuing are GPU availability and
   disk (prune answered+recorded run dirs periodically, keeping the newest ~20
   for re-viewing; never delete a run whose pair is still pending). Interleave
   pair-generation with the `hard_dice` experiments so neither starves.
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

Updated 2026-07-15 (~12:00), 20/29 answered. Per-dimension n is still small
(1–3); treat directions as hypotheses, themes as the real signal.

- `w_air_time` — preferred: tie (n=2). Notes: "too much air cutting, did not cut
  top of part well"; "both follow the contour and dont cut any air, however the
  surface finish is poor as the endmill doesnt follow the surface precisely."
- `w_gouge` — preferred: no clear winner (A=1 B=1 tie=1, n=3). Notes: "less
  jagged surface finish (fewer spikes) however gouging appears in both";
  "better contour following, but surface still jagged due to the cutting not
  overlapping the previous cut enough to remove all the material"; "both pretty
  good, look almost the same."
- `w_residual` — preferred: A=1.0 (n=2, unanimous). Notes: "no gouging, but the
  cutter should follow the surface more consistently and overlap more to remove
  the remaining jagged material"; "better shape following."
- `w_break` — preferred: no clear winner (A=0 B=1 tie=1, n=2). Notes: "better
  because it did not spend time moving while just cutting air, good contour
  following"; "both ok but not great — dont make it all the way around the
  outside of the part and have air cutting."
- `w_time` — preferred: tie (n=1). Notes: "very similar, both great!"
- `k-anneal` — preferred: B=sharper (n=2, UNANIMOUS). Notes: "better contour
  following and less air cutting"; "much better part contour following."
- `init_mode` — preferred: no clear winner (A=1 B=1 tie=1, n=3). Notes: "much
  less time spent cutting air, which is good"; "no time spent air cutting...
  good surface cutting on the top of the part"; "main difference is when tool
  is air cutting... all of which is bad."
- `multidepth_revs` — preferred: no clear winner (A=1 B=1 tie=1, n=3). Notes:
  "a little bit better surface cutting, but not great"; "both have way too much
  air cutting and reasonable part-contour following"; "less air cutting, better
  shape following."
- `loss_shift` — preferred: B=0.5 (n=1). Notes: "follows the contour of the
  part. top surface following could be better."

Recurring themes (the real signal — repeated across unrelated dimensions):
1. **Contour / surface following is the dominant valued quality.** Cited as the
   reason for the preferred side across k-anneal, loss_shift, w_residual,
   multidepth_revs, w_break — "better shape/contour following."
2. **Air cutting is the dominant negative.** Flagged in 6+ notes (init_mode ×2,
   multidepth_revs, w_break, w_air_time, +unstructured) — "air cutting... all of
   which is bad."
3. **Surface finish / precise surface following is a stated deficit.** Flagged
   repeatedly (w_air_time "surface finish poor, endmill doesnt follow surface
   precisely"; loss_shift "top surface following could be better"; init_mode
   "good surface cutting on the top of the part").
4. **NEW (jaggedness from insufficient cut overlap).** Flagged in w_gouge and
   w_residual notes — "surface still jagged due to the cutting not overlapping
   the previous cut enough to remove all the material"; "overlap more to remove
   the remaining jagged material"; "fewer spikes on the surface." This is a
   STEPOVER/cut-overlap signal, distinct from contour following: the user wants
   adjacent passes to overlap enough to clear all material (no scallop/spike
   ridges). Tension with theme 2: finer stepover (more revs) reduces jaggedness
   but adds air cutting — and multidepth_revs A=3.0 (coarser) was preferred for
   "less air cutting." So the ideal is NOT blanket-finer stepover but a
   stepover that overlaps only where there is material (no extra air).

→ next reformulations to try (swept as hard_dice experiments; preference
steering picks what to try, the metric confirms):

**METRIC OUTCOMES (2026-07-15, all three pref-driven reformulations FAILED to
lift hard_dice — strong confirmation that contour/finish preferences are
COSMETIC w.r.t. the deployable carve metric):**
- **k_final=180** (k-anneal B "better contour following"): NEUTRAL/marginal on
  all 5 shapes (sphere +0.002, cyl +0.001, pyramid +0.002, bowl +0.0002, box
  +0.016 1-seed). NOT promoted. See [[pref-signal-largely-cosmetic]].
- **loss_shift=0.5** (loss_shift B "follows the contour"): NEGATIVE on all 6
  runs / 3 shapes (box -0.043 both seeds, sphere -0.02, bowl -0.02 to -0.04).
  NOT promoted; SOTA stays loss_shift=0.0.
- **cut-overlap / multidepth_revs 4.5** (anti-jaggedness "overlap more"):
  NEUTRAL/NEGATIVE — box 3.0 beats 4.5 by +0.009 (finer stepover HURTS, extra
  air), cyl 4.5 +0.0035 (1-seed noise). NOT promoted; SOTA stays revs=3.0.
  The only preference that ALIGNS with hard_dice is "less air cutting"
  (coarser stepover), already captured in SOTA.

**PIVOT (2026-07-15):** further preference steering on contour/finish/stepover
dimensions is unlikely to advance hard_dice — three cosmetically-preferred
changes all failed the metric. The path to actually move hard_dice is NOT more
cosmetic-preference encoding; it is non-cosmetic levers the human does NOT
judge visually, applied where headroom exists (weakest shapes: hole 0.273
stuck, bowl 0.647 seed-unstable). Both weak shapes are GPU atomic-add
nondeterminism-limited (variance run-to-run, not init-seed — simulator not
modifiable), so init tweaks won't help; the lever is the k-anneal SCHEDULE.
Currently testing:
- **k_ramp_delay** (hold k at k_init for a fraction of iters before ramping;
  default 0.0): explicitly motivated by the compute-starved hole — lets it keep
  soft-carving at fixed 5000 iters without the 2x cost of 10k iters. Sweeping
  delay 0.3 / 0.5 on the hole (3-seed / 2-seed) + sphere delay=0.3 (regression
  check on a converged convex shape — risk is under-sharpening). Shape-agnostic.
  This is the active hard_dice experiment; preference steering is paused on
  contour/finish dims (queue still stocked for the separate-objective-layer
  path (a) — the human values the cosmetic qualities even though hard_dice does
  not).

Legacy reformulation entries (kept for the record; all three have now been
metric-tested and REJECTED above):
- **Sharper surface proxy**: k_final=180 → NEUTRAL (rejected).
- **Promote loss_shift=0.5** → NEGATIVE (rejected).
- **Top-surface following**: z-weighted contour term — only worth pursuing if a
  metric-positive lever is found; loss_shift (its proxy) was NEGATIVE, so deprioritize.
- **NEW (cut-overlap / anti-jaggedness)**: multidepth_revs 4.5 → NEUTRAL/
  NEGATIVE (rejected); the structural inter-pass residual scallop loss (b) is
  still untested code-risk, but the stepover sweep failing lowers its prior.
- Air-cutting theme: keep w_air_time=1e-3, multidepth_contour/cavity init,
  revs=3.0 (all confirmed by the metric, not just preferred).

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
