# Direct Free-Form Feedback MVP

## Causal path

The Phase 3 implementation establishes this path:

```text
compare.html first view
  -> snapshot fixed A/B order, prompt, explanations, and displayed metrics
human A/B/tie + required raw critique
  -> immutable answer in pairwise.json
  -> HTTPS server atomically claims the observation
  -> local qwen3.8:latest worker reads that exact pair (not the digest)
  -> Qwen proposes conservative A and stronger B loss-weight adaptations
  -> validator rejects arbitrary code/commands, non-loss edits, bad bounds,
     unrelated A/B terms, and a B that is not stronger in the same direction
  -> append-only event + variant_a.json + variant_b.json
```

The decision log is an audit output. It is written after Qwen consumes the raw
choice/critique and decides the objective change; it is not summarized and fed
back as the decision input.

## Explicit Phase 3 decisions

1. **Persistent process and trigger:** `scripts/serve_web_https.py` is the
   persistent event receiver. The first successful answer POST atomically moves
  the pair from `queued` to `agent_running` and starts a one-shot
  `scripts/direct_feedback_agent.py` worker. Browser retries cannot replace an
  answer or trigger a second worker. The server and CLI enqueuer share an
  OS-level file lock, preventing cross-process pair-store lost updates.
2. **Meaning of direct loss alteration:** the initial experiment changes
   validated `train_csg` loss-weight configuration. This changes the effective
   loss directly without allowing unconstrained source edits. Source editing is
   reserved for a critique that existing terms cannot express.
3. **Meaning of A/B:** both variants adapt the selected parent to the same raw
   critique. A is conservative; B changes the same one or two terms farther in
   the same direction. On a tie, A is the deterministic parent. Unrelated
   controls are fixed from the parent run.
4. **LLM:** local OpenAI-compatible endpoint
   `http://10.1.0.116:11434/v1`, model `qwen3.8:latest`, temperature 0.1. No key
   or paid API is used. Environment overrides are `DIFFCAM_LLM_BASE_URL` and
   `DIFFCAM_LLM_MODEL`.
5. **Training authority:** the Phase 3 worker cannot launch a GPU job. It emits
   validated variant configurations for the controlled Phase 4 runner. This
   keeps a malformed model response from consuming GPU time.

The initial target/critique and numerical meaningful-difference threshold remain
Phase 4/6 decisions and must be fixed before expensive variant runs.

## Event and artifact model

Each answered observation receives an ID of the form
`df_<pair-id>_<answer-time-ms>`. Runtime data is ignored by Git and stored at:

- `autoresearch/tasks/train_csg/direct_feedback_events.jsonl`: append-only event
  stream with an exclusive OS file lock and fsync.
- `autoresearch/tasks/train_csg/direct_feedback/<iteration-id>/agent_input.json`:
  exact pair, raw choice/critique, parent rule, and both run snapshots.
- `agent_response.txt`: exact model output.
- `variant_a.json` and `variant_b.json`: before/after and complete effective loss
  configuration.
- `event.json`: structured interpretation, rationale, uncertainty, fixed
  controls, Git revision, decision hash, and placeholders for commands, child
  runs, metrics, explanations, meaningful-difference result, and next pair.

`pairwise.json` now records `first_view_ts`, fixed `display_order`, and a
`display_snapshot` containing the exact prompt, objective framing,
explanations, and summary metrics shown. Newly answered direct-feedback rows are
immutable. Historical rows remain backward compatible.

## Allowed loss changes

The agent may change one or two of: `w_gouge`, `w_residual`, `w_air`, `w_jerk`,
`w_step`, `w_prox`, `w_traj_prox`, `w_len`, `w_tool_gouge`, `w_time`,
`w_air_time`, `w_air_late`, and `w_break`. Each has a hard numerical bound in
`scripts/direct_feedback_agent.py`. It may not change hard Dice or another
evaluation metric, execute commands, choose a GPU, change geometry/seed/init,
or edit source in this MVP.

## Explanation behavior

New pairs accept `explanation_a` and `explanation_b`, and `compare.html` renders
the text directly under the corresponding option. Historical pairs show a
neutral “no explanation recorded” label. Phase 5 will generate these strings
only after measured child-run metrics are available, clearly separating intent
from observed effects.

## Verification

- Python compilation passes for the worker, server, and pair enqueuer.
- Eight focused tests cover JSON extraction, allowlist/bounds/change-count
  enforcement, immutable answers, required critique queueing, direct use of raw
  critique, conservative/strong variants, and append-only event output.
- The first synthetic local-model chat-completion check timed out after 180
  seconds even though `/v1/models` was healthy. Generation is now capped at 700
  tokens, extended reasoning is disabled in the prompt, and the asynchronous
  worker allows a 660-second process timeout. Any synthetic retry remains
  plumbing validation only and is not human experimental evidence.
- The bounded retry reached the model, but its JSON was truncated at the
  initial 700-token cap. The request now sends only fixed controls, allowlisted
  loss values, and headline metrics; it also requires short fields and requests
  JSON response mode.
- The compact JSON-mode retry passed in about 80 seconds. For the synthetic
  “late motion away from cutting surfaces” fixture, Qwen returned conservative
  `w_prox=1.0` and stronger `w_prox=3.0`; both passed the validator. This is a
  plumbing result, not human experimental evidence.

## Remaining work before Phase 7

- Phase 4 is implemented in `scripts/run_direct_feedback_variants.py`; see
  `PHASE4_CONTROLLED_VARIANTS.md`. A real initial pair/critique is still needed
  before concrete child commands exist.
- Phase 5: produce explanations grounded in actual metrics and trajectory
  measurements.
- Phase 6: compute the predeclared meaningful-difference gate before enqueueing.
- Make the child-run completion/enqueue transition restart-safe, and include
  child run IDs, commands, checksums, results, explanations, next pair ID, and
  final patch/commit identity in the append-only event stream.
