# Phases 5–6: Grounded Explanation and Meaningful-Difference Gate

`scripts/evaluate_direct_feedback.py` evaluates completed controlled children.
It records exact measured metrics, neutral explanation text, physical-space
trajectory differences, quality regressions against the selected parent, and
the predeclared gates from `launch_plan.json`.

For this first iteration, `total_time` is the targeted metric because the raw
human critique said “quicker” and Qwen's interpretation said “save time” before
either child was trained. The mapping is therefore not selected after seeing
the outcomes. Trajectory coordinates are converted from stock-normalized units
to millimetres using the recorded stock dimensions.

Every gate is required: 20% targeted-metric separation, 0.5 mm mean pointwise
path displacement, no more than 0.03 hard-Dice loss from the selected parent,
no more than 10% gouge increase, and visually distinguishable rendered paths.
A failed gate records a `weak_intervention`; such a pair is preserved and must
not be presented as successful adaptation or enqueued for human judgment. A
passing result is marked eligible for Phase 7, where enqueueing and exact
display-order logging occur.

The explanation text separates the intended effect from observed metrics. It
does not call either option better, smoother, or safer.

## First real-feedback result

Iteration `df_p_0090_1788856283997` produced both children successfully. B's
recorded total time was 26.805 s versus A's 30.267 s, an 11.44% difference.
The paths were clearly distinguishable in the physical-space overlay (53.87 mm
mean pointwise displacement), and both children passed the hard-Dice and gouge
regression limits. The targeted-metric gate nevertheless failed because 11.44%
is below the declared 20%; the result is therefore a weak intervention and is
not eligible for enqueueing.

The stronger child already used the minimum possible `w_break` value (zero),
so another same-direction weight step is impossible. The recorded one-time
redesign recommendation is to act directly on `w_time`, using conservative and
stronger increases under the same controls. That would be a new expensive GPU
experiment and is not launched by the Phase 5/6 evaluator.

## One-time redesign outcome

The Phase 7 retry changed the direct differentiable `w_time` loss from the
selected parent's 0.001 to 0.02 (A) and 0.2 (B), leaving all other controls
fixed. The same predeclared gates apply. A failed retry is recorded with
`retry_exhausted: true`; it does not recursively authorize more GPU attempts.

The completed retry measured 36.882 s for A and 33.241 s for B, a 9.87%
difference. Its paths were visually distinct and had 41.35 mm mean pointwise
displacement. Both children passed the hard-Dice and gouge limits, but the
targeted time difference again missed 20%. The retry is therefore recorded as
weak and exhausted, and no follow-up comparison pair was enqueued.

For end-to-end engineering validation only, `--enqueue-weak-demo` may present
an already-recorded weak pair without changing any threshold or scientific
classification. The pair, iteration event, and append-only event are all
marked `experimental_evidence_eligible: false`; this override demonstrates the
UI-to-agent plumbing and must not be counted as evidence of meaningful change.
