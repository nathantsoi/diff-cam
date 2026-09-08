# Prompt to paste into Codex on the SSH/GPU machine

You are working inside the `diff-cam` repository on the SSH/GPU machine. Before doing anything else, read `SSH_CODEX_HANDOFF.md` completely and treat it as the task specification. If present, also read `AUTORESEARCH_CODEBASE_README.md`, `MEETING_REFERENCE.md`, and the source documents listed in the handoff. Then inspect all applicable repository instruction files such as `AGENTS.md`, `CLAUDE.md`, and nested equivalents.

Your job is to complete the two-day research task described in the handoff, not merely propose a plan. Work in these stages:

1. Protect the working tree and report the exact branch, commit, remotes, local changes, GPU/runtime environment, disk space, and scheduler constraints. Fetch remote references safely. Do not reset, clean, or discard any work.
2. Review the current `autoresearch` branch and the earlier branches named in the handoff. Use commit graphs, merge bases, diffs, code, and experiment documents—not branch names or commit messages alone. Write `BRANCH_HISTORY_REVIEW.md`, distinguishing verified results from claims that lack raw artifacts.
3. Set up the intended environment, run a small smoke test, and reproduce one known baseline before launching expensive jobs. Record the exact command, seed, target, GPU, runtime, commit, output path, and metrics.
4. Trace the current A/B feedback flow from `web/compare.html` through the server, JSON storage, digest, agent instructions, and training code. Verify or correct every architectural claim in the handoff.
5. Design and implement the smallest reliable direct-feedback loop: a human A/B/tie choice plus raw free-form critique must be consumed directly by the LLM coding agent; the agent must make a versioned objective change; it must then generate two controlled new trajectories based on that feedback; and the UI must show a short factual explanation under each option grounded in the exact change and measured results.
6. Log the complete causal chain with a stable iteration ID: displayed pair/order/text, raw feedback, structured interpretation, uncertainty, exact code or parameter changes, rationale for A and B, commands, seeds, run IDs, metrics, trajectory-difference checks, generated explanations, and relevant Git revisions or patch hashes. The log is an audit record, not an intermediary summary that replaces the raw feedback.
7. For the first experiment, prefer an already measurable concern such as late air cutting or gouge avoidance. Keep target, seed, initialization family, iteration count, and unrelated settings fixed. Recommended A/B semantics are conservative versus stronger/alternative interpretations of the same critique. Add an unchanged control run if affordable so stochastic variation can be separated from feedback effects.
8. Declare a meaningful-difference criterion before inspecting the new results. Compare the targeted metric, hard Dice, gouge/residual, air/total time, breakage proxy, and a trajectory-space distance. If A and B are effectively identical, log that honestly and revise the intervention once; do not call file changes a successful behavioral change.
9. Run the initial experiment on the GPUs and, if time permits, collect the next human response to demonstrate that the loop continues. Never count synthetic fixture feedback as human experimental evidence.
10. Write `INITIAL_DIRECT_FEEDBACK_EXPERIMENT.md` with the outcome, exact reproduction details, links/paths to artifacts, metric table, limitations/confounds, and next experiment.

Important boundaries:

- Do not alter evaluation metrics to improve results.
- Do not make unversioned or unlogged LLM edits.
- Begin with configuration/weight changes; edit loss source only when needed to express the critique, and preserve an explicit reviewed patch.
- Generated A/B explanations must separate intended effects from measured effects and must not claim “safer,” “smoother,” or “better” without evidence.
- Explanation text may influence the human, so preserve its exact wording and presentation order.
- Inspect the installed Codex CLI and scheduler interfaces rather than guessing commands.
- Keep large artifacts out of Git, but save a small manifest with paths and checksums.
- Do not expose secrets in logs or commits.

Start now with read-only inspection. Give me a concise checkpoint after the branch/environment audit and baseline plan, but continue with safe setup work unless an expensive GPU choice or ambiguous A/B semantics genuinely requires my decision.
