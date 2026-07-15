# idea.md — per-run idea/hypothesis log

This file is the autoresearch agent's per-run working log: hypotheses, the
experiments run to test them, and the findings. It is **reset to this stub at
the start of each run** (see `autoresearch.md` "Setup") and the agent appends to
it as it loops (`LOOP FOREVER`). The prior run's log is preserved in git history
on its run branch — this trunk copy stays a clean stub between runs.

## When starting the next run

Cut a new run branch from `autoresearch`, then overwrite this file with:

- **Branch / tag** — the new run branch name and a short tag.
- **Starting point** — the baseline scenario + any method carried over from the
  prior run (with its `hard_dice`), so every new variation is compared against a
  known baseline.
- **Goal / hypothesis** — what deployability wall this run is trying to break,
  and the lever(s) to try.
- **Plan** — the first few experiments, in order.

Reference the prior run's lessons via the archived memory
(`memory/memory_archive_*/`), used as methodology guardrails — not as claims.

---

## Next run

*(filled in when the next run branch is cut from `autoresearch`)*
