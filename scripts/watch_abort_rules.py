"""Enforce the pre-committed abort rules for the from-scratch curriculum run.

The rules were fixed BEFORE launch, in the curriculum prescription, so that the decision to
stop is made by criteria chosen while calm rather than by hope or impatience while watching
a chart. This watcher only reports; stopping stays a human decision, but the report cites
the rule, so the conversation starts from "rule 2 fired", not from a feeling.

    Rule 1  iter 600:   mean training episode length < 625        -> curriculum failing
    Rule 2  iter 1000:  ep-len > 1250 but median difficulty <= 0.80 -> promotion gate too tight
    Rule 3  any window: median difficulty falls > 0.1 over 500 iters -> promote/demote thrash
    Rule 4  iter 2000:  deterministic falls > 50%                  -> will not converge in budget
    Rule 5  iter 1500:  segment gait match < 0.68                  -> the clock is losing
    Rule 6  iter 1500:  penalty scale pinned at 0.25 and ep-len < 1000 -> leniency is not the bottleneck

EPISODE LENGTH. Rules 1, 2 and 6 are raw STEP counts, so they only mean anything against a
known episode limit. They were written for max_episode_steps = 1000 and were rescaled by 2.5
when that became 2500 (the limit had been 8 s, not the 20 s its comment claimed; see
configs/default.yaml). A step-count threshold that silently outlives the episode length it
was calibrated against is a rule that fires on the wrong thing, which is worse than no rule.

Rule 7 is not an abort rule, it is an instrumentation alarm, and it is here because the six
rules above are only as good as the numbers they read. `evaluate()` used to stop after the
first 32 episodes completed, which under autoreset are the ones that fell, so `fall_rate`
read 1.00 whenever half the environments failed early and ~0.31 otherwise. Rule 4 fired on
that. The signature is that the SAMPLE SIZE moves with the reported fall rate, which cannot
happen if the sample is unbiased, so that is what rule 7 watches.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def rows_of(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> int:
    runs = [p for p in (REPO_ROOT / "runs").iterdir()
            if p.is_dir() and (p / "metrics.jsonl").exists()
            and not p.name.startswith("ABANDONED") and not p.name.startswith("arm-")]
    run = max(runs, key=lambda p: (p / "metrics.jsonl").stat().st_mtime)
    print(f"abort-rule watcher on {run.name}", flush=True)
    fired: set[int] = set()

    while True:
        time.sleep(120)
        rows = rows_of(run / "metrics.jsonl")
        if not rows:
            continue
        it = rows[-1].get("iteration", 0)
        recent = [r.get("episode_length", 0) for r in rows[-30:] if "episode_length" in r]
        ep_len = sum(recent) / max(len(recent), 1)
        evals = [r for r in rows if "eval/episode_return" in r]
        med_s = evals[-1].get("eval/difficulty_median") if evals else None
        p_scale = evals[-1].get("eval/penalty_scale") if evals else None
        falls = evals[-1].get("eval/fall_rate") if evals else None
        gait = [r.get("reward/gait_phase") for r in rows[-30:] if "reward/gait_phase" in r]
        gait_mean = sum(gait) / max(len(gait), 1) if gait else None

        def fire(n: int, msg: str) -> None:
            if n not in fired:
                fired.add(n)
                print(f"[ABORT-RULE {n}] iter {it:,}: {msg}", flush=True)

        if it >= 600 and ep_len < 625:
            fire(1, f"mean episode length {ep_len:.0f} < 625. The curriculum is failing to "
                    f"bootstrap. Prescribed next step: static control at difficulty 0.45.")
        # Recalibrated for difficulty_init = 0.7. The old threshold was 0.55, which sat
        # BELOW the new starting level, so a curriculum that never promoted once would have
        # looked healthy. A stuck curriculum is now the thing this detects.
        if it >= 1000 and ep_len > 1250 and med_s is not None and med_s <= 0.80:
            fire(2, f"episodes healthy ({ep_len:.0f}) but median difficulty {med_s:.2f} has "
                    f"barely moved off its 0.70 start. A promotion gate is unreachable: "
                    f"check reward/gait_phase against promote_gait_match (0.68) and the "
                    f"per-axis errors against promote_vx_err/vy/wz (0.25/0.15/0.25).")
        if len(evals) >= 11:
            s_hist = [e.get("eval/difficulty_median", 0) for e in evals]
            if max(s_hist[-11:]) - s_hist[-1] > 0.1:
                fire(3, f"median difficulty dropped >0.1 over the last ~500 iterations: "
                        f"promote/demote thrash. Demote to -0.05 and require two failures.")
        if it >= 2000 and falls is not None and falls > 0.5:
            fire(4, f"deterministic falls {falls:.0%} at iteration 2000+: not converging "
                    f"in budget.")
        if it >= 1500 and gait_mean is not None and gait_mean < 0.68 * 1.0:
            fire(5, f"gait match {gait_mean:.2f} < 0.68: the clock is losing. Try "
                    f"transition width 0.10 before touching weights.")
        if it >= 1500 and p_scale is not None and p_scale <= 0.251 and ep_len < 1000:
            fire(6, f"penalty scale pinned at floor with episode length {ep_len:.0f}: "
                    f"leniency is not the bottleneck.")
        # Rule 7: the eval sample size must not depend on what the eval measured. If the
        # evals reporting high fall rates are also the ones built from fewer episodes, the
        # sample is selecting on the outcome and every number above is untrustworthy.
        sized = [(e.get("eval/num_episodes", 0), e.get("eval/fall_rate", 0.0))
                 for e in evals if e.get("eval/num_episodes")]
        if len(sized) >= 8:
            small = [f for n, f in sized if n < max(n for n, _ in sized)]
            full = [f for n, f in sized if n == max(n for n, _ in sized)]
            if small and full and (sum(small) / len(small)) - (sum(full) / len(full)) > 0.3:
                fire(7, f"INSTRUMENTATION, not the policy: evals built from fewer episodes "
                        f"report a {sum(small)/len(small):.0%} fall rate against "
                        f"{sum(full)/len(full):.0%} for full-sized ones. The eval is "
                        f"selecting on the outcome it measures. See "
                        f"docs/research/the-eval-metric-was-biased.md; stop and fix the "
                        f"metric before reading any rule above.")


if __name__ == "__main__":
    sys.exit(main())
