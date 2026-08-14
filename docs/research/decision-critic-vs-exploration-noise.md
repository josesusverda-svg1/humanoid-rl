# Decision: the critic is fine; the exploration noise is the defect

Date: 2026-08-13. Context: run `amp-20260813-134908`, iteration 1459, 143M/400M env steps.
Supersedes the working hypothesis "explained variance is too low, fix the critic".

## Verdict

The critic is healthy. On its own on-policy (noisy) distribution it explains **71.8%** of the
true Monte-Carlo discounted return (corr +0.849) and **94.9%** of its literal PPO regression
target. The theoretical ceiling for *any* recalibration of its own output on that data is
**+0.727** — the critic is within 1.2 percentage points of the best score obtainable. The
+0.30 that started this investigation was measured against the *deterministic* policy, which
earns 2.0x the reward the critic was ever fitted on. It is an apples-to-oranges number.

Two real defects, neither of which is critic capacity:

1. **`log_std` receives exactly zero gradient and is frozen at std = 1.0 forever.**
   `networks.py:162` computes `exp(log_std.clamp(-5.0, log_std_max))`. `torch.clamp` passes
   gradient *at* the boundary and zero *above* it. `log_std` initialises at
   `log(1.0) = 0.0 == log_std_max`, the entropy bonus pushes it up on the first step, and it
   is dead from step two. Verified with autograd: d(std)/d(log_std) is 1.0000 at
   `log_std = 0.0` and 0.0000 at `log_std = +1e-9`. Current checkpoint holds +0.0021..+0.0037
   on all 28 dims. `network.init_noise_std: 0.8` in the YAML was silently ignored for this
   entire run because `--init-from` overwrote it. std 1.0 on a hard-clipped [-1,1] action
   space costs ~49% of per-step reward.

2. **`clip_value_loss: true` is a 0.2-absolute-unit-per-iteration speed limit on a critic
   whose typical error is 4.35.** `old_val_b` is fixed at rollout time, so the 0.2 budget
   covers all 5 epochs x 4 minibatches. 96.3% of value targets sit outside it. This is not a
   fit-quality problem — it is a *tracking-speed* problem, and it becomes load-bearing the
   moment the noise fix raises the reward scale by ~75%. Fingerprint in the live logs:
   `weight_change/critic.4.weight` = 0.00099 vs `weight_change/actor.4.weight` = 0.0577, a
   58x gap at the output layer against only 2.3-2.8x in the trunks.

## Ranked changes (one restart, bundled with the pending obs-layout change)

| # | Change | File | Verify |
|---|--------|------|--------|
| 1 | `network.init_noise_std: 0.25`, `learn_noise_std: false`; drop the forward-pass clamp; force `log_std` after warm start | `networks.py`, `train.py`, `configs/amp.yaml` | `action_std` logs 0.2500; `entropy` logs ~0.91; `reward/action_rate` -0.278 -> ~-0.035 within 10 iters |
| 2 | `ppo.entropy_coef: 0.003 -> 0.0` | `configs/amp.yaml` | n/a (removes the only upward force on sigma) |
| 3 | `ppo.clip_value_loss: true -> false` | `configs/amp.yaml` | `weight_change/critic.4.weight` rises from 0.00099 to within 5x of actor.4 |
| 4 | Clip actor and critic gradient norms **separately** | `ppo.py:362` | two logged norms; actor step size no longer tracks `sqrt(value_loss)` |
| 5 | `AdamW(..., weight_decay=0.0)` | `ppo.py:256` | correctness only |
| 6 | Save the AMP discriminator in checkpoints | `train.py:547` | `'discriminator' in torch.load(ckpt)` |
| 7 | Replace the vacuous `explained_variance` (identically `1 - Var(A)/Var(V+A)`) | `ppo.py:399` | new metric reads below 0.949 |

Deferred, explicitly: asymmetric/privileged critic, value-target normalisation,
`max_grad_norm` retune, AMP discriminator rebalance. None of them are needed to unblock this
run, and each is a separate restart's worth of confound.

## Why restart now

`best.pt` has been stuck at iteration 550 / `best_return` 1210.7 for 909 iterations and 89M
env steps — no deterministic-eval improvement across 22% of the total step budget. The
working tree already diverged from the live process (`task_obs_dim` 7 -> 13, obs 102 -> 108,
mid-episode command resampling, per-reset gait randomisation), so the running job cannot be
resumed against current code regardless. This is one forced restart, not a fifth reactive one.
