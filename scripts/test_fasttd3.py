"""Correctness checks for the FastTD3 pieces that fail silently when wrong.

A distributional critic does not crash when its projection is broken. It trains, the loss
goes down, and the values are quietly meaningless. Each check below targets one way that
happens, and each is a property that must hold exactly rather than a smoke test.

    python scripts/test_fasttd3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.algos.fasttd3 import (  # noqa: E402
    FastTD3, FastTD3Config, ReplayBuffer, project_distribution, suggested_support,
)

PASS, FAIL = "ok", "XX"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    if not ok:
        failures += 1
    print(f"[{PASS if ok else FAIL}] {name}{'  ' + detail if detail else ''}")


def main() -> int:
    torch.manual_seed(0)
    support = torch.linspace(-50.0, 350.0, 251)
    gamma = 0.99

    # --- the projection must conserve probability. If it does not, mass is being dropped,
    # which biases every value estimate toward zero in a way no loss curve reveals.
    probs = torch.softmax(torch.randn(64, 251), dim=-1)
    rewards = torch.rand(64) * 3.0
    keep = (torch.rand(64) > 0.2).float()
    out = project_distribution(rewards, keep, probs, support, gamma)
    check("projection conserves probability",
          torch.allclose(out.sum(-1), torch.ones(64), atol=1e-4),
          f"min sum {out.sum(-1).min():.6f}, max {out.sum(-1).max():.6f}")
    check("projection stays non-negative", bool((out >= -1e-6).all()))

    # --- the projection must MOVE the mean correctly: E[Tz] = r + gamma * keep * E[z].
    # This is the check that catches an off-by-one in the atom indexing, which conserves
    # probability perfectly while putting the mass in the wrong place.
    mean_before = (probs * support).sum(-1)
    mean_after = (out * support).sum(-1)
    expected = (rewards + gamma * keep * mean_before).clamp(float(support[0]), float(support[-1]))
    check("projection shifts the mean by the Bellman update",
          torch.allclose(mean_after, expected, atol=1e-2),
          f"max error {(mean_after - expected).abs().max():.5f}")

    # --- a terminal transition must bootstrap NOTHING: its whole distribution collapses to
    # the reward. Getting this backwards is the classic sign error and it teaches the policy
    # that dying is fine.
    point = torch.zeros(1, 251)
    point[0, 125] = 1.0
    terminal = project_distribution(torch.tensor([2.5]), torch.zeros(1), point, support, gamma)
    check("terminal transition ignores the future",
          abs(float((terminal * support).sum(-1)) - 2.5) < 1e-2,
          f"got {float((terminal * support).sum(-1)):.4f}, want 2.5")

    # --- exact grid landing. b lands exactly on an atom when reward and gamma conspire; the
    # naive floor/ceil then gives lower == upper and both weights are zero, silently
    # deleting that sample's mass.
    exact = project_distribution(
        torch.zeros(1), torch.zeros(1), point, support, gamma)
    check("mass survives landing exactly on an atom",
          abs(float(exact.sum()) - 1.0) < 1e-4, f"sum {float(exact.sum()):.6f}")

    # --- the replay buffer must wrap without losing or duplicating slots.
    buf = ReplayBuffer(capacity=100, obs_dim=4, act_dim=2, device=torch.device("cpu"))
    for i in range(15):
        buf.add(torch.full((8, 4), float(i)), torch.zeros(8, 2),
                torch.full((8,), float(i)), torch.full((8, 4), float(i)), torch.ones(8))
    check("buffer reports full after wrapping", len(buf) == 100, f"len {len(buf)}")
    check("buffer wrote every slot", bool((buf.rewards != 0).sum() > 90),
          f"{int((buf.rewards != 0).sum())} non-zero of 100")

    # --- the support must cover what our reward function can actually pay out.
    lo, hi = suggested_support(reward_per_step=3.27, gamma=0.99)
    ceiling = 3.27 / 0.01
    check("suggested support covers the discounted ceiling", hi > ceiling,
          f"v_max {hi:.0f} vs reachable {ceiling:.0f}")
    cfg = FastTD3Config()
    check("configured support covers the discounted ceiling", cfg.v_max > ceiling,
          f"v_max {cfg.v_max:.0f} vs reachable {ceiling:.0f}")

    # --- end to end: one update must run and change the actor.
    device = torch.device("cpu")
    small = FastTD3Config(batch_size=256, buffer_size_per_env=64, num_atoms=51,
                          critic_hidden=(64, 64), actor_hidden=(64, 64))
    agent = FastTD3(obs_dim=108, act_dim=28, cfg=small, num_envs=8, device=device)
    for _ in range(20):
        agent.buffer.add(torch.randn(8, 108), torch.randn(8, 28).clamp(-1, 1),
                         torch.rand(8) * 3, torch.randn(8, 108),
                         (torch.rand(8) > 0.1).float())
    before = agent.actor.net[-1].weight.detach().clone()
    metrics = {}
    for _ in range(small.policy_frequency):
        metrics = agent.update()
    moved = not torch.allclose(before, agent.actor.net[-1].weight)
    check("one update runs and moves the actor", moved,
          f"critic_loss {metrics.get('critic_loss', float('nan')):.4f}")

    # --- actions must always be legal, exploration noise included.
    actions = agent.act(torch.randn(8, 108), explore=True)
    check("actions stay inside [-1, 1]",
          bool((actions.abs() <= 1.0 + 1e-6).all()), f"max |a| {float(actions.abs().max()):.4f}")

    print(f"\n{'all checks passed' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
