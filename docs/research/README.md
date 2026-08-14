# Phase 0 research archive

Verbatim output of the Phase 0 documentation sweep (12 agents, 2026-08-12/13).
Kept in the repo because it was originally produced into session-scoped files and would
otherwise be lost. These are the *raw findings*; the decisions taken from them live in
[DESIGN.md](../../DESIGN.md).

> Caveat: these are research notes written from web sources at a point in time. Where a
> claim here conflicts with a measurement in DESIGN.md, **the measurement wins**. Three
> claims were tested locally and found not to hold on this machine, see DESIGN.md section 9.

## Topic dossiers

- [JAX on Apple Silicon (M3 Max, macOS 26.5, arm64): jax-metal PJRT plugin vs. plain CPU JAX,](topics/jax-on-apple-silicon-m3-max-macos-26-5-arm64-jax-metal-pjrt.md) — status: `usable`, confidence: `high`
- [MJX (MuJoCo XLA) on Apple Silicon M3 Max — viability for AMP/PPO humanoid locomotion train](topics/mjx-mujoco-xla-on-apple-silicon-m3-max-viability-for-amp-ppo.md) — status: `experimental`, confidence: `high`
- [HumanoidBench, dm_control, Gymnasium/MuJoCo envs, and mujoco_menagerie as sources of reusa](topics/humanoidbench-dm-control-gymnasium-mujoco-envs-and-mujoco-me.md) — status: `good`, confidence: `high`
- [MuJoCo Playground and Brax as reusable training ecosystems for humanoid locomotion on Appl](topics/mujoco-playground-and-brax-as-reusable-training-ecosystems-f.md) — status: `usable`, confidence: `high`
- [PyTorch with the MPS (Metal Performance Shaders) backend on M3 Max / macOS 26.5 / Python 3](topics/pytorch-with-the-mps-metal-performance-shaders-backend-on-m3.md) — status: `good`, confidence: `medium`
- [MuJoCo (C engine + official Python bindings) on Apple Silicon CPU — version status, GIL/th](topics/mujoco-c-engine-official-python-bindings-on-apple-silicon-cp.md) — status: `excellent`, confidence: `high`
- [Apple MLX (v0.32.0) as the NN training framework for MuJoCo-based humanoid PPO + AMP on M3](topics/apple-mlx-v0-32-0-as-the-nn-training-framework-for-mujoco-ba.md) — status: `excellent`, confidence: `high`
- [LocoMuJoCo — imitation-learning benchmark for locomotion (viability as base for M3 Max hum](topics/locomujoco-imitation-learning-benchmark-for-locomotion-viabi.md) — status: `usable`, confidence: `high`
- [Mocap datasets (AMASS, CMU, LAFAN1, KIT, 100STYLE, Human3.6M, LocoMuJoCo/dm_control bundle](topics/mocap-datasets-amass-cmu-lafan1-kit-100style-human3-6m-locom.md) — status: `good`, confidence: `high`
- [Mocap datasets, human-proportioned MuJoCo humanoid models, and SMPL/BVH→MuJoCo retargeting](topics/mocap-datasets-human-proportioned-mujoco-humanoid-models-and.md) — status: `good`, confidence: `medium`
- [Motion-imitation algorithms for RL humanoid locomotion (DeepMimic/AMP/ASE/CALM/PULSE/PHC/P](topics/motion-imitation-algorithms-for-rl-humanoid-locomotion-deepm.md) — status: `usable`, confidence: `high`

## Synthesis and critique

- [synthesis.md](synthesis.md) — the stack decision, reuse list, risk register, pinned deps
- [critique.md](critique.md) — adversarial completeness pass over the synthesis
- [amp-motion-imitation.md](amp-motion-imitation.md) — the imitation-algorithm deep dive (run separately after the first agent died on an API error)
