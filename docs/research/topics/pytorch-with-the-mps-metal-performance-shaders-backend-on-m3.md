# PyTorch with the MPS (Metal Performance Shaders) backend on M3 Max / macOS 26.5 / Python 3.12, for small-MLP PPO + humanoid motion imitation

| field | value |
|---|---|
| apple_silicon_status | good |
| current_version | 2.13.0 |
| last_release_date | 2026-07-08 |
| maintenance_status | Actively maintained, very high velocity. torch 2.13.0 tagged and published 2026-07-08; prior 2.12.1 on 2026-06-18, so roughly monthly patch cadence. The MPS backend specifically is under active development by Apple and Meta engineers (malfet, kulinseth respond to MPS issues within hours — see #177819 comments dated 2026-03-19 and 2026-03-23), and 2.13.0 landed a large batch of MPS work including FlexAttention on Metal, SDPA prefill kernels, GQA, and migration of many ops off MPSGraph onto hand-written Metal kernels. |
| license | BSD-3-Clause |
| confidence | medium |
| install | `uv venv --python 3.12 && uv pip install "torch==2.13.0"   # NOTE: use plain PyPI, do NOT pass --index-url .../whl/cpu — the default macOS arm64 wheel (torch-2.13.0-cp312-cp312-macosx_14_0_arm64.whl) already contains MPS; the /whl/cpu index would strip it` |

## Summary

PyTorch 2.13.0 (released 2026-07-08) ships a native macOS arm64 wheel for CPython 3.12 with MPS built in, so nothing here is CUDA-blocked — MPS is a first-class, if still officially "beta", backend. Operator coverage is now good (roughly 20-30 of ~700 tracked ops missing in 2.13.0), fp32/fp16/bf16 all work, but float64 is flatly unsupported on MPS and will raise. The two things that actually matter for this project are both negative: torch.compile on MPS is still an explicitly self-described "early prototype" (tracker #150121 open, 3 of 33 checklist items done, missed its 2.8.0 beta target), and MPS per-op dispatch overhead is roughly 24 microseconds versus about 1 microsecond on CPU, which is measured and confirmed in issue #148219. That overhead claim about tiny networks being slower on MPS than on the ARM CPU is verified, not folklore — MPS only pulls ahead somewhere around 10^5-10^6 elements per op, so a small PPO MLP will lose on MPS unless you make the batch (num_envs x obs) genuinely large. There is also no zero-copy CPU<->MPS path today: .to("mps") still allocates a new device-private buffer and encodes a Metal blit. On the RL side, rsl_rl is the standout — its device is a plain string constructor argument with CUDA calls confined to the multi-GPU path, so it should run on "mps" or "cpu" unmodified.

## Key facts

- torch 2.13.0 is the current stable release, tagged and published 2026-07-08 (GitHub Releases API, not prerelease). Previous stable 2.12.1 was 2026-06-18.
- A native Apple Silicon wheel for Python 3.12 exists: torch-2.13.0-cp312-cp312-macosx_14_0_arm64.whl, uploaded 2026-07-08T16:05:22Z. The macosx_14_0 tag means macOS 14+; the target machine on 26.5 is well clear.
- MPS requires macOS 14.0+ as of PyTorch 2.9, which dropped Ventura support. Apple's own page still describes the MPS backend as being 'in beta phase'.
- Apple's official developer.apple.com/metal/pytorch/ page is STALE — as of this research it still advertises stable PyTorch 2.11.0 while 2.13.0 has shipped. Do not use it as a version source.
- Operator coverage as of 2.13.0: the community coverage matrix reflecting v2.13.0 lists roughly 20-30 ops still unimplemented out of ~700 tracked, i.e. ~3-4% missing. Notable gaps: _ctc_loss, poisson, binomial, take, put, linalg_lstsq, linalg_lu_solve, linalg_matrix_exp, _linalg_svd, _linalg_eigh, and several antialiased upsample backward passes. None of these are on the PPO/MLP hot path.
- PYTORCH_ENABLE_MPS_FALLBACK=1 routes unimplemented ops to CPU. It works but silently inserts device transfers, so it is a correctness crutch, not a performance strategy.
- The MPS op coverage tracking issue #141287 is still OPEN (opened 2024-11-21) and is the canonical place to check op status; it links the coverage matrix at qqaatw.dev/pytorch-mps-ops-coverage/.
- torch.compile on MPS is NOT production ready. Tracker issue #150121 is still open, last updated 2026-06-02, with only 3 of 33 checklist items complete. Its body still reads that torch.compile MPS support 'is an early prototype and attempt to use it to accelerate end-to-end network is likely to fail'. Its stated beta target was the 2.8.0 release, which has long since passed.
- Known open torch.compile/MPS defects: multi-stage Welford reductions unimplemented (blocks ResNet), reduction performance worse than eager for LLMs, no dynamic shape support, shader generation failures on T5Small and M2M100.
- float32 is fully supported and is the correct default on MPS.
- float64 is NOT supported on MPS at all. Attempting it raises 'Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.' This matters for mocap/physics code that defaults to double.
- float16 and bfloat16 both work as storage/compute dtypes on MPS on Apple Silicon with macOS 14+. bf16 autocast was added via PR #139390 (merged, commit 144fde4, shipped in the 2.5 release branch), so torch.autocast(device_type='mps', dtype=torch.bfloat16) is available.
- There is NO zero-copy CPU<->MPS transfer today. Open issue #172987 (filed 2026-01-21) states that on a device move PyTorch 'allocates a new backing buffer in the destination device's private memory' and encodes a Metal blit. Unified memory is not currently exploited.
- Issue #172987 Part 1 (allocate all tensors in unified memory, replace blit with memcpy) is implemented at a 97% test pass rate but the issue is still open and unassigned. Parts 2 and 3 (true zero-copy for read-only tensors and self-assigned moves) are only 'under investigation'.
- MEASURED dispatch overhead, from open issue #148219 (filed 2025-02-28, M1 Pro, 1000 iterations): multiplying a 1-element tensor by 1 costs 24.5 us on MPS of which 24.2 us is pure dispatch, versus 0.9 us on CPU. That is ~99% overhead.
- The 'MPS is slower than CPU for tiny networks' claim is VERIFIED. Per #148219, at 10,000 elements MPS multiply is still 24.5 us versus 1.7 us on CPU. MPS only wins at ~1,000,000 elements, where exp is 32.6 us on MPS versus 300.7 us on CPU. Practical crossover is somewhere between 10^5 and 10^6 elements per op.
- Ops already migrated to hand-written Metal shaders (exp, tanh, erfinv) measurably outperform the generic MPSGraph path at every size, per #148219.
- rsl_rl IS device-agnostic. In rsl_rl/runners/on_policy_runner.py the signature is 'device: str = "cpu"' and tensors move via .to(self.device); in rsl_rl/algorithms/ppo.py everything is .to(self.device) with no hardcoded CUDA. The only torch.cuda call (torch.cuda.set_device) is inside the multi-GPU distributed branch, and there is no torch.amp/GradScaler with CUDA-only assumptions.
- rsl_rl is alive and maintained: repo leggedrobotics/rsl_rl last pushed 2026-07-20, 2882 stars, 9 open issues, not archived. PyPI package is rsl-rl-lib 5.4.2 uploaded 2026-07-15, BSD-3-Clause, requires torch>=2.6.0 and tensordict>=0.7.0, with no NVIDIA/CUDA packages in its dependency metadata.
- tensordict (an rsl_rl dependency) ships tensordict-0.13.0-cp312-cp312-macosx_14_0_arm64.whl (0.13.0, 2026-04-11), so the rsl_rl dependency chain installs natively on arm64.
- stable-baselines3 2.9.0 does NOT auto-detect MPS. Its get_device() docstring literally says 'For now, it supports only cpu and cuda'; device='auto' maps to cuda then silently falls back to cpu. You CAN pass device='mps' explicitly and it is honored, but it is unsupported and users have historically hit missing ops (e.g. aten::amax.out). See issue #914.
- CleanRL does NOT support MPS. ppo_continuous_action.py hardcodes: device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu"). Since CleanRL is single-file-by-design you would edit the line yourself. Repo last pushed 2026-04-20, 110 open issues, 10.2k stars — slowing but not abandoned.
- torchrl 0.13.3 (2026-07-14) ships native macosx_11_0_arm64 wheels for cp310-cp312 and macosx_12_0_arm64 for cp313/cp314, including torchrl-0.13.3-cp312-cp312-macosx_11_0_arm64.whl. It has the best out-of-the-box arm64 packaging of the RL libraries surveyed.
- skrl 2.1.0 (2026-05-10, MIT) does NOT auto-detect MPS either — its parse_device resolves to 'cuda:0' if available else 'cpu' — but device is a plain string parameter so 'mps' can be passed through.
- torch.distributed with gloo/nccl does not work with the mps device; only a single MPS device is usable. Irrelevant for a single-machine build, but it rules out rsl_rl's multi-GPU path.

## Reusable for this project

- rsl_rl (PyPI: rsl-rl-lib 5.4.2, BSD-3-Clause) — the ETH PPO used by legged-gym/IsaacLab. Verified device-agnostic by reading source: OnPolicyRunner takes device: str = 'cpu', PPO uses .to(self.device) throughout, CUDA calls confined to the distributed branch. This is the single highest-value reuse: a battle-tested legged-locomotion PPO you can point at 'mps' or 'cpu' without forking. Includes RND and an ONNX export path.
- torchrl 0.13.3 — native cp312 macOS arm64 wheels, so it installs cleanly. Useful for its replay buffers, TensorDict-based rollout plumbing, and env transforms even if you write the PPO loop yourself.
- tensordict 0.13.0 — native cp312 arm64 wheel; pulled in by rsl_rl anyway. Good for batching heterogeneous observation dicts (proprioception + terrain scan + waypoint goal) without hand-rolled dict handling.
- skrl 2.1.0 (MIT) — modular PPO plus a documented AMP (Adversarial Motion Prior) implementation, which is exactly the motion-imitation algorithm this project needs. Device is a string parameter so 'mps'/'cpu' pass through; only its auto-detect defaults to cuda. Worth reading for the AMP discriminator and motion-dataset handling even if you do not adopt the framework.
- CleanRL ppo_continuous_action.py — single-file reference PPO to read and copy from, not to depend on. Changing one hardcoded device line is trivial; its value is as a known-correct PPO implementation to diff your own against.
- PYTORCH_ENABLE_MPS_FALLBACK=1 — use during bring-up to find which ops are missing, then remove it so silent CPU round-trips cannot hide in your training loop.
- The coverage matrix at qqaatw.dev/pytorch-mps-ops-coverage/ (reflects v2.13.0) — check any exotic op against it before building on it.
- torch.autocast(device_type='mps', dtype=torch.bfloat16) is available if you ever scale to a large enough network to want it. For small PPO MLPs stay in fp32 — mixed precision adds cast ops, and more ops means more 24 us dispatches.

## Blockers

- MAJOR GOTCHA, directly affects how you benchmark: MPS reports as UNAVAILABLE inside agent/CLI sandboxes. Issue #177819 was opened 2026-03-19 claiming macOS 26.3.1 broke MPS (is_built()=True, is_available()=False, device_count()=0, 'The MPS backend is supported on macOS 14.0+' error). It is NOT an OS bug. Multiple reporters and maintainer malfet traced it to processes running under a seatbelt sandbox (CODEX_SANDBOX=seatbelt) that has no GPU access; running the identical Python from a normal Terminal works. Unsetting the env var does not help — the restricted execution context itself is the trigger. Consequence: any Phase-0 hardware-check or benchmark script run by an agent inside a sandbox will report no MPS and silently benchmark CPU, producing misleading numbers. Run those from a real Terminal.
- float64 is completely unsupported on MPS and raises TypeError. Any mocap loading, physics integration, or reference-motion math that defaults to numpy float64 must be explicitly cast to float32 before touching the MPS device.
- Do not plan on torch.compile for MPS. Tracker #150121 remains an open 'early prototype' with 3/33 items done as of 2026-06-02 and a missed 2.8.0 beta target. Worse, there is a confirmed silent-wrong-answer bug: #169738 (open, 2025-12-06) reports AdaptiveMaxPool producing incorrect results under torch.compile on MPS with divergence over 1.0 versus eager. For small PPO MLPs the win would have to come from kernel fusion reducing dispatch count, and given the prototype status this is speculative at best.
- Per-op dispatch overhead of ~24 us (vs ~1 us CPU, issue #148219) means a small PPO actor-critic MLP will very likely run SLOWER on MPS than on the M3 Max ARM CPU. PPO update loops are dispatch-heavy: many small ops, many minibatches, many epochs. Unless num_envs x obs_dim makes each tensor cross the ~10^5-10^6 element threshold, MPS is a pessimization. Benchmark CPU as a serious candidate, not a fallback.
- No zero-copy CPU<->MPS transfer exists (issue #172987, open since 2026-01-21). Despite physically unified memory, .to('mps') allocates a fresh device-private buffer and blits. A design that ping-pongs observations between a CPU physics sim and an MPS policy every step will pay real memcpy cost plus dispatch on every transfer.
- Correctness bug on the transfer path itself: #189690 (open, 2026-07-13) reports that a non_blocking CPU-to-MPS copy can read released storage and silently corrupt data. Avoid non_blocking=True on MPS copies.
- Silent numerical-corruption bugs still being filed against MPS as recently as July 2026: #189960 (torch.cat silently corrupts output past 2^31 elements via integer overflow, also writing out of bounds into unrelated tensors), #179608 (avg_pool1d produces negative values from non-negative input), #170837 (inconsistent batched vs non-batched inference results on BERT/RoBERTa). The backend is beta and it shows — validate MPS results against CPU on any new op you rely on.
- stable-baselines3, CleanRL, and skrl all default away from MPS and none of them test on it. SB3's get_device() explicitly documents 'only cpu and cuda'; CleanRL hardcodes the cuda-or-cpu ternary. Only rsl_rl and torchrl are cleanly device-agnostic, and even rsl_rl has no MPS CI or any MPS issue in its tracker — 'should work' from source reading, not 'known to work'.
- NOT a blocker, stated plainly: nothing about PyTorch MPS is CUDA-dependent. PyTorch itself is fully usable on this machine. The CUDA constraint disqualifies GPU-accelerated simulators (Isaac Gym/Lab) and NVIDIA Warp CUDA paths, but not PyTorch.

## Performance evidence

- Issue #148219 (M1 Pro 16GB, 1000 iterations, opened 2025-02-28, still open): 1-element tensor, multiply by 1 — MPS 24.5 us of which 24.2 us is dispatch; CPU 0.9 us. Roughly 27x slower on GPU.
- Issue #148219: 1-element tensor sqrt — MPS 21 us vs CPU 0.9 us.
- Issue #148219: 10,000-element tensor multiply — MPS 24.5 us vs CPU 1.7 us. Dispatch still ~24 us and still dominant, so MPS is ~14x slower at this size.
- Issue #148219: 1,000,000-element tensor exp — MPS/Metal 32.6 us vs CPU 300.7 us. Here MPS finally wins by ~9x. Broader pattern at 1M elements: MPS 40-60 us vs CPU 190-400 us.
- Issue #148219 qualitative finding: ops implemented as hand-written Metal shaders (exp, tanh, erfinv) beat the generic MPSGraph-backed ops (sqrt, log, trig) at all sizes — and PyTorch 2.13.0 migrated many more ops to Metal kernels, so 2.13 should be better than these M1 Pro numbers suggest.
- PyTorch 2.13.0 release notes: FlexAttention landed on MPS with up to ~12x speedup over SDPA on sparse patterns. Not relevant to a small PPO MLP, but it evidences real ongoing Metal kernel investment.
- NO published benchmark was found for small-MLP PPO training throughput on M3 Max under MPS vs ARM CPU/Accelerate. This gap must be closed empirically in Phase 0 — the dispatch-overhead data strongly predicts CPU wins for small nets, but no one has published the specific measurement.

## Sources

- [PyTorch Releases — v2.13.0 published, v2.12.1 prior](https://github.com/pytorch/pytorch/releases) — 2026-07-08
- [PyPI torch 2.13.0 file listing — torch-2.13.0-cp312-cp312-macosx_14_0_arm64.whl](https://pypi.org/pypi/torch/2.13.0/json) — 2026-07-08
- [MPS backend — PyTorch 2.13 documentation (macOS 14.0+ requirement, availability check)](https://docs.pytorch.org/docs/2.13/notes/mps.html) — 2026-07-08
- [Apple — Accelerated PyTorch training on Mac (states MPS backend is in beta; page is stale at 2.11.0)](https://developer.apple.com/metal/pytorch/) — 2026-08-12
- [MPS operator coverage tracking issue (2.6+ version) — OPEN](https://github.com/pytorch/pytorch/issues/141287) — 2024-11-21
- [PyTorch MPS Ops Coverage Matrix — reflects v2.13.0, ~20-30 of ~700 ops unsupported](http://qqaatw.dev/pytorch-mps-ops-coverage/) — 2026-08-12
- [torch.compile on MPS progress tracker — OPEN, 3/33 items, 'early prototype', missed 2.8.0 beta target](https://github.com/pytorch/pytorch/issues/150121) — 2026-06-02
- [MPS vs Metal vs CPU performance comparison — OPEN, measured 24.2us dispatch vs 0.9us CPU](https://github.com/pytorch/pytorch/issues/148219) — 2025-02-28
- [Leveraging Unified Memory for MPS Tensors on Apple Silicon — OPEN, confirms .to('mps') still allocates + blits, no zero-copy today](https://github.com/pytorch/pytorch/issues/172987) — 2026-01-21
- [MPS unavailable on macOS 26.3.1 arm64 — OPEN, root-caused in comments to seatbelt agent sandbox lacking GPU access, not an OS bug](https://github.com/pytorch/pytorch/issues/177819) — 2026-03-19
- [MPS built but not available on macOS 26 (Tahoe) — CLOSED, reporter fixed by upgrading macOS 26.0 to 26.1](https://github.com/pytorch/pytorch/issues/167679) — 2025-11-12
- [[MPS] non_blocking CPU-to-MPS copy can read released storage and corrupt data — OPEN](https://github.com/pytorch/pytorch/issues/189690) — 2026-07-13
- [[MPS] torch.cat silently corrupts output when exceeding 2^31 elements — OPEN](https://github.com/pytorch/pytorch/issues/189960) — 2026-07-15
- [[MPS] avg_pool1d produces negative values from non-negative input — OPEN](https://github.com/pytorch/pytorch/issues/179608) — 2026-04-07
- [[MPS][Inductor] AdaptiveMaxPool produces incorrect results with torch.compile — OPEN](https://github.com/pytorch/pytorch/issues/169738) — 2025-12-06
- [MPS backend - inconsistent results for batched inference on BERT/RoBERTa — OPEN](https://github.com/pytorch/pytorch/issues/170837) — 2025-12-19
- [[MPS] Extend autocast support to bf16 — CLOSED, implemented by PR #139390 (commit 144fde4, 2.5 release branch)](https://github.com/pytorch/pytorch/issues/139386) — 2024-10-31
- [[MPS] Typo in error message for supported autocast type — documents the fp16/bf16 autocast history](https://github.com/pytorch/pytorch/issues/139190) — 2024-10-29
- [rsl_rl OnPolicyRunner source — device: str = 'cpu' parameter, CUDA confined to distributed path](https://raw.githubusercontent.com/leggedrobotics/rsl_rl/main/rsl_rl/runners/on_policy_runner.py) — 2026-07-20
- [rsl_rl PPO source — all .to(self.device), no hardcoded CUDA, no GradScaler](https://raw.githubusercontent.com/leggedrobotics/rsl_rl/main/rsl_rl/algorithms/ppo.py) — 2026-07-20
- [PyPI rsl-rl-lib 5.4.2 — BSD-3-Clause, torch>=2.6.0, no NVIDIA deps](https://pypi.org/pypi/rsl-rl-lib/json) — 2026-07-15
- [leggedrobotics/rsl_rl repo metadata — pushed 2026-07-20, 2882 stars, not archived](https://api.github.com/repos/leggedrobotics/rsl_rl) — 2026-07-20
- [SB3 get_device() source — 'For now, it supports only cpu and cuda'](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/master/stable_baselines3/common/utils.py) — 2026-08-12
- [SB3 — Supporting PyTorch GPU compatibility on Apple Silicon chips](https://github.com/DLR-RM/stable-baselines3/issues/914) — 2026-08-12
- [CleanRL ppo_continuous_action.py — hardcoded cuda-or-cpu device ternary, no MPS](https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_continuous_action.py) — 2026-04-20
- [CleanRL repo metadata — last push 2026-04-20, 110 open issues, 10.2k stars](https://api.github.com/repos/vwxyzjn/cleanrl) — 2026-04-20
- [PyPI torchrl 0.13.3 — native macosx_11_0_arm64 cp312 wheel present](https://pypi.org/pypi/torchrl/0.13.3/json) — 2026-07-14
- [PyPI tensordict 0.13.0 — tensordict-0.13.0-cp312-cp312-macosx_14_0_arm64.whl](https://pypi.org/pypi/tensordict/json) — 2026-04-11
- [PyPI skrl 2.1.0 — MIT, no CUDA-only requirements](https://pypi.org/pypi/skrl/json) — 2026-05-10
- [skrl parse_device source — defaults 'cuda:0' if available else 'cpu', no MPS auto-detect](https://raw.githubusercontent.com/Toni-SM/skrl/main/skrl/__init__.py) — 2026-08-12
- [PyPI stable-baselines3 2.9.0 — requires torch>=2.8,<3.0, Python >=3.10](https://pypi.org/pypi/stable-baselines3/json) — 2026-08-12
