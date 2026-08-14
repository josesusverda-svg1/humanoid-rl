# Phase 0 adversarial critique

## VERIFIED MYSELF (5 riskiest claims, checked today 2026-08-13)

| Claim | Result |
|---|---|
| `mujoco.Renderer` can be made to pick up `hfield_data` changes via "our own MjrContext" (R5/B10) | **FALSE as designed.** [renderer.py, main](https://raw.githubusercontent.com/google-deepmind/mujoco/main/python/mujoco/rendering/classic/renderer.py) exposes only `model`, `scene`, `height`, `width`, `render`, `update_scene`, `enable/disable_depth_rendering`, `enable/disable_segmentation_rendering`, `close`. No public context, no `update_hfield`. |
| `imageio-ffmpeg==0.6.0` is a stale pin worth flagging | **Wrong direction.** PyPI JSON checked today: 0.6.0 (2025-01-16) IS latest; prior is 0.5.1 (2024-06-03). Package license BSD-2. The bundled ffmpeg binary's license is separate and unexamined. |
| MimicKit `task_location_env.py` / `task_steering_env.py` exist | **TRUE.** `task_location_env.py` 9,936 B, `task_steering_env.py` 12,134 B, `deepmimic_env.py` 41,754 B, `amp_env.py` 17,769 B, plus `task_dodgeball_env.py`, `add_env.py`, `ase_env.py`, `smp_env.py`. Repo root has LICENSE (11,344 B) and `data/`. |
| Offscreen rendering works off the main thread on macOS | **TRUE but under-specified.** Fixed in MuJoCo 2.3.4 by using CGL instead of GLFW/NSOpenGL ([mujoco#742](https://github.com/google-deepmind/mujoco/issues/742), 2.3.4 changelog). CGL contexts are still per-thread-current. |
| Section 11 version pins | **All confirmed latest on PyPI today:** numpy 2.5.2 (2026-08-09), scipy 1.18.0 (2026-06-19), tensorboard 2.21.0 (2026-06-29), torch 2.13.0 (2026-07-08), mujoco 3.11.0 (2026-07-28), tensorboardX 2.6.5 (2026-04-03). Locally verified: macOS 26.5 25F71, arm64, `mp.get_start_method()` = `spawn`, Homebrew ffmpeg 9.0.1 built `--enable-gpl --enable-libx264 --enable-version3`. |

---

## 1. CORRECTIONS

**C-1. R5 and B10 rest on a mechanism that does not exist.** There is no public context accessor on `Renderer` and no `update_hfield`. Worse, the stated mitigation ("construct and hold our own `mujoco.MjrContext` rather than reaching into the private attribute") does not work: `mjr_uploadHField` uploads into *the context you pass it*, and `Renderer.render()` draws through its own `_mjr_context`. Two contexts means two independent copies of GPU-side terrain. The real options are binary: (a) touch `renderer._mjr_context`, or (b) abandon `Renderer` for the offscreen path and drive it yourself: own `GLContext` → `make_current()` → `MjrContext` → `mjv_updateScene` → `mjr_render` → `mjr_readPixels`. Option (b) is correct and should be scheduled as work, not written down as a fallback. Note the asymmetry: the *passive viewer* gained `update_hfield`/`update_mesh`/`update_texture` in 2.3.7 (closing [#812](https://github.com/google-deepmind/mujoco/issues/812), [#958](https://github.com/google-deepmind/mujoco/issues/958), [#965](https://github.com/google-deepmind/mujoco/issues/965)); the offscreen Renderer never did.

**C-2. Offscreen framebuffer size cap is missing entirely.** MuJoCo's offscreen buffer defaults to `<visual><global offwidth="640" offheight="480"/>`. B10 renders at exactly 640x480, which conceals the ceiling. Every procedurally generated terrain scene XML and MimicKit's `humanoid.xml` must carry an enlarged `<global>` or Phase 3 videos are permanently 480p. Bake it into the scene generator now.

**C-3. Threaded rendering is true but the per-thread binding is not in the design.** A CGL context is current on one thread at a time. The render thread must own its `GLContext` and `MjrContext` and be the only thread that calls `make_current`, `mjr_uploadHField`, `mjr_render`. Since terrain reshuffles are performed by physics workers (B4), a reshuffle must post an upload request to the render thread. That cross-thread coupling is unspecified, and its failure mode is exactly the silent-stale-video bug R5 is trying to prevent.

**C-4. imageio-ffmpeg is misdiagnosed.** It is not stale; 0.6.0 is current with no successor. The actual unflagged issues: (a) the bundled ffmpeg executable carries its own license independent of the BSD-2 wrapper and nobody checked it; (b) the suggested remedy "try a system ffmpeg" points at the Homebrew binary already on this machine, verified `--enable-gpl --enable-libx264` and therefore GPL. Invoking it as a subprocess does not infect our code, but this is the only place "GPL contamination" actually appears in the project and it should be recorded deliberately rather than stumbled into.

**C-5. R6 contradicts itself on licensing.** It correctly rules LAFAN1's CC BY-NC-ND a problem because retargeting produces a derivative, then in the same cell recommends LocoMuJoCo's 19 HuggingFace clips as the registration-free day-one corpus. Per the dossier, `huggingface.co/datasets/robfiras/loco-mujoco-datasets` is CC BY-NC-ND 4.0 and those clips are already retargeted derivatives of AMASS and LAFAN1. Identical clause, identical problem. The genuinely clean day-one sources are the dm_control CMU HDF5 (CMU terms allow commercial products, forbid reselling the data) and 100STYLE (CC BY 4.0). Say plainly that anything touching the LocoMuJoCo clips is non-commercial and NoDerivatives-encumbered.

**C-6. MimicKit motion data is a licensing hole, not just an availability one.** R6 notes `data/motions/` is empty behind a SharePoint download. Unstated: the provenance and license of those clips is unknown, and the vendored YAMLs reference them by name. Apache-2.0 covers the code, not the data. Do not put them on the critical path until someone reads the download's terms.

**C-7. Vendoring MimicKit's envs pulls a module-level `import isaacgym.gymapi`.** `mimickit/engines/engine_builder.py` does that import at module scope inside a try/except, and imports the three engine modules lazily inside `build_engine()`. It will not crash, but the plan says "delete `isaac_gym_engine.py`, `isaac_lab_engine.py`, `newton_engine.py`" without noting that `engine_builder.py` itself must be edited. It is the file that decides whether the vendored tree imports.

**C-8. C1 overshoots.** [pytorch#148219](https://github.com/pytorch/pytorch/issues/148219) is about per-op *dispatch latency*; the ground truth measures GFLOP/s and dense-layer batch latency, which do not test that claim. The measurement absolutely settles the device choice, but the row should read "not applicable to our shapes" rather than "claim rejected", because the design consequence (minimize op count per step, no per-env scalar ops on the GPU path) is still live.

**C-9. C6 compares two different things and then declares no conflict.** 403,600 sim-steps/s (ours, derived) versus 650K (`testspeed`, 2x numcore threads, different model, no env logic) is an unvalidated cross-model comparison. The MJX correction is right and worth keeping; the throughput comparison should be labeled as not apples-to-apples rather than as corroboration.

**C-10. B1's E-core measurement is not runnable as written.** `powermetrics` requires root, so the benchmark script cannot collect E-cluster residency and must not silently skip it. Also there is no supported P-core pinning on macOS from Python; threads inherit the creator's QoS and raising it requires `pthread_set_qos_class_self_np` through ctypes. B1 is measuring a scheduler outcome we cannot control, so the decision rule should be throughput only, and the residency number should be an out-of-band `sudo powermetrics` observation.

**C-11. R2's primary mitigation forecloses the jump/climb roadmap.** "Drop `root_h` from the 105-dim frame" removes the single most discriminative feature of a jump. Only the secondary option (height above terrain, sampled from our own heightmap under the root) is compatible with both terrain generalization and future aerial skills. Make it the mandate, not the alternative, and freeze the discriminator observation spec before Phase 3 rather than Phase 4, because its width is baked into every checkpoint.

**C-12. `task_reward_w: 0.0` is the no-task AMP configuration.** It is vendored as "known-good structure" without noting that Phase 5 waypoint following requires a nonzero task weight and a different reward split. Copying it forward produces a policy that ignores the goal and looks like a waypoint bug.

*Minor:* Section 11 asserts an `mlx-0.32.0-cp312-cp312-macosx_26_0_arm64.whl` tag; unverified by me, low stakes since MLX is a fallback lane, but a `macosx_26_0` platform tag also requires a `packaging`/`pip` new enough to parse it.

---

## 2. GAPS

**G-1. The dashboard was not researched at all, and one of its default designs violates the Section 3 architectural law.**
- **Do not run uvicorn in the training process.** FastAPI/uvicorn executes Python bytecode on every request and would contend for the exact GIL that R3 protects. Run the dashboard as a separate process; training writes JSONL or SQLite (WAL mode) plus an mp4 directory, dashboard only reads.
- **Video needs HTTP Range (206) support.** Browsers seeking inside an mp4 require partial content. Starlette's `StaticFiles` handles Range; a naive `FileResponse` route does not.
- **Encoding parameters are unspecified and browser playback depends on them.** H.264 with `-pix_fmt yuv420p` and `+faststart` (moov atom at the front) or the gallery will fail to play or fail to seek.
- **No Node/npm/vite in the pinned list.** Decide now: built React SPA (adds a toolchain, a lockfile, a build step, and a second dependency-rot surface) versus a single static HTML page. Given the constraints, the latter.
- **Metrics transport.** TensorBoard event files are not readable from a browser app. If the dashboard is primary, JSONL/SQLite is the source of truth and TensorBoard is optional.
- Bind `127.0.0.1`, never `0.0.0.0`. Current versions if you want them: fastapi 0.141.1 (2026-07-29), uvicorn 0.52.2 (2026-08-13). Neither is pinned in Section 11.

**G-2. Multi-day laptop runs: power, sleep, and thermal drift are unaddressed.** Run under `caffeinate -dimsu` or the run dies at display sleep. On battery macOS reduces sustained performance, so require AC and log `pmset -g batt`. Thermal decay on an M3 Max over hours shows up as a slow throughput slide that is indistinguishable from a code regression unless you log per-iteration throughput and a rolling median. Add a drift alarm alongside R3's CI throughput gate.

**G-3. Fork safety.** Confirmed here: `multiprocessing.get_start_method()` is `spawn` under Python 3.12 on macOS. Assert it rather than assume it. Any code path that reaches `os.fork` after Metal or CoreFoundation has initialized aborts with the ObjC fork-safety error. The places people reach for `multiprocessing` are exactly the ones in this plan: video encoding, dashboard launch, evaluation harness. Rule: `subprocess` only, never fork, in a process that has touched MPS.

**G-4. MPS memory under unified 96 GB is assumed free.** The MPS allocator and the physics working set draw from the same pool. Unresearched and unpinned: `torch.mps.empty_cache()`, `torch.mps.set_per_process_memory_fraction`, `PYTORCH_MPS_HIGH_WATERMARK_RATIO`. At 4096 envs the rollout buffer plus a 1e5-entry AMP replay buffer persists across iterations and is multiple GB. macOS does not OOM-kill promptly, it compresses and swaps, so the failure mode is a silent throughput collapse. Track RSS and `torch.mps.current_allocated_memory()` per iteration with a hard ceiling.

**G-5. Heightfield amplitude is fixed at model-compile time.** `hfield_data` is normalized elevation in [0,1] scaled by the MJCF `size` z component. A curriculum cannot raise amplitude past the compiled ceiling. Set the compiled z to the maximum curriculum amplitude on day one and scale the written values, otherwise every curriculum step past the ceiling means a model recompile that invalidates every `MjData` and every worker handle. Not covered by Section 7, B3, or B4.

**G-6. Heightfields cannot represent the skills the roadmap promises.** An hfield is single-valued elevation: no overhangs, no vertical risers, no undercuts. "Climb" is not expressible. If climbing or true stairs are on the roadmap, terrain needs composited box geoms, which is a different data structure, a different collision cost curve, and a different curriculum knob than the one mutable array B4 is built around. Decide hfield-only versus hfield-plus-boxes before writing the vec-env model-variant scheme.

**G-7. Goal-conditioned observation width is not frozen.** Adding waypoint conditioning in Phase 5 changes the actor observation dimension and invalidates Phase 2 and 3 checkpoints. Reserve the goal slots from Phase 1 and zero them until Phase 5. Costs nothing, saves a retrain.

**G-8. AMP normalizer semantics are undecided.** The discriminator sees both demo and agent observations. If the normalizer updates from agent data only, the demo distribution drifts underneath the discriminator and style reward decays in a way that looks exactly like adversarial collapse and will be misdiagnosed as R1. The plan vendors both `normalizer.py` and `diff_normalizer.py` without deciding. Write it down: fit the discriminator normalizer on the demo dataset, freeze it, apply it to both sides.

**G-9. Checkpoint portability across the MPS/CPU boundary is untested.** B12 covers seeded divergence but not reload. Evaluation and video export may run on a different device than training.

**G-10. Section 11 is incomplete on transitive dependencies.** `dm-control` is pinned for "loaders and reward reference only" but its install pulls `absl-py`, `labmaze`, `lxml`, `pyopengl`, `glfw`, `protobuf` and more, none of which appear in the pin table, and some of which initialize GL paths. Either import the specific submodules and verify no GL context is created, or apply the project's own rule and vendor the two files.

**G-11. Nothing verifies arm64 purity.** A single x86_64 wheel or a Rosetta-launched Terminal silently halves everything and would be attributed to thermal or code causes. Assert `platform.machine() == "arm64"` and `sysctl sysctl.proc_translated == 0` at startup, and check the loaded extension modules with `lipo -archs`.

---

## 3. MUST-VERIFY-LOCALLY (prioritized for the benchmark script)

1. **V1, gates everything.** `platform.machine()=="arm64"`, `sysctl.proc_translated==0`, `torch.backends.mps.is_available()` True, `mp.get_start_method()=="spawn"`, every loaded `.so` arm64. Hard fail with the "run from a real Terminal, not an agent sandbox" message ([pytorch#177819](https://github.com/pytorch/pytorch/issues/177819)).
2. **V2, gates Phase 3 video.** From a non-main worker thread: create `GLContext` + `MjrContext`, `make_current`, render, read pixels, assert correctness. Then a second thread with its own contexts concurrently. Then `mjr_uploadHField` on that thread and assert the pixels change, and assert they do not change when the upload is skipped. Then render at 1280x720 to force the `offwidth`/`offheight` ceiling and confirm the `<visual><global>` fix.
3. **V3, gates Phase 4.** Write hfield values above the compiled `size` z and confirm the clamping behavior. Then confirm the max-amplitude-at-compile-time plus scaled-data scheme reproduces the full curriculum range exactly, with no recompile.
4. **V4, gates the training budget.** B2 and B3 as written, plus mean active contacts, plus a **60-minute sustained run**. Report median env steps/s over minutes 50 to 60 versus minutes 0 to 10. That delta is the thermal answer and no published source can give it to you.
5. **V5, gates long runs.** RSS, `torch.mps.current_allocated_memory()`, and `vm_stat` swap counters sampled every iteration for 30 minutes at target `n_envs` with the AMP replay buffer full. Any swap activity is a failure.
6. **V6, gates the dashboard.** Encode one clip, `ffprobe` it to confirm `yuv420p` and `faststart`, serve it through the FastAPI static mount, seek to the middle in Safari and Chrome, and assert the response is 206 with `Content-Range`.
7. **V7, gates AMP.** 100 iterations with the discriminator normalizer frozen on demo data; assert demo-side normalized statistics do not drift. Separately, deliberately over-train the discriminator and confirm the R1 accuracy tripwire actually fires.
8. **V8, cheap, do early.** Assert no code path calls `os.fork` after MPS initialization; smoke-test the video encoder and dashboard launch as `subprocess` under a live MPS context.
9. **V9, cheap.** Import the vendored MimicKit tree with `engines/` reduced to `engine.py` + `mujoco_engine.py` and an edited `engine_builder.py`, in a venv with no `isaacgym` and no `gymnasium`.
10. **V10, before Phase 5.** Freeze and unit-test the three observation widths (actor, critic, discriminator) with goal slots reserved and zeroed and terrain-relative root height in place of world z.
11. **V11, deferred.** B11 MLX tripwire, B12 determinism envelope, and cross-device checkpoint reload parity (same observation, same action within tolerance, MPS-saved to CPU-loaded).

Sources: [mujoco renderer.py](https://raw.githubusercontent.com/google-deepmind/mujoco/main/python/mujoco/rendering/classic/renderer.py), [mujoco#742](https://github.com/google-deepmind/mujoco/issues/742), [mujoco#965](https://github.com/google-deepmind/mujoco/issues/965), [mujoco#812](https://github.com/google-deepmind/mujoco/issues/812), [MuJoCo visualization docs](https://mujoco.readthedocs.io/en/stable/programming/visualization.html), [MimicKit envs listing](https://api.github.com/repos/xbpeng/MimicKit/contents/mimickit/envs), [imageio-ffmpeg PyPI](https://pypi.org/pypi/imageio-ffmpeg/json), [FFmpeg legal](https://ffmpeg.org/legal.html), [pytorch#148219](https://github.com/pytorch/pytorch/issues/148219), [pytorch#177819](https://github.com/pytorch/pytorch/issues/177819).
