"""Domain randomisation.

**Domain randomisation** means varying the simulated world during training (friction, link
masses, actuator strength, sensor noise, shoves) so the policy cannot exploit one exact set
of physical constants. Without it, a locomotion policy reliably discovers a gait that
depends on precise ground friction or exact limb inertias, and any change breaks it. Even
staying entirely in simulation it matters here, because Phase 4 changes the terrain
underneath a policy trained on flat ground.

Two kinds, split by what they can touch:

* **Model randomisation** (friction, mass, centre of mass, PD gains, armature) alters
  `MjModel` fields, which are shared across environments in this design.
* **Per-step randomisation** (observation noise, random pushes) alters `MjData` or the
  observation, and is applied per environment per step.

Why a model *pool* rather than per-environment models
-----------------------------------------------------
Measured on this machine: one `MjModel` copy is **5.2 MB**, so 4096 per-environment copies
would cost **21 GB** and destroy cache locality in the physics inner loop, which is the one
thing the whole architecture is built to protect.

Also measured: an `MjData` allocated against one copy steps correctly against any other
copy, because they share an identical structure and differ only in parameter values. So a
pool of K randomised models covers the same ground: each environment is assigned a pool
entry at reset, and over a training run every environment sees many variants. At the
default K = 64 the pool costs about 334 MB, which is free here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class DomainRandConfig:
    """Randomisation ranges. All are multiplicative scales unless noted."""

    enabled: bool = True
    #: Number of distinct randomised models. Memory is about 5.2 MB each.
    model_pool_size: int = 64

    # --- model parameters, resampled per pool entry ---
    #: Ground sliding friction, absolute value not a scale. The single most important
    #: parameter for gait: a policy trained on one friction learns to skate or to stomp.
    friction_range: tuple[float, float] = (0.1, 1.25)
    #: Per-body mass scale.
    mass_scale_range: tuple[float, float] = (0.85, 1.15)
    #: Random offset applied to each body's centre of mass, in metres.
    com_offset: float = 0.015
    #: PD gain scales. Stands in for unmodelled actuator dynamics and manufacturing spread.
    kp_scale_range: tuple[float, float] = (0.85, 1.15)
    kd_scale_range: tuple[float, float] = (0.85, 1.15)
    #: Rotor inertia scale. Small, but it damps unrealistically fast joint motion.
    armature_scale_range: tuple[float, float] = (1.0, 1.1)

    # --- per-step noise, standard deviations in physical units ---
    obs_noise_joint_pos: float = 0.01  # rad
    obs_noise_joint_vel: float = 0.15  # rad/s
    obs_noise_gravity: float = 0.02  # unit vector components
    obs_noise_lin_vel: float = 0.05  # m/s
    obs_noise_ang_vel: float = 0.10  # rad/s

    # --- random pushes ---
    #: Mean seconds between shoves for a given environment. Set to 0 to disable.
    push_interval_s: float = 5.0
    #: Magnitude of the horizontal velocity impulse, in m/s.
    push_vel_xy: float = 0.7


def randomize_model(
    model: mujoco.MjModel,
    base: mujoco.MjModel,
    rng: np.random.Generator,
    cfg: DomainRandConfig,
    floor_geom_id: int,
) -> None:
    """Randomise one model in place, always relative to `base`.

    Scaling relative to the pristine base rather than to the model's current values matters:
    re-randomising in place would compound scales and drift the pool away from realistic
    parameters over time.
    """
    # Ground friction. Only the sliding component; torsional and rolling stay as authored.
    model.geom_friction[floor_geom_id, 0] = rng.uniform(*cfg.friction_range)

    # Body masses and inertias. Inertia scales with mass so the body stays physical.
    scales = rng.uniform(*cfg.mass_scale_range, size=base.nbody)
    model.body_mass[:] = base.body_mass * scales
    model.body_inertia[:] = base.body_inertia * scales[:, None]

    # Centre-of-mass offsets, skipping the world body.
    model.body_ipos[:] = base.body_ipos
    model.body_ipos[1:] += rng.normal(0.0, cfg.com_offset, size=(base.nbody - 1, 3))

    # PD gains. The actuator layout is set by model_prep: gain = kp, bias = (0, -kp, -kv).
    kp_scale = rng.uniform(*cfg.kp_scale_range, size=base.nu)
    kd_scale = rng.uniform(*cfg.kd_scale_range, size=base.nu)
    model.actuator_gainprm[:, 0] = base.actuator_gainprm[:, 0] * kp_scale
    model.actuator_biasprm[:, 1] = base.actuator_biasprm[:, 1] * kp_scale
    model.actuator_biasprm[:, 2] = base.actuator_biasprm[:, 2] * kd_scale

    model.dof_armature[:] = base.dof_armature * rng.uniform(
        *cfg.armature_scale_range, size=base.nv
    )


def build_model_pool(
    base: mujoco.MjModel,
    cfg: DomainRandConfig,
    rng: np.random.Generator,
    floor_geom_id: int,
) -> list[mujoco.MjModel]:
    """Build the pool of randomised models.

    Entry 0 is left as the pristine base, so evaluation and debugging always have access to
    the nominal dynamics and any weirdness can be checked against an unrandomised model.
    """
    if not cfg.enabled or cfg.model_pool_size <= 1:
        return [base]

    pool = [base]
    for _ in range(cfg.model_pool_size - 1):
        variant = copy.copy(base)
        randomize_model(variant, base, rng, cfg, floor_geom_id)
        pool.append(variant)
    return pool


class ObservationNoise:
    """Adds sensor noise to the proprioceptive observation, vectorised over the batch.

    Applied to the observation only, never to the underlying physics state, so rewards and
    terminations still see ground truth. That is the correct split: a real robot has noisy
    sensors but the world itself is not noisy, and rewarding against noisy state would just
    add variance to the gradient.
    """

    def __init__(self, cfg: DomainRandConfig, n_joint_pos: int, n_joint_vel: int) -> None:
        self.cfg = cfg
        # Per-dimension standard deviations over the proprioceptive block, laid out to match
        # vec_env's observation order: joints, joint velocities, gravity, linear velocity,
        # angular velocity, previous action, foot contact.
        self.scale = np.concatenate(
            [
                np.full(n_joint_pos, cfg.obs_noise_joint_pos),
                np.full(n_joint_vel, cfg.obs_noise_joint_vel),
                np.full(3, cfg.obs_noise_gravity),
                np.full(3, cfg.obs_noise_lin_vel),
                np.full(3, cfg.obs_noise_ang_vel),
            ]
        ).astype(np.float32)
        self.width = self.scale.size

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray | None:
        """Draw an (n, width) noise block, or None when randomisation is disabled.

        Returns the block rather than mutating, because the caller indexes into its
        observation buffer with an integer array, and fancy indexing yields a copy.
        """
        if not self.cfg.enabled:
            return None
        return rng.standard_normal((n, self.width), dtype=np.float32) * self.scale
