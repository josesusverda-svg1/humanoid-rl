"""Terrain generation. See `field.py` for the design and the measurements behind it."""

from humanoid_rl.terrain.field import (
    FLOOR_GEOM,
    TerrainConfig,
    TerrainField,
    bake,
    foot_probe_offsets,
    generate,
    inject,
    place_on_terrain,
)

__all__ = ["FLOOR_GEOM", "TerrainConfig", "TerrainField", "bake", "foot_probe_offsets",
           "generate", "inject", "place_on_terrain"]
