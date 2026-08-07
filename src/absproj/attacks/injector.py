"""Facade tying the five attack generators together: given a substrate pool
of real clean tracks, produce `variants_per_class` variants for every attack
class, spanning each class's configured severity range.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from absproj.attacks.drift import generate_position_drift
from absproj.attacks.ghost import generate_ghost
from absproj.attacks.hijack import generate_track_hijack
from absproj.attacks.icao_collision import generate_icao_collision
from absproj.attacks.replay import generate_replay
from absproj.attacks.substrate import SubstratePool
from absproj.attacks.types import AttackClass, AttackedTrack
from absproj.config import AttacksConfig, BBox


def generate_variants(
    attack_class: AttackClass,
    config: AttacksConfig,
    pool: SubstratePool,
    bbox: BBox,
    rng: np.random.Generator,
    base_start_time: datetime,
) -> list[AttackedTrack]:
    n = config.variants_per_class
    srange = config.severity_ranges[attack_class.value]
    severities = np.linspace(srange.min, srange.max, n)

    variants: list[AttackedTrack] = []
    for i, severity in enumerate(severities):
        severity = float(severity)
        variant_id = f"{attack_class.value}_{i:02d}"

        if attack_class == AttackClass.GHOST:
            variants.append(generate_ghost(rng, severity, variant_id, bbox, base_start_time))

        elif attack_class == AttackClass.POSITION_DRIFT:
            base = pool.sample_track(rng)
            mode = "sudden" if i % 2 == 0 else "gradual"
            variants.append(generate_position_drift(base, rng, severity, variant_id, mode=mode))

        elif attack_class == AttackClass.ICAO_COLLISION:
            victim, intruder = pool.sample_two_distinct_tracks(rng)
            variants.append(generate_icao_collision(victim, intruder, rng, severity, variant_id))

        elif attack_class == AttackClass.TRACK_HIJACK:
            base, donor = pool.sample_two_distinct_tracks(rng)
            variants.append(generate_track_hijack(base, donor, rng, severity, variant_id))

        elif attack_class == AttackClass.REPLAY:
            live, source = pool.sample_two_distinct_tracks(rng)
            variants.append(generate_replay(live, source, rng, severity, variant_id))

        else:
            raise ValueError(f"unknown attack class: {attack_class}")

    return variants


def generate_all_variants(
    config: AttacksConfig,
    pool: SubstratePool,
    bbox: BBox,
    rng: np.random.Generator,
    base_start_time: datetime,
) -> dict[AttackClass, list[AttackedTrack]]:
    return {
        attack_class: generate_variants(attack_class, config, pool, bbox, rng, base_start_time)
        for attack_class in AttackClass
    }
