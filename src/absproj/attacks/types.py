"""Shared types for the adversarial testbed.

Every attack generator produces an AttackedTrack: a sequence of *broadcast*
StateVectors (what the detection pipeline actually sees -- this is the only
thing Kalman/NIS/ML ever get to look at, exactly like a real detector) paired
row-for-row with the *true* physical StateVector each one corresponds to (or
None, where there is no real physical aircraft at all). Only the MLAT/radar
verification checks are allowed to see the true_state_vectors -- they're the
simulated "independent sensor" side of the fusion story, and using true
positions to simulate them is not cheating, it's the whole mechanism (see
verification/mlat.py and verification/radar.py docstrings).

`is_attacked` is the ground-truth label used for evaluation (Phase 7) -- it is
never fed to any detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from absproj.ingestion.normalize import StateVector


class AttackClass(str, Enum):
    GHOST = "ghost"
    POSITION_DRIFT = "position_drift"
    ICAO_COLLISION = "icao_collision"
    TRACK_HIJACK = "track_hijack"
    REPLAY = "replay"


@dataclass
class AttackedTrack:
    attack_class: AttackClass
    variant_id: str
    severity: float
    icao24: str
    broadcast_state_vectors: list[StateVector]
    true_state_vectors: list[Optional[StateVector]]  # same length/order as broadcast; None = no physical target
    is_attacked: list[bool]  # same length/order as broadcast
    params: dict = field(default_factory=dict)  # generator-specific bookkeeping (donor icao24s, offsets, etc.)

    def __post_init__(self) -> None:
        n = len(self.broadcast_state_vectors)
        if len(self.true_state_vectors) != n or len(self.is_attacked) != n:
            raise ValueError(
                f"AttackedTrack arrays must be the same length: "
                f"broadcast={n}, true={len(self.true_state_vectors)}, is_attacked={len(self.is_attacked)}"
            )
