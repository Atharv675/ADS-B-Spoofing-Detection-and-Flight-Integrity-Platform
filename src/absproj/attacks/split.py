"""Held-out evaluation split, per the brief's mandatory methodology:

1. Within every attack class except the held-out one, variants are split into
   train/test -- any learned threshold gets tuned only on the train subset,
   then evaluated on the disjoint test subset it never saw.
2. One entire attack class is excluded from training altogether and only
   ever appears in the test set, to check cross-class generalization.

The train/test split within a class is stratified by severity (not a random
shuffle): variants are sorted by severity and evenly-spaced indices go to
train, so both splits span the full easy-to-hard range rather than train
happening to get all the obvious high-severity variants and test getting
stuck with only the subtle ones (or vice versa).
"""
from __future__ import annotations

from dataclasses import dataclass

from absproj.attacks.types import AttackClass, AttackedTrack


@dataclass
class EvaluationSplit:
    train: dict[AttackClass, list[AttackedTrack]]
    test: dict[AttackClass, list[AttackedTrack]]
    holdout_class: AttackClass


def _stratified_split(
    variants: list[AttackedTrack], train_fraction: float
) -> tuple[list[AttackedTrack], list[AttackedTrack]]:
    sorted_variants = sorted(variants, key=lambda v: v.severity)
    n = len(sorted_variants)
    n_train = round(n * train_fraction)

    if n_train <= 0:
        train_idx: set[int] = set()
    elif n_train >= n:
        train_idx = set(range(n))
    elif n_train == 1:
        train_idx = {n // 2}
    else:
        train_idx = {round(i * (n - 1) / (n_train - 1)) for i in range(n_train)}

    train = [v for i, v in enumerate(sorted_variants) if i in train_idx]
    test = [v for i, v in enumerate(sorted_variants) if i not in train_idx]
    return train, test


def build_evaluation_split(
    variants_by_class: dict[AttackClass, list[AttackedTrack]],
    holdout_class: AttackClass,
    train_fraction: float,
) -> EvaluationSplit:
    train: dict[AttackClass, list[AttackedTrack]] = {}
    test: dict[AttackClass, list[AttackedTrack]] = {}

    for attack_class, variants in variants_by_class.items():
        if attack_class == holdout_class:
            train[attack_class] = []
            test[attack_class] = list(variants)
            continue
        tr, te = _stratified_split(variants, train_fraction)
        train[attack_class] = tr
        test[attack_class] = te

    return EvaluationSplit(train=train, test=test, holdout_class=holdout_class)
