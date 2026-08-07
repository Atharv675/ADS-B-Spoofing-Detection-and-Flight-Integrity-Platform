"""Classification metrics and detection latency, shared by every method the
benchmark compares (rule-based, NIS, ML, MLAT, radar, fused).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import numpy as np


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    n: int
    n_positive: int
    n_predicted_positive: int


def compute_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> ClassificationMetrics:
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_pred_arr = np.asarray(y_pred, dtype=bool)

    tp = int(np.sum(y_true_arr & y_pred_arr))
    fp = int(np.sum(~y_true_arr & y_pred_arr))
    fn = int(np.sum(y_true_arr & ~y_pred_arr))
    tn = int(np.sum(~y_true_arr & ~y_pred_arr))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (tp + fp) > 0 and (tp + fn) > 0 and (precision + recall) > 0
        else float("nan")
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    return ClassificationMetrics(
        precision=precision, recall=recall, f1=f1, fpr=fpr, fnr=fnr,
        n=len(y_true_arr), n_positive=int(y_true_arr.sum()), n_predicted_positive=int(y_pred_arr.sum()),
    )


def detection_latency_seconds(
    times: Sequence[datetime], is_attacked: Sequence[bool], predicted: Sequence[bool]
) -> Optional[float]:
    """Seconds from the first attacked row to the first row where the
    detector actually flagged an attacked row. None if the attack was never
    caught, or if there's no attack in this scenario at all."""
    attacked_indices = [i for i, a in enumerate(is_attacked) if a]
    if not attacked_indices:
        return None
    onset_time = times[attacked_indices[0]]

    for i in attacked_indices:
        if predicted[i]:
            return (times[i] - onset_time).total_seconds()
    return None
