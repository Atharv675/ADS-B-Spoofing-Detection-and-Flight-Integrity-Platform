import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.evaluation.metrics import compute_metrics, detection_latency_seconds  # noqa: E402

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_perfect_detection():
    y_true = [False, False, True, True]
    y_pred = [False, False, True, True]
    m = compute_metrics(y_true, y_pred)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.fpr == 0.0
    assert m.fnr == 0.0


def test_known_confusion_matrix():
    # tp=2, fp=1, fn=1, tn=2
    y_true = [True, True, True, False, False, False]
    y_pred = [True, True, False, True, False, False]
    m = compute_metrics(y_true, y_pred)
    assert m.precision == 2 / 3
    assert m.recall == 2 / 3
    assert abs(m.f1 - 2 / 3) < 1e-9
    assert m.fpr == 1 / 3
    assert m.fnr == 1 / 3


def test_no_positives_predicted_precision_is_nan():
    y_true = [True, True, False]
    y_pred = [False, False, False]
    m = compute_metrics(y_true, y_pred)
    assert m.precision != m.precision  # NaN
    assert m.recall == 0.0


def test_no_actual_positives_recall_is_nan():
    y_true = [False, False, False]
    y_pred = [True, False, False]
    m = compute_metrics(y_true, y_pred)
    assert m.recall != m.recall  # NaN
    assert m.fpr == 1 / 3


def test_counts_are_correct():
    y_true = [True, False, True, False]
    y_pred = [True, True, False, False]
    m = compute_metrics(y_true, y_pred)
    assert m.n == 4
    assert m.n_positive == 2
    assert m.n_predicted_positive == 2


def test_detection_latency_immediate():
    times = [T0 + timedelta(seconds=15 * i) for i in range(5)]
    is_attacked = [False, False, True, True, True]
    predicted = [False, False, True, False, False]
    assert detection_latency_seconds(times, is_attacked, predicted) == 0.0


def test_detection_latency_delayed():
    times = [T0 + timedelta(seconds=15 * i) for i in range(5)]
    is_attacked = [False, False, True, True, True]
    predicted = [False, False, False, False, True]
    assert detection_latency_seconds(times, is_attacked, predicted) == 30.0


def test_detection_latency_never_caught():
    times = [T0 + timedelta(seconds=15 * i) for i in range(5)]
    is_attacked = [False, False, True, True, True]
    predicted = [False, False, False, False, False]
    assert detection_latency_seconds(times, is_attacked, predicted) is None


def test_detection_latency_no_attack_rows():
    times = [T0 + timedelta(seconds=15 * i) for i in range(5)]
    is_attacked = [False] * 5
    predicted = [False, True, False, False, False]
    assert detection_latency_seconds(times, is_attacked, predicted) is None
