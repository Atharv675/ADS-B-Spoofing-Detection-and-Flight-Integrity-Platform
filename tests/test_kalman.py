import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.tracking.kalman import (  # noqa: E402
    KalmanFilterCV,
    process_noise_matrix,
    transition_matrix,
)
from absproj.tracking.nis import chi_square_threshold, compute_nis  # noqa: E402


def test_transition_matrix_identity_at_zero_dt():
    F = transition_matrix(0.0)
    assert np.allclose(F, np.eye(6))


def test_transition_matrix_moves_position_by_velocity_times_dt():
    F = transition_matrix(2.0)
    x = np.array([0.0, 0.0, 0.0, 10.0, -5.0, 1.0])
    x_next = F @ x
    assert np.allclose(x_next[:3], [20.0, -10.0, 2.0])
    assert np.allclose(x_next[3:], x[3:])  # velocity unchanged by pure transition


def test_process_noise_matrix_symmetric_and_psd():
    Q = process_noise_matrix(dt=1.0, sigma_a=2.0)
    assert np.allclose(Q, Q.T)
    eigvals = np.linalg.eigvalsh(Q)
    assert np.all(eigvals >= -1e-9)


def test_process_noise_zero_dt_is_zero():
    Q = process_noise_matrix(dt=0.0, sigma_a=5.0)
    assert np.allclose(Q, np.zeros((6, 6)))


def test_process_noise_grows_with_dt():
    Q_small = process_noise_matrix(dt=1.0, sigma_a=2.0)
    Q_large = process_noise_matrix(dt=5.0, sigma_a=2.0)
    assert Q_large[0, 0] > Q_small[0, 0]


def _make_filter(sigma_a=1.0, sigma_pos=20.0):
    x0 = np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0])  # moving east at 100 m/s
    P0 = np.diag([sigma_pos**2, sigma_pos**2, sigma_pos**2, 50.0**2, 50.0**2, 50.0**2])
    R = np.diag([sigma_pos**2, sigma_pos**2, (1.5 * sigma_pos) ** 2])
    return KalmanFilterCV(x0=x0, P0=P0, R=R, sigma_a=sigma_a)


def test_predict_moves_mean_state_by_velocity():
    kf = _make_filter()
    kf.predict(dt=10.0)
    assert np.isclose(kf.x[0], 1000.0)  # 100 m/s * 10s
    assert np.isclose(kf.x[1], 0.0)
    assert np.isclose(kf.x[2], 0.0)


def test_predict_increases_covariance():
    kf = _make_filter()
    trace_before = np.trace(kf.P)
    kf.predict(dt=10.0)
    trace_after = np.trace(kf.P)
    assert trace_after > trace_before


def test_update_decreases_covariance_when_measurement_agrees_with_prediction():
    kf = _make_filter()
    kf.predict(dt=1.0)
    trace_before_update = np.trace(kf.P)
    z = kf.x[:3].copy()  # perfect measurement matching prediction
    kf.update(z)
    assert np.trace(kf.P) < trace_before_update


def test_update_pulls_state_toward_measurement():
    kf = _make_filter(sigma_pos=20.0)
    kf.predict(dt=1.0)
    predicted_x = kf.x[0]
    z = np.array([predicted_x + 500.0, 0.0, 0.0])  # far-off measurement
    kf.update(z)
    # Updated estimate should move toward the measurement, not stay at the prediction.
    assert kf.x[0] > predicted_x
    assert kf.x[0] < z[0]


def test_update_returns_innovation_relative_to_prediction_not_posterior():
    kf = _make_filter()
    kf.predict(dt=1.0)
    predicted_pos = kf.x[:3].copy()
    z = predicted_pos + np.array([10.0, 0.0, 0.0])
    result = kf.update(z)
    assert np.allclose(result.innovation, [10.0, 0.0, 0.0])


def test_filter_self_consistency_mean_nis_near_measurement_dof():
    """Simulates a straight, constant-velocity track with measurement noise
    matching R and process noise matching sigma_a, and checks the resulting
    NIS values behave like a well-tuned filter should: mean NIS close to the
    measurement dimension (3), and the chi-square(alpha=0.01) flag rate close
    to 1%. This validates predict+update+NIS together against a known-truth
    simulation, not just live traffic.
    """
    rng = np.random.default_rng(42)
    sigma_a = 1.0
    sigma_pos = 20.0
    dt = 1.0
    n_steps = 3000

    R = np.diag([sigma_pos**2, sigma_pos**2, sigma_pos**2])
    true_pos = np.zeros(3)
    true_vel = np.array([100.0, 0.0, 0.0])

    x0 = np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0])
    P0 = np.diag([sigma_pos**2] * 3 + [50.0**2] * 3)
    kf = KalmanFilterCV(x0=x0, P0=P0, R=R, sigma_a=sigma_a)

    nis_values = []
    anomalous = 0
    for _ in range(n_steps):
        true_vel = true_vel + rng.normal(0.0, sigma_a * dt, size=3)
        true_pos = true_pos + true_vel * dt
        z = true_pos + rng.normal(0.0, sigma_pos, size=3)

        kf.predict(dt)
        result = kf.update(z)
        nis = compute_nis(result.innovation, result.innovation_cov)
        nis_values.append(nis)
        if nis > chi_square_threshold(dof=3, alpha=0.01):
            anomalous += 1

    mean_nis = float(np.mean(nis_values))
    # For a correctly tuned filter, E[NIS] == measurement dof (3). Allow a wide
    # tolerance since this is a stochastic simulation, not an exact identity.
    assert 2.0 < mean_nis < 5.0

    anomalous_rate = anomalous / n_steps
    assert anomalous_rate < 0.05  # well under, e.g., a 5x blowup of the 1% alpha
