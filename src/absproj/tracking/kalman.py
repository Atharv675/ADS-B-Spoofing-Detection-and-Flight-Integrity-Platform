"""Constant-velocity Kalman filter over a 6D state [x, y, z, vx, vy, vz] in a
local ENU frame (meters, m/s), with a 3D position measurement [x, y, z].

This is the "temporal verification" component: it predicts where a track
should be next based on its own recent motion, and the innovation (predicted
vs. observed position) is the signal the NIS test and the ML layer both
consume. It deliberately does not use raw ADS-B fields beyond position -- the
whole point is a physics-based prediction independent of what the aircraft
*says* its velocity is.

Process noise uses the standard discrete white-noise-acceleration (DWNA)
model: each axis independently accumulates unmodeled acceleration with std
dev `sigma_a` (m/s^2), which is where category-based dynamics constraints
enter (see tracking/categories.py) -- a light aircraft, an airliner, and a
high-performance aircraft get different sigma_a because their plausible
turn/climb accelerations differ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE_DIM = 6
MEAS_DIM = 3

H = np.zeros((MEAS_DIM, STATE_DIM))
H[0, 0] = 1.0
H[1, 1] = 1.0
H[2, 2] = 1.0


def transition_matrix(dt: float) -> np.ndarray:
    F = np.eye(STATE_DIM)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F


def process_noise_matrix(dt: float, sigma_a: float) -> np.ndarray:
    """Block-diagonal DWNA process noise across the three (independent) axes."""
    q = sigma_a ** 2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    block = np.array([
        [dt4 / 4.0, dt3 / 2.0],
        [dt3 / 2.0, dt2],
    ]) * q

    Q = np.zeros((STATE_DIM, STATE_DIM))
    for axis in range(3):
        pos_idx, vel_idx = axis, axis + 3
        Q[pos_idx, pos_idx] = block[0, 0]
        Q[pos_idx, vel_idx] = block[0, 1]
        Q[vel_idx, pos_idx] = block[1, 0]
        Q[vel_idx, vel_idx] = block[1, 1]
    return Q


@dataclass
class UpdateResult:
    innovation: np.ndarray       # shape (3,) -- measured minus predicted position
    innovation_cov: np.ndarray   # shape (3,3) -- S = H P H^T + R
    predicted_state: np.ndarray  # shape (6,) -- state before this update (post-predict)


class KalmanFilterCV:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, R: np.ndarray, sigma_a: float):
        assert x0.shape == (STATE_DIM,)
        assert P0.shape == (STATE_DIM, STATE_DIM)
        assert R.shape == (MEAS_DIM, MEAS_DIM)
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()
        self.R = R.astype(float).copy()
        self.sigma_a = sigma_a

    def predict(self, dt: float) -> None:
        F = transition_matrix(dt)
        Q = process_noise_matrix(dt, self.sigma_a)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray) -> UpdateResult:
        """Applies a position measurement. Must be called after predict().
        Returns the innovation and innovation covariance *before* incorporating
        the measurement -- that's the signal that feeds NIS/ML, not the
        post-update state.
        """
        assert z.shape == (MEAS_DIM,)
        predicted_state = self.x.copy()

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I_KH = np.eye(STATE_DIM) - K @ H
        # Joseph form for numerical stability (keeps P symmetric PSD even with
        # imperfect K, unlike the textbook-simple (I-KH)P).
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return UpdateResult(innovation=y, innovation_cov=S, predicted_state=predicted_state)
