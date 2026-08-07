"""Normalized Innovation Squared (NIS) and the chi-square consistency test.

This is the classical statistical baseline detector: a well-tuned Kalman
filter's NIS should follow a chi-square distribution with degrees of freedom
equal to the *measurement* dimension (3, for our x/y/z position measurement)
-- not the state dimension (6). Using the state dimension here is a common
mistake and would make the test systematically wrong (too permissive), so it's
called out explicitly rather than left implicit.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

MEASUREMENT_DOF = 3


def compute_nis(innovation: np.ndarray, innovation_cov: np.ndarray) -> float:
    S_inv = np.linalg.inv(innovation_cov)
    nis = float(innovation.T @ S_inv @ innovation)
    return nis


def chi_square_threshold(dof: int = MEASUREMENT_DOF, alpha: float = 0.01) -> float:
    """One-sided upper-tail threshold: under H0 (filter well-tuned, no anomaly),
    P(NIS > threshold) = alpha."""
    return float(chi2.ppf(1.0 - alpha, df=dof))


def is_anomalous(nis: float, dof: int = MEASUREMENT_DOF, alpha: float = 0.01) -> bool:
    return nis > chi_square_threshold(dof=dof, alpha=alpha)
