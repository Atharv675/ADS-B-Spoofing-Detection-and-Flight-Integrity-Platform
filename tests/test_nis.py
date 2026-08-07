import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.tracking.nis import chi_square_threshold, compute_nis, is_anomalous  # noqa: E402


def test_compute_nis_identity_covariance():
    innovation = np.array([3.0, 4.0, 0.0])
    S = np.eye(3)
    # NIS = y^T S^-1 y = |y|^2 for identity S
    assert math.isclose(compute_nis(innovation, S), 25.0, rel_tol=1e-9)


def test_compute_nis_diagonal_covariance():
    innovation = np.array([2.0, 0.0, 0.0])
    S = np.diag([4.0, 1.0, 1.0])
    # NIS = 2^2 / 4 = 1.0
    assert math.isclose(compute_nis(innovation, S), 1.0, rel_tol=1e-9)


def test_compute_nis_zero_innovation_is_zero():
    S = np.diag([4.0, 9.0, 1.0])
    assert compute_nis(np.zeros(3), S) == 0.0


def test_chi_square_threshold_known_values_df3():
    # Standard textbook values for chi-square with 3 degrees of freedom.
    assert math.isclose(chi_square_threshold(dof=3, alpha=0.05), 7.8147, abs_tol=1e-3)
    assert math.isclose(chi_square_threshold(dof=3, alpha=0.01), 11.3449, abs_tol=1e-3)


def test_chi_square_threshold_tighter_alpha_is_higher_threshold():
    assert chi_square_threshold(dof=3, alpha=0.001) > chi_square_threshold(dof=3, alpha=0.01)


def test_is_anomalous_boundary():
    threshold = chi_square_threshold(dof=3, alpha=0.01)
    assert is_anomalous(threshold + 0.01, dof=3, alpha=0.01) is True
    assert is_anomalous(threshold - 0.01, dof=3, alpha=0.01) is False
