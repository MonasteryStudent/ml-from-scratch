import numpy as np

from ml_from_scratch.stochastic_approximation.mean_estimation import (
    incremental_mean,
)


def test_incremental_mean_matches_manual_updates():
    samples = [2.0, 4.0, 3.0]

    estimate, estimate_history = incremental_mean(samples)

    expected_history = np.array([
        0.0,
        2.0,
        3.0,
        3.0,
    ])

    np.testing.assert_allclose(
        estimate_history,
        expected_history,
    )
    assert estimate == 3.0


def test_incremental_mean_matches_numpy_mean():
    samples = np.array([3.0, -1.0, 8.0, 2.0])

    estimate, _ = incremental_mean(samples)

    np.testing.assert_allclose(
        estimate,
        np.mean(samples),
    )