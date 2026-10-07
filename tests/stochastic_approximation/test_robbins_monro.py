import numpy as np

from ml_from_scratch.stochastic_approximation.robbins_monro import (
    robbins_monro,
)


def linear_function(value):
    """Return the exact value of g(w) = w - 2."""
    return value - 2


def decreasing_step_size(iteration):
    """Return the step size alpha_k = 1 / (k + 1)."""
    return 1 / (iteration + 1)


def test_robbins_monro_matches_manual_updates():
    estimate, estimate_history = robbins_monro(
        initial_value=0.0,
        noisy_function=linear_function,
        step_size=decreasing_step_size,
        n_iterations=3,
    )

    expected_history = np.array([
        0.0,
        1.0,
        4 / 3,
        1.5,
    ])

    np.testing.assert_allclose(
        estimate_history,
        expected_history,
    )
    np.testing.assert_allclose(
        estimate,
        1.5,
    )


def test_robbins_monro_approaches_root():
    estimate, _ = robbins_monro(
        initial_value=0.0,
        noisy_function=linear_function,
        step_size=decreasing_step_size,
        n_iterations=1000,
    )

    np.testing.assert_allclose(
        estimate,
        2.0,
        atol=0.01,
    )


def test_robbins_monro_approaches_root_with_noise():
    random_generator = np.random.default_rng(0)

    def noisy_linear_function(value):
        noise = random_generator.normal(
            loc=0.0,
            scale=0.5,
        )
        return value - 2 + noise

    estimate, estimate_history = robbins_monro(
        initial_value=0.0,
        noisy_function=noisy_linear_function,
        step_size=decreasing_step_size,
        n_iterations=5000,
    )

    np.testing.assert_allclose(
        estimate,
        2.0,
        atol=0.03,
    )
    assert len(estimate_history) == 5001