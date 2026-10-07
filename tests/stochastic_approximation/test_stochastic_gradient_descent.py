import numpy as np

from ml_from_scratch.stochastic_approximation.stochastic_gradient_descent import (
    stochastic_gradient_descent,
)


def squared_error_gradient(parameters, sample):
    """Return the gradient of 1/2 * (w - x)^2."""
    return parameters - sample


def incremental_step_size(iteration):
    """Return the step size alpha_k = 1 / k."""
    return 1 / iteration


def test_sgd_matches_incremental_mean_updates():
    samples = iter([2.0, 4.0, 3.0])

    def sample_generator():
        return next(samples)

    parameters, parameter_history = (
        stochastic_gradient_descent(
            initial_parameters=0.0,
            stochastic_gradient=squared_error_gradient,
            sample_generator=sample_generator,
            step_size=incremental_step_size,
            n_iterations=3,
        )
    )

    expected_history = np.array([
        0.0,
        2.0,
        3.0,
        3.0,
    ])

    np.testing.assert_allclose(
        parameter_history,
        expected_history,
    )
    np.testing.assert_allclose(
        parameters,
        3.0,
    )


def test_sgd_estimates_expected_sample_value():
    random_generator = np.random.default_rng(0)

    samples = random_generator.normal(
        loc=2.0,
        scale=1.0,
        size=5000,
    )
    sample_iterator = iter(samples)

    def sample_generator():
        return next(sample_iterator)

    parameters, parameter_history = (
        stochastic_gradient_descent(
            initial_parameters=0.0,
            stochastic_gradient=squared_error_gradient,
            sample_generator=sample_generator,
            step_size=incremental_step_size,
            n_iterations=len(samples),
        )
    )

    np.testing.assert_allclose(
        parameters,
        np.mean(samples),
    )
    np.testing.assert_allclose(
        parameters,
        2.0,
        atol=0.05,
    )
    assert len(parameter_history) == len(samples) + 1