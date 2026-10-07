import numpy as np


def stochastic_gradient_descent(
    initial_parameters,
    stochastic_gradient,
    sample_generator,
    step_size,
    n_iterations,
):
    """Minimize an objective using stochastic gradient updates."""
    parameters = np.asarray(
        initial_parameters,
        dtype=float,
    ).copy()

    parameter_history = [parameters.copy()]

    for iteration in range(1, n_iterations + 1):
        sample = sample_generator()
        gradient = stochastic_gradient(
            parameters,
            sample,
        )
        current_step_size = step_size(iteration)

        parameters = (
            parameters
            - current_step_size * gradient
        )
        parameter_history.append(parameters.copy())

    return parameters, np.asarray(parameter_history)