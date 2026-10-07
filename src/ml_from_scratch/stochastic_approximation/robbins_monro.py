import numpy as np


def robbins_monro(
    initial_value,
    noisy_function,
    step_size,
    n_iterations,
):
    """Estimate a root from noisy function observations."""
    estimate = float(initial_value)
    estimate_history = [estimate]

    for iteration in range(1, n_iterations + 1):
        observation = noisy_function(estimate)
        current_step_size = step_size(iteration)

        estimate = (
            estimate
            - current_step_size * observation
        )
        estimate_history.append(estimate)

    return estimate, np.asarray(estimate_history)