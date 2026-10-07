import numpy as np


def incremental_mean(samples):
    """Estimate the sample mean using incremental updates."""
    estimate = 0.0
    estimate_history = [estimate]

    for sample_count, sample in enumerate(samples, start=1):
        step_size = 1 / sample_count

        estimate = estimate - step_size * (estimate - sample)
        estimate_history.append(estimate)

    return estimate, np.asarray(estimate_history)