import numpy as np


def generate_logistic_regression_dataset():
    """Generate a synthetic dataset for binary classification."""

    rng = np.random.default_rng(42)

    m = 40

    study_hours = rng.uniform(1, 20, m)
    practice_tests = rng.uniform(0, 10, m)

    X = np.column_stack(
        (
            study_hours,
            practice_tests,
        )
    )

    # Linear rule used to separate passing and failing students.
    true_w = np.array([0.5, 0.9])
    true_b = -8.0

    noise = rng.normal(0, 1.0, m)
    z = X @ true_w + true_b + noise

    y = (z >= 0).astype(int)

    return X, y