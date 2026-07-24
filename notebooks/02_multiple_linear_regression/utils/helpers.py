import numpy as np
import pandas as pd

from ml_from_scratch.multiple_linear_regression import (
    compute_cost,
    compute_gradient
)


def generate_multiple_linear_regression_dataset():
    """Generate a synthetic dataset for multiple linear regression."""

    rng = np.random.default_rng(42)

    m = 40
    progress = np.linspace(0, 1, m)

    study_hours = 1 + 19 * progress + rng.normal(0, 1.2, m)
    attendance = 55 + 40 * progress + rng.normal(0, 7, m)
    practice_tests = 10 * progress + rng.normal(0, 1.5, m)
    sleep = 6.0 + 2.0 * progress + rng.normal(0, 0.5, m)

    study_hours = np.clip(study_hours, 1, 20)
    attendance = np.clip(attendance, 55, 100)
    practice_tests = np.clip(practice_tests, 0, 10)
    sleep = np.clip(sleep, 5.5, 8.5)

    X = np.column_stack(
        (
            study_hours,
            attendance,
            practice_tests,
            sleep,
        )
    )

    true_w = np.array([1.5, 0.21, 1.2, 1.8])
    true_b = 17.0

    noise = rng.normal(0, 2, m)
    y = X @ true_w + true_b + noise

    return X, y


def create_dataset_table(X, y):
    """
    Create a formatted table of the synthetic dataset.
    """
    
    table = pd.DataFrame(
        {
            "Training Example": np.arange(1, len(X) + 1),
            "Study Hours": X[:, 0].astype(int),
            "Attendance (%)": X[:, 1],
            "Practice Tests": X[:, 2].astype(int),
            "Sleep (h)": X[:, 3],
            "Exam Score": y
        }
    )

    # Shuffle the rows to resemble a more realistic dataset.
    table = table.sample(frac=1, random_state=42).reset_index(drop=True)
    # Renumber the training examples after shuffling.
    table["Training Example"] = np.arange(1, len(table) + 1)

    return table.style.hide(axis="index").format(precision=1)


def standardize_features(X):
    """Standardize each feature using z-score standardization."""

    means = np.mean(X, axis=0)
    standard_deviations = np.std(X, axis=0)
    return (X - means) / standard_deviations


def compute_cost_history(X, y, w_init, b_init, alpha, iterations):
    """Compute the initial cost and the cost after each gradient descent update."""

    w = w_init.copy()
    b = b_init
    cost_history = [compute_cost(X, y, w, b)]

    for _ in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost_history.append(compute_cost(X, y, w, b))
    
    return cost_history


def generate_polynomial_regression_dataset():
    """Generate a synthetic dataset with a quadratic relationship."""

    rng = np.random.default_rng(42)

    m = 40

    sleep = np.linspace(5.5, 12.0, m)

    noise = rng.normal(0, 2, m)

    exam_scores = -3 * (sleep - 8) ** 2 + 92 + noise

    return sleep, exam_scores