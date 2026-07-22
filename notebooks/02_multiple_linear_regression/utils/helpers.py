import numpy as np
import pandas as pd

from ml_from_scratch.multiple_linear_regression import (
    compute_cost,
    compute_gradient
)


def generate_multiple_linear_regression_dataset():
    """
    Generate a synthetic dataset for multiple linear regression.

    Features
    --------
    X[:, 0] : Study hours
    X[:, 1] : Lecture attendance (%)
    X[:, 2] : Completed practice tests
    X[:, 3] : Hours of sleep before the exam

    Returns
    -------
    X : ndarray of shape (20, 4)
        Feature matrix.

    y : ndarray of shape (20,)
        Target vector containing the exam scores.
    """

    np.random.seed(42)

    m = 20
    study_hours = np.linspace(1, 20, m)
    attendance = np.linspace(55, 100, m) + np.random.normal(0, 3, m)
    practice_tests = np.linspace(0, 10, m) + np.random.normal(0, 0.4, m)
    sleep = np.linspace(5.5, 8.5, m) + np.random.normal(0, 0.2, m)

    X = np.column_stack((study_hours, attendance, practice_tests, sleep))

    # Underlying linear model used to generate the synthetic dataset.
    # Note that the magnitudes of the weights cannot be compared directly
    # because the features have different scales.
    true_w = np.array([2.5, 0.35, 2.0, 3.0])
    true_b = 0.0

    noise = np.random.normal(0, 2, m)
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