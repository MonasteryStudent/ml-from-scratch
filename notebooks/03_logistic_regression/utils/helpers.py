import numpy as np


def generate_logistic_regression_dataset():
    """Generate a synthetic dataset for binary classification."""
    rng = np.random.default_rng(42)

    m = 100

    study_hours = rng.uniform(1, 20, m)
    practice_tests = rng.integers(0, 11, m)

    X = np.column_stack(
        (
            study_hours,
            practice_tests,
        )
    )

    # Linear rule used to separate passing and failing students.
    true_w = np.array([0.5, 0.9])
    true_b = -8.0

    z = X @ true_w + true_b

    y = (z >= 0).astype(int)

    return X, y


def train_test_split(X, y, train_ratio):
    """Randomly split a dataset into training and test sets."""
    rng = np.random.default_rng(42)

    indices = rng.permutation(len(X))
    split_index = int(train_ratio * len(X))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test