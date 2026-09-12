import numpy as np

from ml_from_scratch.neural_network import (
    backward_propagation,
    compute_cost,
    fit,
    forward_propagation,
    gradient_descent,
    initialize_parameters,
    predict,
    predict_proba,
    update_parameters,
)


X = np.array([
    [1.0, 2.0],
    [-1.0, 1.0],
])

y = np.array([1, 0])

parameters = {
    "W1": np.array([
        [0.5, -0.4],
        [0.3, 0.2],
    ]),
    "b1": np.array([[0.1, -0.1]]),
    "W2": np.array([
        [0.7],
        [-0.6],
    ]),
    "b2": np.array([[0.05]]),
}


def test_initialize_parameters():
    result = initialize_parameters(n_features=2, n_hidden=4, seed=42)

    assert result["W1"].shape == (2, 4)
    assert result["b1"].shape == (1, 4)
    assert result["W2"].shape == (4, 1)
    assert result["b2"].shape == (1, 1)
    assert not np.allclose(result["W1"], 0.0)
    assert not np.allclose(result["W2"], 0.0)
    assert not np.allclose(result["W1"][:, 0], result["W1"][:, 1])
    assert np.array_equal(result["b1"], np.zeros((1, 4)))
    assert np.array_equal(result["b2"], np.zeros((1, 1)))


def test_initialize_parameters_is_reproducible():
    first = initialize_parameters(n_features=2, n_hidden=4, seed=42)
    second = initialize_parameters(n_features=2, n_hidden=4, seed=42)

    for name in first:
        assert np.array_equal(first[name], second[name])


def test_forward_propagation():
    y_hat, cache = forward_propagation(X, parameters)

    expected_Z1 = np.array([
        [1.2, -0.1],
        [-0.1, 0.5],
    ])
    expected_A1 = np.array([
        [1.2, 0.0],
        [0.0, 0.5],
    ])
    expected_Z2 = np.array([
        [0.89],
        [-0.25],
    ])
    expected_A2 = np.array([
        [0.7088901726],
        [0.4378234991],
    ])

    assert np.allclose(cache["Z1"], expected_Z1)
    assert np.allclose(cache["A1"], expected_A1)
    assert np.allclose(cache["Z2"], expected_Z2)
    assert np.allclose(cache["A2"], expected_A2)
    assert np.allclose(y_hat, expected_A2)


def test_compute_cost_accepts_one_and_two_dimensional_labels():
    y_hat = np.array([
        [0.8],
        [0.3],
    ])
    expected = -0.5 * (np.log(0.8) + np.log(0.7))

    assert np.isclose(compute_cost(y, y_hat), expected)
    assert np.isclose(compute_cost(y.reshape(-1, 1), y_hat), expected)


def test_compute_cost_is_finite_for_extreme_predictions():
    y_hat = np.array([
        [1.0],
        [0.0],
    ])

    assert np.isfinite(compute_cost(y, y_hat))


def test_backward_propagation():
    _, cache = forward_propagation(X, parameters)
    gradients = backward_propagation(X, y, parameters, cache)

    expected_dW1 = np.array([
        [-0.1018884396, 0.1313470497],
        [-0.2037768792, -0.1313470497],
    ])
    expected_db1 = np.array([
        [-0.1018884396, -0.1313470497],
    ])
    expected_dW2 = np.array([
        [-0.1746658964],
        [0.1094558748],
    ])
    expected_db2 = np.array([
        [0.0733568359],
    ])

    assert np.allclose(gradients["dW1"], expected_dW1)
    assert np.allclose(gradients["db1"], expected_db1)
    assert np.allclose(gradients["dW2"], expected_dW2)
    assert np.allclose(gradients["db2"], expected_db2)

    for name in parameters:
        assert gradients[f"d{name}"].shape == parameters[name].shape


def test_update_parameters():
    _, cache = forward_propagation(X, parameters)
    gradients = backward_propagation(X, y, parameters, cache)
    original_parameters = {
        name: value.copy()
        for name, value in parameters.items()
    }

    result = update_parameters(parameters, gradients, alpha=0.1)

    for name in parameters:
        expected = original_parameters[name] - 0.1 * gradients[f"d{name}"]
        assert np.allclose(result[name], expected)
        assert np.array_equal(parameters[name], original_parameters[name])


def test_gradient_descent_reduces_cost():
    initial_y_hat, _ = forward_propagation(X, parameters)
    initial_cost = compute_cost(y, initial_y_hat)

    trained_parameters, cost_history = gradient_descent(
        X,
        y,
        parameters,
        alpha=0.1,
        iterations=10,
    )

    final_y_hat, _ = forward_propagation(X, trained_parameters)
    final_cost = compute_cost(y, final_y_hat)

    assert len(cost_history) == 10
    assert cost_history[-1] < cost_history[0]
    assert final_cost < initial_cost


def test_fit_is_reproducible():
    first_parameters, first_history = fit(
        X,
        y,
        n_hidden=2,
        alpha=0.1,
        iterations=10,
        seed=42,
    )
    second_parameters, second_history = fit(
        X,
        y,
        n_hidden=2,
        alpha=0.1,
        iterations=10,
        seed=42,
    )

    for name in first_parameters:
        assert np.array_equal(first_parameters[name], second_parameters[name])
    assert np.array_equal(first_history, second_history)


def test_predict_proba():
    expected = np.array([0.7088901726, 0.4378234991])

    result = predict_proba(X, parameters)

    assert np.allclose(result, expected)
    assert result.shape == (2,)


def test_predict():
    assert np.array_equal(predict(X, parameters), np.array([1, 0]))
    assert np.array_equal(
        predict(X, parameters, threshold=0.8),
        np.array([0, 0]),
    )