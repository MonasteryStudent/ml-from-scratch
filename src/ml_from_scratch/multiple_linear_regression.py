import numpy as np

def predict(X, w, b):
    """Compute predictions for multiple linear regression."""
    return X @ w + b

def compute_cost(X, y, w, b):
    """Compute the mean squared error cost."""
    m = X.shape[0]
    error = predict(X, w, b) - y
    return np.sum(error**2) / (2 * m)

def compute_gradient(X, y, w, b):
    """Compute the gradient of the cost function."""
    m = X.shape[0]
    error = predict(X, w, b) - y
    dj_dw = X.T @ error / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, iterations):
    """Optimize the model parameters using gradient descent."""
    for _ in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

def fit(X, y, alpha, iterations):
    """Fit a multiple linear regression model."""
    w = np.zeros(X.shape[1])
    b = 0.0
    return gradient_descent(X, y, w, b, alpha, iterations)

def main():

    # X = np.array([
    #     [2, 55, 0, 5.5],
    #     [4, 65, 1, 6.0],
    #     [5, 70, 2, 6.5],
    #     [7, 80, 3, 7.0],
    #     [8, 82, 4, 7.5],
    #     [10, 88, 5, 8.0],
    #     [12, 92, 6, 7.5],
    #     [14, 95, 7, 8.0],
    #     [16, 97, 8, 7.0],
    #     [18, 99, 9, 7.5],
    # ])

    # y = np.array([
    #     45,
    #     55,
    #     62,
    #     74,
    #     80,
    #     89,
    #     94,
    #     98,
    #     100,
    #     100,
    # ])

    # w = np.array([2.0, 2.0, 2.0, 2.0])
    # b = 1.0
    # m = X.shape[0]
    # error = predict(X, w, b) - y
    # print(error)
    # # print(error[..., None])
    # # print(error@X[:0])
    # print(X[..., 0]@error)
    # print(error@X / m)
    # # dj_dw = np.sum(error[...:None]@error) / m
    # print(compute_gradient(X, y, w, b))

    # X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    # y_train = np.array([460, 232, 178])

    # # print(X_train[:, 0])
    # # print(X_train[..., 0])

    # # b_init = 785.1811367994083
    # # w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
    # # print(compute_gradient(X_train, y_train, w_init, b_init))

    # w = np.zeros(X_train.shape[1])
    # b = 0.0


    # alpha = 5.0e-7
    # iterations = 1000
    # w_opt, b_opt = gradient_descent(X_train, y_train, w, b, alpha, iterations)

    # # w, b = linear_regression(X_train, y_train, alpha, iterations)
    # print(f"b, w found by gradient descent: {w_opt}, {b_opt}")
    # print(w)


    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
    ])
    y = np.array([8, 7, 16, 15])

    w = np.zeros(X.shape[1])
    b = 0.0

    initial_cost = compute_cost(X, y, w, b)

    w_final, b_final = gradient_descent(
        X,
        y,
        w,
        b,
        alpha=1e-2,
        iterations=1000,
    )

    final_cost = compute_cost(X, y, w_final, b_final)

    print(initial_cost)
    print(final_cost)


if __name__ == "__main__":
    main()