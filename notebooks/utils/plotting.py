import numpy as np
import matplotlib.pyplot as plt
from ml_from_scratch.linear_regression import (
    predict, 
    compute_cost,
    uni_linear_regression
)
from utils.helpers import compute_cost_surface


def plot_dataset(X, y):
    plt.scatter(X, y)
    plt.xlabel("Study Hours")
    plt.ylabel("Exam Score")
    plt.title("Training Data")

def plot_errors(X, y, w, b):
    y_hat = predict(X, w, b)
    plt.scatter(X, y, zorder=5, label="Training data")
    plt.plot(X, y_hat, zorder=5, label="Model")

    # Draw prediction erros
    for x_i, y_i, y_hat_i in zip(X, y, y_hat):
        plt.plot(
            [x_i, x_i],
            [y_i, y_hat_i],
            "r--"
        )

    # Dummy line for the legend
    plt.plot([], [], "r--", label="Prediction error")

    plt.title("Prediction Errors")

def plot_simple_cost_function(w_values, costs):
    plt.plot(w_values, costs)
    plt.xlabel("w")
    plt.ylabel("J(w)")
    plt.title("Simple Cost Function")

def plot_cost_point(X, y, w, color, s, label=None):
    cost = compute_cost(X, y, w, 0)

    plt.scatter(
        w,
        cost,
        color=color,
        s=s,
        zorder=5,
        label=label
    )

def _plot_model(ax, X, y, model):
    y_pred = model["w"] * X + model["b"]

    ax.scatter(X, y, s=20)
    ax.plot(X, y_pred, color=model["color"])

    for xi, yi, yp in zip(X, y, y_pred):
        ax.plot([xi, xi], [yi, yp], linestyle="--", color="red", linewidth=1)

    ax.set_title(model["title"])
    ax.set_xlabel("Study Hours")
    ax.text(
        0.05,
        0.95,
        f"$w = {model['w']:.2f}$\n$b = {model['b']:.2f}$",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

def plot_model_comparison(X, y):

    params = [
        ("Bad Model", 6.0, 40.0, "red"),
        ("Simple Model", 5.13, 0.0, "blue"),
        ("Good Model", 3.0, 28.74, "green"),
    ]
    models = []

    for name, w, b, color in params:
        cost = compute_cost(X, y, w, b)
        models.append({
            "title": f"{name} J = {cost:.2f}",
            "w": w,
            "b": b,
            "color": color,
        })

    _, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    for ax, model in zip(axes, models):
        _plot_model(ax, X, y, model)

    axes[0].set_ylabel("Exam Score")
    plt.tight_layout()

def plot_tangent(w_tangent, j_tangent):
    plt.plot(w_tangent, j_tangent, label="Tangent")

def plot_cost_history(cost_history):
    plt.plot(cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Gradient Descent Convergence")

def plot_3D_surface(X, y):
    w_values = np.linspace(2, 6, 100)
    b_values = np.linspace(10, 30, 100)

    W, B = np.meshgrid(w_values, b_values)

    J = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            J[i, j] = compute_cost(
                X,
                y,
                W[i, j],
                B[i, j],
            )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        W,
        B,
        J,
        cmap="viridis",
        edgecolor="none",
    )

    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_zlabel("J(w, b)")
    ax.set_title("Cost Function Surface")

    ax.view_init(elev=30, azim=-70)

    fig.colorbar(surface, shrink=0.7)

    plt.tight_layout()

def plot_contour(X, y):

    w_opt, b_opt, _ = uni_linear_regression(X, y, 0.01, 5000)

    w_range = 1.2
    b_range = 6.0

    W, B, J = compute_cost_surface(X, y, w_opt, b_opt, w_range, b_range)

    plt.figure(figsize=(8, 6))

    contour = plt.contour(W, B, J, levels=20)

    plt.clabel(contour, inline=True, fontsize=8)

    plt.xlabel("w")
    plt.ylabel("b")
    plt.title("Contour Plot of the Cost Function")

def plot_gradient_descent_path(X, y, path_w, path_b, arrow_every=150):
    w_opt = path_w[-1]
    b_opt = path_b[-1]

    w_range = 4.0
    b_range = 30.0

    W, B, J = compute_cost_surface(X, y, w_opt, b_opt, w_range, b_range)

    _, ax = plt.subplots(figsize=(8, 6))

    ax.contour(W, B, J, levels=20, zorder=1)

    for i in range(0, len(path_w) - arrow_every, arrow_every):
        ax.annotate(
            "",
            xy=(path_w[i + arrow_every], path_b[i + arrow_every]),
            xytext=(path_w[i], path_b[i]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="red",
                linewidth=2,
                mutation_scale=15,
            ),
            zorder=5,
        )

    ax.plot([], [], color="red", linewidth=2, label="Gradient Descent steps")

    cost_min = compute_cost(X, y, w_opt, b_opt)

    ax.scatter(
        w_opt,
        b_opt,
        color="purple",
        s=80,
        zorder=6,
        label=f"Minimum (J = {cost_min:.2f})",
    )

    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title("Gradient Descent on the Cost Function")
    ax.legend()

    plt.tight_layout()
