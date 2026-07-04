import numpy as np
import matplotlib.pyplot as plt
from ml_from_scratch.linear_regression import (
    predict, 
    compute_cost,
    uni_linear_regression,
)
from utils.helpers import (
    compute_cost_surface_vectorized,
)

def plot_dataset(x, y):
    """Plot the synthetic training dataset."""
    plt.scatter(x, y)
    plt.xlabel("Study Hours")
    plt.ylabel("Exam Score")
    plt.title("Training Data")

def plot_errors(x, y, w, b):
    """Plot prediction errors for a linear regression model."""
    y_hat = predict(x, w, b)
    plt.scatter(x, y, zorder=5, label="Training data")
    plt.plot(x, y_hat, zorder=5, label="Model")
    # Draw prediction errors
    for x_i, y_i, y_hat_i in zip(x, y, y_hat):
        plt.plot([x_i, x_i], [y_i, y_hat_i], "r--")
    # Dummy line for the legend
    plt.plot([], [], "r--", label="Prediction error")
    plt.title("Prediction Errors")

def plot_simple_cost_function(w_values, costs):
    """Plot the simplified cost function J(w)."""
    plt.plot(w_values, costs)
    plt.xlabel("w")
    plt.ylabel("J(w)")
    plt.title("Simple Cost Function")

def plot_cost_point(x, y, w, color, s, label=None):
    """Mark a point on the simplified cost function J(w)."""
    cost = compute_cost(x, y, w, 0)
    plt.scatter(w, cost, color=color, s=s, zorder=5, label=label)

def _plot_model(ax, x, y, model):
    """Plot a single linear regression model including prediction errors."""
    y_pred = model["w"] * x + model["b"]
    ax.scatter(x, y, s=20)
    ax.plot(x, y_pred, color=model["color"])
    for xi, yi, yp in zip(x, y, y_pred):
        ax.plot([xi, xi], [yi, yp], linestyle="--", color="red", linewidth=1)
    ax.set_title(model["title"])
    ax.set_xlabel("Study Hours")
    ax.text(
        0.05, 0.95,
        f"$w = {model['w']:.2f}$\n$b = {model['b']:.2f}$",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

def plot_model_comparison(x, y, model_params):
    """Compare multiple linear regression models side by side."""
    models = []
    for name, w, b, color in model_params:
        cost = compute_cost(x, y, w, b)
        models.append({
            "title": f"{name} J = {cost:.2f}",
            "w": w,
            "b": b,
            "color": color
        })
    _, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, model in zip(axes, models):
        _plot_model(ax, x, y, model)
    axes[0].set_ylabel("Exam Score")
    plt.tight_layout()

def plot_tangent(w_tangent, j_tangent):
    """Plot a tangent line."""
    plt.plot(w_tangent, j_tangent, label="Tangent")

def plot_cost_history(cost_history):
    """Plot the cost history of Gradient Descent."""
    plt.plot(cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Gradient Descent Convergence")

def plot_3D_surface(x, y):
    """Plot the cost function as a 3D surface."""
    w_values = np.linspace(2, 6, 100)
    b_values = np.linspace(10, 30, 100)

    W, B, J = compute_cost_surface_vectorized(x, y, w_values, b_values)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(W, B, J, cmap="viridis", edgecolor="none")

    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_zlabel("J(w, b)")
    ax.set_title("Cost Function Surface")

    ax.view_init(elev=30, azim=-70)

    fig.colorbar(surface, shrink=0.7)

    plt.tight_layout()

def plot_contour(x, y):
    """Plot the cost function as a contour plot."""
    w_opt, b_opt = uni_linear_regression(x, y, 0.01, 5000)
    w_range = 1.2
    b_range = 6.0
    w_values = np.linspace(w_opt - w_range, w_opt + w_range, 100)
    b_values = np.linspace(b_opt - b_range, b_opt + b_range, 100)

    W, B, J = compute_cost_surface_vectorized(x, y, w_values, b_values)

    plt.figure(figsize=(8, 6))

    contour = plt.contour(W, B, J, levels=20)

    plt.clabel(contour, inline=True, fontsize=8)

    plt.xlabel("w")
    plt.ylabel("b")
    plt.title("Contour Plot of the Cost Function")

def _plot_path_arrows(ax, path_w, path_b, arrow_every):
    """Draw Gradient Descent update arrows on a contour plot."""
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

def plot_gradient_descent_path(x, y, path_w, path_b, arrow_every=150):
    """Plot Gradient Descent update steps on the cost function contour plot."""
    w_opt = path_w[-1]
    b_opt = path_b[-1]
    w_range = 4.0
    b_range = 30.0
    w_values = np.linspace(w_opt - w_range, w_opt + w_range, 100)
    b_values = np.linspace(b_opt - b_range, b_opt + b_range, 100)

    W, B, J = compute_cost_surface_vectorized(x, y, w_values, b_values)

    _, ax = plt.subplots(figsize=(8, 6))

    ax.contour(W, B, J, levels=20, zorder=1)

    _plot_path_arrows(ax, path_w, path_b, arrow_every)

    cost_min = compute_cost(x, y, w_opt, b_opt)
    ax.scatter(
        w_opt, b_opt, color="purple", s=80, zorder=6,
        label=f"Minimum (J = {cost_min:.2f})"
    )

    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title("Gradient Descent on the Cost Function")
    ax.legend()

    plt.tight_layout()
