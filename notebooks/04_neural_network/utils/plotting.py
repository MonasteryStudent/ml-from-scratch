import numpy as np
import matplotlib.pyplot as plt


def plot_binary_classes(X, y, title="Two Moons Dataset", ax=None):
    """Plot a binary classification dataset."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    negative_class = y == 0
    positive_class = y == 1

    ax.scatter(
        X[negative_class, 0],
        X[negative_class, 1],
        color="tab:blue",
        marker="o",
        label="Class 0",
    )

    ax.scatter(
        X[positive_class, 0],
        X[positive_class, 1],
        color="tab:red",
        marker="x",
        label="Class 1",
    )

    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend()

    plt.tight_layout()

    return fig, ax


def plot_decision_boundary(
    X,
    y,
    predict_function,
    title="Decision Boundary",
):
    """Plot the decision regions of a binary classifier."""
    x1_min = X[:, 0].min() - 0.5
    x1_max = X[:, 0].max() + 0.5
    x2_min = X[:, 1].min() - 0.5
    x2_max = X[:, 1].max() + 0.5

    x1_values, x2_values = np.meshgrid(
        np.linspace(x1_min, x1_max, 300),
        np.linspace(x2_min, x2_max, 300),
    )

    grid = np.column_stack((
        x1_values.ravel(),
        x2_values.ravel(),
    ))

    predictions = predict_function(grid)
    predictions = predictions.reshape(x1_values.shape)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.contourf(
        x1_values,
        x2_values,
        predictions,
        levels=[-0.5, 0.5, 1.5],
        cmap="coolwarm",
        alpha=0.15,
    )

    ax.contour(
        x1_values,
        x2_values,
        predictions,
        levels=[0.5],
        colors="purple",
        linestyles="--",
    )

    plot_binary_classes(
        X,
        y,
        title=title,
        ax=ax,
    )

    return fig, ax


def plot_activation_functions(z, relu_values, sigmoid_values):
    """Plot the ReLU and sigmoid activation functions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(z, relu_values)
    axes[0].axhline(0, color="gray", linewidth=1)
    axes[0].axvline(0, color="gray", linewidth=1)
    axes[0].set_title("ReLU")
    axes[0].set_xlabel("$z$")
    axes[0].set_ylabel("$g(z)$")
    axes[0].grid(alpha=0.3)

    axes[1].plot(z, sigmoid_values)
    axes[1].axhline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
    )
    axes[1].axvline(
        0,
        color="gray",
        linewidth=1,
    )
    axes[1].set_title("Sigmoid")
    axes[1].set_xlabel("$z$")
    axes[1].set_ylabel("$g(z)$")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    return fig, axes


def plot_binary_cross_entropy(
    probabilities,
    positive_losses,
    negative_losses,
):
    """Plot binary cross-entropy for both target classes."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        probabilities,
        positive_losses,
        label="$y = 1$",
    )
    ax.plot(
        probabilities,
        negative_losses,
        label="$y = 0$",
    )

    ax.set_title("Binary Cross-Entropy Loss")
    ax.set_xlabel("Predicted probability $\\hat{y}$")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    return fig, ax