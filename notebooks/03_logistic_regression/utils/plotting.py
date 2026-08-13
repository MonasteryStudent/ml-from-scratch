import matplotlib.pyplot as plt


def plot_binary_classes(X, y):
    """Plot a binary classification dataset with separate class markers."""

    negative_class = y == 0
    positive_class = y == 1

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        X[positive_class, 0],
        X[positive_class, 1],
        marker="x",
        label="Passed (y = 1)",
    )

    ax.scatter(
        X[negative_class, 0],
        X[negative_class, 1],
        marker="o",
        label="Failed (y = 0)",
    )

    feature_names = ["Study hours", "Practice tests"]

    ax.set_title("Exam outcome by study activity")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend()

    plt.tight_layout()


def plot_sigmoid_function(z, probabilities):
    """Plot the sigmoid function over a range of linear model outputs."""

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(z, probabilities)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle="--", linewidth=1)

    ax.set_title("Sigmoid Function")
    ax.set_xlabel("Linear model output $z$")
    ax.set_ylabel("Probability $\\sigma(z)$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    return fig, ax


def plot_decision_boundary(x1_values, x2_values):
    """Plot a linear decision boundary."""

    plt.plot(
        x1_values,
        x2_values,
        color="purple",
        linestyle="--",
        label="Decision Boundary"
    )