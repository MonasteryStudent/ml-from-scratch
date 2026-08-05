import matplotlib.pyplot as plt


def plot_binary_classes(X, y, feature_names):
    """Plot a binary classification dataset with separate class markers."""

    negative_class = y == 0
    positive_class = y == 1

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        X[positive_class, 0],
        X[positive_class, 1],
        marker="x",
        label="Passed",
    )

    ax.scatter(
        X[negative_class, 0],
        X[negative_class, 1],
        marker="o",
        label="Failed",
    )

    ax.set_title("Exam outcome by study activity")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend()

    plt.tight_layout()