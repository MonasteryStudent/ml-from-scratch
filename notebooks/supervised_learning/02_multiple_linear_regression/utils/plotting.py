import numpy as np
import matplotlib.pyplot as plt


def plot_features_vs_target(X, y, feature_names=None):
    """Plot each feature against the target variable."""

    if feature_names is None:
        feature_names = [
            f"Feature {i + 1}"
            for i in range(X.shape[1])
        ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    fig.suptitle("Relationship Between Features and Exam Score", fontsize=14)
    margin = 5

    for ax, feature, name in zip(axes.flat, X.T, feature_names):
        ax.scatter(feature, y, s=35)

        ax.set_xlabel(name)
        ax.set_ylim(y.min() - margin, y.max() + margin)

    for ax in axes[:, 0]:
        ax.set_ylabel("Exam Score")

    plt.tight_layout(rect=[0, 0, 1, 0.96])


def plot_scaling_comparison(X_subset, X_subset_standardized, feature_names):
    """Plot two features against each other before and after feature scaling."""

    subsets = [X_subset, X_subset_standardized]
    titles = ["Before Scaling", "After Standardization"]

    fig = plt.figure(figsize=(6, 4))

    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.8], wspace=0.05)

    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
    ]

    fig.suptitle("Comparison of Feature Scaling", fontsize=14)

    for ax, subset, title in zip(axes, subsets, titles):
        ax.scatter(subset[:, 0], subset[:, 1], s=35)

        ax.set_title(title)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_aspect("equal", adjustable="box")

    fig.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.97)


def plot_cost_history(cost_history, ax_title):
    """Plot a single cost history for hyperparameter tuning."""

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(cost_history)
    ax.set_title(ax_title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")

    fig.text(
        0.5,
        0.01,
        f"Final Cost: {cost_history[-1]:.3f}",
        ha="center",
    )

    fig.subplots_adjust(
        top=0.85,
        bottom=0.2,
    )


def plot_cost_histories(cost_histories, figure_title, axes_titles, fig_size, sharey=False):
    """Plot multiple cost histories in separate subplots."""

    fig, axes = plt.subplots(
        1, 
        len(cost_histories), 
        figsize=fig_size,
        sharey=sharey
    )

    # Ensure that axes is always iterable.
    axes = np.atleast_1d(axes)

    fig.suptitle(figure_title, fontsize=14)

    for ax, history, title in zip(axes, cost_histories, axes_titles):
        ax.plot(history)
        ax.set_title(title)
        ax.set_xlabel("Iteration")

    axes[0].set_ylabel("Cost")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

