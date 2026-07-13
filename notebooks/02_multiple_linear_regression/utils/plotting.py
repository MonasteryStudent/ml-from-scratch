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
