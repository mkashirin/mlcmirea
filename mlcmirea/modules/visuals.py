import matplotlib.pyplot as plt

from .metrics import ConfusionMatrix


def plot_confusion_matrix(
    confusion_matrix: ConfusionMatrix, title: str
) -> None:
    """Plot any confusion matrix, calculated by the the
    :function:`calculate_confusion_matrix()` function from the
    :module:`metrics` module, labelling each cell, axes, and the matrix itself.
    """
    matrix, _ = confusion_matrix
    figure, axis = plt.subplots()
    ticks = list(range(0, len(matrix)))

    axis.set_yticks(ticks)
    axis.set_xticks(ticks)
    axis.set_xlabel("Actual")
    axis.set_ylabel("Predicted")
    for i, _ in enumerate(matrix):
        for j, _ in enumerate(matrix):
            # fmt: off
            axis.text(
                j, i,
                int(matrix[i, j]),
                verticalalignment="center",
                horizontalalignment="center",
                color="white",
            )
            # fmt: on
    axis.set_title("Confusion Matrix" if title is None else title, fontsize=14)
    to_show = plt.imshow(matrix, cmap="seismic")
    figure.colorbar(to_show, label="Samples")

    plt.tight_layout()
    plt.show()
