from matplotlib import pyplot as plt

class DataVisualizer:
    def __init__(self):
        pass

    def calculate_statistics(self, data_array):
        means = data_array.mean(dim=0)
        stds = data_array.std(dim=0)
        return means, stds

    def plot_statistics(self, means, stds):
        if hasattr(means, "detach"):
            means = means.detach().cpu().numpy()
        if hasattr(stds, "detach"):
            stds = stds.detach().cpu().numpy()

        epochs = range(1, len(means) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, means, color="tab:blue", linewidth=2, label="Mean accuracy")
        ax.fill_between(
            epochs,
            means - stds,
            means + stds,
            color="tab:blue",
            alpha=0.2,
            label=r"±$\sigma$"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy [%]")
        ax.set_title("Model Accuracy During Training")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        return fig, ax