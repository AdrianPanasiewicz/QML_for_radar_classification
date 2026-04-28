from matplotlib import pyplot as plt
import torch

class DataVisualizer:
    def __init__(self, language="english"):

        self.translations = {
            "english": {
                "training_chart": {
                    "mean_accuracy": "Mean accuracy",
                    "epoch": "Epoch",
                    "accuracy": "Accuracy [%]",
                    "title": "Model Accuracy During Training",
                    "std_band": r"±$\sigma$"
                },
                "confusion_matrix": {
                    "title": "Confusion Matrix",
                    "xlabel": "Predicted label",
                    "ylabel": "True label"
                }
            },
            "polish": {
                "training_chart": {
                    "mean_accuracy": "Średnia dokładność",
                    "epoch": "Epoka",
                    "accuracy": "Dokładność [%]",
                    "title": "Krzywa procesu trenowania",
                    "std_band": r"±$\sigma$"
                },
                "confusion_matrix": {
                    "title": "Macierz pomyłek",
                    "xlabel": "Etykieta przewidziana",
                    "ylabel": "Etykieta rzeczywista"
                }
            }
        }

        self.desc_dictionary = self.translations[language.lower()]

    def calculate_statistics(self, data_array):
        means = data_array.mean(dim=0)
        stds = data_array.std(dim=0)
        return means, stds

    def plot_training_chart(self, metrics_dict):
        labels = self.desc_dictionary["training_chart"]

        training_accuracy_array = torch.tensor(
            [metrics_dict["training_data"][i]['accuracy'] for i in range(len(metrics_dict["training_data"]))],
            dtype=torch.float32
        )

        means, stds = self.calculate_statistics(training_accuracy_array)

        epochs = range(1, len(means) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, means, color="tab:blue", linewidth=2, label=labels["mean_accuracy"])
        ax.fill_between(
            epochs,
            means - stds,
            means + stds,
            color="tab:blue",
            alpha=0.2,
            label=labels["std_band"]
        )

        ax.set_xlabel(labels["epoch"])
        ax.set_ylabel(labels["accuracy"])
        ax.set_title(labels["title"])
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        return fig, ax
