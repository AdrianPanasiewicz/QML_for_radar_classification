from Data.Primitives.presets import class_map
from IPython.display import HTML
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
                    "ylabel": "True label",
                    "x_tick_labels": [class_map[0], class_map[1]],
                    "y_tick_labels": [class_map[0], class_map[1]],
                    "colorbar": "Mean count"
                },
                "table_output": {
                    "metric": "Metric",
                    "mean" : "Expectation value",
                    "std" : "Standard deviation",
                    "accuracy": "Accuracy",
                    "balanced_accuracy": "Balanced accuracy",
                    "precision": "Macro-Precision",
                    "recall": "Macro-Recall",
                    "f1": "Macro-F1-score"
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
                    "ylabel": "Etykieta rzeczywista",
                    "x_tick_labels": [class_map[0], class_map[1]],
                    "y_tick_labels": [class_map[0], class_map[1]],
                    "colorbar": "Średnia liczba"
                },
                "table_output": {
                    "metric": "Metryka",
                    "mean": "Wartość oczekiwana",
                    "std": "Odchylenie standardowe",
                    "accuracy": "Dokładność",
                    "balanced_accuracy": "Zbalansowana dokładność",
                    "precision": "Macro-Precyzja",
                    "recall": "Macro-Czułość",
                    "f1": "Macro-Wynik F1"
                }
            }
        }

        self.desc_dictionary = self.translations[language.lower()]

    def calculate_statistics(self, data_array):
        means = data_array.mean(dim=0)
        stds = data_array.std(dim=0)
        return means, stds

    def plot_training_chart(self, metrics_dict, ax=None, color="tab:blue", label_suffix="", include_stds_labels=True, std_transparency=0.2):
        labels = self.desc_dictionary["training_chart"]

        training_accuracy_array = torch.tensor(
            [metrics_dict["training_data"][i]['accuracy'] for i in range(len(metrics_dict["training_data"]))],
            dtype=torch.float32
        )

        means, stds = self.calculate_statistics(training_accuracy_array)
        epochs = range(1, len(means) + 1)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        mean_label = f"{labels['mean_accuracy']} {label_suffix}".strip()
        std_label = f"{labels['std_band']} {label_suffix}".strip()

        ax.plot(epochs, means, color=color, linewidth=2, label=mean_label)
        ax.fill_between(
            epochs,
            means - stds,
            means + stds,
            color=color,
            alpha=std_transparency,
            label=labels["std_band"] if include_stds_labels else None
        )

        ax.set_xlabel(labels["epoch"])
        ax.set_ylabel(labels["accuracy"])
        ax.set_title(labels["title"])
        ax.grid(True, alpha=0.8)

        if fig is not None:
            ax.legend()
            fig.tight_layout()

        return fig, ax

    def plot_confusion_matrix(self, metrics_dict, significant_digits=2, include_std=True):
        labels = self.desc_dictionary["confusion_matrix"]

        if include_std:
            confusion_tensor = {
                key: torch.tensor(
                    [run["confusion_matrix"][key] for run in metrics_dict["testing_data"]],
                    dtype=torch.float32
                )
                for key in ("TP", "TN", "FP", "FN")
            }
            stats = {
                key: self.calculate_statistics(confusion_tensor[key])
                for key in ("TP", "TN", "FP", "FN")
            }

            mean_matrix = torch.tensor([
                [stats["TN"][0], stats["FP"][0]],
                [stats["FN"][0], stats["TP"][0]]
            ], dtype=torch.float32)

            std_matrix = torch.tensor([
                [stats["TN"][1], stats["FP"][1]],
                [stats["FN"][1], stats["TP"][1]]
            ], dtype=torch.float32)
        else:
            cm = metrics_dict['testing_data'][0]["confusion_matrix"]

            mean_matrix = torch.tensor([
                [cm["TN"], cm["FP"]],
                [cm["FN"], cm["TP"]]
            ], dtype=torch.float32)

            std_matrix = torch.zeros_like(mean_matrix)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mean_matrix.numpy(), cmap="Blues")

        ax.set_xticks([0, 1], labels=labels["x_tick_labels"])
        ax.set_yticks([0, 1], labels=labels["y_tick_labels"])
        ax.set_xlabel(labels["xlabel"], labelpad=14)
        ax.set_ylabel(labels["ylabel"], labelpad=14)
        ax.set_title(labels["title"], pad=14)

        threshold = (mean_matrix.max().item() + mean_matrix.min().item()) / 2
        total_inferences = mean_matrix.sum().item()

        for i in range(2):
            for j in range(2):
                value = mean_matrix[i, j].item()
                std = std_matrix[i, j].item()

                percentage = (value / total_inferences) * 100 if total_inferences > 0 else 0
                percentage_std = (std / total_inferences) * 100 if total_inferences > 0 else 0

                text_color = "white" if value > threshold else "black"

                if include_std:
                    text = (
                        f"{percentage:.1f}% ± {percentage_std:.1f}%\n"
                        f"({value:.{significant_digits}f} ± {std:.{significant_digits}f})"
                    )
                else:
                    text = (
                        f"{percentage:.1f}%\n"
                        f"({value:.{significant_digits}f})"
                    )

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold"
                )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(labels["colorbar"], labelpad=14)

        fig.tight_layout()
        return fig, ax

    def get_metrics_table(self, metrics_dict, significant_digits=4):
        labels = self.desc_dictionary["table_output"]
        metrics_tensor = {
            key : torch.tensor([run[key] for run in metrics_dict["testing_data"]], dtype=torch.float32)
            for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1")
        }
        stats = {
            key : self.calculate_statistics(metrics_tensor[key])
            for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1")
        }

        rows = []
        for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
            mean, std = stats[key]
            rows.append(
                f"<tr><td>{labels[key]}</td><td>{mean.item():.{significant_digits}f} ± {std.item():.{significant_digits}f}</td></tr>"
            )

        html = f"""
            <table>
                <thead>
                    <tr>
                        <th>{labels['metric']}</th>
                        <th>{labels['mean']} ± {labels['std']}</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
            """
        return HTML(html)

    def get_metrics(self, metrics_dict):
        labels = self.desc_dictionary["table_output"]
        metrics_tensor = {
            key : torch.tensor([run[key] for run in metrics_dict["testing_data"]], dtype=torch.float32)
            for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1")
        }
        stats = {
            key : self.calculate_statistics(metrics_tensor[key])
            for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1")
        }

        return stats
