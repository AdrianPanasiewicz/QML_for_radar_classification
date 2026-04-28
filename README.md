# Drone Classification from Radar Data

Synthetic radar return signals are generated using the **Martin-Mulgrew model**, which captures the micro-Doppler signatures produced by rotating drone blades. These signals are used to benchmark classical and quantum classification approaches.

The project is structured into three main pipelines: 
1. **Synthetic Dataset Generation** — Simulating radar returns across varying Signal-to-Noise Ratios (SNR) in both the time and frequency domains.
2. **Hyperparameter Tuning** — Finding the best classical and quantum model architectures using `Optuna`.
3. **Statistical Evaluation** — Running multiple training trials to calculate metrics and visualize performance.

## Methods

### Signal & Dataset Generation
The Martin-Mulgrew model parametrises each drone by blade count `N`, blade lengths `L_1`/`L_2`, and rotor frequency `f_rot`. Signals are generated synthetically for five drone classes: DJI Matrice 300 RTK, DJI Mavic Air 2, DJI Mavic Mini, DJI Phantom 4, and Parrot Disco.

The `SyntheticDatasetGenerator` handles:
- **Context definitions:** Varying radar parameters such as distance (`R`), radial velocity (`V_rad`), viewing angles (`θ`, `Φ_p`), and SNR.
- **Noise injection:** Applying Additive White Gaussian Noise (AWGN) to simulate real-world radar conditions.
- **Domain representations:** Generating datasets in both the **time domain** (raw complex signals) and **frequency domain** (spectrograms via STFT).

### Classification Models
The following classifiers are implemented and evaluated:
- **Classical Neural Network (CNN / DNN)** — PyTorch-based models for baseline classification.
- **Quantum Neural Network (QNN)** — Hybrid models built with PennyLane/Qiskit, supporting varying ansatz designs (basic, entangling, random) and encoding schemes (angle, amplitude).
- **Quantum Support Vector Machine (QSVM)** *(Planned/WIP)*

### Training, Tuning & Visualization
The training pipeline features:
- **Optuna Integration:** Automated hyperparameter optimization for model architecture and training parameters via `HyperparameterTrainer`.
- **Statistical Evaluation:** `StatisticalTrainer` for executing multiple full training loops to gather statistically significant performance metrics.
- **Visualization:** `DataVisualizer` for plotting training curves, spectrograms, and formatted confusion matrices.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/AdrianPanasiewicz/QML_for_radar_classification.git
cd QML_for_radar_classification
pip install -r requirements.txt
```

**Core Dependencies:**
- `numpy`, `sympy`, `scipy` — Signal generation, STFT, and mathematics.
- `qiskit`, `pennylane` — Quantum circuit implementations and simulators.
- `torch` — Classical and hybrid quantum-classical neural networks.
- `ray[tune]`, `optuna` — Hyperparameter tuning and trial scheduling.
- `matplotlib`, `scikit-learn` — Visualization and metric calculations.

## Usage Examples

### 1. Generating a Synthetic Dataset
```python
from Data.Primitives.presets import drones_array, default_radar
from Data.Generators.synthetic_dataset_generator import DatasetMetadata, DataRequest, SyntheticDatasetGenerator
from Data.Primitives.environment_classes import Context
from Data.Primitives.noise_models import AdditiveWhiteGaussianNoise

# Define the radar context
context = Context(R=1000, V_rad=25, θ=0.39, Φ_p=0.39, A_r=1, snr=20, t_start=0, t_stop=0.1, dt=0.0001)

# Initialize dataset generator
md = DatasetMetadata.create_from_path("Datasets/time_domain/training_dataset.pkl")
dataset_gen = SyntheticDatasetGenerator(dataset_metadata=md)

# Generate requests for all drone classes
requests = [
    DataRequest(f"label={drone.name}", drone, default_radar, context, AdditiveWhiteGaussianNoise(), sample_size=70)
    for drone in drones_array
]

dataset_gen.append_data_requests(requests)
dataset_gen.generate_signal_data(stft_form=False) # Set to True for frequency domain
```

### 2. Hyperparameter Tuning with Optuna
```python
import optuna
from torch import nn
from MachineLearning.Trainers.hyperparameter_trainer import HyperparameterTrainer
from MachineLearning.Models.experiment_pure.quantum_neural_network import QuantumNeuralNetwork

trainer = HyperparameterTrainer(
    training_path="Datasets/time_domain/training_dataset.pkl",
    validating_path="Datasets/time_domain/validating_dataset.pkl",
    testing_path="Datasets/time_domain/testing_dataset.pkl",
    criterion=nn.BCELoss()
)

def objective(trial):
    config = {
        model_config = {
            "n_qubits": 10,
            "layers": trial.suggest_int("layers", 1, 5),
            "encoding": trial.suggest_categorical("encoding", ["angle", "amplitude"]),
            "ansatz": trial.suggest_categorical("ansatz", ["basic", "entangling", "random"]),
            "simulator": "default.qubit",
        },
        "training_config": {
            "batch_size": trial.suggest_categorical("batch_size", ),[4][5][6]
            "device": "cuda",
            "epochs": 20,
            "optimizer": {"name": "Adam", "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True), "weight_decay": 1e-5},
            "number_of_training_workers": 4,
            "number_of_validating_workers": 2,
            "regularization": {"type": "none", "lambda": None}
        }
    }
    return trainer.train_model(trial, config, QuantumNeuralNetwork)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### 3. Statistical Evaluation & Visualization
Once the best hyperparameters are found, use the `StatisticalTrainer` to train the model multiple times. This allows you to evaluate stability and generate visuals like confusion matrices and loss curves.

```python
import torch
from torch import nn
from MachineLearning.Trainers.statistical_trainer import StatisticalTrainer
from MachineLearning.Processing.data_visualizer import DataVisualizer
from MachineLearning.Models.experiment_pure.quantum_neural_network import QuantumNeuralNetwork

config = {
        model_config = {
        "n_qubits"  : 10,
        "layers"    : 2,
        "encoding"  : "angle",
        "ansatz"    : "basic",
        "simulator" : 'default.qubit',
    },
    "training_config": {
        "number_of_training_workers": 4,
        "number_of_validating_workers": 2,
        "number_of_testing_workers": 2,
        "batch_size": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 100,
        "number_of_trials": 10, # Run training 10 separate times
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "momentum": 0.8,
            "weight_decay": 1e-6
        },
        "regularization": {
            "type": "l1",
            "lambda": 1e-6
        },
    }
}

# Run the statistical trainer
trainer = StatisticalTrainer(
    training_path="Datasets/time_domain/training_dataset.pkl",
    validating_path="Datasets/time_domain/validating_dataset.pkl",
    testing_path="Datasets/time_domain/testing_dataset.pkl", 
    criterion=nn.BCELoss()
)
net, metrics_dict = trainer.train_model(QuantumNeuralNetwork, config)

# Visualize results
plotter = DataVisualizer(language="english")

# Display the confusion matrix
plotter.plot_confusion_matrix(metrics_dict, significant_digits=1)

# Display tabular metrics (Accuracy, Precision, Recall, F1)
print(plotter.get_metrics_table(metrics_dict, significant_digits=3))

# Plot training vs validation accuracy/loss over epochs
plotter.plot_training_chart(metrics_dict)
```
