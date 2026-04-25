# Drone Classification from Radar Data

Synthetic radar return signals are generated using the **Martin-Mulgrew model**, which captures the micro-Doppler signatures produced by rotating drone blades. These signals are used to benchmark classical and quantum classification approaches.

The project is structured into two main pipelines: 
1. **Synthetic Dataset Generation** — Simulating radar returns across varying Signal-to-Noise Ratios (SNR) in both the time and frequency domains.
2. **Model Training & Evaluation** — Tuning and evaluating classical and quantum neural networks using `Optuna` and `Ray Tune`.

## Methods

### Signal & Dataset Generation
The Martin-Mulgrew model parametrises each drone by blade count `N`, blade lengths `L_1`/`L_2`, and rotor frequency `f_rot`. Signals are generated synthetically for five drone classes: DJI Matrice 300 RTK, DJI Mavic Air 2, DJI Mavic Mini, DJI Phantom 4, and Parrot Disco.

The `SyntheticDatasetGenerator` pipeline handles:
- **Context definitions:** Varying radar parameters such as distance (`R`), radial velocity (`V_rad`), viewing angles (`θ`, `Φ_p`), and SNR.
- **Noise injection:** Applying Additive White Gaussian Noise (AWGN) to simulate real-world radar conditions.
- **Domain representations:** Generating datasets in both the **time domain** (raw complex signals) and **frequency domain** (spectrograms via STFT).

### Classification Models
Three primary classifiers are implemented and evaluated:
- **Classical Neural Network (DNN)** — PyTorch-based models for baseline classification.
- **Quantum Neural Network (QNN)** — Quantum models built with PennyLane, supporting varying ansatz designs (basic, entangling, random) and encoding schemes (angle, amplitude).
- **Quantum Support Vector Machine (QSVM)** *(Planned/WIP)*

### Training & Hyperparameter Tuning
The training pipeline features:
- **Optuna Integration:** Automated hyperparameter optimization for model architecture (layers, neurons, dropout, quantum ansatz) and training parameters (learning rate, batch size, regularization).
- **Statistical Evaluation:** `StatisticalTrainer` for running multiple training trials to calculate performance metrics (accuracy, precision, recall, F1, and confusion matrices).
- **Visualization:** `DataVisualizer` for plotting spectrograms and statistical training results (means and standard deviations across runs).

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
from MachineLearning.Models.experiment_pure.classical_neural_network import ClassicalNeuralNetwork

trainer = HyperparameterTrainer(
    training_path="Datasets/time_domain/training_dataset.pkl",
    validating_path="Datasets/time_domain/validating_dataset.pkl",
    testing_path="Datasets/time_domain/testing_dataset.pkl",
    criterion=nn.CrossEntropyLoss()
)

def objective(trial):
    # Suggest model and training hyperparameters
    config = {
        "model_config": {
            "layers": trial.suggest_categorical("layers", ),[1][2][3][4][5]
            "neurons_per_layer": 256,
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5)
        },
        "training_config": {
            "batch_size": trial.suggest_categorical("batch_size", ),[3][4][5][6]
            "device": "cuda",
            "epochs": 20,
            "optimizer": {"name": "Adam", "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True), "weight_decay": 1e-5},
            "number_of_training_workers": 4,
            "number_of_validating_workers": 2,
            "regularization": {"type": "none", "lambda": None}
        }
    }
    return trainer.train_model(trial, config, ClassicalNeuralNetwork)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```
