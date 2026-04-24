import pennylane as qml
from torch import nn

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encodings = {
            "angle" : self.angle_encoding,
            "amplitude": self.amplitude_embedding
        }
        self.ansatzes = {
            "basic" : self.basic_layers,
            "entangling": self.entangling_layers,
            "random": self.random_layers
        }

        self.init_kwargs = {
            "n_qubits": config['n_qubits'],
            "layers": config['layers'],
            "encoding": config["encoding"],
            "ansatz": config["ansatz"],
            "simulator": config["simulator"],
        }
        
        self.model_name = self.__class__.__name__

        self.dev = qml.device(self.init_kwargs["simulator"], wires=config["n_qubits"])

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def classifier(inputs, weights):
            self.encodings[self.init_kwargs['encoding']](inputs)
            self.ansatzes[self.init_kwargs['ansatz']](weights)
            return qml.expval(qml.PauliZ(0))

        ansatz = self.init_kwargs["ansatz"]
        n_layers = self.init_kwargs["layers"]
        n_qubits = self.init_kwargs["n_qubits"]

        if ansatz == "basic":
            weight_shapes = {
                "weights": qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
            }
        elif ansatz == "entangling":
            weight_shapes = {
                "weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
            }
        elif ansatz == "random":
            weight_shapes = {
                "weights": qml.RandomLayers.shape(n_layers=n_layers, n_rotations=n_qubits)
            }
        else:
            raise ValueError(f"Unknown ansatz: {ansatz}")

        self.qnode = qml.qnn.TorchLayer(classifier, weight_shapes)


    def angle_encoding(self, x):
        n_qubits = self.init_kwargs['n_qubits']
        qml.AngleEmbedding(features=x, wires=range(n_qubits), rotation='X')

    def amplitude_embedding(self, x):
        n_qubits = self.init_kwargs['n_qubits']
        qml.AmplitudeEmbedding(features=x, wires=range(n_qubits), pad_with=0.0, normalize=True)

    def basic_layers(self, weights):
        n_layers, n_qubits = weights.shape[0], weights.shape[1]
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    def entangling_layers(self, weights):
        n_layers, n_qubits = weights.shape[0], weights.shape[1]
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    def random_layers(self, weights):
       n_layers, n_qubits = weights.shape[0], weights.shape[1]
       qml.RandomLayers(weights, wires=range(n_qubits))

    def forward(self, x):
        return self.qnode(x)