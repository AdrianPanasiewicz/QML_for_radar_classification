import pennylane as qml
from cv2 import QRCodeDetectorAruco
from torch import nn
import torch

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits, device='default.qubit'):
        super(QuantumNeuralNetwork, self).__init__()
        self.init_kwargs = {
            "n_qubits": n_qubits,
            "layers": 1,
            "encoding": None,
            "Ansatz": None,
            "simulator": 'lightning.qubit', # Try "default.mixed"
            "gpu": False
            }
        self.model_name = self.__class__.__name__
        self.n_qubits = n_qubits
        self.dev = qml.device(self.init_kwargs["simulator"], wires=n_qubits)

        def quantum_circuit(inputs, weights):
            pass

        self.qnode = qml.QNode(quantum_circuit, self.dev, interface='torch')


    def forward(self, x):
        pass

qnn = QuantumNeuralNetwork(n_qubits=2)