import pennylane as qml
from torch import nn
import torch

class QuantumSupportVectorMachine(nn.Module):
    def __init__(self, n_qubits, device='default.qubit'):
        super(QuantumSupportVectorMachine, self).__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)

        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = qml.QNode(quantum_circuit, self.dev, interface='torch')

        self.quantum_weights = nn.Parameter(torch.rand((3, self.n_qubits), requires_grad=True))
        self.classical_layer = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        x_quantum = torch.tensor(x, dtype=torch.float32)
        quantum_out = self.qnode(x_quantum, self.quantum_weights)
        quantum_out = torch.stack(quantum_out, dim=1).to(torch.float32)
        out = self.classical_layer(quantum_out)
        return out