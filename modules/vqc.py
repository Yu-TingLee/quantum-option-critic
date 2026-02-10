import torch
import torch.nn as nn
import pennylane as qml

def preprocess_obs(obs):
    """
    Normalize obs into [-pi, pi].
    """
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)

    o1, o2, o3, o4 = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]

    # Bounded variables scaling
    lam1 = torch.pi * o1 / 4.8
    lam2 = torch.pi * o2 / 0.418

    # Unbounded variables scaling via arctan
    lam3 = 2.0 * torch.atan(o3)
    lam4 = 2.0 * torch.atan(o4)

    return torch.stack([lam1, lam2, lam3, lam4], dim=1)

class VQC(nn.Module):
    """
    RX -> CNOT -> RZ-RY-RZ -> Measure
    """
    def __init__(self, n_qubits=4, layers=2, device = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.device = device
        
        self.theta = nn.Parameter(
            0.01 * torch.randn(self.layers, self.n_qubits, 3, dtype=torch.float32, device=self.device)
        )
        
        self.qdev = qml.device("default.qubit", wires=self.n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def circuit(inputs, theta):
            # Encoding layer
            for q in range(self.n_qubits):
                qml.RX(inputs[..., q], wires=q)

            for l in range(self.layers):
                # CNOT entanglement
                for q in range(self.n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

                # Trainable RZ-RY-RZ block
                for q in range(self.n_qubits):
                    qml.RZ(theta[l, q, 0], wires=q)
                    qml.RY(theta[l, q, 1], wires=q)
                    qml.RZ(theta[l, q, 2], wires=q)

            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_vals = self.circuit(x, self.theta) 
        return torch.stack([p for p in exp_vals], dim=1)