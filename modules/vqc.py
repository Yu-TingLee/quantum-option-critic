import torch
import torch.nn as nn
import pennylane as qml

class Preprocessor(nn.Module):
    """
    Normalize continuous obs into [-pi, pi]. Booleans are normalized to {0,pi}.
    """
    def __init__(self, env_name):
        super().__init__()
        self.env_name = env_name
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.env_name == 'CartPole-v1':
            return self._cartpole(obs)
        elif self.env_name == 'Acrobot-v1':
            return self._acrobot(obs)
        elif self.env_name == 'LunarLander-v3':
            return self._lunarlander(obs)
        else:
            assert False, f"Preprocessing not defined for env {self.env_name}"
        
    def _cartpole(self, obs):
        o1, o2, o3, o4 = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]

        y1 = torch.pi * o1 / 4.8
        y2 = 2.0 * torch.atan(o2)
        y3 = torch.pi * o3 / 0.418
        y4 = 2.0 * torch.atan(o4)

        return torch.stack([y1, y2, y3, y4], dim=1)
    
    def _acrobot(self, obs):
        o1, o2, o3, o4, o5, o6 = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4], obs[:, 5]
        
        y1 = torch.pi * o1
        y2 = torch.pi * o2
        y3 = torch.pi * o3
        y4 = torch.pi * o4
        y5 = o5 / 4.0
        y6 = o6 / 9.0
        
        return torch.stack([y1, y2, y3, y4, y5, y6], dim=1)
    
    def _lunarlander(self, obs):
        o1, o2, o3, o4, o5, o6, o7, o8 = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4], obs[:, 5], obs[:, 6], obs[:, 7]
        
        y1 = torch.pi * o1 / 2.5
        y2 = torch.pi * o2 / 2.5
        y3 = torch.pi * o3 / 10.0
        y4 = torch.pi * o4 / 10.0
        y5 = o5 / 2.0
        y6 = torch.pi * o6 / 10.0
        y7 = torch.pi * o7
        y8 = torch.pi * o8
        
        return torch.stack([y1, y2, y3, y4, y5, y6, y7, y8], dim=1)
        
    
class VQC(nn.Module):
    """
    lambda*RX -> CNOT -> RZ-RY-RZ -> Measurement
    """
    def __init__(self, n_qubits=4, layers=1, device = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.device = device
        
        self.theta = nn.Parameter(
            0.01 * torch.randn(self.layers, self.n_qubits, 2, dtype=torch.float32, device=self.device)
        )
        
        self.lam = nn.Parameter(
            torch.ones(self.layers, self.n_qubits, dtype=torch.float32, device=self.device)
        )
        
        self.qdev = qml.device("default.qubit", wires=self.n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.qdev, interface="torch", diff_method="backprop")
        def circuit(inputs, theta, lam):
            for l in range(self.layers):
                # Encoding/re-upload
                for q in range(self.n_qubits):
                    qml.RX(lam[l, q] * inputs[..., q], wires=q)
                # CNOT entanglement
                for q in range(self.n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

                # Trainable RZ-RY-RZ block
                for q in range(self.n_qubits):
                    qml.RY(theta[l, q, 0], wires=q)
                    qml.RZ(theta[l, q, 1], wires=q)

            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_vals = self.circuit(x, self.theta, self.lam)
        return torch.stack([p for p in exp_vals], dim=1)