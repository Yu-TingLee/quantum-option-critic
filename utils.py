
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from pathlib import Path
import pennylane as qml
import matplotlib.pyplot as plt
from modules.vqc import preprocess_obs

def make_env(env_name):
    env = gym.make(env_name)
    if isinstance(env.observation_space, gym.spaces.Dict):
        try:
            # Keep only the image observation (drop mission text), then flatten.
            from minigrid.wrappers import ImgObsWrapper
            env = ImgObsWrapper(env)
        except Exception:
            pass

        env = FlattenObservation(env)

    return env


def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs


def print_param(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable)

    def count(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def type_str(is_quantum):
        return "(Quantum)" if is_quantum else "(Classical)"

    def vqc_meta(module):
        n_vqcs = 0
        for m in module.modules():
            if m.__class__.__name__ == "VQC":
                n_vqcs += 1
                
        target = module
        if hasattr(module, "vqc"):
            target = module.vqc
        elif hasattr(module, "__len__") and len(module) > 0 and hasattr(module[0], "vqc"):
            target = module[0].vqc

        n_qubits = getattr(target, "n_qubits", "-")
        layers = getattr(target, "layers", "-")
        return n_qubits, layers, n_vqcs

    def row(name, is_q, module):
        n_qubits, layers, n_vqcs = ("-", "-", "-")
        if is_q:
            n_qubits, layers, n_vqcs = vqc_meta(module)
        return (name, type_str(is_q), count(module), n_qubits, layers, n_vqcs)

    rows = [
        row("Feature Trunk", model.Qfeats, model.features),
        row("Q-Head", model.Qhead, model.Q),
        row("Terminations", model.Qterm, model.terminations),
        row("Intra-Option Policies", model.Qoption, model.option_policies),
    ]

    print("-" * 72)
    print(f"Whole model trainable parameters: {total_params}")
    print(f"{'Component':<22} | {'Type':<12} | {'Params':<6} | {'Qubits':<6} | {'Layers':<6} | {'#VQCs':<5}")
    print("-" * 72)

    for name, typ, params, n_qubits, layers, n_vqcs in rows:
        print(f"{name:<22} | {typ:<12} | {params:<6} | {str(n_qubits):<6} | {str(layers):<6} | {str(n_vqcs):<5}")

    print("-" * 72)

def plot_circuits(model, obs_shape, device, save_dir="plots"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    qml.drawer.use_style("black_white")

    def save(vqc, x, name):
        fig, ax = qml.draw_mpl(vqc.circuit)(x, vqc.theta)
        fig.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_dir}/{name}.png")

    any_vqc = False

    # 1) Feature trunk VQC
    if hasattr(model.features, "vqc"):
        any_vqc = True
        dummy_obs = torch.zeros((1, *obs_shape), dtype=torch.float32, device=device)
        x = preprocess_obs(dummy_obs).to(dtype=torch.float32)
        save(model.features.vqc, x, "circuit_Qfeats")

    def head_x(vqc):
        s = torch.zeros((1, vqc.n_qubits), dtype=torch.float32, device=device)
        return (torch.pi * torch.tanh(s)).to(dtype=torch.float32)

    # 2) Q-head VQC
    if hasattr(model.Q, "vqc"):
        any_vqc = True
        save(model.Q.vqc, head_x(model.Q.vqc), "circuit_Qhead")

    # 3) Terminations VQC
    if hasattr(model.terminations, "vqc"):
        any_vqc = True
        save(model.terminations.vqc, head_x(model.terminations.vqc), "circuit_Qterm")

    # 4) Option policy VQC
    if len(model.option_policies) > 0 and hasattr(model.option_policies[0], "vqc"):
        any_vqc = True
        save(model.option_policies[0].vqc, head_x(model.option_policies[0].vqc), "circuit_Qoption")

    if not any_vqc:
        print("No VQC modules found to plot.")