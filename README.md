# Quantum Hierarchical Reinforcement Learning via Variational Quantum Circuits

Official implementation for the paper: **Quantum Hierarchical Reinforcement Learning via Variational Quantum Circuits**.

We implement a hybrid hierarchical RL agent based on the option-critic architecture, where each component (feature extractor, option-value function, termination functions, intra-option policies) can be instantiated as either a classical component or a variational quantum circuit (VQC). VQCs are implemented in PennyLane with a data-reuploading ansatz.

## Setup

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run all experiments
**WARNING:** `run.sh` executes sequentially for simplicity. We highly recommend parallelizing these runs, or execution may take months to complete.

```sh
bash run.sh
```

Logs go to `runs/`; plots go to `plots/`.
