# Quantum Hierarchical Reinforcement Learning via Variational Quantum Circuits

Official implementation for the paper: **Quantum Hierarchical Reinforcement Learning via Variational Quantum Circuits**.

We implement a hybrid hierarchical RL agent based on the option-critic architecture, where each component can be instantiated as either a classical NN or a variational quantum circuit (VQC). VQCs are implemented in PennyLane with a data-reuploading ansatz.
The classical baselines are adapted from [lweitkamp/option-critic-pytorch](https://github.com/lweitkamp/option-critic-pytorch).

## Setup

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run all experiments
**WARNING:** `run.sh` executes sequentially for simplicity. We highly recommend parallelizing these run.

```sh
bash run.sh
```

Logs go to `runs/`; plots go to `plots/`.
