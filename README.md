# Quantum Hierarchical Reinforcement Learning via Variational Quantum Circuits

Official implementation for the paper: **Quantum Hierarchical Reinforcement Learning via Variational Quantum Circuits**.

We implement a hybrid hierarchical RL (HRL) agent based on the option-critic architecture, where each component can be instantiated as either a classical NN or a variational quantum circuit (VQC). VQCs are implemented in PennyLane with a data re-uploading ansatz.
The classical baselines are adapted from [lweitkamp/option-critic-pytorch](https://github.com/lweitkamp/option-critic-pytorch).

## Short Abstract
**TL;DR:** We show that VQCs can effectively enhance a HRL agent with less trainable parameters.

While parameterized quantum computations have shown success in standard RL, their adaptability to HRL remains a critical open question. We show that VQCs can enhance a hybrid option-critic agent. Using a quantum feature extractor, our hybrid agent outperforms classical baselines with fewer parameters. We also identify an architectural bottleneck that quantum option-value estimation severely degrades learning. Our ablations reveal how quantum circuit design affects performance.

## Usage

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash run.sh
```

Logs go to `runs/`; plots go to `plots/`.