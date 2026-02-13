# A Quantum Option-Critic Architecture for Reinforcement Learning
The is the official implementation of the paper: **A Quantum Option-Critic Architecture for Reinforcement Learning**.
Specifically, hybrid quantum-classical agents based on the option-critic framework are implemented.

## Overview
- **option_critic_run.py**: Main training script for option-critic.
- **random_run.py**: Runs random baseline.
- **modules/**: Contains modules for VQCs, option-critic and experience replay.
- **plot.py**: Plots learning curves (reward vs. steps; reward vs. episodes).
- **plot_arch.py**: Plots the model architecture.
- **utils.py**: Utilities.

Run all the experiments:
```sh
bash run.sh
```
Logs are saved to `runs/`; plots are saved in `plots/`.