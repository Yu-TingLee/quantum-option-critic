# A Quantum Option-Critic Architecture for Reinforcement Learning
The is the official implementation of the paper: **A Quantum Option-Critic Architecture for Reinforcement Learning**.
Specifically, hybrid quantum-classical agents based on the option-critic framework are implemented.

## Overview
- **option_critic_run.py**: Main training script for option-critic.
- **modules/**: Contains VQC and option-critic modules.
- **utils.py**: Utilities.

Run all the experiments:
```sh
bash run.sh
```
Logs are saved to `outputs/<model>`; plots are saved in `plots/`.