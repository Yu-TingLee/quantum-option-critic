#!/usr/bin/env bash
set -euo pipefail

ENVS=(
  "CartPole-v1"
)

MODELS=(
  ""  # Classical
  "--Qfeats"
  "--Qhead"
  "--Qterm"
  "--Qoption"
  "--Qfeats --Qhead"
  "--Qfeats --Qterm"
  "--Qfeats --Qoption"
  "--Qhead --Qhead_scaling"
  "--Qfeats --Qhead --Qterm --Qoption"
)

NUM_OPTIONS=(
  3
  4
)

python plot_arch.py

for seed in {0..9}; do
  for env in "${ENVS[@]}"; do
    python random_run.py --env "${env}" --seed "${seed}"
  done
done

for seed in {0..9}; do
  for model in "${MODELS[@]}"; do
    for env in "${ENVS[@]}"; do
      python option_critic_run.py --env "${env}" ${model} --seed "${seed}"
    done
  done
done

for seed in {0..9}; do
  for model in "${MODELS[@]}"; do
    for env in "${ENVS[@]}"; do
      for num_options in "${NUM_OPTIONS[@]}"; do
        python option_critic_run.py --env "${env}" ${model} --num-options "${num_options}" --seed "${seed}" --exp="-Op${num_options}"
      done
    done
  done
done

python plot.py