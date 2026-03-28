#!/usr/bin/env bash
set -euo pipefail

ENVS=(
  "CartPole-v1"
  "Acrobot-v1"
)

MODELS=(
  ""  # Classical
  "--Qfeats"
  "--Qoption_value"
  "--Qterm"
  "--Qoption_policies"
  "--Qfeats --Qoption_value"
  "--Qfeats --Qterm"
  "--Qfeats --Qoption_policies"
  "--Qfeats --Qoption_value --Qterm --Qoption_policies"
)

# Main experiment

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

# classical with varying hidden neurons

HIDDEN_NEURONS=(
  16
  24
  32
)

for seed in {0..9}; do
  for hidden_neuron in "${HIDDEN_NEURONS[@]}"; do
    for env in "${ENVS[@]}"; do
      python option_critic_run.py --env "${env}" --seed "${seed}" --hidden_neuron "${hidden_neuron}" --exp="-hn${hidden_neuron}"
    done
  done
done

# Ablation study: varying the number of options

MODELS_MORE_OPTIONS=(
  ""  # Classical
  "--Qfeats --Qoption_policies"
)

NUM_OPTIONS=(
  3
  4
)

for seed in {0..9}; do
  for model in "${MODELS_MORE_OPTIONS[@]}"; do
    for env in "${ENVS[@]}"; do
      for num_options in "${NUM_OPTIONS[@]}"; do
        python option_critic_run.py --env "${env}" ${model} --num-options "${num_options}" --seed "${seed}" --exp="-Op${num_options}"
      done
    done
  done
done

# Ablation study: no learnable input scaling, no entanglement, and varying number of layers in Quantum Feature Extractor

MODELS_ABLATION=(
  "--Qfeats --no_scaling"
  "--Qfeats --no_entanglement"
)

for env in "${ENVS[@]}"; do
  for model in "${MODELS_ABLATION[@]}"; do
    for seed in {0..9}; do
    if [ "${env}" == "Acrobot-v1" ]; then
      LAYER_H=2
      LAYER_F=5
    else
      LAYER_F=4
      LAYER_H=1
    fi
    python option_critic_run.py --env "${env}" ${model} --seed "${seed}" --layer_F "${LAYER_F}" --layer_H "${LAYER_H}" 
    done
  done
done

LAYERS_CartPole=(
  "2"
  "6"
)

LAYERS_Acrobot=(
  "3"
  "7"
)

for env in "${ENVS[@]}"; do
  if [ "${env}" == "Acrobot-v1" ]; then
    LAYERS=("${LAYERS_Acrobot[@]}")
  elif [ "${env}" == "CartPole-v1" ]; then
    LAYERS=("${LAYERS_CartPole[@]}")
  fi
  for layer_f in "${LAYERS[@]}"; do
    for seed in {0..9}; do
      python option_critic_run.py --env "${env}" --exp="-${layer_f}layers" --Qfeats --seed "${seed}" --logdir="runs_ab"
    done
  done
done

python plot.py