#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-option_critic_run.py}
COMMON_ARGS=${COMMON_ARGS:-""}  # e.g. "--env CartPole-v1 --seed 0"

runs=(
  # ""
  "--Qoption"
  "--Qhead"
  "--Qfeats"
  "--Qterm"
  # "--Qfeats --Qhead"
  # "--Qfeats --Qoption"
  # "--Qhead --Qoption"
  # "--Qfeats --Qhead --Qoption"
)

for flags in "${runs[@]}"; do
  echo "Running: ${PYTHON} ${SCRIPT} ${COMMON_ARGS} ${flags}"
  # shellcheck disable=SC2086
  ${PYTHON} ${SCRIPT} ${COMMON_ARGS} ${flags}
done

python plot.py