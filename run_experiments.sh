#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  "mnist"
  "cifar10"
  "breastmnist"
  "organsmnist"
  "skinl_derm"
  "organs_axial"
)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

# Optional extra config can be passed without editing this script, for example:
# RUN_CONFIG_EXTRA="teste=true num_clients=4" ./run_experiments.sh
RUN_CONFIG_EXTRA="${RUN_CONFIG_EXTRA:-}"

for dataset in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "Starting experiment for dataset: ${dataset}"
  echo "============================================================"

  run_config="dataset='${dataset}'"
  if [[ -n "$RUN_CONFIG_EXTRA" ]]; then
    run_config="${run_config} ${RUN_CONFIG_EXTRA}"
  fi

  exp_name="${dataset}"
  
  if flwr run . --stream --run-config "$run_config" 2>&1 | tee "logs/${exp_name}.log"; then
    echo "Experiment for dataset: ${dataset} completed successfully."
  else
    echo "Experiment for dataset: ${dataset} failed."
  fi

  echo "Finished experiment for dataset: ${dataset}"
  echo
done

echo "All experiments finished."
