#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  "tissuemnist"
  "pathmnist"
  "organsmnist"
  "octmnist"
  "bloodmnist"
)

MEDMNIST_SIZES=(
  "64"
  "128"
  "224"
)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

# Optional extra config can be passed without editing this script, for example:
# RUN_CONFIG_EXTRA="teste=true num_clients=4" ./run_experiments.sh
RUN_CONFIG_EXTRA="${RUN_CONFIG_EXTRA:-}"

for dataset in "${DATASETS[@]}"; do
  for medmnist_size in "${MEDMNIST_SIZES[@]}"; do
    echo "============================================================"
    echo "Starting experiment for dataset: ${dataset}, medmnist_size: ${medmnist_size}"
    echo "============================================================"

    run_config="dataset='${dataset}' medmnist_size=${medmnist_size}"
    if [[ -n "$RUN_CONFIG_EXTRA" ]]; then
      run_config="${run_config} ${RUN_CONFIG_EXTRA}"
    fi

    exp_name="${dataset}_size_${medmnist_size}"

    if flwr run . --stream --run-config "$run_config" 2>&1 | tee "logs/${exp_name}.log"; then
      echo "Experiment for dataset: ${dataset}, medmnist_size: ${medmnist_size} completed successfully."
    else
      echo "Experiment for dataset: ${dataset}, medmnist_size: ${medmnist_size} failed."
    fi

    echo "Finished experiment for dataset: ${dataset}, medmnist_size: ${medmnist_size}"
    echo
  done
done

echo "All experiments finished."
