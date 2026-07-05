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

BASELINES=(
  "false"
  "true"
)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

# Optional extra config can be passed without editing this script, for example:
# RUN_CONFIG_EXTRA="teste=true num_clients=4" ./run_experiments.sh
RUN_CONFIG_EXTRA="${RUN_CONFIG_EXTRA:-}"

for dataset in "${DATASETS[@]}"; do
  for medmnist_size in "${MEDMNIST_SIZES[@]}"; do
    for baseline in "${BASELINES[@]}"; do
      echo "============================================================"
      echo "Starting experiment for dataset: ${dataset}, medmnist_size: ${medmnist_size}, baseline: ${baseline}"
      echo "============================================================"

      run_config="dataset='${dataset}' medmnist_size=${medmnist_size} baseline=${baseline}"
      if [[ "$baseline" == "true" ]]; then
        run_config="${run_config} patience=50"
      fi
      if [[ -n "$RUN_CONFIG_EXTRA" ]]; then
        run_config="${run_config} ${RUN_CONFIG_EXTRA}"
      fi

      method="fleg"
      if [[ "$baseline" == "true" ]]; then
        method="baseline"
      fi
      exp_name="${dataset}_size_${medmnist_size}_${method}"

      if flwr run . --stream --run-config "$run_config" 2>&1 | tee "logs/${exp_name}.log"; then
        echo "Experiment for dataset: ${dataset}, medmnist_size: ${medmnist_size}, baseline: ${baseline} completed successfully."
      else
        echo "Experiment for dataset: ${dataset}, medmnist_size: ${medmnist_size}, baseline: ${baseline} failed."
      fi

      echo "Finished experiment for dataset: ${dataset}, medmnist_size: ${medmnist_size}, baseline: ${baseline}"
      echo
    done
  done
done

echo "All experiments finished."
