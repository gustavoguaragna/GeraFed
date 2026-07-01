#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  "breastmnist"
  "organsmnist"
  "bloodmnist"
  "tissuemnist"
  "pathmnist"
  "octmnist"
  "dermamnist"
  "pneumoniamnist"
  "organamnist"
  "organcmnist"
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
  for baseline in "${BASELINES[@]}"; do
    echo "============================================================"
    echo "Starting experiment for dataset: ${dataset}, baseline: ${baseline}"
    echo "============================================================"

    run_config="dataset='${dataset}' baseline=${baseline}"
    if [[ "$baseline" == "true" ]]; then
      run_config="${run_config} patience=50"
    fi
    if [[ -n "$RUN_CONFIG_EXTRA" ]]; then
      run_config="${run_config} ${RUN_CONFIG_EXTRA}"
    fi

    exp_name="${dataset}_baseline_${baseline}"

    if flwr run . --stream --run-config "$run_config" 2>&1 | tee "logs/${exp_name}.log"; then
      echo "Experiment for dataset: ${dataset}, baseline: ${baseline} completed successfully."
    else
      echo "Experiment for dataset: ${dataset}, baseline: ${baseline} failed."
    fi

    echo "Finished experiment for dataset: ${dataset}, baseline: ${baseline}"
    echo
  done
done

echo "All experiments finished."
