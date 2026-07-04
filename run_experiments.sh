#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  "bloodmnist"
  "octmnist"
  "organamnist"
  "organcmnist"
  "organsmnist"
  "pathmnist"
)

CVAE_EPOCHS=(
  "40"
  "70"
  "100"
)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

# Optional extra config can be passed without editing this script, for example:
# RUN_CONFIG_EXTRA="teste=true num_clients=4" ./run_experiments.sh
RUN_CONFIG_EXTRA="${RUN_CONFIG_EXTRA:-}"

for dataset in "${DATASETS[@]}"; do
  for epocas_gen in "${CVAE_EPOCHS[@]}"; do
    echo "============================================================"
    echo "Starting experiment for dataset: ${dataset}, epocas_gen: ${epocas_gen}"
    echo "============================================================"

    run_config="dataset='${dataset}' epocas_gen=${epocas_gen}"
    if [[ -n "$RUN_CONFIG_EXTRA" ]]; then
      run_config="${run_config} ${RUN_CONFIG_EXTRA}"
    fi

    exp_name="${dataset}_epocas_gen_${epocas_gen}"

    if flwr run . --stream --run-config "$run_config" 2>&1 | tee "logs/${exp_name}.log"; then
      echo "Experiment for dataset: ${dataset}, epocas_gen: ${epocas_gen} completed successfully."
    else
      echo "Experiment for dataset: ${dataset}, epocas_gen: ${epocas_gen} failed."
    fi

    echo "Finished experiment for dataset: ${dataset}, epocas_gen: ${epocas_gen}"
    echo
  done
done

echo "All experiments finished."
