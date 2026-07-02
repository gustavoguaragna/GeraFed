#!/usr/bin/env bash
set -euo pipefail

CVAE_EPOCHS=(
  "70"
  "100"
)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

# Optional extra config can be passed without editing this script, for example:
# RUN_CONFIG_EXTRA="teste=true num_clients=4" ./run_experiments.sh
RUN_CONFIG_EXTRA="${RUN_CONFIG_EXTRA:-}"

for epocas_gen in "${CVAE_EPOCHS[@]}"; do
  echo "============================================================"
  echo "Starting experiment for epocas_gen: ${epocas_gen}"
  echo "============================================================"

  run_config="epocas_gen=${epocas_gen}"
  if [[ -n "$RUN_CONFIG_EXTRA" ]]; then
    run_config="${run_config} ${RUN_CONFIG_EXTRA}"
  fi

  exp_name="epocas_gen_${epocas_gen}"

  if flwr run . --stream --run-config "$run_config" 2>&1 | tee "logs/${exp_name}.log"; then
    echo "Experiment for epocas_gen: ${epocas_gen} completed successfully."
  else
    echo "Experiment for epocas_gen: ${epocas_gen} failed."
  fi

  echo "Finished experiment for epocas_gen: ${epocas_gen}"
  echo
done

echo "All experiments finished."
