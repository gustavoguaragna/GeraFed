#!/usr/bin/env bash
set -euo pipefail

LESSLVLS=(
  "true"
  "false"
)

SEEDS=(
  "42"
  "30"
  "20"
)

DATASETS=(
  "mnist"
  "cifar10"
  "tissuemnist"
  "bloodmnist"
  "octmnist"
  "organsmnist"
  "pathmnist"
)

PARTITIONERS=(
  "Class"
  "Dir01"
  "Dir05"
)

BASELINES=(
  "false"
  "true"
)

CLASSIFIER_OPTIMIZERS=(
  "sgd"
  "adam"
)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

# Optional extra config can be passed without editing this script, for example:
# RUN_CONFIG_EXTRA="teste=true num_clients=4" ./run_experiments.sh
RUN_CONFIG_EXTRA="${RUN_CONFIG_EXTRA:-}"

should_skip_experiment() {
  local lesslvl="$1"
  local seed="$2"
  local dataset="$3"
  local partitioner="$4"
  local baseline="$5"
  local classifier_optimizer="$6"

  if [[ "$lesslvl" == "false" && "$baseline" == "true" ]]; then
    return 0
  fi

  if [[ "$seed" != "42" ]]; then
    return 1
  fi

  if [[ "$dataset" == "mnist" || "$dataset" == "cifar10" ]]; then
    if [[ "$partitioner" == "Class" || "$partitioner" == "Dir01" ]]; then
      return 0
    fi
    return 1
  fi

  if [[ "$partitioner" == "Class" && "$classifier_optimizer" == "adam" ]]; then
    return 0
  fi
  if [[ "$partitioner" == "Dir01" && "$classifier_optimizer" == "sgd" ]]; then
    return 0
  fi

  return 1
}

for lesslvl in "${LESSLVLS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for partitioner in "${PARTITIONERS[@]}"; do
        for baseline in "${BASELINES[@]}"; do
          for classifier_optimizer in "${CLASSIFIER_OPTIMIZERS[@]}"; do
            if should_skip_experiment "$lesslvl" "$seed" "$dataset" "$partitioner" "$baseline" "$classifier_optimizer"; then
              echo "Skipping lesslvl=${lesslvl}, seed=${seed}, dataset=${dataset}, partitioner=${partitioner}, baseline=${baseline}, classifier_optimizer=${classifier_optimizer}"
              continue
            fi

            echo "============================================================"
            echo "Starting experiment: lesslvl=${lesslvl}, seed=${seed}, dataset=${dataset}, partitioner=${partitioner}, baseline=${baseline}, classifier_optimizer=${classifier_optimizer}"
            echo "============================================================"

            run_config="lesslvl=${lesslvl} seed=${seed} dataset='${dataset}' partitioner='${partitioner}' baseline=${baseline} classifier_optimizer='${classifier_optimizer}'"
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
            exp_name="lesslvl_${lesslvl}_seed_${seed}_${dataset}_${partitioner}_${method}_${classifier_optimizer}"

            if flwr run . --stream --run-config "$run_config" 2>&1 | tee "logs/${exp_name}.log"; then
              echo "Experiment completed: lesslvl=${lesslvl}, seed=${seed}, dataset=${dataset}, partitioner=${partitioner}, baseline=${baseline}, classifier_optimizer=${classifier_optimizer}"
            else
              echo "Experiment failed: lesslvl=${lesslvl}, seed=${seed}, dataset=${dataset}, partitioner=${partitioner}, baseline=${baseline}, classifier_optimizer=${classifier_optimizer}"
            fi

            echo "Finished experiment: lesslvl=${lesslvl}, seed=${seed}, dataset=${dataset}, partitioner=${partitioner}, baseline=${baseline}, classifier_optimizer=${classifier_optimizer}"
            echo
          done
        done
      done
    done
  done
done

echo "All experiments finished."
