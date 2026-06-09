"""Flower server app for the CVAE version of FLEG."""

from __future__ import annotations

import os

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from Simulation.CVAE.strategy import FLEG_CVAE
from Simulation.CVAE.task import (
    create_full_model,
    get_weights,
    load_data,
    normalize_dataset_name,
)


def _get(run_config, key, default=None):
    try:
        return run_config[key]
    except KeyError:
        return default


def _alpha_from_partitioner(partitioner: str, run_config):
    if partitioner in {"Dir01", "Dirichlet01"}:
        return 0.1
    if partitioner in {"Dir05", "Dirichlet05"}:
        return 0.5
    return None


def server_fn(context: Context):
    run_config = context.run_config

    dataset = normalize_dataset_name(_get(run_config, "dataset", "mnist"))
    partitioner = _get(run_config, "partitioner", "Class")
    strategy_name = _get(run_config, "strategy", "fedavg")
    seed = _get(run_config, "seed", 42)
    num_clients = _get(run_config, "num_clients", 10)
    batch_size = _get(run_config, "tam_batch", 32)
    teste = _get(run_config, "teste", False)
    cvae_epochs = _get(run_config, "epocas_gen", 25)
    patience = _get(run_config, "patience", 10)
    levels = _get(run_config, "levels", 4)
    syn_input = _get(run_config, "syn_input", "dynamic")
    alpha_dir = _alpha_from_partitioner(partitioner, run_config)
    if seed == 42:
        trial = 1
    elif seed == 30:
        trial = 2
    elif seed == 20:
        trial = 3
    else:
        trial = seed

    folder = (
        f"{_get(run_config, 'Exp_name_folder', 'Experimentos/Flwr_run/')}CVAE/"
        f"{dataset}_{partitioner}_{strategy_name}_"
        f"cvaeepochs{cvae_epochs}_{syn_input}_fleg_trial{trial}"
    )
    os.makedirs(folder, exist_ok=True)

    classifier = create_full_model(dataset, seed=seed)
    parameters = ndarrays_to_parameters(get_weights(classifier))

    _, valloader, _ = load_data(
        partition_id=0,
        num_partitions=num_clients,
        batch_size=batch_size,
        dataset=dataset,
        teste=teste,
        partitioner_type=partitioner,
        alpha_dir=alpha_dir,
        seed=seed,
    )

    # Flower needs a finite upper bound. The strategy stops itself when patience/levels finish.
    num_rounds = patience * max(1, levels) * 10 + cvae_epochs * max(1, levels) + levels + 5

    strategy = FLEG_CVAE(
        fraction_fit_alvo=_get(run_config, "fraction_fit_alvo", 1.0),
        fraction_fit_cvae=_get(run_config, "fraction_fit_gen", 1.0),
        fraction_evaluate_alvo=1.0,
        min_available_clients=min(2, num_clients),
        min_fit_clients=min(2, num_clients),
        min_evaluate_clients=min(2, num_clients),
        initial_parameters=parameters,
        dataset=dataset,
        folder=folder,
        strategy_name=strategy_name,
        mu=_get(run_config, "mu", 0.5),
        seed=seed,
        patience=patience,
        levels=levels,
        lesslvl=_get(run_config, "lesslvl", False),
        baseline=_get(run_config, "baseline", False),
        cvae_epochs=cvae_epochs,
        cvae_beta=_get(run_config, "cvae_beta", 1.0),
        cvae_lr=_get(run_config, "learn_rate_gen", 0.001),
        normalization=_get(run_config, "cvae_normalization", "minmax"),
        resblock=_get(run_config, "cvae_resblock", False),
        anealing=_get(run_config, "cvae_anealing", False),
        latent_dim_mode=syn_input,
        latent_dim=_get(run_config, "tam_ruido", 100),
        num_syn=syn_input,
        mixup_type=_get(run_config, "cvae_mixup_type", "none"),
        valloader=valloader,
        num_clients=num_clients,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
