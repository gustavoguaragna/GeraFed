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
    normalize_stop_criterion,
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
    data_root = _get(run_config, "data_root", "data")
    download_datasets = _get(run_config, "download_datasets", True)
    medmnist_size = int(_get(run_config, "medmnist_size", 224))
    cvae_epochs = _get(run_config, "epocas_gen", 25)
    cvae_local_epochs = _get(run_config, "epocas_locais_gen",1)
    patience = _get(run_config, "patience", 10)
    levels = _get(run_config, "levels", 4)
    syn_input = _get(run_config, "syn_input", "dynamic")
    latent_dim_mode = _get(run_config, "cvae_latent_dim_mode", "fixed")
    stop_criterion = normalize_stop_criterion(
        _get(run_config, "criterio_parada", "global_test_acc")
    )
    fixed_classifier_rounds = int(_get(run_config, "num_epocas", patience))
    alpha_dir = _alpha_from_partitioner(partitioner, run_config)
    if seed == 42:
        trial = 1
    elif seed == 30:
        trial = 2
    elif seed == 20:
        trial = 3
    else:
        trial = seed

    dataset_folder_name = dataset
    if dataset.endswith("mnist") and dataset not in {"mnist"}:
        dataset_folder_name = f"{dataset}_size{medmnist_size}"
    baseline = _get(run_config, "baseline", False)
    method = "baseline" if baseline else "fleg"

    folder = (
        f"{_get(run_config, 'Exp_name_folder', 'Experimentos/Flwr_run/')}CVAE/"
        f"{dataset_folder_name}_{partitioner}_{strategy_name}_"
        f"cvaeepochs{cvae_epochs}_{syn_input}_{method}_trial{trial}"
    )
    os.makedirs(folder, exist_ok=True)

    _, valloader, _, _ = load_data(
        partition_id=0,
        num_partitions=num_clients,
        batch_size=batch_size,
        dataset=dataset,
        teste=teste,
        partitioner_type=partitioner,
        alpha_dir=alpha_dir,
        seed=seed,
        data_root=data_root,
        download_datasets=download_datasets,
        medmnist_size=medmnist_size,
    )

    classifier = create_full_model(dataset, seed=seed)
    parameters = ndarrays_to_parameters(get_weights(classifier))

    # Flower needs a finite upper bound. The strategy stops itself when patience/levels finish.
    classifier_rounds_bound = (
        fixed_classifier_rounds
        if stop_criterion == "fixed_rounds"
        else patience * 10
    )
    num_rounds = (
        classifier_rounds_bound * max(1, levels)
        + cvae_epochs * max(1, levels)
        + levels
        + 5
    )

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
        stop_criterion=stop_criterion,
        fixed_classifier_rounds=fixed_classifier_rounds,
        reset_best_metric_per_level=_get(
            run_config,
            "reset_best_metric_per_level",
            True,
        ),
        levels=levels,
        lesslvl=_get(run_config, "lesslvl", False),
        baseline=baseline,
        cvae_epochs=cvae_epochs,
        cvae_local_epochs=cvae_local_epochs,
        cvae_beta=_get(run_config, "cvae_beta", 1.0),
        cvae_lr=_get(run_config, "learn_rate_gen", 0.001),
        normalization=_get(run_config, "cvae_normalization", "minmax"),
        resblock=_get(run_config, "cvae_resblock", False),
        annealing=_get(run_config, "cvae_annealing", False),
        latent_dim_mode=latent_dim_mode,
        latent_dim=_get(run_config, "tam_ruido", 100),
        num_syn=syn_input,
        mixup_type=_get(run_config, "cvae_mixup_type", "none"),
        valloader=valloader,
        num_clients=num_clients,
        resume_from_checkpoint=_get(run_config, "resume_from_checkpoint", False),
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
