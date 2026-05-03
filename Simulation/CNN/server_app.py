"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from Simulation.CNN.task import Net, Net_CIFAR, get_weights, load_data
from Simulation.CNN.strategy import GeraFed
from typing import List, Tuple
from flwr.common.typing import Metrics
import os

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    net_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "net_loss_epoch": sum(net_losses) / sum(examples)
            }

def server_fn(context: Context):
    # Read from config
    num_chunks   = context.run_config["num_chunks"]
    try:
        num_rounds = context.run_config["num_epocas"] * num_chunks
    except KeyError:
        num_rounds = context.run_config["num_rodadas"]
    fraction_fit = context.run_config["fraction_fit_alvo"]
    seed         = context.run_config["seed"]
    dataset      = context.run_config["dataset"]
    teste        = context.run_config["teste"]
    batch_size   = context.run_config["tam_batch"]
    num_clients  = context.run_config["num_clients"]
    partitioner  = context.run_config["partitioner"]
    if partitioner == "Dir01":
        alpha_dir = 0.1
    elif partitioner == "Dir05":
        alpha_dir = 0.5
    else:
        alpha_dir = None
    folder       = f"{context.run_config['Exp_name_folder']}CNN/{dataset}/{partitioner}/{context.run_config['strategy']}/{num_clients}_clients"
    os.makedirs(folder, exist_ok=True)

    # Initialize model parameters
    if dataset == "mnist":
        classifier = Net(seed=seed)
    elif dataset == "cifar10":
        classifier = Net_CIFAR(seed=seed)
    else:
        raise ValueError(f"Dataset {dataset} nao identificado. Deveria ser 'mnist' ou 'cifar10'")

    ndarrays = get_weights(classifier)
    parameters = ndarrays_to_parameters(ndarrays)

    _, valloader, _ = load_data(
        partition_id=0,
        num_partitions=num_clients,
        batch_size=batch_size,
        dataset=dataset,
        teste=teste,
        partitioner_type=partitioner,
        num_chunks=num_chunks,
        alpha_dir=alpha_dir
    )

    # Define strategy
    strategy = GeraFed(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=weighted_average,
        num_chunks=num_chunks,
        folder=folder,
        dataset=dataset,
        valloader=valloader,
        model_save_interval=10,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
