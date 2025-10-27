"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from Simulation.CNN.task import Net, get_weights
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
        "net_loss_chunk": sum(net_losses) / sum(examples)
            }

def server_fn(context: Context):
    # Read from config
    num_rounds   = context.run_config["num_rodadas"]
    fraction_fit = context.run_config["fraction_fit_alvo"]
    seed         = context.run_config["seed"]
    num_chunks   = context.run_config["num_chunks"]
    dataset      = context.run_config["dataset"]
    folder       = f"{context.run_config['Exp_name_folder']}CNN/{dataset}/{context.run_config['partitioner']}/{context.run_config['strategy']}/{context.run_config['num_clients']}_clients"
    os.makedirs(folder, exist_ok=True)

    # Initialize model parameters
    ndarrays = get_weights(Net(seed=seed))
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = GeraFed(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        num_chunks=num_chunks,
        folder=folder,
        dataset=dataset
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
