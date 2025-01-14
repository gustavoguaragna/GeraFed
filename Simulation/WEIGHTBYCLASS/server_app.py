"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from Simulation.WEIGHTBYCLASS.task import Net, get_weights
from Simulation.WEIGHTBYCLASS.strategy import GeraFed


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num_rodadas"]
    fraction_fit = context.run_config["fraction_fit_alvo"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = GeraFed(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
