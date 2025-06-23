"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from Simulation.Multi_Models.task import Net, CGAN, get_weights
from Simulation.Multi_Models.strategy import GeraFed
import random
import numpy as np
import torch
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Para garantir determinismo total em operações com CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num_rodadas"]
    dataset = str(context.run_config["dataset"])
    img_size = context.run_config["tam_img"]
    latent_dim = context.run_config["tam_ruido"]
    teste = context.run_config["teste"]

    # Initialize model parameters
    ndarrays_alvo = get_weights(Net())
    parameters_alvo = ndarrays_to_parameters(ndarrays_alvo)
    ndarrays_gen = get_weights(CGAN(dataset=dataset,      #type: ignore
                                    img_size=img_size,
                                    latent_dim=latent_dim))
    parameters_gen = ndarrays_to_parameters(ndarrays_gen)

    client_counter = Counter()

    # Define strategy
    strategy = GeraFed(
        fraction_evaluate_alvo=1.0,
        initial_parameters_alvo=parameters_alvo,
        initial_parameters_gen=parameters_gen,
        dataset=dataset,
        img_size=img_size,
        latent_dim=latent_dim,
        client_counter=client_counter,
        teste=teste
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
