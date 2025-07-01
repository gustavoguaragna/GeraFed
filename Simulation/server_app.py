"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import os
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from Simulation.task import Net, CGAN, F2U_GAN,get_weights
from Simulation.strategy import GeraFed
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
    fraction_fit_alvo = context.run_config["fraction_fit_alvo"]
    fraction_fit_gen = context.run_config["fraction_fit_gen"]
    dataset = context.run_config["dataset"]
    img_size = context.run_config["tam_img"]
    latent_dim = context.run_config["tam_ruido"]
    agg = context.run_config["agg"]
    model = context.run_config["model"]
    gan_arq = context.run_config["gan_arq"]
    fid = context.run_config["fid"]
    teste = context.run_config["teste"]
    folder = context.run_config["Exp_name_folder"]
    os.makedirs(folder, exist_ok=True)
    # Initialize model parameters
    ndarrays_alvo = get_weights(Net())
    parameters_alvo = ndarrays_to_parameters(ndarrays_alvo)
    if gan_arq == "simple_cnn":
        ndarrays_gen = get_weights(CGAN(dataset=dataset,
                                        img_size=img_size,
                                        latent_dim=latent_dim))
    elif gan_arq == "f2u_gan":
        ndarrays_gen = get_weights(F2U_GAN(dataset=dataset,
                                            img_size=img_size,
                                            latent_dim=latent_dim))
    parameters_gen = ndarrays_to_parameters(ndarrays_gen)

    client_counter = Counter()

    # Define strategy
    strategy = GeraFed(
        fraction_fit_alvo=fraction_fit_alvo,
        fraction_fit_gen=fraction_fit_gen,
        fraction_evaluate_alvo=1.0,
        initial_parameters_alvo=parameters_alvo,
        initial_parameters_gen=parameters_gen,
        dataset=dataset,
        img_size=img_size,
        latent_dim=latent_dim,
        client_counter=client_counter,
        agg=agg,
        model=model,
        gan_arq=gan_arq,
        fid=fid,
        teste=teste,
        folder=folder
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
