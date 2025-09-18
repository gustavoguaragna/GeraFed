"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import os
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from Simulation.GeraFed_F2U.task import Net, CGAN, F2U_GAN, get_weights
from Simulation.GeraFed_F2U.strategy import GeraFed
from typing import List, Tuple
from flwr.common.typing import Metrics
import torch

# import random
# import numpy as np
# import torch
# from collections import Counter
 
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)
#     # Para garantir determinismo total em operações com CUDA
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    d_losses = [num_examples * m["avg_d_loss"] for num_examples, m in metrics]
    net_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "d_loss_chunk": sum(d_losses) / sum(examples),
        "net_loss_chunk": sum(net_losses) / sum(examples)
            }


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num_rodadas"]
    fraction_fit_alvo = context.run_config["fraction_fit_alvo"]
    fraction_fit_gen = context.run_config["fraction_fit_gen"]
    dataset = context.run_config["dataset"]
    img_size = context.run_config["tam_img"]
    latent_dim = context.run_config["tam_ruido"]
    gan_arq = context.run_config["gan_arq"]
    teste = context.run_config["teste"]
    folder = context.run_config["Exp_name_folder"]
    num_chunks = context.run_config["num_chunks"]
    os.makedirs(folder, exist_ok=True)
    continue_epoch = context.run_config["continue_epoch"]
    
    # Initialize model parameters
    classifier = Net()

    if gan_arq == "simple_cnn":
        gen = CGAN(
                dataset=dataset,
                img_size=img_size,
                latent_dim=latent_dim
                )
    elif gan_arq == "f2u_gan":
        gen = F2U_GAN(
                dataset=dataset,
                img_size=img_size,
                latent_dim=latent_dim
                )
        
    optimGstate_dict = None

    if continue_epoch != 0:
        checkpoint = torch.load(f"{folder}/checkpoint_epoch{continue_epoch}.pth")
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        gen.load_state_dict(checkpoint['gen_state_dict'])
        optimGstate_dict = checkpoint['optim_G_state_dict']

    

    ndarrays_alvo = get_weights(classifier)
    # ndarrays_gen  = get_weights(gen)
    parameters_alvo = ndarrays_to_parameters(ndarrays_alvo)
    # parameters_gen = ndarrays_to_parameters(ndarrays_gen)

    # Define strategy
    strategy = GeraFed(
        fraction_fit_alvo=fraction_fit_alvo,
        fraction_fit_gen=fraction_fit_gen,
        fraction_evaluate_alvo=1.0,
        initial_parameters_alvo=parameters_alvo,
        gen=gen,
        optimG_state_dict=optimGstate_dict,
        dataset=dataset,
        img_size=img_size,
        latent_dim=latent_dim,
        gan_arq=gan_arq,
        teste=teste,
        folder=folder,
        num_chunks=num_chunks,
        fit_metrics_aggregation_fn=weighted_average,
        continue_epoch=continue_epoch
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
