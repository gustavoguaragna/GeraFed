"""FLEG: um framework para balancear dados heterogêneos em aprendizado federado, com precupações com a privacidade."""

from flwr.server import ServerApp, ServerConfig
from strategy import FlegStrategy
import torch

def server_fn(context):
    # Parâmetros de configuração
    num_partitions = 4
    dataset_name = "mnist" # ou "cifar10"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Estratégia FLEG
    strategy = FlegStrategy(
        num_partitions=num_partitions,
        dataset_name=dataset_name,
        device=device,
        total_levels=4,
        gan_epochs_per_level=5, # Reduzido para teste rápido; use 25 para full
        patience=3
    )

    config = ServerConfig(num_rounds=50) # Número alto de rounds, a Strategy para quando acabar os níveis

    return ServerApp(
        config=config,
        strategy=strategy,
    )

# Para rodar com flwr run:
# flwr run .


# Create ServerApp
app = ServerApp(server_fn=server_fn)
