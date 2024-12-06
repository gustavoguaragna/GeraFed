"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from Simulation.task import Net, CGAN, get_weights, load_data, set_weights, test, train_alvo, train_gen


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net_alvo, net_gen, trainloader, valloader, local_epochs_alvo, local_epochs_gen):
        self.net_alvo = net_alvo
        self.net_gen = net_gen
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs_alvo = local_epochs_alvo
        self.local_epochs_gen = local_epochs_gen
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_alvo.to(self.device)
        self.net_gen.to(self.device)

    def fit(self, parameters, config):
        if config["model"] == "alvo":
            set_weights(self.net_alvo, parameters)
            train_loss = train_alvo(
                self.net_alvo,
                self.trainloader,
                self.local_epochs_alvo,
                self.device,
            )
            return (
                get_weights(self.net_alvo),
                len(self.trainloader.dataset),
                {"train_loss": train_loss, "modelo": "alvo"},
            )
        elif config["model"] == "gen":
            set_weights(self.net_gen, parameters)
            train_loss = train_gen(
                self.net_gen,
                self.trainloader,
                self.local_epochs_gen,
                self.device,
            )
            return (
                get_weights(self.net_gen),
                len(self.trainloader.dataset),
                {"train_loss": train_loss, "modelo": "gen"},
            )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    dataset = context.run_config["dataset"]
    net_gen = CGAN(dataset=dataset)
    net_alvo = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs_alvo = context.run_config["epocas_alvo"]
    local_epochs_gen = context.run_congig["epocas_gen"]

    # Return Client instance
    return FlowerClient(net_gen, net_alvo, trainloader, valloader, local_epochs_alvo, local_epochs_gen).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
