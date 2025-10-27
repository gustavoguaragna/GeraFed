"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from Simulation.CNN.task import (
    Net, 
    Net_CIFAR,
    get_weights, 
    load_data, 
    set_weights, 
    test, 
    train
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs, dataset, lr):
        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.dataset = dataset
        self.lr = lr

    def fit(self, parameters, config):

        # Atualiza pesos do modelo classificador
        set_weights(self.net, parameters)

         # Define o dataloader
        if isinstance(self.trainloader, list):
            chunk_idx = config["round"] % len(self.trainloader)
            trainloader_chunk = self.trainloader[chunk_idx]
        else:
            trainloader_chunk = self.trainloader

        train_loss = train(
            net=self.net,
            trainloader=trainloader_chunk,
            epochs=self.local_epochs,
            device=self.device,
            dataset=self.dataset,
            lr=self.lr
        )

        return (
            get_weights(self.net),
            len(trainloader_chunk.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device, dataset=self.dataset)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partitioner    = context.run_config["partitioner"]
    if partitioner == "Dir01":
        alpha_dir       = 0.1
    elif partitioner == "Dir05":
        alpha_dir       = 0.5
    else:
        alpha_dir       = None
    batch_size     = context.run_config["tam_batch"]
    local_epochs   = context.run_config["epocas_alvo"]
    dataset        = context.run_config["dataset"]
    seed           = context.run_config["seed"]
    teste          = context.run_config["teste"]
    num_chunks     = context.run_config["num_chunks"]
    lr             = context.run_config["learn_rate_alvo"]

    if dataset == "mnist":
        net = Net(seed=seed)
    elif dataset == "cifar10":
        net = Net_CIFAR(seed=seed)
    else:
        raise ValueError(f"Dataset {dataset} nao identificado. Deveria ser 'mnist' ou 'cifar10'")

    trainloader, testloader, testloader_local = load_data(partition_id=partition_id,
                                       num_partitions=num_partitions,
                                       dataset=dataset,
                                       teste=teste,
                                       partitioner_type=partitioner,
                                       num_chunks=num_chunks,
                                       batch_size=batch_size
                                      )

    # Return Client instance
    return FlowerClient(net, 
                        trainloader, 
                        testloader, 
                        local_epochs,
                        dataset=dataset,
                        lr=lr).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)

