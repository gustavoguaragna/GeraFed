"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import torch
from Simulation.task import CGAN, get_weights, load_data, set_weights, test, train

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


class CGANClient(NumPyClient):
    def __init__(self, trainloader, testloader, local_epochs, learning_rate, dataset, img_size, latent_dim):
        self.latent_dim = latent_dim
        self.net = CGAN(dataset=dataset, img_size=img_size, latent_dim=self.latent_dim)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # cudnn.benchmark = True
        self.dataset = dataset

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        train(
            self.net,
            self.trainloader,
            epochs=self.local_epochs,
            learning_rate=self.lr,
            device=self.device,
            dataset=self.dataset,
            latent_dim=self.latent_dim
        )
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        g_loss, d_loss = test(self.net, self.testloader, self.device, dataset=self.dataset)
        return g_loss, len(self.testloader), {}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read the run_config to fetch hyperparameters relevant to this run
    dataset = context.run_config["dataset"]  # Novo parâmetro
    img_size = context.run_config["tam_img"]
    batch_size = context.run_config["tam_batch"]
    trainloader, testloader = load_data(partition_id, num_partitions, dataset=dataset, img_size=img_size, batch_size=batch_size)
    local_epochs = context.run_config["epocas_gen"]
    learning_rate = context.run_config["learn_rate_gen"]
    noise_dim = context.run_config["tam_ruido"]

    return CGANClient(trainloader, testloader, local_epochs, learning_rate, dataset, img_size=img_size, latent_dim=noise_dim).to_client()


app = ClientApp(client_fn=client_fn)
