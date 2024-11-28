# client_app.py

"""fedvaeexample: A Flower / PyTorch app for Federated Variational Autoencoder."""

import torch
from fedvaeexample.task import Net, get_weights, load_data, set_weights, test, train

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


class FedVaeClient(NumPyClient):
    def __init__(self, trainloader, testloader, local_epochs, learning_rate, dataset):
        self.net = Net(dataset=dataset)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            dataset=self.dataset
        )
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss = test(self.net, self.testloader, self.device, dataset=self.dataset)
        return float(loss), len(self.testloader), {}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read the run_config to fetch hyperparameters relevant to this run
    dataset = context.run_config["dataset"]  # Novo parâmetro
    trainloader, testloader = load_data(partition_id, num_partitions, dataset=dataset)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    return FedVaeClient(trainloader, testloader, local_epochs, learning_rate, dataset).to_client()


app = ClientApp(client_fn=client_fn)
