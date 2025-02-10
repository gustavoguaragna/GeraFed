import torch
from ctgan import CTGAN
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays

from Simulation.CTGAN.task import load_data

class CTGANClient(NumPyClient):
    def __init__(self, data, num_train):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CTGAN(epochs=5)  # Define a CTGAN local
        self.data = data
        self.num_train = num_train

    def fit(self, parameters, config):
        """Treina a CTGAN no cliente e retorna os novos pesos."""
        self.model.load_state_dict(parameters_to_ndarrays(parameters))
        self.model.fit(self.data)  # Treina a CTGAN com os dados locais
        return ndarrays_to_parameters(self.model.state_dict()), self.num_train, {}

    def evaluate(self, parameters, config):
        """Avalia a qualidade dos dados sintéticos gerados."""
        self.model.load_state_dict(parameters_to_ndarrays(parameters))
        synthetic_data = self.model.sample(1000)  # Gerar dados sintéticos
        return 0.0, len(synthetic_data), {}  # Retorna uma métrica fictícia
    
def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    niid = context.run_config["niid"]
    alpha_dir = context.run_config["alpha_dir"]
    data, _, num_train, _, _ = load_data(
        partition_id=partition_id, num_clients=num_partitions, niid=niid, alpha_dir=alpha_dir
    )

    return(CTGANClient(data=data, num_train=num_train))


# Iniciar Cliente Federado
app = ClientApp(client_fn=client_fn)
