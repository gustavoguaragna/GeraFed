"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, array_from_numpy
from Simulation.task import Net, CGAN, get_weights, load_data, set_weights, test, train_alvo, train_gen

import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Para garantir determinismo total em operações com CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net_alvo, net_gen, trainloader, valloader, local_epochs_alvo, local_epochs_gen, dataset, lr_alvo, lr_gen, latent_dim, context: Context):
        self.net_alvo = net_alvo
        self.net_gen = net_gen
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs_alvo = local_epochs_alvo
        self.local_epochs_gen = local_epochs_gen
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_alvo.to(self.device)
        self.net_gen.to(self.device)
        self.dataset = dataset
        self.lr_alvo = lr_alvo
        self.lr_gen = lr_gen
        self.latent_dim = latent_dim
        self.client_state = (
            context.state
        ) 


    def fit(self, parameters, config):
        if config["modelo"] == "alvo":
            set_weights(self.net_alvo, parameters)
            train_loss = train_alvo(
                net=self.net_alvo,
                trainloader=self.trainloader,
                epochs=self.local_epochs_alvo,
                lr=self.lr_alvo,
                device=self.device,
            )
            return (
                get_weights(self.net_alvo),
                len(self.trainloader.dataset),
                {"train_loss": train_loss, "modelo": "alvo"},
            )
        elif config["modelo"] == "gen":
            set_weights(self.net_gen, parameters)
            train_loss = train_gen(
                net=self.net_gen,
                trainloader=self.trainloader,
                epochs=self.local_epochs_gen,
                lr=self.lr_gen,
                device=self.device,
                dataset=self.dataset,
                latent_dim=self.latent_dim
            )
            p_record = ParametersRecord()
            for k, v in self.net_gen.state_dict().items():
                p_record[k] = array_from_numpy(v.detach().cpu().numpy())
            self.client_state.parameters_records["net_parameters"] = p_record
            return (
                get_weights(self.net_gen),
                len(self.trainloader.dataset),
                {"train_loss": train_loss, "modelo": "gen"},
            )

    def evaluate(self, parameters, config):
        set_weights(self.net_alvo, parameters)
        loss, accuracy = test(self.net_alvo, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    dataset = context.run_config["dataset"]
    img_size = context.run_config["tam_img"]
    latent_dim = context.run_config["tam_ruido"]
    net_gen = CGAN(dataset=dataset, img_size=img_size, latent_dim=latent_dim)
    net_alvo = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    niid = context.run_config["niid"]
    alpha_dir = context.run_config["alpha_dir"]
    batch_size = context.run_config["tam_batch"]
    trainloader, valloader = load_data(partition_id=partition_id,
                                       num_partitions=num_partitions,
                                       niid=niid,
                                       alpha_dir=alpha_dir,
                                       batch_size=batch_size
                                      )
    local_epochs_alvo = context.run_config["epocas_alvo"]
    local_epochs_gen = context.run_config["epocas_gen"]
    lr_gen = context.run_config["learn_rate_gen"]
    lr_alvo = context.run_config["learn_rate_alvo"]
    latent_dim = context.run_config["tam_ruido"]

    # Return Client instance
    return FlowerClient(net_alvo=net_alvo, 
                        net_gen=net_gen, 
                        trainloader=trainloader, 
                        valloader=valloader, 
                        local_epochs_alvo=local_epochs_alvo, 
                        local_epochs_gen=local_epochs_gen,
                        dataset=dataset,
                        lr_alvo=lr_alvo,
                        lr_gen=lr_gen,
                        latent_dim=latent_dim,
                        context=context).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn
)
