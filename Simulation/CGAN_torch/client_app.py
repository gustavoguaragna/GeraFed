"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import torch
from Simulation.CGAN_torch.task import CGAN, get_weights, get_weights_gen, load_data, set_weights, test, train, generate_images
from collections import OrderedDict
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common import Context, ParametersRecord, array_from_numpy

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

class CGANClient(NumPyClient):
    def __init__(self, 
                 cid,
                 trainloader,
                 testloader,
                 local_epochs, 
                 learning_rate, 
                 dataset, 
                 latent_dim,
                 agg,
                 batch_size):
        self.cid = cid
        self.latent_dim = latent_dim
        self.net = CGAN(dataset=dataset, latent_dim=self.latent_dim)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"DEVICE CLIENT: {self.device}")
        # cudnn.benchmark = True
        self.dataset = dataset
        self.agg = agg
        self.batch_size = batch_size

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        if self.agg == "full":
            set_weights(self.net, parameters)
            # Gera imagens do modelo agregado do round anterior
            if self.cid == 0:
                figura = generate_images(net=self.net)
                figura.savefig(f"mnist_CGAN_r{config['server_round']-1}_{self.local_epochs}e_{self.batch_size}_100z_10c_{self.lr}lr_niid_01dir.png")
            train_loss = train(
            net=self.net,
            trainloader=self.trainloader,
            epochs=self.local_epochs,
            lr=self.lr,
            device=self.device,
            dataset=self.dataset,
            latent_dim=self.latent_dim
        )
            figura = generate_images(net=self.net)
            figura.savefig(f"mnist_CGAN_r{config['server_round']}_{self.local_epochs}e_{self.batch_size}_100z_10c_{self.lr}lr_niid_01dir_cliente{self.cid}.png")
            return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "modelo": "gen"},
        )

        elif self.agg == "disc":
            # Supondo que net seja o modelo e parameters seja a lista de parâmetros fornecida
            state_keys = [k for k in self.net.state_dict().keys() if 'generator' not in k]
        
            # Criando o OrderedDict com as chaves filtradas e os parâmetros fornecidos
            disc_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(state_keys, parameters)})

            state_dict = {}
            # Extract record from context
            if "net_parameters" in self.client_state.parameters_records:
                p_record = self.client_state.parameters_records["net_parameters"]

                # Deserialize arrays
                for k, v in p_record.items():
                    state_dict[k] = torch.from_numpy(v.numpy())

                # Apply state dict to a new model instance
                model_ = CGAN()
                model_.load_state_dict(state_dict)

                new_state_dict = {}

                for name, param in self.net.state_dict().items():
                    if 'generator' in name:
                        new_state_dict[name] = model_.state_dict()[name]
                    elif 'discriminator' in name or 'label' in name:
                        new_state_dict[name] = disc_dict[name]
                    else:
                        new_state_dict[name] = param

                self.net.load_state_dict(new_state_dict)
            
            figura = generate_images(net=self.net)
            figura.savefig(f"mnist_CGAN_r{config['server_round']-1}_{self.local_epochs}e_{self.batch_size}_100z_10c_{self.lr}lr_niid_01dir_cliente{self.cid}.png")

            train_loss = train(
                net=self.net,
                trainloader=self.trainloader,
                epochs=self.local_epochs,
                lr=self.lr,
                device=self.device,
                dataset=self.dataset,
                latent_dim=self.latent_dim
            )
            # Save all elements of the state_dict into a single RecordSet
            p_record = ParametersRecord()
            for k, v in self.net.state_dict().items():
                # Convert to NumPy, then to Array. Add to record
                p_record[k] = array_from_numpy(v.detach().cpu().numpy())
            # Add to a context
            self.client_state.parameters_records["net_parameters"] = p_record

            model_path = f"modelo_gen_round_{config['round']}_client_{self.cid}.pt"
            torch.save(self.net.state_dict(), model_path)

            figura = generate_images(net=self.net)
            figura.savefig(f"mnist_CGAN_r{config['server_round']}_{self.local_epochs}e_{self.batch_size}_100z_10c_{self.lr}lr_niid_01dir_cliente{self.cid}.png")

            return (
                get_weights_gen(self.net),
                len(self.trainloader.dataset),
                {"train_loss": train_loss, "modelo": "gen"},
            )


    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        print("ENTROU EVALUATE NO CLIENTE")
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
    batch_size = context.run_config["tam_batch"]
    niid = context.run_config["niid"]
    alpha_dir = context.run_config["alpha_dir"]
    trainloader, testloader = load_data(partition_id, 
                                        num_partitions, 
                                        dataset=dataset, 
                                        niid=niid,
                                        alpha_dir=alpha_dir, 
                                        batch_size=batch_size)
    local_epochs = context.run_config["epocas_gen"]
    learning_rate = context.run_config["learn_rate_gen"]
    noise_dim = context.run_config["tam_ruido"]
    agg = context.run_config["agg"]

    return CGANClient(cid=partition_id,
                      trainloader=trainloader, 
                      testloader=testloader, 
                      local_epochs=local_epochs, 
                      learning_rate=learning_rate, 
                      dataset=dataset, 
                      latent_dim=noise_dim,
                      agg=agg,
                      batch_size=batch_size).to_client()


app = ClientApp(client_fn=client_fn)
