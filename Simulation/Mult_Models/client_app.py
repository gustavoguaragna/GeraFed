"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, array_from_numpy
from Simulation.task import (
    Net, 
    CGAN, 
    get_weights, 
    load_data, 
    set_weights, 
    test, 
    train_alvo, 
    train_gen, 
    generate_plot
)
import random
import numpy as np
import pickle

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
    def __init__(self,
                cid: int, 
                net_alvo,
                net_gen, 
                trainloader, 
                valloader,
                local_epochs_alvo: int, 
                local_epochs_gen: int, 
                dataset: str, 
                lr_alvo: float, 
                lr_gen: float, 
                latent_dim: int, 
                context: Context,
                num_partitions: int,
                niid: bool,
                alpha_dir: float,
                batch_size: int,
                teste: bool):
        self.cid=cid
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_alvo = net_alvo.to(self.device)
        self.net_gen = net_gen.to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs_alvo = local_epochs_alvo
        self.local_epochs_gen = local_epochs_gen
        self.dataset = dataset
        self.lr_alvo = lr_alvo
        self.lr_gen = lr_gen
        self.latent_dim = latent_dim
        self.client_state = (
            context.state
        ) 
        self.num_partitions = num_partitions
        self.niid = niid
        self.alpha_dir = alpha_dir
        self.batch_size = batch_size
        self.teste = teste

    def fit(self, parameters, config):
        set_weights(self.net_alvo, parameters)
        train_loss = train_alvo(
            net=self.net_alvo,
            trainloader=self.trainloader,
            epochs=self.local_epochs_alvo,
            lr=self.lr_alvo,
            device=self.device,
        )

        state_dict = {}
        # Extract record from context
        if "net_parameters" in self.client_state.parameters_records:
            p_record = self.client_state.parameters_records["net_parameters"]

            # Deserialize arrays
            for k, v in p_record.items():
                state_dict[k] = torch.from_numpy(v.numpy())

            self.net_gen.load_state_dict(state_dict)

            figura = generate_plot(net=self.net_gen, device=self.device, round_number=config["round"])
            figura.savefig(f"mnist_CGAN_r{config['round']-1}_{self.local_epochs_gen}e_{self.batch_size}b_100z_4c_{self.lr_gen}lr_{'niid' if self.niid else 'iid'}_{self.alpha_dir if self.niid else ''}.png")

        train_gen(
            net=self.net_gen,
            trainloader=self.trainloader,
            epochs=self.local_epochs_gen,
            lr=self.lr_gen,
            device=self.device,
            dataset=self.dataset,
            latent_dim=self.latent_dim
        )
        # Save all elements of the state_dict into a single RecordSet
        p_record = ParametersRecord()
        for k, v in self.net_gen.state_dict().items():
            # Convert to NumPy, then to Array. Add to record
            p_record[k] = array_from_numpy(v.detach().cpu().numpy())
        # Add to a context
        self.client_state.parameters_records["gan_parameters"] = p_record

        model_path = f"modelo_gen_round_{config['round']}_client_{self.cid}.pt"
        torch.save(self.net_gen.state_dict(), model_path)

        figura = generate_plot(net=self.net_gen, device=self.device, round_number=config["round"], client_id=self.cid)
        figura.savefig(f"mnist_CGAN_r{config['round']}_{self.local_epochs_gen}e_{self.batch_size}b_100z_4c_{self.lr_gen}lr_{'niid' if self.niid else 'iid'}_{self.alpha_dir if self.niid else ''}_cliente{self.cid}.png")
        
        gan_params = get_weights(self.net_gen)
        
        return (
            get_weights_gen(self.net_alvo),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "gan": pickle.dumps(gan_params)},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net_alvo, parameters)
        loss, accuracy = test(self.net_alvo, self.valloader, self.device, model=self.model)
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
    partitioner = context.run_config["partitioner"]
    alpha_dir = context.run_config["alpha_dir"]
    batch_size = context.run_config["tam_batch"]
    teste = context.run_config["teste"]
    num_chunks = context.run_config["num_chunks"]
    # pretrained_cgan = CGAN()
    # pretrained_cgan.load_state_dict(torch.load("model_round_10_mnist.pt"))
    trainloader, valloader = load_data(partition_id=partition_id,
                                       num_partitions=num_partitions,
                                       partitioner=partitioner,
                                       alpha_dir=alpha_dir,
                                       batch_size=batch_size,
                                       teste=teste,
                                       num_chunks=num_chunks)
    local_epochs_alvo = context.run_config["epocas_alvo"]
    local_epochs_gen = context.run_config["epocas_gen"]
    lr_gen = context.run_config["learn_rate_gen"]
    lr_alvo = context.run_config["learn_rate_alvo"]
    latent_dim = context.run_config["tam_ruido"]
    agg = context.run_config["agg"]
    model = context.run_config["model"]
    niid = False if partitioner == "IID" else True 

    # Return Client instance
    return FlowerClient(cid=partition_id,
                        net_alvo=net_alvo, 
                        net_gen=net_gen, 
                        trainloader=trainloader, 
                        valloader=valloader, 
                        local_epochs_alvo=local_epochs_alvo, 
                        local_epochs_gen=local_epochs_gen,
                        dataset=dataset,
                        lr_alvo=lr_alvo,
                        lr_gen=lr_gen,
                        latent_dim=latent_dim,
                        context=context,
                        agg=agg,
                        model=model,
                        num_partitions=num_partitions,
                        niid=niid,
                        alpha_dir=alpha_dir,
                        batch_size=batch_size,
                        teste=teste).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn
)
