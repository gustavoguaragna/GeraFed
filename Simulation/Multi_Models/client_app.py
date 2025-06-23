"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, array_from_numpy
from Simulation.Multi_Models.task import (
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
                teste: bool,
                partitioner: str):
        self.cid=cid
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_alvo = net_alvo.to(self.device)
        self.net_gen = net_gen.to(self.device)
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
        self.partitioner = partitioner

    def fit(self, parameters, config):

        ### CONFERIR SE A CID MANTEM PARA NAO GERAR DADOS DO PROPRIO CLIENTE

        gans_dict = pickle.loads(config["gans"])
        gan_list = [CGAN() for k in gans_dict.keys() if k != self.cid]
        for gan, v in zip(gan_list, gans_dict.values()):
            # Deserialize the GAN parameters and set them to the CGAN instance
            set_weights(gan, pickle.loads(v))

        trainloader, _, trainloader_gen = load_data(partition_id=self.cid,
                                       num_partitions=self.num_partitions,
                                       partitioner=self.partitioner,
                                       alpha_dir=self.alpha_dir,
                                       batch_size=self.batch_size,
                                       teste=self.teste,
                                       syn_samples=config["round"]*10,
                                       gans=gan_list,
                                       classifier=self.net_alvo)

        set_weights(self.net_alvo, parameters)
        train_loss = train_alvo(
            net=self.net_alvo,
            trainloader=trainloader,
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
            trainloader=trainloader_gen,
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
            get_weights(self.net_alvo),
            len(trainloader.dataset),
            {"train_loss": train_loss, "gan": pickle.dumps(gan_params), "cid": self.cid},
        )

    def evaluate(self, parameters, config):
        _, valloader, _ = load_data(partition_id=self.cid,
                                       num_partitions=self.num_partitions,
                                       partitioner=self.partitioner,
                                       alpha_dir=self.alpha_dir,
                                       batch_size=self.batch_size,
                                       teste=self.teste) 
        set_weights(self.net_alvo, parameters)
        loss, accuracy = test(self.net_alvo, valloader, self.device, model=self.net_alvo)
        return loss, len(valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    dataset = str(context.run_config["dataset"])
    img_size = context.run_config["tam_img"]
    latent_dim = int(context.run_config["tam_ruido"])
    partitioner = context.run_config["partitioner"]
    alpha_dir = float(context.run_config["alpha_dir"])
    batch_size = int(context.run_config["tam_batch"])
    teste = bool(context.run_config["teste"])
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    net_gen = CGAN(dataset=dataset, img_size=img_size, latent_dim=latent_dim) #type: ignore
    net_alvo = Net()
    # pretrained_cgan = CGAN()
    # pretrained_cgan.load_state_dict(torch.load("model_round_10_mnist.pt"))
    local_epochs_alvo = int(context.run_config["epocas_alvo"])
    local_epochs_gen = int(context.run_config["epocas_gen"])
    lr_gen = float(context.run_config["learn_rate_gen"])
    lr_alvo = float(context.run_config["learn_rate_alvo"])
    latent_dim = int(context.run_config["tam_ruido"])
    niid = False if partitioner == "IID" else True 

    # Return Client instance
    return FlowerClient(cid=partition_id,
                        net_alvo=net_alvo, 
                        net_gen=net_gen, 
                        local_epochs_alvo=local_epochs_alvo, 
                        local_epochs_gen=local_epochs_gen,
                        dataset=dataset,
                        lr_alvo=lr_alvo,
                        lr_gen=lr_gen,
                        latent_dim=latent_dim,
                        context=context,
                        num_partitions=num_partitions,
                        niid=niid,
                        alpha_dir=alpha_dir,
                        batch_size=batch_size,
                        teste=teste,
                        partitioner=partitioner).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn
)
