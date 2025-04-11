"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import torch
from flwr.common.typing import Parameters
from collections import OrderedDict
from flwr.client import ClientApp, NumPyClient
from torch.utils.data import DataLoader
from flwr.common import Context, ParametersRecord, array_from_numpy, parameters_to_ndarrays, bytes_to_ndarray
from Simulation.GeraFed_LoRa.task import (
    Net, 
    CGAN, 
    get_weights,  
    load_data, 
    set_weights, 
    test, 
    train_alvo, 
    train_gen, 
    generate_plot,
    LoRALinear,
    add_lora_to_model,
    prepare_model_for_lora,
    get_lora_adapters,
    set_lora_adapters,
    get_lora_weights_from_list,
    GeneratedDataset

)
import random
import numpy as np
import json

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
                agg: str,
                model: str,
                num_partitions: int,
                niid: bool,
                alpha_dir: float,
                batch_size: int,
                teste: bool):
        self.cid=cid
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
        self.agg = agg
        self.model = model
        self.num_partitions = num_partitions
        self.niid = niid
        self.alpha_dir = alpha_dir
        self.batch_size = batch_size
        self.teste = teste

    def fit(self, parameters, config):
        if config["modelo"] == "alvo":

            print("ENTROU TREINAR ALVO")

            if self.teste:
                num_samples = 1200
            else:
                num_samples = 12000

            generated_datasets = []

            # Reconstruct generator parameters
            gen_tensors = []
            j = 0
            while f"gen_{j}" in config:
                gen_tensors.append(config[f"gen_{j}"])
                j += 1
            gen_params = Parameters(tensors=gen_tensors, tensor_type="numpy.ndarray")
            set_weights(self.net_gen, parameters_to_ndarrays(gen_params))
            add_lora_to_model(self.net_gen)
            # Reconstruct LoRA parameters
            for i in range(0, self.num_partitions):
                lora_bytes = []
                j = 0
                while f"lora_{i}_{j}" in config:
                    lora_bytes.append(config[f"lora_{i}_{j}"])
                    j += 1
                lora_ndarrays = [bytes_to_ndarray(tensor) for tensor in lora_bytes]
                lora_tensors = [torch.from_numpy(ndarray).to(self.device) for ndarray in lora_ndarrays]
                lora_params = []
                for i in range(0, len(lora_tensors), 2):
                    lora_A = lora_tensors[i]
                    lora_B = lora_tensors[i + 1]
                    lora_params.append((torch.nn.Parameter(lora_A), torch.nn.Parameter(lora_B)))

                set_lora_adapters(self.net_gen, lora_params)

                generated_dataset = GeneratedDataset(generator=self.net_gen, num_samples=num_samples, device=self.device)
                concat_dataset = torch.utils.data.ConcatDataset([self.trainloader.dataset, generated_dataset])
                self.trainloader = DataLoader(concat_dataset, batch_size=self.batch_size, shuffle=True) 
                 
                
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
            print("ENTROU TREINAR GEN")
            if self.agg == "full":
                set_weights(self.net_gen, parameters)

                #Gera imagens do modelo agregado do round anterior
                generate_plot(net=self.net_gen, device=self.device, round_number=config["round"])
                #figura.savefig(f"mnist_CGAN_r{config['round']-1}_{self.local_epochs_gen}e_{self.batch_size}b_100z_4c_{self.lr_gen}lr_{'niid' if self.niid else 'iid'}_{self.alpha_dir if self.niid else ''}.png")
               
                # Adiciona LoRA ao modelo
                if config["round"] > 1:
                    add_lora_to_model(self.net_gen)
                    prepare_model_for_lora(self.net_gen)
                
                train_gen(
                net=self.net_gen,
                trainloader=self.trainloader,
                epochs=self.local_epochs_gen,
                lr=self.lr_gen,
                device=self.device,
                dataset=self.dataset,
                latent_dim=self.latent_dim,
                cid=self.cid,
                logfile="lora_train.txt",
                round_number=config["round"],
            )
                
                #generate_plot(net=self.net_gen, device=self.device, round_number=config["round"]+10, client_id=self.cid)
                #figura.savefig(f"mnist_CGAN_r{config['round']}_{self.local_epochs_gen}e_{self.batch_size}b_100z_4c_{self.lr_gen}lr_{'niid' if self.niid else 'iid'}_{self.alpha_dir if self.niid else ''}_cliente{self.cid}.png")
                
                if config["round"] > 1:
                    lora = get_lora_adapters(self.net_gen)

                    return (
                    get_lora_weights_from_list(lora),
                    len(self.trainloader.dataset),
                    {"modelo": "gen"},
                )
                return (
                    get_weights(self.net_gen),
                    len(self.trainloader.dataset),
                    {"modelo": "gen"},
                )


    def evaluate(self, parameters, config):
        if config["round"] < 3:
            return 0.0, 1, {}
        else:
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
    alpha_dir = context.run_config["alpha_dir"]
    batch_size = context.run_config["tam_batch"]
    teste = context.run_config["teste"]
    partitioner = context.run_config["partitioner"]
    # pretrained_cgan = CGAN()
    # pretrained_cgan.load_state_dict(torch.load("model_round_10_mnist.pt"))
    trainloader, valloader = load_data(partition_id=partition_id,
                                       num_partitions=num_partitions,
                                       alpha_dir=alpha_dir,
                                       batch_size=batch_size,
                                       teste=teste,
                                       partitioner=partitioner
                                      )
    local_epochs_alvo = context.run_config["epocas_alvo"]
    local_epochs_gen = context.run_config["epocas_gen"]
    lr_gen = context.run_config["learn_rate_gen"]
    lr_alvo = context.run_config["learn_rate_alvo"]
    latent_dim = context.run_config["tam_ruido"]
    agg = context.run_config["agg"]
    model = context.run_config["model"]
    niid = True if partitioner == "Dir" or partitioner == "Class" else False

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
