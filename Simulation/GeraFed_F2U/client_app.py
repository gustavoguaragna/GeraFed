"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, array_from_numpy
from Simulation.GeraFed_F2U.task import (
    Net, 
    CGAN, 
    F2U_GAN,
    get_weights, 
    get_weights_gen, 
    load_data, 
    set_weights, 
    test, 
    train_alvo, 
    train_gen,
    local_test
)
import random
import numpy as np
import math
from flwr.common.typing import UserConfigValue

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
                cid: UserConfigValue, 
                net_alvo,
                net_gan, 
                local_epochs_alvo: UserConfigValue, 
                local_epochs_gen: UserConfigValue, 
                dataset: UserConfigValue, 
                lr_alvo: UserConfigValue, 
                lr_gen: UserConfigValue, 
                latent_dim: UserConfigValue, 
                context: Context,
                agg: UserConfigValue,
                model: UserConfigValue,
                num_partitions: UserConfigValue,
                niid: bool,
                alpha_dir: UserConfigValue,
                batch_size: UserConfigValue,
                teste: UserConfigValue,
                folder: UserConfigValue = ".",
                num_chunks: UserConfigValue = 1,
                partitioner: UserConfigValue = "Class"):
        self.cid=cid
        self.net_alvo = net_alvo
        self.net_gen = net_gan
        self.net_disc = net_gan
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
        self.folder = folder
        self.num_chunks = num_chunks
        self.partitioner = partitioner
  

    def fit(self, parameters, config):
        
        # Atualiza pesos do modelo generativo
        set_weights(net=self.net_gen, parameters=config["gan"])

        # Calcula numero de amostras sinteticas
        num_syn = int(13 * (math.exp(0.01*config["round"]) - 1) / (math.exp(0.01*50) - 1)) * 10

        # Carrega dados
        trainloader_real, trainloader_syn, _, _ = load_data(partition_id=self.cid,
                                num_partitions=self.num_partitions,
                                partitioner_type=self.partitioner,
                                alpha_dir=self.alpha_dir,
                                batch_size=self.batch_size,
                                teste=self.teste,
                                num_chunks=self.num_chunks,
                                syn_samples=num_syn,
                                gan=self.net_gen,)

        # Atualiza pesos do modelo classificador
        set_weights(self.net_alvo, parameters)

        # Define o dataloader
        if isinstance(trainloader_syn, list):
            chunk_idx = config["round"] % len(trainloader_syn)
            trainloader_syn_chunk = trainloader_syn[chunk_idx]
        else:
            trainloader_syn_chunk = trainloader_syn

        # Treina o modelo classificador
        train_loss = train_alvo(
            net=self.net_alvo,
            trainloader=trainloader_syn_chunk,
            epochs=self.local_epochs_alvo,
            lr=self.lr_alvo,
            device=self.device,
        )

        # Cria state_dict para a disc
        state_dict = {}
        # Carrega parametros da disc do estado do cliente
        if "net_parameters" in self.client_state.parameters_records:
            p_record = self.client_state.parameters_records["net_parameters"]

            # Deserialize arrays
            for k, v in p_record.items():
                state_dict[k] = torch.from_numpy(v.numpy())

            # Apply state dict to disc
            self.net_disc.load_state_dict(state_dict)

        # Define o dataloader
        if isinstance(trainloader_real, list):
            chunk_idx = config["round"] % len(trainloader_real)
            trainloader_real_chunk = trainloader_real[chunk_idx]
        else:
            trainloader_real_chunk = trainloader_real

        train_gen(
        disc=self.net_disc,
        gen=self.net_gen,
        trainloader=trainloader_real_chunk,
        epochs=self.local_epochs_gen,
        lr=self.lr_gen,
        device=self.device,
        dataset=self.dataset,
        latent_dim=self.latent_dim
        )
        # Save all elements of the state_dict into a single RecordSet
        p_record = ParametersRecord()
        for k, v in self.net_disc.state_dict().items():
            # Convert to NumPy, then to Array. Add to record
            p_record[k] = array_from_numpy(v.detach().cpu().numpy())
        # Add to a context
        self.client_state.parameters_records["net_parameters"] = p_record
        
        return (
        get_weights(self.net_alvo),
        len(trainloader_syn_chunk.dataset),
        {"train_loss": train_loss, "disc": get_weights_gen(self.net_disc)},
        )

    def evaluate(self, parameters, config):

        # Carrega dados
        _, _, testloader, local_testloader = load_data(partition_id=self.cid,
                                num_partitions=self.num_partitions,
                                partitioner_type=self.partitioner,
                                alpha_dir=self.alpha_dir,
                                batch_size=self.batch_size,
                                teste=self.teste)

        set_weights(self.net_alvo, parameters)
        loss, accuracy = test(self.net_alvo, testloader, self.device, model=self.model)

        if config["round"] % self.num_chunks == 0 or config["round"] == 1:
            local_test(net=self.net_alvo,
                    testloader=local_testloader,
                    device=self.device,
                    acc_filepath=f"{self.folder}/accuracy_report.txt",
                    epoch=int(config["round"]/10),
                    cliente=self.cid)

        return loss, len(testloader.dataset), {"accuracy": accuracy}



def client_fn(context: Context):
    """Client function to be used in the Flower ClientApp."""
    # Load model and data
    dataset            = context.run_config["dataset"]
    gan_arq            = context.run_config["gan_arq"]

    if gan_arq == "simple_cnn":
        # Use a simple CNN architecture for the generator
        net_gan        = CGAN()
    elif gan_arq == "f2u_gan":
        net_gan        = F2U_GAN()

    net_alvo           = Net()
    partition_id       = context.node_config["partition-id"]
    num_partitions     = context.node_config["num-partitions"]
    partitioner        = context.run_config["partitioner"]
    alpha_dir          = context.run_config["alpha_dir"]
    batch_size         = context.run_config["tam_batch"]
    teste              = context.run_config["teste"]
    num_chunks         = context.run_config["num_chunks"]

    # pretrained_cgan  = CGAN()
    # pretrained_cgan.load_state_dict(torch.load("model_round_10_mnist.pt"))

    local_epochs_alvo  = context.run_config["epocas_alvo"]
    local_epochs_gen   = context.run_config["epocas_gen"]
    lr_gen             = context.run_config["learn_rate_gen"]
    lr_alvo            = context.run_config["learn_rate_alvo"]
    latent_dim         = context.run_config["tam_ruido"]
    agg                = context.run_config["agg"]
    model              = context.run_config["model"]
    niid               = False if partitioner == "IID" else True 
    folder             = context.run_config["Exp_name_folder"]

    #os.makedirs(folder, exist_ok=True)


    # Return Client instance
    return FlowerClient(cid=partition_id,
                        net_alvo=net_alvo, 
                        net_gan=net_gan,  
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
                        teste=teste,
                        folder=folder,
                        num_chunks=num_chunks,
                        partitioner=partitioner).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn
)
