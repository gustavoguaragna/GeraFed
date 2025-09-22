"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import torch
from torch.utils.data import DataLoader
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, array_from_numpy
from Simulation.GeraFed_F2U.task import (
    Net,
    Net_CIFAR,
    CGAN, 
    F2U_GAN,
    F2U_GAN_CIFAR,
    get_weights,
    load_data, 
    set_weights,
    syn_input,
    test, 
    train_alvo, 
    train_disc,
    local_test
)
# import random
# import numpy as np
import math
from flwr.common.typing import UserConfigValue
from typing import Union, List
##import time
import pickle
import copy

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)
#     # Para garantir determinismo total em operações com CUDA
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self,
                cid: UserConfigValue, 
                net_alvo,
                net_gan,
                optim_D,
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
                trainloader: Union[DataLoader, List],
                testloader: DataLoader,
                testloader_local: DataLoader,
                folder: UserConfigValue = ".",
                num_chunks: UserConfigValue = 1,
                partitioner: UserConfigValue = "Class",
                continue_epoch: UserConfigValue = 0,
                num_epochs: UserConfigValue = 100):
        self.cid=cid
        self.net_alvo = net_alvo
        self.net_gen = copy.deepcopy(net_gan)
        self.net_disc = copy.deepcopy(net_gan)
        self.optim_D = optim_D
        self.local_epochs_alvo = local_epochs_alvo
        self.local_epochs_gen = local_epochs_gen
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_alvo.to(self.device)
        self.lr_alvo = lr_alvo
        self.net_gen.to(self.device)
        self.lr_gen = lr_gen
        self.dataset = dataset
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
        self.continue_epoch = continue_epoch
        self.num_epochs = num_epochs
        self.trainloader = trainloader
        self.testloader = testloader
        self.testloader_local = testloader_local

    def fit(self, parameters, config):
        
        # Atualiza pesos do modelo generativo
        set_weights(net=self.net_gen, parameters=pickle.loads(config["gan"]))


        # Calcula numero de amostras sinteticas
        num_syn = int(13 * (math.exp(0.01*config["round"]) - 1) / (math.exp(0.01*self.num_epochs/2) - 1)) * 10
    

        # Atualiza pesos do modelo classificador
        set_weights(self.net_alvo, parameters)


        # Define o dataloader
        if isinstance(self.trainloader, list):
            chunk_idx = config["round"] % len(self.trainloader)
            trainloader_chunk = self.trainloader[chunk_idx]
        else:
            trainloader_chunk = self.trainloader

        if num_syn > 0:
            trainloader_syn = syn_input(
                                    num_samples=num_syn,
                                    gan=self.net_gen,
                                    round=config["round"],
                                    num_chunks=self.num_chunks,
                                    folder=self.folder,
                                    dataloader=trainloader_chunk,
                                    partition_id=self.cid,
                                    dataset=self.dataset
                                    )

        # Treina o modelo classificador
        ##train_alvo_start_time = time.time()
        train_loss = train_alvo(
            net=self.net_alvo,
            trainloader=trainloader_syn,
            epochs=self.local_epochs_alvo,
            lr=self.lr_alvo,
            device=self.device,
            dataset=self.dataset
        )
        ##train_classifier_time  = time.time() - train_alvo_start_time

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

        # Load optimizer state for the discriminator
        if "optim_parameter0" in self.client_state.parameters_records:

            for p in self.optim_D.state_dict()['state'].keys():
                # Carrega parametros do estado do parametro p do optim da disc
                optim_record = self.client_state.parameters_records[f"optim_parameter{p}"]

                # Deserialize arrays and substitute for current value
                for _, v in optim_record.items():
                   self.optim_D.state_dict()['state'][p] = torch.from_numpy(v.numpy())


        ##train_gen_start_time = time.time()
        # Treina o modelo generativo
        avg_d_loss = train_disc(
        disc=self.net_disc,
        gen=self.net_gen,
        trainloader=trainloader_chunk,
        epochs=self.local_epochs_gen,
        device=self.device,
        dataset=self.dataset,
        latent_dim=self.latent_dim,
        optim=self.optim_D
        )
        ##train_gen_time = time.time() - train_gen_start_time

        # Save all elements of the state_dict into a single RecordSet
        p_record = ParametersRecord()
        for k, v in self.net_disc.state_dict().items():
            # Convert to NumPy, then to Array. Add to record
            p_record[k] = array_from_numpy(v.detach().cpu().numpy())
        # Add to a context
        self.client_state.parameters_records["net_parameters"] = p_record

        # Save all elements of the optim.state_dict into a single RecordSet
        
        optim_records = [ParametersRecord() for _ in self.optim_D.state_dict()['state'].keys()]
        for p in self.optim_D.state_dict()['state'].keys():
            for k, v in self.optim_D.state_dict()['state'][p].items():
                # Convert to NumPy, then to Array. Add to record
                optim_records[p][k] = array_from_numpy(v.detach().cpu().numpy())
            # Add to a context
            self.client_state.parameters_records[f"optim_parameter{p}"] = optim_records[p]


        disc_params = get_weights(self.net_disc)
        
        return (
        get_weights(self.net_alvo),
        len(trainloader_syn.dataset),
        {"train_loss": train_loss,
         "disc": pickle.dumps(disc_params),
         "avg_d_loss": avg_d_loss,
         "optimDs_state_dict": pickle.dumps(self.optim_D.state_dict()),
         "cid": self.cid
         ##"tempo_treino_alvo": train_classifier_time,
         ##"tempo_treino_gen": train_gen_time
        },
        )

    def evaluate(self, parameters, config):


        set_weights(self.net_alvo, parameters)
        ##test_time_start = time.time()
        # Avalia o modelo classificador
        loss, accuracy = test(self.net_alvo, self.testloader, self.device, model=self.model, dataset=self.dataset)
        ##test_time = time.time() - test_time_start


        test_local = config["round"] % self.num_chunks == 0
        if test_local:
            ##local_test_start_time = time.time()
            # Avalia o modelo classificador localmente
            local_test(net=self.net_alvo,
                    testloader=self.testloader_local,
                    device=self.device,
                    acc_filepath=f"{self.folder}/accuracy_report.txt",
                    epoch=int(config["round"]/self.num_chunks),
                    cliente=self.cid,
                    continue_epoch=self.continue_epoch,
                    dataset=self.dataset)
            ##local_test_time = time.time() - local_test_start_time

        return (loss,
                len(self.testloader.dataset),
                {"accuracy": accuracy,
                 ##"test_time": test_time,
                 ##"local_test_time": local_test_time if test_local else None
                 }
            )


def client_fn(context: Context):
    """Client function to be used in the Flower ClientApp."""
    # Load model and data
    partition_id       = context.node_config["partition-id"]
    num_partitions     = context.node_config["num-partitions"]
    partitioner        = context.run_config["partitioner"]
    alpha_dir          = context.run_config["alpha_dir"]
    batch_size         = context.run_config["tam_batch"]
    teste              = context.run_config["teste"]
    num_chunks         = context.run_config["num_chunks"]
    continue_epoch     = context.run_config["continue_epoch"]
    dataset            = context.run_config["dataset"]
    gan_arq            = context.run_config["gan_arq"]

    local_epochs_alvo  = context.run_config["epocas_alvo"]
    local_epochs_gen   = context.run_config["epocas_gen"]
    lr_gen             = context.run_config["learn_rate_gen"]
    lr_disc            = context.run_config["learn_rate_disc"]
    lr_alvo            = context.run_config["learn_rate_alvo"]
    latent_dim         = context.run_config["tam_ruido"]
    agg                = context.run_config["agg"]
    model              = context.run_config["model"]
    niid               = False if partitioner == "IID" else True 
    folder             = f"{context.run_config['Exp_name_folder']}FedGenIA_F2U_{num_partitions}clients/{dataset}/{partitioner}"
    num_epochs         = int(context.run_config["num_rodadas"]/num_chunks)
    
    if dataset == "mnist":
        net_alvo = Net()
        if gan_arq == "simple_cnn":
            # Use a simple CNN architecture for the generator
            gan = CGAN()
        elif gan_arq == "f2u_gan":
            gan = F2U_GAN()
    elif dataset == "cifar10":
        net_alvo = Net_CIFAR()
        if gan_arq == "simple_cnn":
            # Use a simple CNN architecture for the generator
            raise ValueError(f"cGAN nao implementada para CIFAR10")
        elif gan_arq == "f2u_gan":
            gan = F2U_GAN_CIFAR()
    else:
        raise ValueError(f"{dataset} nao identificado")

    optim_D = torch.optim.Adam(gan.discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))



    if continue_epoch != 0:
        checkpoint = torch.load(f"{folder}/checkpoint_epoch{continue_epoch}.pth")
        gan.load_state_dict(checkpoint['discs_state_dict'][partition_id])
        optim_D.load_state_dict(checkpoint['optimDs_state_dict'][partition_id])


    trainloader, testloader, testloader_local = load_data(
                                partition_id=partition_id,
                                num_partitions=num_partitions,
                                batch_size=batch_size,
                                dataset=dataset,
                                teste=teste,
                                partitioner_type=partitioner,
                                num_chunks=num_chunks,

    )


    # Return Client instance
    return FlowerClient(cid=partition_id,
                        net_alvo=net_alvo, 
                        net_gan=gan,  
                        optim_D=optim_D,
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
                        partitioner=partitioner,
                        continue_epoch=continue_epoch,
                        num_epochs=num_epochs,
                        trainloader=trainloader,
                        testloader=testloader,
                        testloader_local=testloader_local).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn
)
