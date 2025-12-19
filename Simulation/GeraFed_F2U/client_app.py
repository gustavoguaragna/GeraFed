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
import math
from flwr.common.typing import UserConfigValue
from typing import Union, List
import pickle
import copy
import time
import io
import numpy as np

# import random
# import numpy as np

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
                local_epochs_alvo: UserConfigValue, 
                local_epochs_disc: UserConfigValue, 
                dataset: UserConfigValue, 
                lr_alvo: UserConfigValue,
                lr_disc: UserConfigValue,
                latent_dim: UserConfigValue, 
                context: Context,
                trainloader: Union[DataLoader, List],
                testloader: DataLoader,
                testloader_local: DataLoader,
                folder: UserConfigValue = ".",
                num_chunks: UserConfigValue = 1,
                continue_epoch: UserConfigValue = 0,
                num_epochs: UserConfigValue = 100):
        self.cid=cid
        self.net_alvo = net_alvo
        self.net_gen = copy.deepcopy(net_gan)
        self.net_disc = copy.deepcopy(net_gan)
        self.local_epochs_alvo = local_epochs_alvo
        self.local_epochs_disc = local_epochs_disc
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_alvo.to(self.device)
        self.net_gen.to(self.device)
        self.net_disc.to(self.device)
        self.lr_alvo = lr_alvo
        self.lr_disc = lr_disc
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.client_state = (
            context.state
        ) 
        self.folder = folder
        self.num_chunks = num_chunks
        self.continue_epoch = continue_epoch
        self.num_epochs = num_epochs
        self.trainloader = trainloader
        self.testloader = testloader
        self.testloader_local = testloader_local
        self.num_classes = 10 if dataset in ["mnist", "cifar10"] else None

        self.optim_D = torch.optim.Adam(list(self.net_disc.discriminator.parameters())+list(self.net_disc.label_embedding.parameters()), lr=self.lr_disc, betas=(0.5, 0.999))

    def fit(self, parameters, config):
        
        # Atualiza pesos do modelo generativo
        set_weights(net=self.net_gen, parameters=pickle.loads(config["gan"]))

        # Atualiza pesos do modelo classificador
        set_weights(self.net_alvo, parameters)

        # Define o dataloader
        if isinstance(self.trainloader, list):
            chunk_idx = config["round"] % len(self.trainloader)
            trainloader_chunk = self.trainloader[chunk_idx]
        else:
            trainloader_chunk = self.trainloader

        # Calcula numero de amostras sinteticas
        num_syn = int(math.ceil(len(trainloader_chunk.dataset)) * (math.exp(0.01*int(config["round"]/self.num_chunks)) - 1) / (math.exp(0.01*self.num_epochs/2) - 1))

        img_syn_time = 0
        trainloader_aug = trainloader_chunk

        if num_syn > 0:
            img_syn_start_time = time.time()
            # Gera dados sinteticos
            trainloader_aug = syn_input(
                                    num_samples=num_syn,
                                    gan=self.net_gen,
                                    round=config["round"],
                                    num_chunks=self.num_chunks,
                                    folder=self.folder,
                                    dataloader=trainloader_chunk,
                                    partition_id=self.cid,
                                    dataset=self.dataset,
                                    continue_epoch=self.continue_epoch
                                    )
            img_syn_time = time.time() - img_syn_start_time

        # Treina o modelo classificador
        train_alvo_start_time = time.time()
        train_loss = train_alvo(
            net=self.net_alvo,
            trainloader=trainloader_aug,
            epochs=self.local_epochs_alvo,
            lr=self.lr_alvo,
            device=self.device,
            dataset=self.dataset
        )
        train_classifier_time  = time.time() - train_alvo_start_time

        # --- Restore discriminator model parameters ---
        if "disc_state_dict" in self.client_state.parameters_records:
            rec_model = self.client_state.parameters_records["disc_state_dict"]
            arr_model = rec_model["state_bytes"].numpy()
            buf_model = io.BytesIO(arr_model.tobytes())
            state_dict = torch.load(buf_model, map_location=self.device)
            self.net_disc.load_state_dict(state_dict)

        # --- Restore optimizer state ---
        if "disc_optim_state_dict" in self.client_state.parameters_records:
            rec_optim = self.client_state.parameters_records["disc_optim_state_dict"]
            arr_optim = rec_optim["state_bytes"].numpy()
            buf_optim = io.BytesIO(arr_optim.tobytes())
            optim_state_dict = torch.load(buf_optim, map_location=self.device)
            self.optim_D.load_state_dict(optim_state_dict)

        # # Cria state_dict para a disc
        # state_dict = {}
        # # Carrega parametros da disc do estado do cliente
        # if "net_parameters" in self.client_state.parameters_records:
        #     p_record = self.client_state.parameters_records["net_parameters"]

        #     # Deserialize arrays
        #     for k, v in p_record.items():
        #         state_dict[k] = torch.from_numpy(v.numpy())

        #     # Apply state dict to disc
        #     self.net_disc.load_state_dict(state_dict)

        # # Load optimizer state for the discriminator
        # if "optim_parameter0" in self.client_state.parameters_records:

        #     for p in self.optim_D.state_dict()['state'].keys():
        #         # Carrega parametros do estado do parametro p do optim da disc
        #         optim_record = self.client_state.parameters_records[f"optim_parameter{p}"]

        #         # Deserialize arrays and substitute for current value
        #         for _, v in optim_record.items():
        #            self.optim_D.state_dict()['state'][p] = torch.from_numpy(v.numpy())

        train_disc_start_time = time.time()
        # Treina o modelo generativo
        avg_d_loss = train_disc(
        disc=self.net_disc,
        gen=self.net_gen,
        trainloader=trainloader_chunk,
        epochs=self.local_epochs_disc,
        device=self.device,
        dataset=self.dataset,
        latent_dim=self.latent_dim,
        optim=self.optim_D,
        cid=self.cid, 
        round=config["round"]
        )

        train_disc_time = time.time() - train_disc_start_time

        # Save all elements of the state_dict into a single RecordSet
        p_record = ParametersRecord()
        for k, v in self.net_disc.state_dict().items():
            # Convert to NumPy, then to Array. Add to self.client_state
            p_record[k] = array_from_numpy(v.detach().cpu().numpy())
        # Add to a context
        self.client_state.parameters_records["net_parameters"] = p_record

        # # Save all elements of the optim.state_dict into a single RecordSet    
        # optim_records = [ParametersRecord() for _ in self.optim_D.state_dict()['state'].keys()]
        # for p in self.optim_D.state_dict()['state'].keys():
        #     for k, v in self.optim_D.state_dict()['state'][p].items():
        #         # Convert to NumPy, then to Array. Add to self.client_state
        #         optim_records[p][k] = array_from_numpy(v.detach().cpu().numpy())
        #     # Add to a context
        #     self.client_state.parameters_records[f"optim_parameter{p}"] = optim_records[p]
        
        # Save optimizer state_dict fully (state + param_groups)
        # --- Save discriminator model parameters ---
        buf_model = io.BytesIO()
        torch.save(self.net_disc.state_dict(), buf_model)

        # Convert bytes → numpy array (uint8)
        arr_model = np.frombuffer(buf_model.getvalue(), dtype=np.uint8)

        # Store in ParametersRecord
        rec_model = ParametersRecord()
        rec_model["state_bytes"] = array_from_numpy(arr_model)
        self.client_state.parameters_records["disc_state_dict"] = rec_model

        # --- Save optimizer state (Adam, etc.) ---
        buf_optim = io.BytesIO()
        torch.save(self.optim_D.state_dict(), buf_optim)

        arr_optim = np.frombuffer(buf_optim.getvalue(), dtype=np.uint8)
        rec_optim = ParametersRecord()
        rec_optim["state_bytes"] = array_from_numpy(arr_optim)
        self.client_state.parameters_records["disc_optim_state_dict"] = rec_optim


        disc_params = get_weights(self.net_disc)
        
        return (
            get_weights(self.net_alvo),
            len(trainloader_aug.dataset),
            {"train_loss": train_loss,
            "disc": pickle.dumps(disc_params),
            "avg_d_loss": avg_d_loss,
            "optimDs_state_dict": pickle.dumps(self.optim_D.state_dict()),
            "cid": self.cid,
            "tempo_treino_alvo": train_classifier_time,
            "tempo_treino_disc": train_disc_time,
            "img_syn_time": img_syn_time
            },
        )

    def evaluate(self, parameters, config):

        set_weights(self.net_alvo, parameters)
        test_time_start = time.time()
        # Avalia o modelo classificador
        loss, accuracy = test(self.net_alvo, self.testloader, self.device, dataset=self.dataset)
        test_time = time.time() - test_time_start


        test_local = config["round"] % self.num_chunks == 0
        if test_local:
            local_test_start_time = time.time()
            # Avalia o modelo classificador localmente
            local_test(net=self.net_alvo,
                    testloader=self.testloader_local,
                    device=self.device,
                    acc_filepath=f"{self.folder}/accuracy_report.txt",
                    epoch=int(config["round"]/self.num_chunks),
                    cliente=self.cid,
                    continue_epoch=self.continue_epoch,
                    dataset=self.dataset)
            local_test_time = time.time() - local_test_start_time

        return (loss,
                len(self.testloader.dataset),
                {"accuracy": accuracy,
                 "test_time": test_time,
                 "local_test_time": local_test_time if test_local else 0
                 }
            )


def client_fn(context: Context):
    """Client function to be used in the Flower ClientApp."""
    # Load model and data
    partition_id       = context.node_config["partition-id"]
    num_partitions     = context.node_config["num-partitions"]
    partitioner        = context.run_config["partitioner"]
    if partitioner == "Dir01":
        alpha_dir       = 0.1
    elif partitioner == "Dir05":
        alpha_dir       = 0.5
    else:
        alpha_dir       = None
    batch_size         = context.run_config["tam_batch"]
    teste              = context.run_config["teste"]
    num_chunks         = context.run_config["num_chunks"]
    continue_epoch     = context.run_config["continue_epoch"]
    dataset            = context.run_config["dataset"]
    gan_arq            = context.run_config["gan_arq"]
    strategy           = context.run_config["strategy"]

    local_epochs_alvo  = context.run_config["epocas_alvo"]
    local_epochs_disc  = context.run_config["epocas_disc"]
    lr_disc            = context.run_config["learn_rate_disc"]
    lr_alvo            = context.run_config["learn_rate_alvo"]
    latent_dim         = context.run_config["tam_ruido"]
    seed               = context.run_config["seed"]
    folder             = f"{context.run_config['Exp_name_folder']}FedGenIA_F2U/{dataset}/{partitioner}/{strategy}/{num_partitions}_clients"
    num_epochs         = context.run_config["num_epocas"]
    patience           = context.run_config["patience"]
    
    if dataset == "mnist":
        net_alvo = Net(seed=seed)
        if gan_arq == "simple_cnn":
            # Use a simple CNN architecture for the generator
            gan = CGAN()
        elif gan_arq == "f2u_gan":
            gan = F2U_GAN(seed=seed)
    elif dataset == "cifar10":
        net_alvo = Net_CIFAR(seed=seed)
        if gan_arq == "simple_cnn":
            # Use a simple CNN architecture for the generator
            raise ValueError(f"cGAN nao implementada para CIFAR10")
        elif gan_arq == "f2u_gan":
            gan = F2U_GAN_CIFAR(seed=seed)
    else:
        raise ValueError(f"Dataset {dataset} nao identificado. Deveria ser 'mnist' ou 'cifar10'")


    if continue_epoch != 0:
        checkpoint = torch.load(f"{folder}/checkpoint_epoch{continue_epoch}.pth", map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
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
                                alpha_dir=alpha_dir

    )


    # Return Client instance
    return FlowerClient(cid=partition_id,
                        net_alvo=net_alvo, 
                        net_gan=gan,
                        local_epochs_alvo=local_epochs_alvo, 
                        local_epochs_disc=local_epochs_disc,
                        dataset=dataset,
                        lr_alvo=lr_alvo,
                        lr_disc=lr_disc,
                        latent_dim=latent_dim,
                        context=context,
                        folder=folder,
                        num_chunks=num_chunks,
                        continue_epoch=continue_epoch,
                        num_epochs=num_epochs,
                        trainloader=trainloader,
                        testloader=testloader,
                        testloader_local=testloader_local).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn
)
