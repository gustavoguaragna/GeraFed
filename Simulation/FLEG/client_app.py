"""FLEG: um framework para balancear dados heterogêneos em aprendizado federado, com precupações com a privacidade."""

import torch
from torch.utils.data import DataLoader
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, array_from_numpy
from Simulation.FLEG.task import (
    augment_client_with_generated,
    Net,
    Net_Cifar,
    ClassifierHead1, ClassifierHead2, ClassifierHead3, ClassifierHead4,
    ClassifierHead1_Cifar, ClassifierHead2_Cifar, ClassifierHead3_Cifar, ClassifierHead4_Cifar,
    EmbeddingGAN1, EmbeddingGAN2, EmbeddingGAN3, EmbeddingGAN4,
    EmbeddingGAN1_Cifar, EmbeddingGAN2_Cifar, EmbeddingGAN3_Cifar, EmbeddingGAN4_Cifar,
    EmbeddingPairDataset,
    FeatureExtractor1, FeatureExtractor2, FeatureExtractor3, FeatureExtractor4,
    FeatureExtractor1_Cifar, FeatureExtractor2_Cifar, FeatureExtractor3_Cifar, FeatureExtractor4_Cifar,
    get_label_counts,
    get_weights,
    load_data, 
    set_weights,
    test, 
    train_alvo, 
    train_disc,
    unpack_batch
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
                local_epochs_alvo: UserConfigValue, 
                local_epochs_disc: UserConfigValue, 
                dataset: UserConfigValue, 
                batch_size: UserConfigValue,
                lr_alvo: UserConfigValue,
                lr_disc: UserConfigValue,
                latent_dim: UserConfigValue, 
                context: Context,
                folder: UserConfigValue = ".",
                num_chunks: UserConfigValue = 1,
                num_partitions: UserConfigValue = 4,
                partitioner: UserConfigValue = "ClassPartitioner",
                alpha: Union(float, None) = None,
                continue_epoch: UserConfigValue = 0,
                num_epochs: UserConfigValue = 100,
                seed: UserConfigValue = 42,
                teste: UserConfigValue = False):
        self.cid=cid
        self.local_epochs_alvo = local_epochs_alvo
        self.local_epochs_disc = local_epochs_disc
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.lr_alvo = lr_alvo
        self.lr_disc = lr_disc
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.client_state = (
            context.state
        ) 
        self.folder = folder
        self.num_chunks = num_chunks
        self.num_partitions = num_partitions
        self.partitioner = partitioner
        self.alpha = alpha
        self.continue_epoch = continue_epoch
        self.num_epochs = num_epochs
        self.num_classes = 10 if dataset in ["mnist", "cifar10"] else None
        self.seed = seed
        self.teste = teste


    def fit(self, parameters, config):

        if config["model"] == "gan":
            if config["level"] == 0:
                if self.dataset == "mnist":
                    self.gen = EmbeddingGAN1(seed=self.seed)
                    self.disc = EmbeddingGAN1(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.gen = EmbeddingGAN1_Cifar(seed=self.seed)
                    self.disc = EmbeddingGAN1_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")

            elif config["level"] == 1:
                if self.dataset == "mnist":
                    self.gen = EmbeddingGAN2(seed=self.seed)
                    self.disc = EmbeddingGAN2(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.gen = EmbeddingGAN2_Cifar(seed=self.seed)
                    self.disc = EmbeddingGAN2_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")

            elif config["level"] == 2:
                if self.dataset == "mnist":
                    self.gen = EmbeddingGAN3(seed=self.seed)
                    self.disc = EmbeddingGAN3(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.gen = EmbeddingGAN3_Cifar(seed=self.seed)
                    self.disc = EmbeddingGAN3_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")

            elif config["level"] == 3:
                if self.dataset == "mnist":
                    self.gen = EmbeddingGAN4(seed=self.seed)
                    self.disc = EmbeddingGAN4(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.gen = EmbeddingGAN4_Cifar(seed=self.seed)
                    self.disc = EmbeddingGAN4_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")
            
            else:
                raise ValueError(f"Treino da GAN vai até nível 3 (4° nível), não deveria receber config['level'] {config['level']}.")

            self.optim_D = torch.optim.Adam(list(self.disc.discriminator.parameters())+list(self.disc.label_embedding.parameters()), lr=self.lr_disc, betas=(0.5, 0.999))

            # Atualiza pesos do modelo generativo
            set_weights(net=self.gen, parameters=parameters)

            # --- Restore discriminator model parameters ---
            if "disc_state_dict" in self.client_state.parameters_records:
                rec_model = self.client_state.parameters_records["disc_state_dict"]
                arr_model = rec_model["state_bytes"].numpy()
                buf_model = io.BytesIO(arr_model.tobytes())
                state_dict = torch.load(buf_model, map_location=self.device)
                self.disc.load_state_dict(state_dict)

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

            # Define o dataloader
            if isinstance(self.trainloader, list):
                chunk_idx = config["round"] % len(self.trainloader)
                trainloader_chunk = self.trainloader[chunk_idx]
            else:
                trainloader_chunk = self.trainloader

            train_disc_start_time = time.time()
            # Treina o modelo generativo
            avg_d_loss = train_disc(
            disc=self.disc,
            gen=self.gen,
            trainloader=trainloader_chunk,
            epochs=self.local_epochs_disc,
            device=self.device,
            dataset=self.dataset,
            latent_dim=self.latent_dim,
            optim=self.optim_D,
            )

            train_disc_time = time.time() - train_disc_start_time

            # Save all elements of the state_dict into a single RecordSet
            p_record = ParametersRecord()
            for k, v in self.net_disc.state_dict().items():
                # Convert to NumPy, then to Array. Add to self.client_state
                p_record[k] = array_from_numpy(v.detach().cpu().numpy())
            # Add to a context
            self.client_state.parameters_records["disc_parameters"] = p_record

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
            disc_params,
            len(trainloader_chunk.dataset),
            {"train_loss": avg_d_loss,
            "tempo_treino_disc": train_disc_time,
            },
        )

        elif config["model"] == "classifier":
            if config["level"] == 0:
                if self.dataset == "mnist":
                    self.net = Net(seed=self.seed)
                    self.feature_extractor = None
                elif self.dataset == "cifar10":
                    self.net = Net_Cifar(seed=self.seed)
                    self.feature_extractor = None
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido") 

            elif config["level"] == 1:
                if self.dataset == "mnist":
                    self.feature_extractor = FeatureExtractor1(seed=self.seed)
                    self.classifier = ClassifierHead1(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.feature_extractor = FeatureExtractor1_Cifar(seed=self.seed)
                    self.classifier = ClassifierHead1_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")

            elif config["level"] == 2:
                if self.dataset == "mnist":
                    self.feature_extractor = FeatureExtractor2(seed=self.seed)
                    self.classifier = ClassifierHead2(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.feature_extractor = FeatureExtractor2_Cifar(seed=self.seed)
                    self.classifier = ClassifierHead2_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")

            elif config["level"] == 3:
                if self.dataset == "mnist":
                    self.feature_extractor = FeatureExtractor3(seed=self.seed)
                    self.classifier = ClassifierHead3(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.feature_extractor = FeatureExtractor3_Cifar(seed=self.seed)
                    self.classifier = ClassifierHead3_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")

            elif config["level"] == 4:
                if self.dataset == "mnist":
                    self.feature_extractor = FeatureExtractor4(seed=self.seed)
                    self.classifier = ClassifierHead4(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.feature_extractor = FeatureExtractor4_Cifar(seed=self.seed)
                    self.classifier = ClassifierHead4_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")
            
            else:
                raise ValueError(f"Treino da GAN vai até nível 3 (4° nível), não deveria receber config['level'] {config['level']}.")

        else:
            raise ValueError(f"config[model] deveria ser 'gan' ou 'classifier', mas obteve {config['model']}")
        
        # Atualiza pesos do modelo classificador
        set_weights(self.net, parameters)

        if config["best_model"]:
            # Save all elements of the state_dict into a single RecordSet
            net_record = ParametersRecord()
            for k, v in self.net.state_dict().items():
                # Convert to NumPy, then to Array. Add to self.client_state
                net_record[k] = array_from_numpy(v.detach().cpu().numpy())

            # Add to a context
            self.client_state.parameters_records["net_parameters"] = net_record

            # --- Save discriminator model parameters ---
            buf_net = io.BytesIO()
            torch.save(self.net.state_dict(), buf_net)

            # Convert bytes → numpy array (uint8)
            arr_net = np.frombuffer(buf_net.getvalue(), dtype=np.uint8)

            # Store in ParametersRecord
            rec_net = ParametersRecord()
            rec_net["state_bytes"] = array_from_numpy(arr_net)
            self.client_state.parameters_records["net_state_dict"] = rec_model
        
        trainloader, testloader, testloader_local = load_data(
            partition_id=self.cid,
            num_partitions=self.num_partitions,
            batch_size=self.batch_size,
            dataset=self.dataset,
            teste=self.teste,
            partitioner_type=self.partitioner,
            num_chunks=1,
            alpha_dir=self.alpha_dir
        )

        # Synthetic Data Augmentation
        # if checkpoint_level is not None and level == checkpoint_level:
        #     generated_dataset = checkpoint_loaded['generated_dataset']


        if len(config["embds"]) > 0:    
            all_embeddings = []
            all_labels = []

            counts = get_label_counts(trainloader.dataset)
            
            with torch.no_grad():
                self.feature_extractor.eval()
                for batch in trainloader:
                    images, labels = unpack_batch(batch, dataset=self.dataset)
                    embeddings = self.feature_extractor(images.to(self.device))
                    embeddings = embeddings.view(embeddings.size(0), -1)
                    all_embeddings.append(embeddings)
                    all_labels.append(labels.to(self.device))
            final_embeddings = torch.cat(all_embeddings, dim=0)
            final_labels = torch.cat(all_labels, dim=0)
            embedding_dataset = EmbeddingPairDataset(final_embeddings, final_labels,
                                            asset_col_name=config["embds"].asset_col_name,
                                            label_col_name=config["embds"].label_col_name)
        
            combined_ds, stats = augment_client_with_generated(
                client_train=embedding_dataset,
                gen_dataset=config["embds"],
                counts=counts,    
                strategy='threshold', 
                fill_to=int(len(embedding_dataset)/10),
                threshold=int(len(embedding_dataset)/10),
                rng_seed=42
            )

            print(f"Cliente {self.cid}: adicionou {stats['gen_selected_count']} amostras geradas para as classes {stats['desired_labels']}")
            
            trainloader = DataLoader(combined_ds, batch_size=self.batch_size, shuffle=True)

        # Treina o modelo classificador
        train_alvo_start_time = time.time()
        train_loss = train_alvo(
            net=self.net,
            trainloader=trainloader,
            epochs=self.local_epochs_alvo,
            lr=self.lr_alvo,
            device=self.device,
            dataset=self.dataset
        )
        train_classifier_time  = time.time() - train_alvo_start_time

        return (
            get_weights(self.net),
            len(trainloader.dataset),
            {"train_loss": train_loss,
            "tempo_treino_alvo": train_classifier_time,
            },
        )

    def evaluate(self, parameters, config):

        set_weights(self.net, parameters)

        # --- Restore discriminator model parameters ---
        if "net_state_dict" in self.client_state.parameters_records:
            rec_net = self.client_state.parameters_records["net_state_dict"]
            arr_net = rec_net["state_bytes"].numpy()
            buf_net = io.BytesIO(arr_net.tobytes())
            state_dict = torch.load(buf_net, map_location=self.device)
            self.feature_extractor.load_state_dict(state_dict)

        # test_time_start = time.time()

        # Avalia o modelo classificador
        #loss, accuracy = test(self.net, self.testloader, self.device, dataset=self.dataset)
        # test_time = time.time() - test_time_start

        # Avalia o modelo classificador localmente
        local_test_start_time = time.time()
        local_acc = local_test(
            net=self.net,
            feature_extractor=self.feature_extractor,
            testloader=self.testloader_local,
            device=self.device,
            acc_filepath=f"{self.folder}/accuracy_report.txt",
            epoch=int(config["round"]),
            cliente=self.cid,
            continue_epoch=self.continue_epoch,
            dataset=self.dataset
        )

        local_test_time = time.time() - local_test_start_time

        return (0.0,
                len(self.testloader_local.dataset),
                {
                 "local_test_time": local_test_time
                 "local_accuracy": local_acc
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
    strategy           = context.run_config["strategy"]

    local_epochs_alvo  = context.run_config["epocas_alvo"]
    local_epochs_disc  = context.run_config["epocas_disc"]
    lr_disc            = context.run_config["learn_rate_disc"]
    lr_alvo            = context.run_config["learn_rate_alvo"]
    latent_dim         = context.run_config["tam_ruido"]
    seed               = context.run_config["seed"]
    if seed == 42:
        trial = 1
    elif seed == 30:
        trial = 2
    elif seed == 20:
        trial = 3
    else:
        trial = seed
    folder             = f"{context.run_config['Exp_name_folder']}FLEG/{dataset}_{partitioner}_{strategy}_numchunks{num_chunks}_ganepochs{local_epochs_disc}_trial{trial}"
    num_epochs         = context.run_config["num_epocas"]


    if continue_epoch != 0:
        checkpoint = torch.load(f"{folder}/checkpoint_epoch{continue_epoch}.pth", map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        gan.load_state_dict(checkpoint['discs_state_dict'][partition_id])
        optim_D.load_state_dict(checkpoint['optimDs_state_dict'][partition_id])


    # Return Client instance
    return FlowerClient(cid=partition_id,
                        local_epochs_alvo=local_epochs_alvo, 
                        local_epochs_disc=local_epochs_disc,
                        dataset=dataset,
                        batch_size=batch_size,
                        lr_alvo=lr_alvo,
                        lr_disc=lr_disc,
                        latent_dim=latent_dim,
                        context=context,
                        folder=folder,
                        num_chunks=num_chunks,
                        num_partitions=num_partitions,
                        partitioner=partitioner,
                        alpha=alpha_dir,
                        continue_epoch=continue_epoch,
                        num_epochs=num_epochs,
                        seed=seed,
                        teste=teste).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn
)
