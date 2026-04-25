"""FLEG: um framework para balancear dados heterogêneos em aprendizado federado, com precupações com a privacidade."""

import torch
from torch.utils.data import DataLoader, ConcatDataset
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, array_from_numpy
from Simulation.FLEG.task import (
    DictTensorDataset,
    augment_client_with_generated,
    Net,
    Net_Cifar,
    ClassifierHead1, ClassifierHead2, ClassifierHead3, ClassifierHead4,
    ClassifierHead1_Cifar, ClassifierHead2_Cifar, ClassifierHead3_Cifar, ClassifierHead4_Cifar,
    EmbeddingGAN0, EmbeddingGAN1, EmbeddingGAN2, EmbeddingGAN3,
    EmbeddingGAN0_Cifar, EmbeddingGAN1_Cifar, EmbeddingGAN2_Cifar, EmbeddingGAN3_Cifar,
    EmbeddingPairDataset,
    FeatureExtractor1, FeatureExtractor2, FeatureExtractor3, FeatureExtractor4,
    FeatureExtractor1_Cifar, FeatureExtractor2_Cifar, FeatureExtractor3_Cifar, FeatureExtractor4_Cifar,
    get_label_counts,
    get_weights,
    get_weights_disc,
    load_data, 
    local_test,
    set_weights,
    set_weights_gen,
    test, 
    train_alvo, 
    train_disc,
    unpack_batch
)
from typing import Union
import pickle
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
                cid: int, 
                local_epochs_alvo: int, 
                local_epochs_disc: int, 
                dataset: str, 
                batch_size: int,
                lr_alvo: float,
                lr_disc: float,
                latent_dim: int, 
                context: Context,
                trainloader: Union[DataLoader, list],
                testloader_local: DataLoader,
                folder: str = ".",
                num_chunks: int = 1,
                num_partitions: int = 4,
                partitioner: str = "ClassPartitioner",
                alpha: Union[float, None] = None,
                continue_epoch: int = 0,
                seed: int = 42,
                ):
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
        self.num_classes = 10 if dataset in ["mnist", "cifar10"] else None
        self.seed = seed
        
        self.trainloader = trainloader
        self.testloader_local = testloader_local     

        if self.dataset == "mnist":
            self.net = Net(seed=self.seed)
            self.asset_name = "image"
        elif self.dataset == "cifar10":
            self.net = Net_Cifar(seed=self.seed)
            self.asset_name = "img"  


    def fit(self, parameters, config):

        if config["model"] == "gan":
            if config["level"] == 0:
                if self.dataset == "mnist":
                    self.gen = EmbeddingGAN0(seed=self.seed)
                    self.disc = EmbeddingGAN0(seed=self.seed)
                    self.feature_extractor = FeatureExtractor1(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.gen = EmbeddingGAN0_Cifar(seed=self.seed)
                    self.disc = EmbeddingGAN0_Cifar(seed=self.seed)
                    self.feature_extractor = FeatureExtractor1_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")

            elif config["level"] == 1:
                if self.dataset == "mnist":
                    self.gen = EmbeddingGAN1(seed=self.seed)
                    self.disc = EmbeddingGAN1(seed=self.seed)
                    self.feature_extractor = FeatureExtractor2(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.gen = EmbeddingGAN1_Cifar(seed=self.seed)
                    self.disc = EmbeddingGAN1_Cifar(seed=self.seed)
                    self.feature_extractor = FeatureExtractor2_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")

            elif config["level"] == 2:
                if self.dataset == "mnist":
                    self.gen = EmbeddingGAN2(seed=self.seed)
                    self.disc = EmbeddingGAN2(seed=self.seed)
                    self.feature_extractor = FeatureExtractor3(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.gen = EmbeddingGAN2_Cifar(seed=self.seed)
                    self.disc = EmbeddingGAN2_Cifar(seed=self.seed)
                    self.feature_extractor = FeatureExtractor3_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")

            elif config["level"] == 3:
                if self.dataset == "mnist":
                    self.gen = EmbeddingGAN3(seed=self.seed)
                    self.disc = EmbeddingGAN3(seed=self.seed)
                    self.feature_extractor = FeatureExtractor4(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.gen = EmbeddingGAN3_Cifar(seed=self.seed)
                    self.disc = EmbeddingGAN3_Cifar(seed=self.seed)
                    self.feature_extractor = FeatureExtractor4_Cifar(seed=self.seed)
                else:
                    raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {self.dataset} não reconhecido")
            
            else:
                raise ValueError(f"Treino da GAN vai até nível 3 (4° nível), não deveria receber config['level'] {config['level']}.")

            self.optim_D = torch.optim.Adam(list(self.disc.discriminator.parameters())+list(self.disc.label_embedding.parameters()), lr=self.lr_disc, betas=(0.5, 0.999))

            # Atualiza pesos do modelo generativo
            set_weights_gen(net=self.gen, parameters=parameters)

            # --- Restore discriminator model parameters ---
            if "disc_state_dict" in self.client_state.parameters_records and not config["new_lvl"]:
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

            # --- Restore feature extractor parameters ---
            if "net_state_dict" in self.client_state.parameters_records:
                rec_net = self.client_state.parameters_records["net_state_dict"]
                arr_net = rec_net["state_bytes"].numpy()
                buf_net = io.BytesIO(arr_net.tobytes())
                state_dict = torch.load(buf_net, map_location=self.device)
                self.feature_extractor.load_state_dict(state_dict, strict=False)

            # Define o dataloader
            if isinstance(self.trainloader, list):
                chunk_idx = config["round"] % len(self.trainloader)
                trainloader_chunk = self.trainloader[chunk_idx]
            else:
                trainloader_chunk = self.trainloader

            train_disc_start_time = time.time()
            # Treina a discriminadora
            avg_d_loss = train_disc(
            disc=self.disc,
            gen=self.gen,
            feature_extractor=self.feature_extractor,
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
            for k, v in self.disc.state_dict().items():
                # Convert to NumPy, then to Array. Add to self.client_state
                p_record[k] = array_from_numpy(v.detach().cpu().numpy())
            # Add to a context
            self.client_state.parameters_records["disc_parameters"] = p_record
            
            # Save optimizer state_dict fully (state + param_groups)
            # --- Save discriminator model parameters ---
            buf_model = io.BytesIO()
            torch.save(self.disc.state_dict(), buf_model)

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


            disc_params = get_weights_disc(self.disc)

            return (
            disc_params,
            len(trainloader_chunk.dataset),
            {"train_d_loss": avg_d_loss,
            "tempo_treino_disc": train_disc_time,
            "cid": self.cid,
            },
        )

        elif config["model"] == "classifier":

            if isinstance(self.trainloader, list):
                trainloader = DataLoader(
                    ConcatDataset([dl.dataset for dl in self.trainloader]),
                    batch_size=self.batch_size,
                    shuffle=True
                )
            else:
                trainloader = self.trainloader

            if config["level"] == 0:
                if self.dataset == "mnist":
                    self.classifier = Net(seed=self.seed)
                elif self.dataset == "cifar10":
                    self.classifier = Net_Cifar(seed=self.seed)
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
                raise ValueError(f"Treino vai até nível 4 (5° nível), não deveria receber config['level'] {config['level']}.")
            

            # Atualiza pesos do modelo classificador
            set_weights(self.classifier, parameters)

            # --- Restore model parameters ---
            if "net_state_dict" in self.client_state.parameters_records:
                rec_net = self.client_state.parameters_records["net_state_dict"]
                arr_net = rec_net["state_bytes"].numpy()
                buf_net = io.BytesIO(arr_net.tobytes())
                state_dict = torch.load(buf_net, map_location=self.device)
                self.net.load_state_dict(state_dict)
                if config["level"] > 0:
                    self.feature_extractor.load_state_dict(state_dict, strict=False)
                if config["use_best_model"]:
                    self.classifier.load_state_dict(state_dict, strict=False)
            
            if config["best_model"]:
                set_weights(self.net, parameters)
                # Save all elements of the state_dict into a single RecordSet
                net_record = ParametersRecord()
                for k, v in self.net.state_dict().items():
                    # Convert to NumPy, then to Array. Add to self.client_state
                    net_record[k] = array_from_numpy(v.detach().cpu().numpy())

                # Add to a context
                self.client_state.parameters_records["net_parameters"] = net_record

                # --- Save model parameters ---
                buf_net = io.BytesIO()
                torch.save(self.net.state_dict(), buf_net)

                # Convert bytes → numpy array (uint8)
                arr_net = np.frombuffer(buf_net.getvalue(), dtype=np.uint8)

                # Store in ParametersRecord
                rec_net = ParametersRecord()
                rec_net["state_bytes"] = array_from_numpy(arr_net)
                self.client_state.parameters_records["net_state_dict"] = rec_net
            
            # Synthetic Data Augmentation
            # if checkpoint_level is not None and level == checkpoint_level:
                # generated_dataset = checkpoint_loaded['generated_dataset']

            # 1. Carrega os dados recebidos via pickle (arrays puros)
            dados_recebidos = pickle.loads(config["embds"])

            if len(dados_recebidos["assets"]) > 0:    
                
                # 2. Converte para Tensores
                gen_assets = torch.from_numpy(dados_recebidos["assets"]).to(self.device)
                gen_labels = torch.from_numpy(dados_recebidos["labels"]).to(self.device)

                #Salvar no contexto
                record_data = ParametersRecord()
                record_data["assets"] = array_from_numpy(dados_recebidos["assets"])
                record_data["labels"] = array_from_numpy(dados_recebidos["labels"])
                self.client_state.parameters_records["emb_sin"] = record_data

                
                # 3. Instancia o Wrapper (você define os nomes aqui UMA única vez)

                embds_wrapper = DictTensorDataset(
                    assets=gen_assets,
                    labels=gen_labels,
                    asset_col_name=self.asset_name, # Ajuste se no seu caso original era "embedding"
                    label_col_name="label"
                )

                # Junta com dados locais
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
                
                # 4. Puxando os atributos diretamente da instância do wrapper
                embedding_dataset = EmbeddingPairDataset(
                    final_embeddings, 
                    final_labels,
                    asset_col_name=embds_wrapper.asset_col_name, 
                    label_col_name=embds_wrapper.label_col_name   
                )

                combined_ds, stats = augment_client_with_generated(
                    client_train=embedding_dataset,
                    gen_dataset=embds_wrapper,
                    counts=counts,    
                    strategy='threshold', 
                    fill_to=int(len(embedding_dataset)/10),
                    threshold=int(len(embedding_dataset)/10),
                    rng_seed=42
                )

                print(f"Cliente {self.cid}: adicionou {stats['gen_selected_count']} amostras geradas para as classes {stats['desired_labels']}")
                
                trainloader_aug = DataLoader(combined_ds, batch_size=self.batch_size, shuffle=True)

            elif "emb_sin" in self.client_state.parameters_records:
                record_data = self.client_state.parameters_records["emb_sin"]
            
                # Recupera para PyTorch
                assets_tensor = torch.from_numpy(record_data["assets"].numpy()).to(self.device)
                labels_tensor = torch.from_numpy(record_data["labels"].numpy()).to(self.device)
                
                # Cria um dataset simples do PyTorch para ser usado no DataLoader
                embds_wrapper = DictTensorDataset(
                    assets=assets_tensor, 
                    labels=labels_tensor,
                    asset_col_name=self.asset_name, # Ajuste se você usou nomes diferentes na classe original
                    label_col_name="label"
                )
                print(f"Dataset recuperado do contexto! Total de amostras: {len(dataset)}")

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
                
                # 4. USO PURISTA: Puxando os atributos diretamente da instância do wrapper!
                # Repare que isso é 100% idêntico ao seu código original.
                embedding_dataset = EmbeddingPairDataset(
                    final_embeddings, 
                    final_labels,
                    asset_col_name=embds_wrapper.asset_col_name, 
                    label_col_name=embds_wrapper.label_col_name   
                )

                combined_ds, stats = augment_client_with_generated(
                    client_train=embedding_dataset,
                    gen_dataset=embds_wrapper,
                    counts=counts,    
                    strategy='threshold', 
                    fill_to=int(len(embedding_dataset)/10),
                    threshold=int(len(embedding_dataset)/10),
                    rng_seed=42
                )

                print(f"Cliente {self.cid}: adicionou {stats['gen_selected_count']} amostras geradas para as classes {stats['desired_labels']}")
                
                trainloader_aug = DataLoader(combined_ds, batch_size=self.batch_size, shuffle=True)
            else:
                trainloader_aug = trainloader

            # Treina o modelo classificador
            train_alvo_start_time = time.time()
            train_loss = train_alvo(
                net=self.classifier,
                trainloader=trainloader_aug,
                epochs=self.local_epochs_alvo,
                lr=self.lr_alvo,
                device=self.device,
                dataset=self.dataset
            )
            train_classifier_time  = time.time() - train_alvo_start_time

            return (
                get_weights(self.classifier),
                len(trainloader.dataset),
                {"train_loss": train_loss,
                "tempo_treino_alvo": train_classifier_time,
                },
            )

        else:
            raise ValueError(f"config[model] deveria ser 'gan' ou 'classifier', mas obteve {config['model']}")

    def evaluate(self, parameters, config):

        # --- Restore model parameters ---
        if "net_state_dict" in self.client_state.parameters_records:
            rec_net = self.client_state.parameters_records["net_state_dict"]
            arr_net = rec_net["state_bytes"].numpy()
            buf_net = io.BytesIO(arr_net.tobytes())
            state_dict = torch.load(buf_net, map_location=self.device)
            self.net.load_state_dict(state_dict, strict=False)

        set_weights(self.net, parameters)

        # test_time_start = time.time()

        # Avalia o modelo classificador
        #loss, accuracy = test(self.net, self.testloader, self.device, dataset=self.dataset)
        # test_time = time.time() - test_time_start

        # Avalia o modelo classificador localmente
        local_test_start_time = time.time()
        local_acc = local_test(
            net=self.net,
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
                 "local_test_time": local_test_time,
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
    syn_input         = context.run_config["syn_input"]

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
    folder             = f"{context.run_config['Exp_name_folder']}FLEG/{dataset}_{partitioner}_{strategy}_numchunks{num_chunks}_ganepochs{local_epochs_disc}_{syn_input}_fleg_trial{trial}"


    if continue_epoch != 0:
        checkpoint = torch.load(f"{folder}/checkpoint_epoch{continue_epoch}.pth", map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        gan.load_state_dict(checkpoint['discs_state_dict'][partition_id])
        optim_D.load_state_dict(checkpoint['optimDs_state_dict'][partition_id])

    trainloader, _, testloader_local = load_data(
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
                        seed=seed,
                        trainloader=trainloader,
                        testloader_local=testloader_local).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn
)
