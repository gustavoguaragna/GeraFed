"""FLEG: um framework para balancear dados heterogêneos em aprendizado federado, com precupações com a privacidade."""

import torch
from torch.utils.data import DataLoader, random_split, Subset
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import numpy as np
from collections import OrderedDict
import random
import math

# Imports do task.py e do script original adaptados
from task import (
    Net, Net_Cifar,
    ClassifierHead1, ClassifierHead2, ClassifierHead3, ClassifierHead4,
    ClassifierHead1_Cifar, ClassifierHead2_Cifar, ClassifierHead3_Cifar, ClassifierHead4_Cifar,
    FeatureExtractor1, FeatureExtractor2, FeatureExtractor3, FeatureExtractor4,
    FeatureExtractor1_Cifar, FeatureExtractor2_Cifar, FeatureExtractor3_Cifar, FeatureExtractor4_Cifar,
    EmbeddingGAN1, EmbeddingGAN2, EmbeddingGAN3, EmbeddingGAN4,
    EmbeddingGAN1_Cifar, EmbeddingGAN2_Cifar, EmbeddingGAN3_Cifar, EmbeddingGAN4_Cifar,
    GeneratedAssetDataset, EmbeddingPairDataset, augment_client_with_generated,
    get_label_counts, unpack_batch
)

# Configuração de device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(partition_id, num_partitions, dataset_name, alpha=0.1):
    """Carrega a partição de dados para este cliente."""
    # Recria o particionador usado no servidor para garantir consistência
    # Para simulação eficiente, usamos flwr_datasets
    
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=alpha,
        min_partition_size=0,
        self_balancing=False
    )
    
    fds = FederatedDataset(
        dataset=dataset_name,
        partitioners={"train": partitioner}
    )
    
    partition = fds.load_partition(partition_id, split="train")
    
    # Transformações
    from torchvision.transforms import Compose, ToTensor, Normalize
    if dataset_name == "mnist":
        transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        img_key = "image"
    else:
        transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img_key = "img"

    def apply_transforms(batch):
        batch[img_key] = [transforms(img) for img in batch[img_key]]
        return batch

    partition = partition.with_transform(apply_transforms)
    
    # Split local train/test (FLEG usa local_test_frac=0.2)
    # Como flwr_datasets retorna Dataset, convertemos para lógica PyTorch
    # Para simplificar na simulação, usamos train_test_split do dataset ou indices
    # Aqui retornamos o dataset completo, o split pode ser feito no fit se necessário
    return partition, img_key

class FlegClient(NumPyClient):
    def __init__(self, context: Context, dataset_name, num_partitions):
        self.context = context
        self.partition_id = int(context.node_config["partition-id"])
        self.dataset_name = dataset_name
        self.dataset, self.img_key = load_data(self.partition_id, num_partitions, dataset_name)
        self.seed = 42
        
        # Carrega dados em memória (para processamento rápido com DataLoaders)
        # Nota: Em datasets gigantes, isso deve ser evitado, mas MNIST/CIFAR cabe.
        self.train_loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.counts = get_label_counts(self.dataset) # Função auxiliar do task.py

    def _get_classes_for_level(self, level):
        """Helper para instanciar classes baseadas no nível e dataset."""
        suffix = "_Cifar" if self.dataset_name == "cifar10" else ""
        return (
            globals()[f"FeatureExtractor{level}{suffix}"],
            globals()[f"ClassifierHead{level}{suffix}"],
            globals()[f"EmbeddingGAN{level}{suffix}"]
        )

    def _deserialize_model(self, model_class, params_ndarrays):
        """Reconstrói um modelo a partir de ndarrays."""
        model = model_class(self.seed).to(DEVICE)
        params_dict = zip(model.state_dict().keys(), params_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v).to(DEVICE) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        level = config["level"]
        phase = config["phase"]
        
        # --- FASE 1: CLASSIFICAÇÃO ---
        if phase == "classification":
            if level == 0:
                # Treino da Net Completa
                net_class = Net if self.dataset_name == "mnist" else Net_Cifar
                model = self._deserialize_model(net_class, parameters)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                model.train()
                
                # Treinamento padrão
                for _ in range(1): # 1 época local por round
                    for batch in self.train_loader:
                        images = batch[self.img_key].to(DEVICE)
                        labels = batch["label"].to(DEVICE)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                        loss.backward()
                        optimizer.step()
                
                return [val.cpu().numpy() for _, val in model.state_dict().items()], len(self.dataset), {}

            else:
                # Treino do ClassifierHead (Level > 0)
                FeatClass, HeadClass, GanClass = self._get_classes_for_level(level)
                
                # Deserializa Feature Extractor (fixo)
                # No flwr, Parameters é serializado. Precisa converter de volta se veio no config.
                from flwr.common import parameters_to_ndarrays
                fe_params = parameters_to_ndarrays(config["feature_extractor_params"])
                feature_extractor = self._deserialize_model(FeatClass, fe_params)
                feature_extractor.eval() # Fixo

                # Deserializa Classifier Head (treinável)
                class_head = self._deserialize_model(HeadClass, parameters)
                optimizer = torch.optim.Adam(class_head.parameters(), lr=0.01)

                # --- DATA AUGMENTATION ---
                # Se houver gerador no config, gera dados sintéticos
                train_loader = self.train_loader
                if "generator_params" in config:
                    gen_params = parameters_to_ndarrays(config["generator_params"])
                    # Instancia o GAN completo para pegar o generator
                    full_gan = GanClass(condition=True, seed=self.seed).to(DEVICE)
                    
                    # Carrega pesos no generator
                    gen_state_dict = OrderedDict({k: torch.tensor(v).to(DEVICE) for k, v in zip(full_gan.generator.state_dict().keys(), gen_params)})
                    full_gan.generator.load_state_dict(gen_state_dict)
                    
                    # Gera Dataset Sintético
                    num_syn = 1000 # Simplificação
                    gen_dataset = GeneratedAssetDataset(
                        generator=full_gan.generator,
                        num_samples=num_syn,
                        latent_dim=128,
                        num_classes=10,
                        asset_shape=(full_gan.embedding_dim,),
                        device=DEVICE
                    )
                    
                    # Prepara dataset de embeddings reais
                    # (Precisamos converter as imagens locais em embeddings para misturar com os sintéticos)
                    all_embs, all_lbls = [], []
                    with torch.no_grad():
                        for batch in self.train_loader:
                            imgs = batch[self.img_key].to(DEVICE)
                            lbls = batch["label"].to(DEVICE)
                            embs = feature_extractor(imgs).view(imgs.size(0), -1)
                            all_embs.append(embs)
                            all_lbls.append(lbls)
                    
                    real_emb_ds = EmbeddingPairDataset(torch.cat(all_embs), torch.cat(all_lbls))
                    
                    # Combina (Augmentation)
                    combined_ds, _ = augment_client_with_generated(
                        client_train=real_emb_ds,
                        gen_dataset=gen_dataset,
                        counts=self.counts,
                        strategy='threshold',
                        threshold=int(len(real_emb_ds)/10),
                        rng_seed=42
                    )
                    train_loader = DataLoader(combined_ds, batch_size=32, shuffle=True)

                # Treino do Head com dados (Reais+Sintéticos) ou só Reais
                class_head.train()
                for batch in train_loader:
                    # Se for dataset combinado, ele retorna dicionário ou tupla de embeddings
                    # Se for loader original, retorna imagens.
                    # Lógica para tratar os dois casos:
                    if isinstance(batch, dict) or (isinstance(batch, list) and len(batch)==2 and batch[0].dim() == 2):
                        # É embedding
                        if isinstance(batch, dict):
                            inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE) # "image" key em EmbeddingPairDataset é o embedding
                        else:
                            inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                        
                        optimizer.zero_grad()
                        outputs = class_head(inputs)
                        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    else:
                        # É imagem crua (não houve augmentation ou falha)
                        imgs, lbls = batch[self.img_key].to(DEVICE), batch["label"].to(DEVICE)
                        optimizer.zero_grad()
                        embs = feature_extractor(imgs) # Passa pelo extrator on-the-fly
                        outputs = class_head(embs)
                        loss = torch.nn.CrossEntropyLoss()(outputs, lbls)
                        loss.backward()
                        optimizer.step()

                return [val.cpu().numpy() for _, val in class_head.state_dict().items()], len(self.dataset), {}

        # --- FASE 2: GAN ---
        elif phase == "gan":
            # Nível + 1 porque a Strategy já atualizou a arquitetura para o próximo nível
            # Mas o config['level'] ainda pode estar refletindo o contador. 
            # Assumimos que a Strategy enviou os params corretos para o Discriminador
            # Precisamos instanciar o modelo correto.
            # Se a strategy diz "gan" e "level 0" (mas preparando para 1), precisamos carregar GAN1.
            # Vamos confiar que a strategy enviou params compatíveis com GAN(level+1)
            
            target_level = level + 1 
            FeatClass, _, GanClass = self._get_classes_for_level(target_level)
            
            # Carrega Feature Extractor (Fixo)
            from flwr.common import parameters_to_ndarrays
            fe_params = parameters_to_ndarrays(config["feature_extractor_params"])
            feature_extractor = self._deserialize_model(FeatClass, fe_params)
            feature_extractor.eval()

            # Carrega Generator (Fixo nesta etapa local - gera fakes)
            gen_params = parameters_to_ndarrays(config["generator_params"])
            full_gan_temp = GanClass(condition=True, seed=self.seed).to(DEVICE)
            gen_state = OrderedDict({k: torch.tensor(v).to(DEVICE) for k, v in zip(full_gan_temp.generator.state_dict().keys(), gen_params)})
            full_gan_temp.generator.load_state_dict(gen_state)
            generator = full_gan_temp.generator
            generator.eval()

            # Carrega Discriminador (Treinável)
            # Os parâmetros recebidos no `fit` são do Discriminador
            discriminator = full_gan_temp.discriminator
            disc_params = zip(discriminator.state_dict().keys(), parameters)
            disc_state = OrderedDict({k: torch.tensor(v).to(DEVICE) for k, v in disc_params})
            discriminator.load_state_dict(disc_state)
            
            optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
            discriminator.train()
            
            # Loop de Treino do Discriminador (Real vs Fake)
            # FLEG original: itera sobre chunks. Aqui iteramos sobre o dataset local.
            for batch in self.train_loader:
                real_imgs = batch[self.img_key].to(DEVICE)
                real_labels = batch["label"].to(DEVICE)
                bs = real_imgs.size(0)
                if bs == 1: continue

                # Features Reais
                with torch.no_grad():
                    real_embs = feature_extractor(real_imgs)
                    if real_embs.dim() == 4: real_embs = real_embs.view(bs, -1)

                # Fake
                z = torch.randn(bs, 128, device=DEVICE)
                fake_labels = torch.randint(0, 10, (bs,), device=DEVICE)
                with torch.no_grad():
                    fake_embs = generator(z, fake_labels)

                # Train D
                optim_D.zero_grad()
                
                # Real Loss
                # Discriminator espera inputs concatenados se condicional
                # O EmbeddingGAN.forward lida com isso se labels forem passados
                out_real = discriminator(real_embs, real_labels)
                loss_real = torch.nn.BCEWithLogitsLoss()(out_real, torch.full_like(out_real, 1.0))
                
                # Fake Loss
                out_fake = discriminator(fake_embs, fake_labels)
                loss_fake = torch.nn.BCEWithLogitsLoss()(out_fake, torch.full_like(out_fake, 0.0))
                
                loss_d = (loss_real + loss_fake) / 2
                loss_d.backward()
                optim_D.step()

            return [val.cpu().numpy() for _, val in discriminator.state_dict().items()], len(self.dataset), {}

def client_fn(context: Context):
    return FlegClient(context, dataset_name="mnist", num_partitions=4).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn
)