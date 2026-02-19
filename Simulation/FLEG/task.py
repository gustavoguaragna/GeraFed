"""FLEG: um framework para balancear dados heterogêneos em aprendizado federado, com precupações com a privacidade."""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from typing import Optional, List, Dict, Tuple
from collections import defaultdict, Counter, OrderedDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, Partitioner
from typing import Optional, List, Union
from datasets import Dataset
import numpy as np
import random

class Net(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor1(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(FeatureExtractor1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        return x

# Parte 2: O novo classificador (o restante da arquitetura)
class ClassifierHead1(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(ClassifierHead1, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        if x.dim() == 2:
              x = x.view(-1, 6, 12, 12)

        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor2(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(FeatureExtractor2, self).__init__()
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4) # Achatamento (Flatten)
        return x

# Parte 2: O novo classificador (o restante da arquitetura)
class ClassifierHead2(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(ClassifierHead2, self).__init__()
        # A camada de entrada aqui deve ter o mesmo tamanho do embedding de saída
        # Camadas totalmente conectadas (até a que gera o embedding)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84) # A saída desta camada será nosso embedding
        self.fc3 = nn.Linear(84, 10) # Entrada 84, saída 10

    def forward(self, x):
        # x aqui é o embedding gerado pelo FeatureExtractor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) # Embedding de 84 dimensões
        x = self.fc3(x)
        return x
    
class FeatureExtractor3(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(FeatureExtractor3, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4) 
        x = F.relu(self.fc1(x))
        return x

# Parte 2: O novo classificador (o restante da arquitetura)
class ClassifierHead3(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(ClassifierHead3, self).__init__()
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor4(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(FeatureExtractor4, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Parte 2: O novo classificador (o restante da arquitetura)
class ClassifierHead4(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(ClassifierHead4, self).__init__()
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = self.fc3(x)
        return x
    
class Net_Cifar(nn.Module):
    def __init__(self,seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor1_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(FeatureExtractor1_Cifar, self).__init__()
        # Input: 3 channels (RGB)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # 32x32 -> 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))
        return x

class ClassifierHead1_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(ClassifierHead1_Cifar, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Reshape if input comes flattened from a GAN (Batch, 1176) -> (Batch, 6, 14, 14)
        if x.dim() == 2:
            x = x.view(-1, 6, 14, 14)

        # 14x14 -> 10x10 -> 5x5
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # or x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor2_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(FeatureExtractor2_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # Output size: 16*5*5 = 400
        return x

class ClassifierHead2_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(ClassifierHead2_Cifar, self).__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x is the flattened embedding (size 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor3_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(FeatureExtractor3_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

class ClassifierHead3_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(ClassifierHead3_Cifar, self).__init__()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class FeatureExtractor4_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(FeatureExtractor4_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ClassifierHead4_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(ClassifierHead4_Cifar, self).__init__()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.fc3(x)
        return x

# --- CLASSE BASE PARA REUTILIZAR O FORWARD E LOSS ---
class BaseEmbeddingGAN(nn.Module):
    def __init__(self):
        super(BaseEmbeddingGAN, self).__init__()
        self.adv_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_tensor, labels=None):
        # Se for 4D (ex: vindo do FeatureExtractor1), achata
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.view(input_tensor.size(0), -1)

        if input_tensor.shape[1] == self.latent_dim: # Modo Gerador
            # --- Generator Forward Pass ---
            if self.condition and labels is not None:
                embedded_labels = self.label_embedding(labels)
                gen_input = torch.cat((input_tensor, embedded_labels), dim=1)
                return self.generator(gen_input)
            else:
                return self.generator(input_tensor)

        elif input_tensor.shape[1] == self.embedding_dim: # Modo Discriminador
            # --- Discriminator Forward Pass ---
            if self.condition and labels is not None:
                embedded_labels = self.label_embedding(labels)
                disc_input = torch.cat((input_tensor, embedded_labels), dim=1)
                return self.discriminator(disc_input)
            else:
                return self.discriminator(input_tensor)
        else:
            raise ValueError(f"Input tensor shape {input_tensor.shape} invalid. Expected dim {self.latent_dim} (Gen) or {self.embedding_dim} (Disc).")

    def loss(self, output, label):
        return self.adv_loss(output, label)

# --- GAN NÍVEL 1 (Embedding Dim: 864) ---
class EmbeddingGAN1(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=864, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN1, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024), # LayerNorm aceita Batch Size 1
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

# --- GAN NÍVEL 2 (Embedding Dim: 256) ---
class EmbeddingGAN2(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=256, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN2, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim # 16 * 4 * 4 = 256
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LayerNorm(512), # Correção para Batch Size 1
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

# --- GAN NÍVEL 3 (Embedding Dim: 120) ---
class EmbeddingGAN3(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=120, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN3, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

# --- GAN NÍVEL 4 (Embedding Dim: 84) ---
class EmbeddingGAN4(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=84, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN4, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )

# --- GAN NÍVEL 1 CIFAR (Embedding Dim: 1176) ---
class EmbeddingGAN1_Cifar(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=1176, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN1_Cifar, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        # Generator: Latent -> 1176
        self.generator = nn.Sequential(
            nn.Linear(gen_in, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, self.embedding_dim)
        )

        # Discriminator: 1176 -> Real/Fake
        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

class EmbeddingGAN2_Cifar(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=400, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN2_Cifar, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

class EmbeddingGAN3_Cifar(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=120, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN3_Cifar, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )
    
class EmbeddingGAN4_Cifar(BaseEmbeddingGAN):
    def __init__(self, latent_dim=128, embedding_dim=84, condition=True, seed=42):
        if seed is not None: torch.manual_seed(seed)
        super(EmbeddingGAN4_Cifar, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.condition = condition
        self.classes = 10
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None

        gen_in = latent_dim + self.classes if condition else latent_dim
        disc_in = embedding_dim + self.classes if condition else embedding_dim

        self.generator = nn.Sequential(
            nn.Linear(gen_in, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.embedding_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(disc_in, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )

class GeneratedAssetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 generator,
                 num_samples,
                 latent_dim=128,
                 num_classes=10,
                 asset_shape=None,
                 desired_classes=None,
                 device=torch.device("cpu"),
                 asset_col_name="image",
                 label_col_name="label"):
        """
        Generates a dataset of assets (images or embeddings) using a conditional generative model.

        Args:
            generator: The pre-trained generative model.
            num_samples (int): Total number of assets to generate.
            latent_dim (int): Dimension of the latent space vector (z).
            num_classes (int): Total classes the generator was trained on.
            asset_shape (tuple, optional): The shape of a single generated asset.
                                           Crucial for handling empty datasets correctly.
                                           e.g., (1, 28, 28) for MNIST images, (256,) for embeddings.
            desired_classes (list[int], optional): List of class indices to generate. Defaults to all.
            device (torch.device): Device to run generation on.
            asset_col_name (str): Name for the generated asset column.
            label_col_name (str): Name for the label column.
        """
        self.generator = generator
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.total_num_classes = num_classes
        self.asset_shape = asset_shape
        self.device = device
        self.asset_col_name = asset_col_name
        self.label_col_name = label_col_name

        # This logic for selecting classes
        if desired_classes is not None and len(desired_classes) > 0:
            if not all(0 <= c < self.total_num_classes for c in desired_classes):
                raise ValueError(f"All desired classes must be integers between 0 and {self.total_num_classes - 1}")
            self._actual_classes_to_generate = sorted(list(set(desired_classes)))
        else:
            self._actual_classes_to_generate = list(range(self.total_num_classes))

        self.classes = self._actual_classes_to_generate
        self.num_generated_classes = len(self.classes)

        if self.num_generated_classes == 0 and self.num_samples > 0:
             raise ValueError("Cannot generate samples with an empty list of desired classes.")
        elif self.num_samples == 0:
             print("Warning: num_samples is 0. Dataset will be empty.")
             self.assets = torch.empty(0, *self.asset_shape) if self.asset_shape else torch.empty(0)
             self.labels = torch.empty(0, dtype=torch.long)
        else:
             if self.asset_shape is None:
                 raise ValueError("asset_shape must be provided when num_samples > 0.")
             self.assets, self.labels = self.generate_data()


    def generate_data(self):
        """Generates assets and corresponding labels for the specified classes."""
        self.generator.eval()
        self.generator.to(self.device)

        # Label generation logic
        generated_labels_list = []
        if self.num_generated_classes > 0:
            samples_per_class = self.num_samples // self.num_generated_classes
            remainder = self.num_samples % self.num_generated_classes
            for cls in self._actual_classes_to_generate:
                generated_labels_list.extend([cls] * samples_per_class)
            if remainder > 0:
                generated_labels_list.extend(random.choices(self._actual_classes_to_generate, k=remainder))
            random.shuffle(generated_labels_list)
        labels = torch.tensor(generated_labels_list, dtype=torch.long, device=self.device)

        z = torch.randn(self.num_samples, self.latent_dim, device=self.device)
        generated_assets_list = []
        batch_size = min(1024, self.num_samples) if self.num_samples > 0 else 1

        with torch.no_grad():
            for i in range(0, self.num_samples, batch_size):
                z_batch = z[i : i + batch_size]
                labels_batch = labels[i : i + batch_size]
                if z_batch.shape[0] == 0: continue

                # --- Simplified Generator Call ---
                # This single call works for your EmbeddingGAN and other common conditional GANs
                # that take (z, labels) as input.
                gen_assets = self.generator(z_batch, labels_batch)
                generated_assets_list.append(gen_assets)

        if generated_assets_list:
            all_gen_assets = torch.cat(generated_assets_list, dim=0)
        else:
            # Use asset_shape for a robust empty tensor
            print("Warning: No images generated. Returning empty tensor for images.")
            all_gen_assets = torch.empty(0, *self.asset_shape, device=self.device)

        return all_gen_assets, labels

    def __len__(self):
        return self.assets.shape[0]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Dataset index out of range")
        return {
            self.asset_col_name: self.assets[idx], # MODIFIED
            self.label_col_name: int(self.labels[idx])
        }

class ClassPartitioner(Partitioner):
    """Partitions a dataset by class, ensuring each class appears in exactly one partition.

    Attributes:
        num_partitions (int): Total number of partitions to create
        seed (int, optional): Random seed for reproducibility
        label_column (str): Name of the column containing class labels
    """

    def __init__(
        self,
        num_partitions: int,
        seed: Optional[int] = None,
        label_column: str = "label"
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._seed = seed
        self._label_column = label_column
        self._partition_indices: Optional[List[List[int]]] = None

    def _create_partitions(self) -> None:
        """Create class-based partitions and store indices."""
        # Extract labels from dataset
        labels = self.dataset[self._label_column]

        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)

        classes = list(class_indices.keys())
        num_classes = len(classes)

        # Validate number of partitions
        if self._num_partitions > num_classes:
            raise ValueError(
                f"Cannot create {self._num_partitions} partitions with only {num_classes} classes. "
                f"Reduce partitions to ≤ {num_classes}."
            )

        # Shuffle classes for random distribution
        rng = random.Random(self._seed)
        rng.shuffle(classes)

        # Split classes into partitions
        partition_classes = np.array_split(classes, self._num_partitions)

        # Create index lists for each partition
        self._partition_indices = []
        for class_group in partition_classes:
            indices = []
            for cls in class_group:
                indices.extend(class_indices[cls])
            self._partition_indices.append(indices)

    @property
    def dataset(self) -> Dataset:
        return super().dataset

    @dataset.setter
    def dataset(self, value: Dataset) -> None:
        # Use parent setter for basic validation
        super(ClassPartitioner, ClassPartitioner).dataset.fset(self, value)

        # Create partitions once dataset is set
        self._create_partitions()

    def load_partition(self, partition_id: int) -> Dataset:
        """Load a partition containing exclusive classes.

        Args:
            partition_id: The ID of the partition to load (0-based index)

        Returns:
            Dataset: Subset of the dataset containing only the specified partition's data
        """
        if not self.is_dataset_assigned():
            raise RuntimeError("Dataset must be assigned before loading partitions")
        if partition_id < 0 or partition_id >= self.num_partitions:
            raise ValueError(f"Invalid partition ID: {partition_id}")

        return self.dataset.select(self._partition_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    def __repr__(self) -> str:
        return (f"ClassPartitioner(num_partitions={self._num_partitions}, "
                f"seed={self._seed}, label_column='{self._label_column}')")


def _get_item_label(item, label_key='label'):
    """Return integer label from a dataset item.
       Supports: dict-like with label_key, tuple (x,y) where y is label, or single label.
    """
    if isinstance(item, dict):
        return int(item[label_key])
    if isinstance(item, (list, tuple)):
        # common pattern: (x, y) or (x, y, ...)
        maybe_label = item[1] if len(item) > 1 else item[0]
        return int(maybe_label)
    # fallback
    return int(item)

def get_label_counts(dataset, label_key='label', max_samples=None) -> Counter:
    """Return a Counter mapping label -> count for any torch Dataset or Subset.
       If max_samples is set, stops after examining that many samples (useful for huge datasets).
    """
    # Handle Subset
    indices = None
    if isinstance(dataset, Subset):
        indices = dataset.indices
        underlying = dataset.dataset
    else:
        indices = None
        underlying = dataset

    counts = Counter()
    n = 0
    if indices is not None:
        for idx in indices:
            item = underlying[idx]
            counts[_get_item_label(item, label_key)] += 1
            n += 1
            if max_samples and n >= max_samples:
                break
    else:
        # iterate directly (some datasets may be large; optionally limit by max_samples)
        for item in dataset:
            counts[_get_item_label(item, label_key)] += 1
            n += 1
            if max_samples and n >= max_samples:
                break
    return counts

def choose_minority_labels(counts: Counter,
                           total_num_classes: int = 10,
                           method: str = 'topk',
                           k: Optional[int] = None,
                           threshold: Optional[int] = None,
                           ratio: Optional[float] = None) -> List[int]:
    """Select labels considered 'minority' according to `method`.
       - method='topk': pick k labels with smallest counts (k required)
       - method='threshold': pick labels with counts <= threshold (threshold required)
       - method='ratio': pick labels whose count <= ratio * max_count (ratio required)
    """

    full_counts = {lbl: counts.get(lbl, 0) for lbl in range(total_num_classes)}

    if method == 'topk':
        if k is None:
            raise ValueError("k must be provided for topk method")
        # return k labels with smallest counts (ties arbitrary)
        return [l for l, _ in sorted(full_counts.items(), key=lambda kv: kv[1])][:k]
    elif method == 'threshold':
        if threshold is None:
            raise ValueError("threshold must be provided for threshold method")
        return [l for l, c in full_counts.items() if c <= threshold]
    elif method == 'ratio':
        if ratio is None:
            raise ValueError("ratio must be provided for ratio method")
        maxc = max(full_counts.values()) if full_counts else 0
        return [l for l, c in full_counts.items() if c <= ratio * maxc]
    else:
        raise ValueError(f"Unknown method '{method}'.")


def build_label_index_map(dataset, label_key='label') -> Dict[int, List[int]]:
    """Precompute a mapping from label -> list of indices for a dataset.
       Works for any dataset that returns dicts/tuples with a label.
    """
    label_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        lbl = _get_item_label(item, label_key)
        label_to_indices[int(lbl)].append(idx)
    return label_to_indices

def sample_generated_indices_for_labels(gen_dataset,
                                        desired_labels: List[int],
                                        num_per_label: Dict[int,int],
                                        label_key='label',
                                        rng_seed=None) -> List[int]:
    """Return a list of indices from gen_dataset sampling num_per_label[l] indices for each label l.
       If not enough generated samples exist for a label, it will sample as many as available and warn.
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    label_map = build_label_index_map(gen_dataset, label_key=label_key)
    chosen = []
    for lbl in desired_labels:
        available = label_map.get(lbl, [])
        need = num_per_label.get(lbl, 0)
        if need <= 0:
            continue
        if len(available) == 0:
            print(f"Warning: no generated samples for label {lbl}")
            continue
        if len(available) < need:
            print(f"Warning: requested {need} for label {lbl} but only {len(available)} available; taking all.")
            chosen.extend(available)
        else:
            chosen.extend(random.sample(available, need))
    return chosen

class _IndexWrappingDataset(torch.utils.data.Dataset):
    """Wrap a dataset subset (selected indices) and optionally remap output keys / shapes.
       This wrapper returns exactly what the original dataset returns for each index.
    """
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = list(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.base[self.indices[idx]]
    

def unpack_batch(batch, dataset):
    """
    Accepts either:
     - a dict batch -> {'image': tensor, 'label': tensor}
     - a tuple/list batch -> (images, labels, ...)
     - a pair (images, labels)
    Returns (images_tensor, labels_tensor)
    """
    if dataset == "mnist":
        image_key = "image"
    elif dataset == "cifar10":
        image_key = "img"
    else:
        raise ValueError(f"self.dataset deveria ser mnist ou cifar10, {dataset} não reconhecido")
            
    label_key = "label"

    if isinstance(batch, dict):
        images = batch[image_key]
        labels = batch[label_key]
    elif isinstance(batch, (list, tuple)):
        # common: (images, labels) or ((images, ...), labels) etc.
        if len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            raise ValueError("Tuple batch with unexpected length")
    else:
        raise ValueError("Unsupported batch type: %s" % type(batch))
    return images, labels

# Optionally, you can wrap it to return dicts similar to generated dataset:
class EmbeddingPairDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels, asset_col_name='image', label_col_name='label'):
        self.emb = embeddings
        self.lbl = labels
        self.asset_col_name = asset_col_name
        self.label_col_name = label_col_name
    def __len__(self):
        return self.emb.size(0)
    def __getitem__(self, idx):
        return {self.asset_col_name: self.emb[idx], self.label_col_name: int(self.lbl[idx])}
    
def augment_client_with_generated(client_train,
                                  gen_dataset,
                                  counts,
                                  label_key_client='label',
                                  label_key_gen='label',
                                  strategy='fill_to_max',
                                  fill_to: Optional[int] = None,
                                  k: Optional[int] = None,
                                  threshold: Optional[int] = None,
                                  ratio: Optional[float] = None,
                                  rng_seed: Optional[int] = None) -> Tuple[torch.utils.data.Dataset, Dict]:
    """
    Combine client_train with selected samples from gen_dataset.

    strategy options:
      - 'fill_to_max' (default): for each label, augment until it has as many samples as the *max* label count in the client.
         (i.e., balance up to current max class size)
      - 'fill_to': provide fill_to (int), augment each minority label up to `fill_to` samples.
      - 'topk': choose k least frequent labels and augment each up to `fill_to` (you must give k and fill_to).
      - 'threshold': choose labels with count <= threshold and fill them to fill_to (needs threshold and fill_to).
      - 'ratio': choose labels with count <= ratio * max_count and fill to fill_to (needs ratio and fill_to).

    Returns: (combined_dataset, stats_dict)
    """
    # Step 1: counts
    if len(counts)==0:
        print("Warning: client_train has zero samples.")
    max_count = max(counts.values()) if counts else 0

    # Step 2: decide labels
    if strategy == 'fill_to_max':
        # pick labels that are below max_count
        desired_labels = [l for l, c in counts.items() if c < max_count]
        per_label_target = {l: max_count for l in desired_labels}
    elif strategy == 'fill_to':
        if fill_to is None:
            raise ValueError("fill_to must be provided for 'fill_to' strategy")
        desired_labels = [l for l, c in counts.items() if c < fill_to]
        per_label_target = {l: fill_to for l in desired_labels}
    elif strategy == 'topk':
        if k is None or fill_to is None:
            raise ValueError("k and fill_to required for topk")
        desired_labels = choose_minority_labels(counts, method='topk', k=k)
        per_label_target = {l: fill_to for l in desired_labels}
    elif strategy == 'threshold':
        if threshold is None or fill_to is None:
            raise ValueError("threshold and fill_to required for threshold")
        desired_labels = choose_minority_labels(counts, method='threshold', threshold=threshold)
        per_label_target = {l: fill_to for l in desired_labels}
    elif strategy == 'ratio':
        if ratio is None or fill_to is None:
            raise ValueError("ratio and fill_to required for ratio")
        desired_labels = choose_minority_labels(counts, method='ratio', ratio=ratio)
        per_label_target = {l: fill_to for l in desired_labels}
    else:
        raise ValueError("Unknown strategy")

    # compute how many extra samples we need per label
    need_per_label = {}
    for l in desired_labels:
        need = per_label_target[l] - counts.get(l, 0)
        if need > 0:
            need_per_label[l] = need

    # Step 3: sample from generated dataset
    chosen_gen_indices = sample_generated_indices_for_labels(
        gen_dataset,
        desired_labels=desired_labels,
        num_per_label=need_per_label,
        label_key=label_key_gen,
        rng_seed=rng_seed
    )

    gen_subset = _IndexWrappingDataset(gen_dataset, chosen_gen_indices) if chosen_gen_indices else None

    # Step 4: combine
    if gen_subset is None or len(gen_subset) == 0:
        combined = client_train  # nothing to add
    else:
        # ConcatDataset expects Dataset instances; Subset is acceptable.
        combined = ConcatDataset([client_train, gen_subset])

    stats = {
        'client_counts': counts,
        'desired_labels': desired_labels,
        'need_per_label': need_per_label,
        'gen_selected_count': len(chosen_gen_indices)
    }
    return combined, stats

fds = None  # Cache FederatedDataset
gen_img_part = None # Cache GeneratedDataset

def load_data(partition_id: int, 
              num_partitions: int,
              batch_size: int = 32,
              dataset: str = "mnist",
              teste: bool = False,
              partitioner_type: str = "IID",
              alpha_dir: float = 0.1, 
              num_chunks: int = 1) -> tuple[Union[DataLoader, List], DataLoader, DataLoader]:
    
    """Carrega MNIST com splits de treino e teste separados. Se syn_samples > 0, inclui dados gerados."""
   
    global fds

    if fds is None:
        print(f"Carregamento dos Dados - {dataset}")
        if partitioner_type == "Dir":
            print("Dados por Dirichlet")
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha_dir,
                min_partition_size=0,
                self_balancing=False
            )
        elif partitioner_type == "Class":
            print("Dados por classe")
            partitioner = ClassPartitioner(num_partitions=num_partitions, seed=42)
        else:
            print("Dados IID")
            partitioner = IidPartitioner(num_partitions=num_partitions)

        fds = FederatedDataset(
            dataset=dataset,
            partitioners={"train": partitioner}
        )

    # Carrega a partição de treino e teste separadamente
    test_partition = fds.load_split("test")
    train_partition = fds.load_partition(partition_id, split="train")

    from collections import Counter
    labels = train_partition["label"]
    class_distribution = Counter(labels)
    print(f"CID {partition_id}: {class_distribution}")
        
    if teste:
        print("reduzindo dataset para modo teste")
        num_samples = int(len(train_partition)/10)
        train_partition = train_partition.select(range(num_samples))

    if dataset == "mnist":
        pytorch_transforms = Compose([
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
        image = "image"
    elif dataset == "cifar10":
        pytorch_transforms = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = "img"
    else:
        raise ValueError(f"{dataset} not identified")
    
    def apply_transforms(batch):
        batch[image] = [pytorch_transforms(img) for img in batch[image]]
        return batch

    train_partition = train_partition.with_transform(apply_transforms)
    test_partition = test_partition.with_transform(apply_transforms)

    test_frac = 0.2

    total     = len(train_partition)
    test_size = int(total * test_frac)
    train_size = total - test_size

    client_train, client_test = random_split(
        train_partition,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    client_dataset = {
        "train": client_train,
        "test":  client_test,
    }


    if num_chunks > 1:
        n_real = len(client_dataset["train"])

        # 1) embaralha os índices com seed fixa
        indices_real = list(range(n_real))
        random.seed(42)
        random.shuffle(indices_real)

        # 2) calcula tamanho aproximado de cada chunk
        chunk_size_real = math.ceil(n_real / num_chunks)

        # 3) divide em chunks usando fatias dos índices embaralhados
        chunks_real = []
        for i in range(num_chunks):
            start = i * chunk_size_real
            end = min((i + 1) * chunk_size_real, n_real)
            chunk_indices = indices_real[start:end]
            chunk_subset = Subset(client_dataset["train"], chunk_indices)
            chunks_real.append(chunk_subset)

        trainloader_real = [DataLoader(chunk, batch_size=batch_size, shuffle=True) for chunk in chunks_real if len(chunk) > 0]

    else:
        trainloader_real = DataLoader(client_dataset["train"], batch_size=batch_size, shuffle=True)

    testloader = DataLoader(test_partition, batch_size=64, shuffle=True)
    testloader_local = DataLoader(client_dataset["test"], batch_size=64, shuffle=True)

    return trainloader_real, testloader, testloader_local

def train_alvo(net, trainloader, epochs, lr, device, dataset):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    if dataset == "mnist":
        image = "image"
    elif dataset == "cifar10":
        image = "img"
    else:
        raise ValueError(f"Dataset {dataset} nao identificado. Deveria ser 'mnist' ou 'cifar10'")
    
    #treinou = False
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[image]
            labels = batch["label"]
            # if images.size(0) == 1:
            #     print("Batch size is 1, skipping batch")
            #     continue
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss

def train_disc(gen, disc, trainloader, feature_extractor, epochs, device, optim, dataset="mnist", latent_dim=128):
    """Train the network on the training set."""
    if dataset == "mnist":
      image = "image"
    elif dataset == "cifar10":
      image = "img"
    else:
        raise ValueError(f"{dataset} nao identificado, deveria ser mnist ou cifar10")
    
    gen.to(device)  # move model to GPU if available
    disc.to(device)  # move model to GPU if available

    d_loss_b = 0
    total_samples = 0
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch[image].to(device), batch["label"].to(device)
            batch_size = images.size(0)
            if batch_size == 1:
                print("Batch size is 1, skipping batch")
                continue

            images = feature_extractor(images)
            
            real_ident = torch.full((batch_size, 1), 1., device=device)
            fake_ident = torch.full((batch_size, 1), 0., device=device)

            # Train D
            optim.zero_grad()

            # Dados reais
            y_real = disc(images, labels)
            d_real_loss = disc.loss(y_real, real_ident)

            # Dados falsos
            z_noise = torch.randn(batch_size, latent_dim, device=device)
            x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
            x_fake = gen(z_noise, x_fake_labels).detach()
            y_fake_d = disc(x_fake, x_fake_labels)
            d_fake_loss = disc.loss(y_fake_d, fake_ident)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optim.step()

            d_loss_b += d_loss.item() * batch_size
            total_samples += batch_size

            if batch_idx % 50 == 0 and batch_idx > 0:
                print('Epoch {} [{}/{}] loss_D_treino: {:.4f}'.format(
                        epoch, batch_idx, len(trainloader),
                        d_loss.mean().item())) 
            
                
    avg_d_loss = d_loss_b / total_samples if total_samples != 0 else 0.0
    return avg_d_loss            
        
                
def train_G(net: nn.Module, discs: list, device: str, lr: float, epochs: int, batch_size: int, latent_dim: int, optim_state_dict = None):
    net.to(device)  # move model to GPU if available

    optim_G = torch.optim.Adam(list(net.generator.parameters())+list(net.label_embedding.parameters()), lr=lr, betas=(0.5, 0.999))
    if optim_state_dict:
        optim_G.load_state_dict(optim_state_dict)

    g_losses = []

    for _ in range(epochs):
        # Train G
        optim_G.zero_grad()

        z_noise = torch.randn(batch_size, latent_dim, device=device)
        x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)

        x_fake = net(z_noise, x_fake_labels)

        # Calcula a média das saídas dos discriminadores
        y_fake_gs = [disc(x_fake.detach(), x_fake_labels) for disc in discs]
        y_fake_g_means = [torch.mean(y).item() for y in y_fake_gs]

        # Escolhe o discriminador com a maior média
        Dmax = discs[y_fake_g_means.index(max(y_fake_g_means))]

        real_ident = torch.full((batch_size, 1), 1., device=device)
        y_fake_g = Dmax(x_fake, x_fake_labels)
        
        g_loss = net.loss(y_fake_g, real_ident)
        g_loss.backward()
        optim_G.step()

        g_losses.append(g_loss.item())

    return np.mean(g_losses), optim_G.state_dict()

def test(net, testloader, device, dataset, level):
    """Validate the model on the test set."""
    net.to(device)
    if dataset == "mnist":
        image = "image"
    elif dataset == "cifar10":
        image = "img"
    else:
        raise ValueError(f"Dataset {dataset} nao identificado. Deveria ser 'mnist' ou 'cifar10'")
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[image].to(device)
            labels = batch["label"].to(device)
            if level == 0:
                outputs = net(images)
            else:
                outputs = class_head(feature_extractor(images))
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
    
def local_test(net: nn.Module,
               feature_extractor: nn.Module,
               testloader: DataLoader,
               device: str,
               acc_filepath: str,
               epoch: int,
               cliente: str,
               num_classes: int = 10,
               continue_epoch: int = 0,
               dataset: str = "mnist"):
    client_eval_time = time.time()
    # Evaluation in client test
    # Initialize counters
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    predictions_counter = defaultdict(int)

    net.eval()
    net.to(device)

    if dataset == "mnist":
        image = "image"
    elif dataset == "cifar10":
        image = "img"
    else:
        raise ValueError(f"dataset nao identificado")
    
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch[image].to(device), batch["label"].to(device)
            if feature_extractor:
                feature_extractor.eval()
                images = feature_extractor(images)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            # Update counts for each sample in batch
            for true_label, pred_label in zip(labels, predicted):
                true_idx = true_label.item()
                pred_idx = pred_label.item()

                class_total[true_idx] += 1
                predictions_counter[pred_idx] += 1

                if true_idx == pred_idx:
                    class_correct[true_idx] += 1

        # Create results dictionary
        results_metrics = {
            "class_metrics": {},
            "overall_accuracy": None,
            "prediction_distribution": dict(predictions_counter)
        }

        # Calculate class-wise metrics
        for i in range(num_classes):
            metrics = {
                "samples": class_total[i],
                "predictions": predictions_counter[i],
                "accuracy": class_correct[i] / class_total[i] if class_total[i] > 0 else "N/A"
            }
            results_metrics["class_metrics"][f"class_{i}"] = metrics

        # Calculate overall accuracy
        total_samples = sum(class_total.values())
        results_metrics["overall_accuracy"] = sum(class_correct.values()) / total_samples

        # Save to txt file
        with open(acc_filepath, "a") as f:
            f.write(f"Epoch {epoch+continue_epoch} - Client {cliente}\n")
            # Header with fixed widths
            f.write("{:<10} {:<10} {:<10} {:<10}\n".format(
                "Class", "Accuracy", "Samples", "Predictions"))
            f.write("-"*45 + "\n")

            # Class rows with consistent formatting
            for cls in range(num_classes):
                metrics = results_metrics["class_metrics"][f"class_{cls}"]

                # Format accuracy (handle "N/A" case)
                accuracy = (f"{metrics['accuracy']:.4f}"
                            if isinstance(metrics['accuracy'], float)
                            else "  N/A  ")

                f.write("{:<10} {:<10} {:<10} {:<10}\n".format(
                    f"Class {cls}",
                    accuracy,
                    metrics['samples'],
                    metrics['predictions']
                ))

            # Footer with alignment
            f.write("\n{:<20} {:.4f}".format("Overall Accuracy:", results_metrics["overall_accuracy"]))
            f.write("\n{:<20} {}".format("Total Samples:", total_samples))
            f.write("\n{:<20} {}".format("Total Predictions:", sum(predictions_counter.values())))
            f.write("\n{:<20} {:.4f}".format("Client Evaluation Time:", time.time() - client_eval_time))
            f.write("\n")
            f.write("\n")

    print("Results saved to accuracy_report.txt")

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def get_weights_gen(net):
    return [val.cpu().numpy() for key, val in net.state_dict().items() if 'discriminator' in key or 'label' in key]

def set_weights(net, parameters):
    device = next(net.parameters()).device
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)