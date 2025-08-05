"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, Partitioner
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from torchvision.transforms.functional import to_pil_image
from datasets import Features, ClassLabel, Image
import random
import torchvision
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from PIL import Image as IMG
import os
from scipy import linalg
import time
from tqdm import tqdm
from flwr.common import parameters_to_ndarrays
from collections import defaultdict, OrderedDict
from typing import Optional, List
import math
from torchvision.utils import save_image

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Para garantir determinismo total em operações com CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, seed=42):
        if seed is not None:
          torch.manual_seed(seed)
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Define the GAN model
class CGAN(nn.Module):
    def __init__(self, dataset="mnist", img_size=28, latent_dim=100):
        super(CGAN, self).__init__()
        if dataset == "mnist":
            self.classes = 10
            self.channels = 1
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.label_embedding = nn.Embedding(self.classes, self.classes)
        self.adv_loss = torch.nn.BCELoss()


        self.generator = nn.Sequential(
            *self._create_layer_gen(self.latent_dim + self.classes, 128, False),
            *self._create_layer_gen(128, 256),
            *self._create_layer_gen(256, 512),
            *self._create_layer_gen(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            *self._create_layer_disc(self.classes + int(np.prod(self.img_shape)), 1024, False, True),
            *self._create_layer_disc(1024, 512, True, True),
            *self._create_layer_disc(512, 256, True, True),
            *self._create_layer_disc(256, 128, False, False),
            *self._create_layer_disc(128, 1, False, False),
            nn.Sigmoid()
        )

        #self._initialize_weights()

    def _create_layer_gen(self, size_in, size_out, normalize=True):
        layers = [nn.Linear(size_in, size_out)]
        if normalize:
            layers.append(nn.BatchNorm1d(size_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    
    def _create_layer_disc(self, size_in, size_out, drop_out=True, act_func=True):
        layers = [nn.Linear(size_in, size_out)]
        if drop_out:
            layers.append(nn.Dropout(0.4))
        if act_func:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def _initialize_weights(self):
        # Itera sobre todos os módulos da rede geradora
        for m in self.generator:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, input, labels):
        # device = input.device  # Ensure all tensors are on the same device
        # labels = labels.to(device)  # Move labels to the same device as input
        
        if input.dim() == 2:
            z = torch.cat((self.label_embedding(labels), input), -1)
            x = self.generator(z)
            x = x.view(x.size(0), *self.img_shape) #Em
            return x 
        elif input.dim() == 4:
            x = torch.cat((input.view(input.size(0), -1), self.label_embedding(labels)), -1)
            return self.discriminator(x)

    def loss(self, output, label):
        return self.adv_loss(output, label)
    
class F2U_GAN(nn.Module):
    def __init__(self, dataset="mnist", img_size=28, latent_dim=128, condition=True, seed=42):
        if seed is not None:
          torch.manual_seed(seed)
        super(F2U_GAN, self).__init__()
        if dataset == "mnist":
            self.classes = 10
            self.channels = 1
        else:
            raise NotImplementedError("Only MNIST is supported")

        self.condition = condition
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None
        #self.label_embedding_disc = nn.Embedding(self.classes, self.img_size*self.img_size) if condition else None
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.input_shape_gen = self.latent_dim + self.label_embedding.embedding_dim if condition else self.latent_dim
        self.input_shape_disc = self.channels + self.classes if condition else self.channels

        self.adv_loss = torch.nn.BCEWithLogitsLoss()

        # Generator (unchanged) To calculate output shape of convtranspose layers, we can use the formula:
        # output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding (or dilation * (kernel_size - 1) + 1 inplace of kernel_size if using dilation)
        self.generator = nn.Sequential(
            nn.Linear(self.input_shape_gen, 256 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (256,7,7) -> (128,14,14)
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (128,14,14) -> (64,28,28)
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.channels, kernel_size=3, stride=1, padding=1), # (64,28,28) -> (1,28,28)
            nn.Tanh()
        )

        # Discriminator (corrected) To calculate output shape of conv layers, we can use the formula:
        # output_shape = ⌊(input_shape - kernel_size + 2 * padding) / stride + 1⌋ (or (dilation * (kernel_size - 1) - 1) inplace of kernel_size if using dilation)
        self.discriminator = nn.Sequential(
        # Camada 1: (1,28,28) -> (32,13,13)
        nn.utils.spectral_norm(nn.Conv2d(self.input_shape_disc, 32, kernel_size=3, stride=2, padding=0)),
        nn.LeakyReLU(0.2, inplace=True),

        # Camada 2: (32,14,14) -> (64,7,7)
        nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),

        # Camada 3: (64,7,7) -> (128,3,3)
        nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)),
        nn.LeakyReLU(0.2, inplace=True),

        # Camada 4: (128,3,3) -> (256,1,1)
        nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)),  # Padding 0 aqui!
        nn.LeakyReLU(0.2, inplace=True),

        # Achata e concatena com as labels
        nn.Flatten(), # (256,1,1) -> (256*1*1,)
        nn.utils.spectral_norm(nn.Linear(256 * 1 * 1, 1))  # 256 (features)
        )

    def forward(self, input, labels=None):
        if input.dim() == 2:
            # Generator forward pass (unchanged)
            if self.condition:
                embedded_labels = self.label_embedding(labels)
                gen_input = torch.cat((input, embedded_labels), dim=1)
                x = self.generator(gen_input)
            else:
                x = self.generator(input)
            return x.view(-1, *self.img_shape)

        elif input.dim() == 4:
            # Discriminator forward pass
            if self.condition:
                embedded_labels = self.label_embedding(labels)
                image_labels = embedded_labels.view(embedded_labels.size(0), self.label_embedding.embedding_dim, 1, 1).expand(-1, -1, self.img_size, self.img_size)
                x = torch.cat((input, image_labels), dim=1)
            else:
                x = input
            return self.discriminator(x)

    def loss(self, output, label):
        return self.adv_loss(output, label)


# def generate_images(cgan, examples_per_class):
#     cgan.eval()
#     device = next(cgan.parameters()).device
#     latent_dim = cgan.latent_dim
#     classes = cgan.classes
    
#     # Cria um vetor de rótulos balanceado
#     labels = torch.tensor([i for i in range(classes) for _ in range(examples_per_class)], device=device)
#     # Número total de imagens geradas
#     num_samples = examples_per_class * classes
#     # Gera vetores latentes aleatórios
#     z = torch.randn(num_samples, latent_dim, device=device)

#     with torch.no_grad():
#         cgan.eval()
#         gen_imgs = cgan(z, labels)  # [N, C, H, W]

#     # Retorna como lista para aplicar transforms depois
#     # Convertendo para CPU para aplicar transforms com PIL, se necessário
#     gen_imgs_list = [img.cpu() for img in gen_imgs]
#     gen_labels_list = labels.cpu().tolist()

#     #converte para PIL
#     gen_imgs_pil = [to_pil_image((img * 0.5 + 0.5).clamp(0,1)) for img in gen_imgs_list]

#     #Monta dataset como do FederatedDataset
#     features = Features({
#     "image": Image(),
#     "label": ClassLabel(names=[str(i) for i in range(classes)])
#     })

#     # Cria um dicionário com os dados
#     gen_dict = {"image": gen_imgs_pil, "label": gen_labels_list}

#     # Cria o dataset Hugging Face
#     gen_dataset_hf = datasets.Dataset.from_dict(gen_dict, features=features)
#     return gen_dataset_hf

# def split_balanced(gen_dataset_hf, num_clientes):
#     # Identificar as classes
#     class_label = gen_dataset_hf.features["label"]
#     num_classes = class_label.num_classes
    
#     # Cria um mapeamento classe -> índices
#     class_to_indices = {c: [] for c in range(num_classes)}
#     for i in range(len(gen_dataset_hf)):
#         lbl = gen_dataset_hf[i]["label"]
#         class_to_indices[lbl].append(i)
    
#     # Verifica quantos exemplos por classe temos
#     examples_per_class = len(class_to_indices[0])
#     # Garantir que o número de exemplos por classe seja divisível por num_clientes 
#     # (ou lidar com o resto de alguma forma)
#     if examples_per_class % num_clientes != 0:
#         raise ValueError("O número de exemplos por classe não é divisível igualmente pelo número de clientes.")
        
#     # Número de exemplos por classe por cliente
#     examples_per_class_per_client = examples_per_class // num_clientes

#     # Agora, distribui igualmente os índices para cada cliente
#     client_indices = [[] for _ in range(num_clientes)]
#     for c in range(num_classes):
#         # Índices dessa classe
#         idxs = class_to_indices[c]
#         # Divide em partes iguais
#         for i in range(num_clientes):
#             start = i * examples_per_class_per_client
#             end = (i+1) * examples_per_class_per_client
#             client_indices[i].extend(idxs[start:end])
    
#     # Agora criamos um DatasetDict com um subset para cada cliente
#     # Cada cliente terá a mesma quantidade total de exemplos (classes * examples_per_class_per_client)
#     client_datasets = {}
#     for i in range(num_clientes):
#         # Ordena os índices do cliente (opcional, mas pode manter a ordem)
#         client_indices[i].sort()
#         # Seleciona os exemplos
#         client_subset = gen_dataset_hf.select(client_indices[i])
#         client_datasets[i] = client_subset
    
#     return client_datasets


#######################################################################################

# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Class-based partitioner for Hugging Face Datasets."""

 # Assuming this is in the package structure


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
    def dataset(self) -> datasets.Dataset:
        return super().dataset

    @dataset.setter
    def dataset(self, value: datasets.Dataset) -> None:
        # Use parent setter for basic validation
        super(ClassPartitioner, ClassPartitioner).dataset.fset(self, value)

        # Create partitions once dataset is set
        self._create_partitions()

    def load_partition(self, partition_id: int) -> datasets.Dataset:
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


fds = None  # Cache FederatedDataset
gen_img_part = None # Cache GeneratedDataset

def load_data(partition_id: int, 
              num_partitions: int,  
              alpha_dir: float, 
              batch_size: int,
              filter_classes=None,
              teste: bool = False,
              partitioner_type: str = "IID",
              num_chunks: int = 1,
              syn_samples: int = 0,
              gan: Optional[nn.Module] = None) -> tuple[DataLoader, Optional[DataLoader], DataLoader, DataLoader]:
    
    """Carrega MNIST com splits de treino e teste separados. Se syn_samples > 0, inclui dados gerados."""
   
    global fds

    if fds is None:
        print("Carregamento dos Dados")
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
            dataset="mnist",
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

    if filter_classes is not None:
        print("filtrando classes no dataset")
        train_partition = train_partition.filter(lambda x: x["label"] in filter_classes)
        print(f"selecionadas classes: {filter_classes}")
    
    pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    train_partition = train_partition.with_transform(apply_transforms)
    test_partition = test_partition.with_transform(apply_transforms)

    test_frac = 0.2
    client_dataset = []

    total     = len(train_partition)
    test_size = int(total * test_frac)
    train_size = total - test_size

    client_train, client_test = random_split(
        train_partition,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    client_dataset.append({
        "train": client_train,
        "test":  client_test,
    })

    if gan and syn_samples > 0:
        generated_dataset = GeneratedDataset(
            generator=gan.to("cpu"),  # Move GAN to CPU for generation
            num_samples=syn_samples,
            latent_dim=gan.latent_dim,
            device="cpu",  # Use CPU for generation
            image_col_name="image"
        )

        # Cria a pasta de saída se não existir
        os.makedirs("syn_samples", exist_ok=True)
        # Escolhe até 5 índices aleatórios dentro do tamanho do dataset
        num_to_save = min(5, len(generated_dataset))
        sample_idxs = random.sample(range(len(generated_dataset)), num_to_save)

        for i, idx in enumerate(sample_idxs, start=1):
            sample = generated_dataset[idx]
            img = sample["image"]       # tensor C×H×W
            lbl = sample["label"]       # inteiro
            # Aqui você pode incluir qualquer informação no nome do arquivo
            filename = f"syn_samples/syn_img_{i:02d}_idx{idx}_lbl{lbl}_r{round}.png"
            save_image(img, filename)

        train_partition = ConcatDataset([client_dataset["train"], generated_dataset])

    if num_chunks > 1:
        n_gen = len(train_partition)
        n_real = len(client_dataset["train"])

        # 1) embaralha os índices com seed fixa
        indices_gen = list(range(n_gen))
        indices_real = list(range(n_real))
        random.seed(42)
        random.shuffle(indices_gen)
        random.shuffle(indices_real)

        # 2) calcula tamanho aproximado de cada chunk
        chunk_size_gen = math.ceil(n_gen / num_chunks)
        chunk_size_real = math.ceil(n_real / num_chunks)

        # 3) divide em chunks usando fatias dos índices embaralhados
        chunks_gen = []
        chunks_real = []
        for i in range(num_chunks):
            start = i * chunk_size_gen
            end = min((i + 1) * chunk_size_gen, n_gen)
            chunk_indices = indices_gen[start:end]
            chunks_gen.append(Subset(train_partition, chunk_indices))

            start = i * chunk_size_real
            end = min((i + 1) * chunk_size_real, n_real)
            chunk_indices = indices_real[start:end]
            chunks_real.append(Subset(client_dataset["train"], chunk_indices))


        trainloader_syn = [DataLoader(chunk, batch_size=batch_size, shuffle=True) for chunk in chunks_gen if len(chunk) > 0]
        trainloader_real = [DataLoader(chunk, batch_size=batch_size, shuffle=True) for chunk in chunks_real if len(chunk) > 0]

    else:
        trainloader_real = DataLoader(client_dataset["train"], batch_size=batch_size, shuffle=True)
        trainloader_syn = DataLoader(train_partition, batch_size=batch_size, shuffle=True)

    testloader = DataLoader(test_partition, batch_size=batch_size, shuffle=True)
    testloader_local = DataLoader(client_dataset["test"], batch_size=batch_size, shuffle=True)

    return trainloader_real, trainloader_syn, testloader, testloader_local


def train_alvo(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    #treinou = False
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            # if images.size(0) == 1:
            #     print("Batch size is 1, skipping batch")
            #     continue
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #treinou = True

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss

def train_gen(net, trainloader, epochs, lr, device, dataset="mnist", latent_dim=100, f2a: bool = False):
    """Train the network on the training set."""
    if dataset == "mnist":
      imagem = "image"
    elif dataset == "cifar10":
      imagem = "img"
    
    net.to(device)  # move model to GPU if available
    optim_G = torch.optim.Adam(net.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(net.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []


    for epoch in range(epochs):
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch[imagem].to(device), batch["label"].to(device)
            batch_size = images.size(0)
            real_ident = torch.full((batch_size, 1), 1., device=device)
            fake_ident = torch.full((batch_size, 1), 0., device=device)

            if not f2a: 
                # Train G
                optim_G.zero_grad()
                z_noise = torch.randn(batch_size, latent_dim, device=device)
                x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
                x_fake = net(z_noise, x_fake_labels)
                y_fake_g = net(x_fake, x_fake_labels)
                g_loss = net.loss(y_fake_g, real_ident)
                g_loss.backward()
                optim_G.step()

                # Train D
                optim_D.zero_grad()
                y_real = net(images, labels)
                d_real_loss = net.loss(y_real, real_ident)
                y_fake_d = net(x_fake.detach(), x_fake_labels)
                d_fake_loss = net.loss(y_fake_d, fake_ident)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optim_D.step()

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

                if batch_idx % 100 == 0 and batch_idx > 0:
                    print('Epoch {} [{}/{}] loss_D_treino: {:.4f} loss_G_treino: {:.4f}'.format(
                                epoch, batch_idx, len(trainloader),
                                d_loss.mean().item(),
                                g_loss.mean().item())) 

            else:
                # Train D
                optim_D.zero_grad()
                y_real = net(images, labels)
                d_real_loss = net.loss(y_real, real_ident)
                z_noise = torch.randn(batch_size, latent_dim, device=device)
                x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
                x_fake = net(z_noise, x_fake_labels).detach()
                y_fake_d = net(x_fake.detach(), x_fake_labels)
                d_fake_loss = net.loss(y_fake_d, fake_ident)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optim_D.step()

                d_losses.append(d_loss.item())

                if batch_idx % 10 == 0 and batch_idx > 0:
                    print('Epoch {} [{}/{}] loss_D_treino: {:.4f}'.format(
                            epoch, batch_idx, len(trainloader),
                            d_loss.mean().item())) 
                
def train_G(net: CGAN, discs: list, device: str, lr: float, epochs: int, batch_size: int, latent_dim: int):
    net.to(device)  # move model to GPU if available
    optim_G = torch.optim.Adam(net.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        # Train G
        net.zero_grad()
        z_noise = torch.randn(batch_size, latent_dim, device=device)
        x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
        x_fake = net(z_noise, x_fake_labels)
        y_fake_gs = [disc(x_fake.detach(), x_fake_labels) for disc in discs]
        y_fake_g_means = [torch.mean(y).item() for y in y_fake_gs]
        Dmax = discs[y_fake_g_means.index(max(y_fake_g_means))]
        real_ident = torch.full((batch_size, 1), 1., device=device)
        y_fake_g = Dmax(x_fake, x_fake_labels)
        g_loss = net.loss(y_fake_g, real_ident)
        g_loss.backward()
        optim_G.step()

def test(net, testloader, device, model):
    """Validate the model on the test set."""
    net.to(device)
    if model == "gen":
        imagem = "image"
        g_losses = []
        d_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(testloader):
                images, labels = batch[imagem].to(device), batch["label"].to(device)
                batch_size = images.size(0)
                real_ident = torch.full((batch_size, 1), 1., device=device)
                fake_ident = torch.full((batch_size, 1), 0., device=device)
                
                #Gen loss
                z_noise = torch.randn(batch_size, 100, device=device)
                x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
                x_fake = net(z_noise, x_fake_labels)
                y_fake_g = net(x_fake, x_fake_labels)
                g_loss = net.loss(y_fake_g, real_ident)

                #Disc loss
                y_real = net(images, labels)
                d_real_loss = net.loss(y_real, real_ident)
                y_fake_d = net(x_fake.detach(), x_fake_labels)
                d_fake_loss = net.loss(y_fake_d, fake_ident)
                d_loss = (d_real_loss + d_fake_loss) / 2

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

                if batch_idx % 100 == 0 and batch_idx > 0:
                    print('[{}/{}] loss_D_teste: {:.4f} loss_G_teste: {:.4f}'.format(
                                batch_idx, len(testloader),
                                d_loss.mean().item(),
                                g_loss.mean().item()))
        return np.mean(g_losses), np.mean(d_losses)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for batch in testloader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader)
        return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def get_weights_gen(net):
    return [val.cpu().numpy() for key, val in net.state_dict().items() if 'discriminator' in key or 'label' in key]


def set_weights(net, parameters):
    device = next(net.parameters()).device
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# FID

def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`"""
    try:
        version = tuple(map(int, torchvision.__version__.split(".")[:2]))
    except ValueError:
        # Just a caution against weird version strings
        version = (0,)

    # Skips default weight inititialization if supported by torchvision
    # version. See https://github.com/mseitzer/pytorch-fid/issues/28.
    if version >= (0, 6):
        kwargs["init_weights"] = False

    # Backwards compatibility: `weights` argument was handled by `pretrained`
    # argument prior to version 0.13.
    if version < (0, 13) and "weights" in kwargs:
        if kwargs["weights"] == "DEFAULT":
            kwargs["pretrained"] = True
        elif kwargs["weights"] is None:
            kwargs["pretrained"] = False
        else:
            raise ValueError(
                "weights=={} not supported in torchvision {}".format(
                    kwargs["weights"], torchvision.__version__
                )
            )
        del kwargs["weights"]

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, weights=None)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url("https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth", progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3,  # Final average pooling features
    }

    def __init__(
        self,
        output_blocks=(DEFAULT_BLOCK_INDEX,),
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
        use_fid_inception=True,
    ):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, "Last possible output block index is 3"

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(weights="DEFAULT")

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

class GeneratedDataset(Dataset):
    def __init__(self,
                 generator,
                 num_samples,
                 latent_dim=100,
                 num_classes=10, # Total classes the generator model knows
                 desired_classes=None, # Optional: List of specific class indices to generate
                 device="cpu",
                 image_col_name="image",
                 label_col_name="label"):
        """
        Generates a dataset using a conditional generative model, potentially
        focusing on a subset of classes.

        Args:
            generator: The pre-trained generative model.
            num_samples (int): Total number of images to generate across the desired classes.
            latent_dim (int): Dimension of the latent space vector (z).
            num_classes (int): The total number of classes the generator was trained on.
                               This is crucial for correct label conditioning (e.g., one-hot dim).
            desired_classes (list[int], optional): A list of integer class indices to generate.
                                                  If None or empty, images for all classes
                                                  (from 0 to num_classes-1) will be generated,
                                                  distributed as evenly as possible.
                                                  Defaults to None.
            device (str): Device to run generation on ('cpu' or 'cuda').
            image_col_name (str): Name for the image column in the output dictionary.
            label_col_name (str): Name for the label column in the output dictionary.
        """
        self.generator = generator
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        # Store the total number of classes the generator understands
        self.total_num_classes = num_classes
        self.device = device
        self.model_type = type(self.generator).__name__ # Get generator class name
        self.image_col_name = image_col_name
        self.label_col_name = label_col_name

        # Determine the actual classes to generate based on desired_classes
        if desired_classes is not None and len(desired_classes) > 0:
            # Validate that desired classes are within the generator's known range
            if not all(0 <= c < self.total_num_classes for c in desired_classes):
                raise ValueError(f"All desired classes must be integers between 0 and {self.total_num_classes - 1}")
            # Use only the unique desired classes, sorted for consistency
            self._actual_classes_to_generate = sorted(list(set(desired_classes)))
        else:
            # If no specific classes desired, generate all classes
            self._actual_classes_to_generate = list(range(self.total_num_classes))

        # The 'classes' attribute of the dataset reflects only those generated
        self.classes = self._actual_classes_to_generate
        self.num_generated_classes = len(self.classes) # Number of classes being generated

        if self.num_generated_classes == 0 and self.num_samples > 0:
             raise ValueError("Cannot generate samples with an empty list of desired classes.")
        elif self.num_samples == 0:
             print("Warning: num_samples is 0. Dataset will be empty.")
             self.images = torch.empty(0) # Adjust shape if known
             self.labels = torch.empty(0, dtype=torch.long)
        else:
             # Generate the data only if needed
             self.images, self.labels = self.generate_data()


    def generate_data(self):
        """Generates images and corresponding labels for the specified classes."""
        self.generator.eval()
        self.generator.to(self.device)

        # --- Create Labels ---
        generated_labels_list = []
        if self.num_generated_classes > 0:
            # Distribute samples as evenly as possible among the desired classes
            samples_per_class = self.num_samples // self.num_generated_classes
            for cls in self._actual_classes_to_generate:
                generated_labels_list.extend([cls] * samples_per_class)

            # Handle remaining samples if num_samples is not perfectly divisible
            num_remaining = self.num_samples - len(generated_labels_list)
            if num_remaining > 0:
                # Add remaining samples by randomly choosing from the desired classes
                remainder_labels = random.choices(self._actual_classes_to_generate, k=num_remaining)
                generated_labels_list.extend(remainder_labels)

            # Shuffle labels for better distribution in batches later
            random.shuffle(generated_labels_list)

        # Convert labels list to tensor
        labels = torch.tensor(generated_labels_list, dtype=torch.long, device=self.device)

        # Double check label count (should match num_samples due to logic above)
        if len(labels) != self.num_samples:
             # This indicates an unexpected issue, potentially if num_generated_classes was 0 initially
             # but num_samples > 0. Raise error or adjust. Let's adjust defensively.
             print(f"Warning: Label count mismatch. Expected {self.num_samples}, got {len(labels)}. Adjusting size.")
             if len(labels) > self.num_samples:
                 labels = labels[:self.num_samples]
             else:
                 # Pad if too few (less likely with current logic unless num_generated_classes=0)
                 num_needed = self.num_samples - len(labels)
                 if self.num_generated_classes > 0:
                      padding = torch.tensor(random.choices(self._actual_classes_to_generate, k=num_needed), dtype=torch.long, device=self.device)
                      labels = torch.cat((labels, padding))
                 # If no classes to generate from, labels tensor might remain smaller

        # --- Create Latent Noise ---
        z = torch.randn(self.num_samples, self.latent_dim, device=self.device)

        # --- Generate Images in Batches ---
        generated_images_list = []
        # Consider making batch_size configurable
        batch_size = min(1024, self.num_samples) if self.num_samples > 0 else 1

        with torch.no_grad():
            for i in range(0, self.num_samples, batch_size):
                z_batch = z[i : min(i + batch_size, self.num_samples)]
                labels_batch = labels[i : min(i + batch_size, self.num_samples)]

                # Skip if batch is empty (can happen if num_samples = 0)
                if z_batch.shape[0] == 0:
                    continue

                # --- Condition the generator based on its type ---
                if self.model_type == 'Generator': # Assumes input: concat(z, one_hot_label)
                    # One-hot encode labels using the TOTAL number of classes the generator knows
                    labels_one_hot_batch = F.one_hot(labels_batch, num_classes=self.total_num_classes).float()
                    generator_input = torch.cat([z_batch, labels_one_hot_batch], dim=1)
                    gen_imgs = self.generator(generator_input)
                elif self.model_type in ('CGAN', 'F2U_GAN', 'F2U_GAN_CIFAR'): # Assumes input: z, label_index
                    gen_imgs = self.generator(z_batch, labels_batch)
                else:
                    # Handle other potential generator architectures or raise an error
                    raise NotImplementedError(f"Generation logic not defined for model type: {self.model_type}")

                generated_images_list.append(gen_imgs.cpu()) # Move generated images to CPU

        self.generator.cpu() # Move generator back to CPU after generation

        # Concatenate all generated image batches
        if generated_images_list:
            all_gen_imgs = torch.cat(generated_images_list, dim=0)
        else:
            # If no images were generated (e.g., num_samples = 0)
            # Create an empty tensor. Shape needs care - determine from generator or use placeholder.
            # Let's attempt a placeholder [0, C, H, W] - requires knowing C, H, W.
            # For now, a simple empty tensor. User might need to handle this downstream.
            print("Warning: No images generated. Returning empty tensor for images.")
            all_gen_imgs = torch.empty(0)

        return all_gen_imgs, labels.cpu() # Return images and labels (on CPU)

    def __len__(self):
        # Return the actual number of samples generated
        return self.images.shape[0]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Dataset index out of range")
        return {
            self.image_col_name: self.images[idx],
            self.label_col_name: int(self.labels[idx]) # Return label as standard Python int
        }
    
class ImagePathDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = IMG.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    
def select_samples_per_class(dataset, num_samples):
    """
    Selects a specified number of samples per class from the dataset and returns them as tensors.

    Parameters:
    dataset (torch.utils.data.Dataset): The dataset to select samples from.
    num_samples (int): The number of samples to select per class.

    Returns:
    dict: A dictionary where each key corresponds to a class and the value is a tensor of shape [num_samples, 1, 28, 28].
    """
    class_samples = {i: [] for i in range(len(dataset.classes))}
    class_counts = {i: 0 for i in range(len(dataset.classes))}

    for img, label in dataset:
        if class_counts[label] < num_samples:
            class_samples[label].append(img)
            class_counts[label] += 1
        if all(count >= num_samples for count in class_counts.values()):
            break
    else:
        print("Warning: Not all classes have the requested number of samples.")

    # Convert lists of tensors to a single tensor per class
    for label in class_samples:
        if class_samples[label]:  # Check if the list is not empty
            class_samples[label] = torch.stack(class_samples[label], dim=0)
            class_samples[label] = (class_samples[label] + 1) / 2
            class_samples[label] = class_samples[label].repeat(1, 3, 1, 1)
        else:
            # Handle empty classes if necessary; here we leave an empty tensor
            class_samples[label] = torch.Tensor()

    return class_samples

def calculate_fid(instance: str, model_gen: CGAN, dims: int = 2048, param_model=None, samples: int = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    if instance == "server":
        ndarrays = parameters_to_ndarrays(param_model)
        set_weights(model_gen, ndarrays)
    generated_dataset = GeneratedDataset(generator=model_gen, num_samples=samples, latent_dim=100, num_classes=10, device=device)
    gen_dataset = generated_dataset.images
    for c in gen_dataset.keys():
        gen_dataset[c] = (gen_dataset[c] + 1) / 2 #intervalo entre 0 e 1
        gen_dataset[c] = gen_dataset[c].repeat(1, 3, 1, 1) # 3 canais como na rede inception

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity is not available under Windows, use
        # os.cpu_count instead (which may not return the *available* number
        # of CPUs).
        num_cpus = os.cpu_count()
    num_workers = min(8, num_cpus)
    num_workers = 0
    print(f"DEVICE: {device}")
    print(f"DEVICE=cuda? {device=='cuda'}")
    if device == "cuda":
      print("DEVICE CUDA")
      num_workers = 0

    dataloaders = [torch.utils.data.DataLoader(gen_dataset[c], batch_size=64, num_workers=num_workers, shuffle=False) for c in range(10)]

    mus_gen = []
    sigmas_gen = []

    for c in range(10):
        pred_arr = np.empty((len(gen_dataset[c]), dims))
        start_idx = 0
        for batch in tqdm(dataloaders[c]):
                batch = batch.to(device)

                with torch.no_grad():
                    pred = model(batch)[0]

                # If model output is not scalar, apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

                pred = pred.squeeze(3).squeeze(2).cpu().numpy()

                pred_arr[start_idx : start_idx + pred.shape[0]] = pred

                start_idx += pred.shape[0]

        mus_gen.append(np.mean(pred_arr, axis=0))
        sigmas_gen.append(np.cov(pred_arr, rowvar=False))

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    img_reais = select_samples_per_class(testset, num_samples=samples)
    dataloaders = [torch.utils.data.DataLoader(img_reais[c], batch_size=50, num_workers=num_workers, shuffle=False) for c in range(10)]
    mus_real = []
    sigmas_real = []
    
    for c in range(10):
        model = InceptionV3([block_idx]).to(device)
        model.eval()
        pred_arr = np.empty((len(img_reais[c]), dims))
        start_idx = 0
        for batch in tqdm(dataloaders[c]):
                batch = batch.to(device)

                with torch.no_grad():
                    pred = model(batch)[0]

                # If model output is not scalar, apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

                pred = pred.squeeze(3).squeeze(2).cpu().numpy()

                pred_arr[start_idx : start_idx + pred.shape[0]] = pred

                start_idx += pred.shape[0]  # Corrected the operator to +=
        mus_real.append(np.mean(pred_arr, axis=0))
        sigmas_real.append(np.cov(pred_arr, rowvar=False))

    mus_gen = [np.atleast_1d(mu_gen) for mu_gen in mus_gen]
    mus_real = [np.atleast_1d(mu_real) for mu_real in mus_real]

    sigmas_gen = [np.atleast_2d(sigma_gen) for sigma_gen in sigmas_gen]
    sigmas_real = [np.atleast_2d(sigma_real) for sigma_real in sigmas_real]

    for mu_gen, mu_real, sigma_gen, sigma_real in zip(mus_gen, mus_real, sigmas_gen, sigmas_real):
        assert (
            mu_gen.shape == mu_real.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma_gen.shape == sigma_real.shape
        ), "Training and test covariances have different dimensions"

    diffs = [mu_gen - mu_real for mu_gen, mu_real in zip(mus_gen, mus_real)]

    # Product might be almost singular
    covmeans = [linalg.sqrtm(sigmas_gen.dot(sigmas_real), disp=False)[0] for sigmas_gen, sigmas_real in zip(sigmas_gen, sigmas_real)]
    for covmean, sigma_gen, sigma_real in zip(covmeans, sigmas_gen, sigmas_real):
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % 1e-6
            print(msg)
            offset = np.eye(sigma_gen.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma_gen + offset).dot(sigma_real + offset))

    # Numerical error might give slight imaginary component
    for i, covmean in enumerate(covmeans):
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmeans[i] = covmean.real

    tr_covmeans = [np.trace(covmean) for covmean in covmeans]

    fids = [diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_real) - 2 * tr_covmean for diff, sigma_gen, sigma_real, tr_covmean in zip(diffs, sigmas_gen, sigmas_real, tr_covmeans)]
    
    return fids

def generate_plot(net, device, round_number, client_id = None, examples_per_class: int=5, classes: int=10, latent_dim: int=100, server: bool=False):
    """Gera plot de imagens de cada classe"""
    if server:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    else:
        import matplotlib.pyplot as plt

    net.to(device) 
    net.eval()
    batch_size = examples_per_class * classes

    latent_vectors = torch.randn(batch_size, latent_dim, device=device)
    labels = torch.tensor([i for i in range(classes) for _ in range(examples_per_class)], device=device)

    with torch.no_grad():
        generated_images = net(latent_vectors, labels).cpu()

    # Criar uma figura com 10 linhas e 5 colunas de subplots
    fig, axes = plt.subplots(classes, examples_per_class, figsize=(5, 9))

    # Adiciona título no topo da figura
    if isinstance(client_id, int):
        fig.text(0.5, 0.98, f"Round: {round_number} | Client: {client_id}", ha="center", fontsize=12)
    else:
        fig.text(0.5, 0.98, f"Round: {round_number-1}", ha="center", fontsize=12)

    # Exibir as imagens nos subplots
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i, 0, :, :], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    # Ajustar o layout antes de calcular as posições
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])

    # Reduzir espaço entre colunas
    # plt.subplots_adjust(wspace=0.05)

    # Adicionar os rótulos das classes corretamente alinhados
    fig.canvas.draw()  # Atualiza a renderização para obter posições corretas
    for row in range(classes):
        # Obter posição do subplot em coordenadas da figura
        bbox = axes[row, 0].get_window_extent(fig.canvas.get_renderer())
        pos = fig.transFigure.inverted().transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        center_y = (pos[0, 1] + pos[1, 1]) / 2  # Centro exato da linha

        # Adicionar o rótulo
        fig.text(0.04, center_y, str(row), va='center', fontsize=12, color='black')
    
    return fig