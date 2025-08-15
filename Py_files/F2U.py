import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner.partitioner import Partitioner
from collections import Counter, defaultdict
from typing import Optional, List
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm
import math
import random
import numpy as np
import time
import os
import json


class F2U_GAN(nn.Module):
    def __init__(self, dataset="mnist", img_size=28, latent_dim=128, condition=True, seed=None):
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

class GeneratedDataset(torch.utils.data.Dataset):
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
             tqdm.write("Warning: num_samples is 0. Dataset will be empty.")
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
             tqdm.write(f"Warning: Label count mismatch. Expected {self.num_samples}, got {len(labels)}. Adjusting size.")
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
            tqdm.write("Warning: No images generated. Returning empty tensor for images.")
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
    
def generate_plot(net, device, round_number, client_id = None, examples_per_class: int=5, classes: int=10, latent_dim: int=100, folder: str="."):
    """Gera plot de imagens de cada classe"""

    net_type = type(net).__name__
    net.to(device)
    net.eval()
    batch_size = examples_per_class * classes
    dataset = "mnist" if  not net_type == "F2U_GAN_CIFAR" else "cifar10"

    latent_vectors = torch.randn(batch_size, latent_dim, device=device)
    labels = torch.tensor([i for i in range(classes) for _ in range(examples_per_class)], device=device)

    with torch.no_grad():
        if net_type == "Generator":
            labels_one_hot = torch.nn.functional.one_hot(labels, 10).float().to(device)
            generated_images = net(torch.cat([latent_vectors, labels_one_hot], dim=1))
        else:
            generated_images = net(latent_vectors, labels)

    # Criar uma figura com 10 linhas e 5 colunas de subplots
    fig, axes = plt.subplots(classes, examples_per_class, figsize=(5, 9))

    # Adiciona título no topo da figura
    if isinstance(client_id, int):
        fig.text(0.5, 0.98, f"Round: {round_number} | Client: {client_id}", ha="center", fontsize=12)
    else:
        fig.text(0.5, 0.98, f"Round: {round_number}", ha="center", fontsize=12)

    # Exibir as imagens nos subplots
    for i, ax in enumerate(axes.flat):
        if dataset == "mnist":
            ax.imshow(generated_images[i, 0, :, :], cmap='gray')
        else:
            images = (generated_images[i] + 1)/2
            ax.imshow(images.permute(1, 2, 0).clamp(0,1))
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

    if isinstance(client_id, int):
        fig.savefig(f"{folder}/{dataset}{net_type}_r{round_number}_c{client_id}.png")
        tqdm.write("Imagem do cliente salva")
    else:
        fig.savefig(f"{folder}/{dataset}{net_type}_r{round_number}.png")
        tqdm.write("Imagem do servidor salva")
    plt.close(fig)
    return
    
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

num_partitions = 4
alpha_dir = 0.1

partitioner = ClassPartitioner(num_partitions=num_partitions, seed=42, label_column="label")

fds = FederatedDataset(
    dataset="mnist",
    partitioners={"train": partitioner}
)

train_partitions = [fds.load_partition(i, split="train") for i in range(num_partitions)]

# num_samples = [int(len(train_partition)/100) for train_partition in train_partitions]
# train_partitions = [train_partition.select(range(n)) for train_partition, n in zip(train_partitions, num_samples)]

min_lbl_count = 0.05
class_labels = train_partitions[0].info.features["label"]
labels_str = class_labels.names
label_to_client = {lbl: [] for lbl in labels_str}
for idx, ds in enumerate(train_partitions):
    counts = Counter(ds['label'])
    for label, cnt in counts.items():
        if cnt / len(ds) >= min_lbl_count:
            label_to_client[class_labels.int2str(label)].append(idx)

pytorch_transforms = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

def apply_transforms(batch):
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

train_partitions = [train_partition.with_transform(apply_transforms) for train_partition in train_partitions]

testpartition = fds.load_split("test")
testpartition = testpartition.with_transform(apply_transforms)
testloader = DataLoader(testpartition, batch_size=64)

test_frac = 0.2
client_datasets = []

for train_part in train_partitions:
    total     = len(train_part)
    test_size = int(total * test_frac)
    train_size = total - test_size

    client_train, client_test = random_split(
        train_part,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    client_datasets.append({
        "train": client_train,
        "test":  client_test,
    })

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = [F2U_GAN(condition=True, seed=42) for i in range(num_partitions)]
gen = F2U_GAN(condition=True, seed=42).to(device)
optim_G = torch.optim.Adam(gen.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

optim_Ds = [
    torch.optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    for model in models
]

for num_chunks in [1, 10, 100, 1000]:
    seed = 42  # escolha qualquer inteiro para reprodutibilidade
    client_chunks = []

    for train_partition in client_datasets:
        dataset = train_partition["train"]
        n = len(dataset)

        # 1) embaralha os índices com seed fixa
        indices = list(range(n))
        random.seed(seed)
        random.shuffle(indices)

        # 2) calcula tamanho aproximado de cada chunk
        chunk_size = math.ceil(n / num_chunks)

        # 3) divide em chunks usando fatias dos índices embaralhados
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            chunk_indices = indices[start:end]
            chunks.append(Subset(dataset, chunk_indices))

        client_chunks.append(chunks)

    batch_size = 64
    client_test_loaders = [DataLoader(dataset=ds["test"], batch_size=batch_size, shuffle=True) for ds in client_datasets]

    wgan = False
    f2a = False
    epochs = 10
    losses_dict = {"g_losses_chunk": [],
                "d_losses_chunk": [],
                "g_losses_round": [],
                "d_losses_round": [],
                "time_chunk": [],
                "time_round": [],
                "disc_time": [],
                "gen_time": [],
                "track_mismatch_time": []
                }

    epoch_bar = tqdm(range(0, epochs), desc="Treinamento", leave=True, position=0)

    batch_size_gen = 1
    batch_tam = 32
    extra_g_e = 20
    latent_dim = 128
    num_classes = 10
 
    folder = f"num_chunks{num_chunks}"
    os.makedirs(folder, exist_ok=True)
    loss_filename = f"{folder}/losses.json"
    dmax_mismatch_log = f"{folder}/dmax_mismatch.txt"
    lambda_log = "lambda_log.txt"

    for epoch in epoch_bar:
        epoch_start_time = time.time()
        mismatch_count = 0
        total_checked = 0
        g_loss_c = 0.0
        d_loss_c = 0.0
        total_d_samples = 0  # Amostras totais processadas pelos discriminadores
        total_g_samples = 0  # Amostras totais processadas pelo gerador

        chunk_bar = tqdm(range(num_chunks), desc="Chunks", leave=True, position=1)

        for chunk_idx in chunk_bar:
            chunk_start_time = time.time()
            # ====================================================================
            # Treino dos Discriminadores (clientes) no bloco atual
            # ====================================================================
            d_loss_b = 0
            total_chunk_samples = 0


            client_bar = tqdm(enumerate(zip(models, client_chunks)), desc="Clients", leave=True, position=2)

            for cliente, (disc, chunks) in client_bar:
                # Carregar o bloco atual do cliente
                chunk_dataset = chunks[chunk_idx]
                if len(chunk_dataset) == 0:
                    tqdm.write(f"Chunk {chunk_idx} for client {cliente} is empty, skipping.")
                    continue
                chunk_loader = DataLoader(chunk_dataset, batch_size=batch_tam, shuffle=True)

                # Treinar o discriminador no bloco
                disc.to(device)
                optim_D = optim_Ds[cliente]

                batch_bar = tqdm(chunk_loader, desc="Batches", leave=True, position=4)

                start_disc_time = time.time()
                for batch in chunk_loader:
                    images, labels = batch["image"].to(device), batch["label"].to(device)
                    batch_size = images.size(0)
                    if batch_size == 1:
                        tqdm.write("Batch size is 1, skipping batch")
                        continue

                    real_ident = torch.full((batch_size, 1), 1., device=device)
                    fake_ident = torch.full((batch_size, 1), 0., device=device)

                    z_noise = torch.randn(batch_size, latent_dim, device=device)
                    x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)

                    # Train D
                    optim_D.zero_grad()

                    if wgan:
                        labels = torch.nn.functional.one_hot(labels, 10).float().to(device)
                        x_fake_l = torch.nn.functional.one_hot(x_fake_labels, 10).float()

                        # Adicionar labels ao images para treinamento do Discriminador
                        image_labels = labels.view(labels.size(0), 10, 1, 1).expand(-1, -1, 28, 28)
                        image_fake_labels = x_fake_l.view(x_fake_l.size(0), 10, 1, 1).expand(-1, -1, 28, 28)

                        images = torch.cat([images, image_labels], dim=1)

                        # Treinar Discriminador
                        z = torch.cat([z_noise, x_fake_l], dim=1)
                        fake_images = gen(z).detach()
                        fake_images = torch.cat([fake_images, image_fake_labels], dim=1)

                        d_loss = discriminator_loss(disc(images), disc(fake_images)) + 10 * gradient_penalty(disc, images, fake_images)

                    else:
                        # Dados Reais
                        y_real = disc(images, labels)
                        d_real_loss = disc.loss(y_real, real_ident)

                        # Dados Falsos
                        x_fake = gen(z_noise, x_fake_labels).detach()
                        y_fake_d = disc(x_fake, x_fake_labels)
                        d_fake_loss = disc.loss(y_fake_d, fake_ident)

                        # Loss total e backprop
                        d_loss = (d_real_loss + d_fake_loss) / 2

                    d_loss.backward()
                    #torch.nn.utils.clip_grad_norm_(disc.discriminator.parameters(), max_norm=1.0)
                    optim_D.step()
                    d_loss_b += d_loss.item()
                    total_chunk_samples += 1
                disc_time = time.time() - start_disc_time  


            # Média da perda dos discriminadores neste chunk
            avg_d_loss_chunk = d_loss_b / total_chunk_samples if total_chunk_samples > 0 else 0.0
            losses_dict["d_losses_chunk"].append(avg_d_loss_chunk)
            d_loss_c += avg_d_loss_chunk * total_chunk_samples
            total_d_samples += total_chunk_samples

            chunk_g_loss = 0.0

            epoch_gen_bar = tqdm(range(extra_g_e), desc="Gerador", leave=True, position=2)

            start_gen_time = time.time()
            for g_epoch in epoch_gen_bar:
                # Train G
                optim_G.zero_grad()

                # Gera dados falsos
                z_noise = torch.randn(batch_size_gen, latent_dim, device=device)
                x_fake_labels = torch.randint(0, 10, (batch_size_gen,), device=device)
                label = int(x_fake_labels.item())

                if wgan:
                    x_fake_labels = torch.nn.functional.one_hot(x_fake_labels, 10).float()
                    z_noise = torch.cat([z_noise, x_fake_labels], dim=1)
                    fake_images = gen(z_noise)

                    # Seleciona o melhor discriminador (Dmax)
                    image_fake_labels = x_fake_labels.view(x_fake_labels.size(0), 10, 1, 1).expand(-1, -1, 28, 28)
                    fake_images = torch.cat([fake_images, image_fake_labels], dim=1)

                    y_fake_gs = [model(fake_images.detach()) for model in models]

                else:
                    x_fake = gen(z_noise, x_fake_labels)

                    if f2a:
                        y_fakes = []
                        for D in models:
                            D = D.to(device)
                            y_fakes.append(D(x_fake, x_fake_labels))  # each is [B,1]
                        # stack into [N_discriminators, B, 1]
                        y_stack = torch.stack(y_fakes, dim=0)

                        # 4) Compute λ = ReLU(lambda_star) to enforce λ ≥ 0
                        lam = relu(lambda_star)

                        # 5) Soft‐max weights across the 0th dim (discriminators)
                        #    we want S_i = exp(λ D_i) / sum_j exp(λ D_j)
                        #    shape remains [N, B, 1]
                        S = torch.softmax(lam * y_stack, dim=0)

                        # 6) Weighted sum: D_agg shape [B,1]
                        D_agg = (S * y_stack).sum(dim=0)

                        # 7) Compute your generator loss + β λ² regularizer
                        real_ident = torch.full((batch_size_gen, 1), 1., device=device)
                        adv_loss   = gen.loss(D_agg, real_ident)       # BCEWithLogitsLoss or whatever
                        reg_loss   = beta * lam.pow(2)                 # β λ²
                        g_loss     = adv_loss + reg_loss

                    else:
                        # Seleciona o melhor discriminador (Dmax)
                        y_fake_gs = [model(x_fake.detach(), x_fake_labels) for model in models]
                        y_fake_g_means = [torch.mean(y).item() for y in y_fake_gs]
                        dmax_index = y_fake_g_means.index(max(y_fake_g_means))
                        Dmax = models[dmax_index]

                        start_track_mismatch_time = time.time()
                        #Track mismatches
                        expected_indexes = label_to_client[class_labels.int2str(x_fake_labels.item())] ##PEGA SOMENTE A PRIMEIRA LABEL, SE BATCH_SIZE_GEN FOR DIFERENTE DE 1 VAI DAR ERRO
                        if dmax_index not in expected_indexes:
                            mismatch_count += 1
                            total_checked +=1
                            percent_mismatch =  mismatch_count / total_checked
                            with open(dmax_mismatch_log, "a") as mismatch_file:
                                mismatch_file.write(f"{epoch+1} {x_fake_labels.item()} {expected_indexes} {dmax_index} {percent_mismatch:.2f}\n")
                        else:
                            total_checked += 1
                            if g_epoch == extra_g_e - 1 and chunk_idx == num_chunks - 1:
                                percent_mismatch =  mismatch_count / total_checked
                                with open(dmax_mismatch_log, "a") as mismatch_file:
                                    mismatch_file.write(f"{epoch+1} {x_fake_labels.item()} {expected_indexes} {dmax_index} {percent_mismatch:.2f}\n")
                        track_mismatch_time = time.time() - start_track_mismatch_time

                        # Calcula a perda do gerador
                        real_ident = torch.full((batch_size_gen, 1), 1., device=device)
                        if wgan:
                            y_fake_g = Dmax(fake_images)
                            g_loss = generator_loss(y_fake_g)

                        else:
                            y_fake_g = Dmax(x_fake, x_fake_labels)  # Detach explícito
                            g_loss = gen.loss(y_fake_g, real_ident)

                g_loss.backward()
                #torch.nn.utils.clip_grad_norm_(gen.generator.parameters(), max_norm=1.0)
                optim_G.step()
                gen.to(device)
                chunk_g_loss += g_loss.item()
            gen_time = time.time() - start_gen_time

            losses_dict["g_losses_chunk"].append(chunk_g_loss / extra_g_e)
            g_loss_c += chunk_g_loss /extra_g_e

            losses_dict["time_chunk"].append(time.time() - chunk_start_time)
            losses_dict["disc_time"].append(disc_time)
            losses_dict["gen_time"].append(gen_time)
            losses_dict["track_mismatch_time"].append(track_mismatch_time)


        g_loss_e = g_loss_c/num_chunks
        d_loss_e = d_loss_c / total_d_samples if total_d_samples > 0 else 0.0

        losses_dict["g_losses_round"].append(g_loss_e)
        losses_dict["d_losses_round"].append(d_loss_e)

        if (epoch+1)%1==0:
            checkpoint = {
                    'epoch': epoch+1,  # número da última época concluída
                    'gen_state_dict': gen.state_dict(),
                    'optim_G_state_dict': optim_G.state_dict(),
                    'discs_state_dict': [model.state_dict() for model in models],
                    'optim_Ds_state_dict:': [optim_d.state_dict() for optim_d in optim_Ds]
                }
            checkpoint_file = f"{folder}/checkpoint_epoch{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_file)
            tqdm.write(f"Global net saved to {checkpoint_file}")

            if f2a:
                current_lambda_star = lambda_star.item()
                current_lam         = F.relu(lambda_star).item()

                with open(lambda_log, "a") as f:
                 f.write(f"{current_lambda_star},{current_lam}\n")

        tqdm.write(f"Época {epoch+1} completa")
        generate_plot(gen, "cpu", epoch+1, latent_dim=128, folder=folder)
        gen.to(device)

        losses_dict["time_round"].append(time.time() - epoch_start_time)

        try:
            with open(loss_filename, 'w', encoding='utf-8') as f:
                json.dump(losses_dict, f, ensure_ascii=False, indent=4) # indent makes it readable
            tqdm.write(f"Losses dict successfully saved to {loss_filename}")
        except Exception as e:
            tqdm.write(f"Error saving losses dict to JSON: {e}")
