"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, Partitioner
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from typing import Optional
import random
import os
from collections import defaultdict
from typing import Optional, List
from datasets import Dataset
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

    def __init__(self):
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
    def __init__(self, dataset="mnist", img_size=28, latent_dim=100, condition=False):
        super(CGAN, self).__init__()
        if dataset == "mnist":
            self.classes = 10
            self.channels = 1
        self.condition = condition
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None
        self.adv_loss = torch.nn.BCELoss()
        self.input_shape_gen = self.latent_dim + self.label_embedding.embedding_dim if condition else self.latent_dim
        self.input_shape_disc = int(np.prod(self.img_shape)) + self.classes if condition else int(np.prod(self.img_shape))


        self.generator = nn.Sequential(
            *self._create_layer_gen(self.input_shape_gen, 128, False),
            *self._create_layer_gen(128, 256),
            *self._create_layer_gen(256, 512),
            *self._create_layer_gen(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            *self._create_layer_disc(self.input_shape_disc, 1024, False, True),
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

    def forward(self, input, labels=None):
        device = input.device  # Ensure all tensors are on the same device
        if self.condition:
            labels = labels.to(device)  # Move labels to the same device as input
        
        if input.dim() == 2:
            if self.condition:
                z = torch.cat((self.label_embedding(labels), input), -1)
                x = self.generator(z)
            else:
                x = self.generator(input)
            x = x.view(x.size(0), *self.img_shape) #Em
            return x 
        
        elif input.dim() == 4:
            if self.condition:
                x = torch.cat((input.view(input.size(0), -1), self.label_embedding(labels)), -1)
            else:
                x = input.view(input.size(0), -1)
            return self.discriminator(x)

    def loss(self, output, label):
        return self.adv_loss(output, label)


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
    
class DictWrapper(torch.utils.data.Dataset):
    def __init__(self, tensor_dataset):
        self.tensor_dataset = tensor_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, idx):
        img, label = self.tensor_dataset[idx]
        # ensure tensors
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.clone().detach().long()
        return {"image": img, "label": label}

# Custom collate_fn to stack dict batches
def _collate_dict(batch):
    images = torch.stack([item['image'] for item in batch], dim=0)
    labels = torch.stack([item['label'] for item in batch], dim=0)
    return {'image': images, 'label': labels}


fds = None  # Cache FederatedDataset
gen_img_part = None # Cache GeneratedDataset

def load_data(partition_id: int, 
              num_partitions: int,  
              alpha_dir: float, 
              batch_size: int, 
              teste: bool = False,
              partitioner: str = "IID",
              syn_samples: int = 0,
              gans: Optional[List[CGAN]] = None,
              classifier: Optional[Net] = None,
              conf_threshold: float = 0.8) -> tuple[DataLoader, DataLoader]:

    """Carrega MNIST com splits de treino e teste separados. Se syn_samples > 0, inclui dados gerados."""
   
    global fds

    if fds is None:
        print("Carregando os Dados")
        if partitioner == "Dir":
            print("Dados por Dirichlet")
            partitioner = DirichletPartitioner( #type:ignore
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha_dir,
                min_partition_size=0,
                self_balancing=False
            )
        elif partitioner == "Class":
            print("Dados por classe")
            partitioner = ClassPartitioner(num_partitions=num_partitions, seed=42) #type:ignore
        else:
            print("Dados IID")
            partitioner = IidPartitioner(num_partitions=num_partitions) #type:ignore

        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": partitioner} #type:ignore
        )

    # Carrega a partição de treino e teste separadamente
    test_partition = fds.load_split("test")
    train_partition = fds.load_partition(partition_id, split="train")

    if syn_samples == 10:
        from collections import Counter
        labels = train_partition["label"]
        class_distribution = Counter(labels)
        with open("class_distribution.txt", "a") as f:
            f.write(f"CID {partition_id}: {class_distribution}\n")
    
    pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    train_partition = train_partition.with_transform(apply_transforms)
    test_partition = test_partition.with_transform(apply_transforms)

    if teste:
        print("reduzindo dataset para modo teste")
        num_samples = int(len(train_partition)/10)
        train_partition = train_partition.select(range(num_samples))
    
    # Generate fake examples per class if requested
    if syn_samples > 0 and gans and classifier:
        # Divide samples equally among GANs
        num_gans = len(gans)
        samples_per_gan = syn_samples // num_gans
        all_fake_imgs, all_fake_labels, all_fake_probs = [], [], []

        classifier.eval()
        for gan in gans:
            gan.eval()
            device = next(gan.parameters()).device
            classifier.to(device)
            with torch.no_grad():
                # Sample noise and generate
                z = torch.randn(samples_per_gan, gan.latent_dim, device=device)
                fake_imgs = gan(z)  # shape: (M, C, H, W)

                # Classify in batches to avoid OOM
                all_probs = []
                batch_size_clf = 256
                for i in range(0, fake_imgs.size(0), batch_size_clf):
                    batch = fake_imgs[i:i + batch_size_clf].to(device)
                    outputs = classifier(batch)
                    all_probs.append(torch.softmax(outputs, dim=1))
                probs = torch.cat(all_probs, dim=0)
                maxp, preds = probs.max(1)

                # Keep only confident predictions
                mask = maxp > conf_threshold
                filtered_imgs = fake_imgs[mask].cpu()
                filtered_labels = preds[mask].cpu()
                filtered_probs = maxp[mask].cpu()

                all_fake_imgs.append(filtered_imgs)
                all_fake_labels.append(filtered_labels)
                all_fake_probs.append(filtered_probs)

        if all_fake_imgs:
            fake_imgs_tensor = torch.cat(all_fake_imgs, dim=0)
            fake_labels_tensor = torch.cat(all_fake_labels, dim=0)
            fake_probs_tensor = torch.cat(all_fake_probs, dim=0)

            # Save 5 random samples
            os.makedirs("syn_samples", exist_ok=True)
            sample_idxs = random.sample(range(len(fake_imgs_tensor)), min(5, len(fake_imgs_tensor)))
            for idx in sample_idxs:
                img = fake_imgs_tensor[idx]
                lbl = fake_labels_tensor[idx].item()
                prob = fake_probs_tensor[idx].item()
                filename = f"syn_samples/syn_img_c{partition_id}_lbl{lbl}_prob{prob:.2f}_r{syn_samples/10}.png"
                save_image(img, filename)

            fake_dataset = DictWrapper(TensorDataset(fake_imgs_tensor, fake_labels_tensor))
            combined_dataset = ConcatDataset([train_partition, fake_dataset])
        else:
            combined_dataset = train_partition
    else:
        combined_dataset = train_partition
    
    def _collate_dict(batch):
        # batch: list of dicts {"image": tensor, "label": int or tensor}
        images = torch.stack([item['image'] for item in batch], dim=0)
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        return {'image': images, 'label': labels}

    # Create DataLoaders
    trainloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_dict)
    trainloader_gen = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(test_partition,   batch_size=batch_size)

    return trainloader, testloader, trainloader_gen


def train_alvo(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

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

            # Train G
            net.zero_grad()
            z_noise = torch.randn(batch_size, latent_dim, device=device)
            #x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
            x_fake = net(z_noise)
            y_fake_g = net(x_fake)
            g_loss = net.loss(y_fake_g, real_ident)
            g_loss.backward()
            optim_G.step()

            # Train D
            net.zero_grad()
            y_real = net(images)
            d_real_loss = net.loss(y_real, real_ident)
            y_fake_d = net(x_fake.detach())
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


def test(net, testloader, device, model=None):
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


def set_weights(net, parameters):
    device = next(net.parameters()).device
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class GeneratedDataset(Dataset):
    def __init__(self, generator, num_samples, latent_dim, num_classes, device):
        self.generator = generator
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device
        self.model = type(self.generator).__name__
        self.images = self.generate_data()
        self.classes = [i for i in range(self.num_classes)]


    def generate_data(self):
        gen_imgs = {}
        self.generator.to(self.device)
        self.generator.eval()
        labels = {c: torch.tensor([c for i in range(self.num_samples)], device=self.device) for c in range(self.num_classes)}
        for c, label in labels.items():
          if self.model == 'Generator':
              labels_one_hot = F.one_hot(label, self.num_classes).float().to(self.device) #
          z = torch.randn(self.num_samples, self.latent_dim, device=self.device)
          with torch.no_grad():
              if self.model == 'Generator':
                  gen_imgs_class = self.generator(torch.cat([z, labels_one_hot], dim=1))
              elif self.model == 'CGAN':
                  gen_imgs_class = self.generator(z, label)
          gen_imgs[c] = gen_imgs_class

        return gen_imgs

    def __len__(self):
        return self.num_samples * self.num_classes

    def __getitem__(self, idx):
        # Mapear o índice global para (classe, índice interno)
        class_idx = idx // self.num_samples
        sample_idx = idx % self.num_samples
        # Retorna apenas a imagem (sem o rótulo)
        return self.images[class_idx][sample_idx]

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