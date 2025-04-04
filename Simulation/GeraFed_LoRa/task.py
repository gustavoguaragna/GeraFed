"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, Partitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from torchvision.transforms.functional import to_pil_image
from datasets import Dataset, Features, ClassLabel, Image
import random
import os
from collections import defaultdict
from typing import Optional, List
import matplotlib.pyplot as plt
from collections import defaultdict
import time

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
        device = input.device  # Ensure all tensors are on the same device
        labels = labels.to(device)  # Move labels to the same device as input
        
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

def generate_images(cgan, examples_per_class):
    cgan.eval()
    device = next(cgan.parameters()).device
    latent_dim = cgan.latent_dim
    classes = cgan.classes
    
    # Cria um vetor de rótulos balanceado
    labels = torch.tensor([i for i in range(classes) for _ in range(examples_per_class)], device=device)
    # Número total de imagens geradas
    num_samples = examples_per_class * classes
    # Gera vetores latentes aleatórios
    z = torch.randn(num_samples, latent_dim, device=device)

    with torch.no_grad():
        cgan.eval()
        gen_imgs = cgan(z, labels)  # [N, C, H, W]

    # Retorna como lista para aplicar transforms depois
    # Convertendo para CPU para aplicar transforms com PIL, se necessário
    gen_imgs_list = [img.cpu() for img in gen_imgs]
    gen_labels_list = labels.cpu().tolist()

    #converte para PIL
    gen_imgs_pil = [to_pil_image((img * 0.5 + 0.5).clamp(0,1)) for img in gen_imgs_list]

    #Monta dataset como do FederatedDataset
    features = Features({
    "image": Image(),
    "label": ClassLabel(names=[str(i) for i in range(classes)])
    })

    # Cria um dicionário com os dados
    gen_dict = {"image": gen_imgs_pil, "label": gen_labels_list}

    # Cria o dataset Hugging Face
    gen_dataset_hf = Dataset.from_dict(gen_dict, features=features)
    return gen_dataset_hf

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

fds = None  # Cache FederatedDataset
gen_img_part = None # Cache GeneratedDataset

def load_data(partition_id: int, 
              num_partitions: int,
              alpha_dir: float, 
              batch_size: int, 
              cgan=None, 
              examples_per_class=5000,
              partitioner: str = "IID",
              filter_classes=None,
              teste: bool = False):
    
    """Carrega MNIST com splits de treino e teste separados. Se examples_per_class > 0, inclui dados gerados."""
   
    global fds

    if fds is None:
        print("Carregamento dos Dados")
        if partitioner == "Dir":
            print("Dados por Dirichlet")
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha_dir,
                min_partition_size=0,
                self_balancing=False
            )
        elif partitioner == "Class":
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
    
    pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    train_partition = train_partition.with_transform(apply_transforms)
    test_partition = test_partition.with_transform(apply_transforms)
    
    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=batch_size)

    return trainloader, testloader


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

def train_gen(net, trainloader, epochs, lr, device, cid, logfile, dataset="mnist", latent_dim=100):
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
        G_loss = 0
        D_loss = 0
        start_time = time().time()

        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch[imagem].to(device), batch["label"].to(device)
            batch_size = images.size(0)
            real_ident = torch.full((batch_size, 1), 1., device=device)
            fake_ident = torch.full((batch_size, 1), 0., device=device)

            # Train D
            net.zero_grad()

            y_real = net(images, labels)
            d_real_loss = net.loss(y_real, real_ident)

            z_noise = torch.randn(batch_size, latent_dim, device=device)
            x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
            x_fake = net(z_noise, x_fake_labels).detach()
            y_fake_d = net(x_fake, x_fake_labels)
            d_fake_loss = net.loss(y_fake_d, fake_ident)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optim_D.step()

            # Train G
            net.zero_grad()

            z_noise = torch.randn(batch_size, latent_dim, device=device)
            x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
            x_fake = net(z_noise, x_fake_labels)
            y_fake_g = net(x_fake, x_fake_labels)
            g_loss = net.loss(y_fake_g, real_ident)

            g_loss.backward()
            optim_G.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            if batch_idx % 100 == 0 and batch_idx > 0:
                print('Epoch {} [{}/{}] loss_D_treino: {:.4f} loss_G_treino: {:.4f}'.format(
                            epoch, batch_idx, len(trainloader),
                            d_loss.mean().item(),
                            g_loss.mean().item())) 
        G_loss = np.mean(g_losses[epoch * len(trainloader):(epoch + 1) * len(trainloader)])
        D_loss = np.mean(d_losses[epoch * len(trainloader):(epoch + 1) * len(trainloader)])
        
        end_time = time.time()

        with open(logfile, "a") as f:
            f.write(f"Cliente {cid+1}, Epoca {epoch+1}: G_loss={G_loss}, D_loss={D_loss}, Tempo={end_time - start_time}\n")
    
        generate_plot(net=net, device="cpu", round_number=epoch+1, latent_dim=latent_dim, client_id=cid+1)
        net.to(device)

                

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
        num_classes = 10
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        predictions_counter = defaultdict(int)

        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0

        with torch.no_grad():
            for batch in testloader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                predicted = torch.max(outputs.data, 1)[1]
                correct += (predicted == labels).sum().item()

                for true_label, pred_label in zip(labels, predicted):
                    true_idx = true_label.item()
                    pred_idx = pred_label.item()

                    class_total[true_idx] += 1
                    predictions_counter[pred_idx] += 1

                    if true_idx == pred_idx:
                        class_correct[true_idx] += 1

        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader)
        
        results = {
            "class_metrics": {},
            "overall_accuracy": accuracy,
            "prediction_distribution": dict(predictions_counter)
        }

        for i in range(num_classes):
            metrics = {
                "samples": class_total[i],
                "predictions": predictions_counter[i],
                "accuracy": class_correct[i] / class_total[i] if class_total[i] > 0 else "N/A"
            }
            results["class_metrics"][f"class_{i}"] = metrics
        
        # Save to txt file
        with open("accuracy_report.txt", "a") as f:
            # Header with fixed widths
            f.write("{:<10} {:<10} {:<10} {:<10}\n".format(
                "Class", "Accuracy", "Samples", "Predictions"))
            f.write("-"*45 + "\n")

            # Class rows with consistent formatting
            for cls in range(num_classes):
                metrics = results["class_metrics"][f"class_{cls}"]
                
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
            f.write("\n{:<20} {:.4f}".format("Overall Accuracy:", results["overall_accuracy"]))
            f.write("\n{:<20} {}".format("Total Samples:", len(testloader.dataset)))
            f.write("\n{:<20} {}".format("Total Predictions:", sum(predictions_counter.values())))
            
            f.write("-"*45 + "\n")
        
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
    if client_id:
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

    # Salvar a figura
    IN_COLAB = False
    try:
        # Tenta importar um módulo específico do Colab
        import google.colab
        IN_COLAB = True
    except:
        pass
    if IN_COLAB:
        if client_id:
             fig.savefig(os.path.join(save_dir, f"mnist_{net_type}_r{round_number}_c{client_id}.png"))
        else:
            fig.savefig(os.path.join(save_dir, f"mnist_{net_type}_r{round_number}.png"))
    else:
        if client_id:
            fig.savefig(f"mnist_{net_type}_r{round_number}_c{client_id}.png")
        else:
            fig.savefig(f"mnist_{net_type}_r{round_number}.png")
    plt.close(fig)
    return

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()

        self.orig_linear = orig_linear
        self.r = r
        self.alpha = alpha
        # Criando os Adapters: A (out_features x r) and B (r x in_features), onde r será o rank intermediário
        self.lora_A = nn.Parameter(torch.zeros(orig_linear.out_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, orig_linear.in_features))
        # Inicializando: usando Kaiming-uniform para a matriz A e zeros para a matriz B. Dessa forma, no inicio não ocorre mudanças no modelo.
        #nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        #with normal
        nn.init.normal_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Compute original output plus LoRA adjustment.
        # F.linear computes x @ weight.T + bias.
        lora_update = F.linear(x, self.lora_A @ self.lora_B)
        return self.orig_linear(x) + self.alpha * lora_update
    
def add_lora_to_model(model: nn.Module, r: int = 8, alpha: float = 16):

    for name, module in model.named_children():
        # Recursively process children modules.
        add_lora_to_model(module, r, alpha)

        # Troca apara a camada do LoRA quando o nome da camada é igual a algum dos target_module_names
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha))

    return model

def prepare_model_for_lora(model):

    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True