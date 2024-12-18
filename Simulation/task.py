"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from torchvision.transforms.functional import to_pil_image
from datasets import Dataset, Features, ClassLabel, Image
import random

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

def split_balanced(gen_dataset_hf, num_clientes):
    # Identificar as classes
    class_label = gen_dataset_hf.features["label"]
    num_classes = class_label.num_classes
    
    # Cria um mapeamento classe -> índices
    class_to_indices = {c: [] for c in range(num_classes)}
    for i in range(len(gen_dataset_hf)):
        lbl = gen_dataset_hf[i]["label"]
        class_to_indices[lbl].append(i)
    
    # Verifica quantos exemplos por classe temos
    examples_per_class = len(class_to_indices[0])
    # Garantir que o número de exemplos por classe seja divisível por num_clientes 
    # (ou lidar com o resto de alguma forma)
    if examples_per_class % num_clientes != 0:
        raise ValueError("O número de exemplos por classe não é divisível igualmente pelo número de clientes.")
        
    # Número de exemplos por classe por cliente
    examples_per_class_per_client = examples_per_class // num_clientes

    # Agora, distribui igualmente os índices para cada cliente
    client_indices = [[] for _ in range(num_clientes)]
    for c in range(num_classes):
        # Índices dessa classe
        idxs = class_to_indices[c]
        # Divide em partes iguais
        for i in range(num_clientes):
            start = i * examples_per_class_per_client
            end = (i+1) * examples_per_class_per_client
            client_indices[i].extend(idxs[start:end])
    
    # Agora criamos um DatasetDict com um subset para cada cliente
    # Cada cliente terá a mesma quantidade total de exemplos (classes * examples_per_class_per_client)
    client_datasets = {}
    for i in range(num_clientes):
        # Ordena os índices do cliente (opcional, mas pode manter a ordem)
        client_indices[i].sort()
        # Seleciona os exemplos
        client_subset = gen_dataset_hf.select(client_indices[i])
        client_datasets[i] = client_subset
    
    return client_datasets


fds = None  # Cache FederatedDataset
gen_img_part = None # Cache GeneratedDataset

def load_data(partition_id: int, 
              num_partitions: int, 
              niid: bool, 
              alpha_dir: float, 
              batch_size: int, 
              cgan=None, 
              examples_per_class=0):
    
    """Carrega MNIST com splits de treino e teste separados. Se examples_per_class > 0, inclui dados gerados."""
   
    global fds

    if fds is None:
        print("ENTROU FDS NONE")
        if niid:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha_dir,
                min_partition_size=0,
                self_balancing=False
            )
        else:
            partitioner = IidPartitioner(num_partitions=num_partitions)

        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": partitioner}
        )

    # Carrega a partição de treino e teste separadamente
    test_partition = fds.load_split("test")

    global gen_img_part

    if cgan is not None and examples_per_class > 0:
        if gen_img_part is None:
            print("ENTRA GEN_IMG_PART NONE")
            generated_images = generate_images(cgan, examples_per_class)
            gen_img_part = split_balanced(gen_dataset_hf=generated_images, num_clientes=num_partitions)
        train_partition = gen_img_part[partition_id]
    else:
        train_partition = fds.load_partition(partition_id, split="train")
    
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

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

def train_gen(net, trainloader, epochs, lr, device, dataset="mnist", latent_dim=100):
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
            x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
            x_fake = net(z_noise, x_fake_labels)
            y_fake_g = net(x_fake, x_fake_labels)
            g_loss = net.loss(y_fake_g, real_ident)
            g_loss.backward()
            optim_G.step()

            # Train D
            net.zero_grad()
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
    return np.mean(g_losses)   


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
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
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
