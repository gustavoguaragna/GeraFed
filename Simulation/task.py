"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import numpy as np


# Define the GAN model
class CGAN(nn.Module):
    def __init__(self, dataset, img_size, latent_dim):
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
            *self._create_layer(self.latent_dim + self.classes, 128, False),
            *self._create_layer(128, 256),
            *self._create_layer(256, 512),
            *self._create_layer(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            *self._create_layer(self.classes + int(np.prod(self.img_shape)), 1024, False, True),
            *self._create_layer(1024, 512, True, True),
            *self._create_layer(512, 256, True, True),
            *self._create_layer(256, 128, False, False),
            *self._create_layer(128, 1, False, False),
            nn.Sigmoid()
        )

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

    def forward_gen(self, noise, labels):
        z = torch.cat((self.label_embedding(labels), noise), -1)
        x = self.model(z)
        x = x.view(x.size(0), *self.img_shape) #Em
        return x

    def forward_disc(self, image, labels):
        x = torch.cat((image.view(image.size(0), -1), self.label_embedding(labels)), -1)
        return self.model(x)

    def loss(self, output, label):
        return self.adv_loss(output, label)
    

fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions, dataset="mnist", img_size=64, batch_size=32):
    """Load partition dataset (MNIST or CIFAR10)."""
    # Only initialize FederatedDataset once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        if dataset == "mnist":
            fds = FederatedDataset(
                dataset="mnist",
                partitioners={"train": partitioner},
            )
        elif dataset == "cifar10":
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        else:
            raise ValueError(f"Dataset {dataset} not supported")
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    if dataset == "mnist":
        pytorch_transforms = Compose(
            [Resize(img_size), ToTensor(), Normalize((0.5,), (0.5,))]  # MNIST has 1 channel
        )
    elif dataset == "cifar10":
        pytorch_transforms = Compose(
            [Resize(img_size), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # CIFAR-10 has 3 channels
        )

    def apply_transforms(batch, dataset=dataset):
        if dataset == "mnist":
          imagem = "image"
        elif dataset == "cifar10":
          imagem = "img"
        """Apply transforms to the partition from FederatedDataset."""
        batch[imagem] = [pytorch_transforms(img) for img in batch[imagem]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def train(net, trainloader, epochs, learning_rate, device, dataset="mnist"):
    """Train the network on the training set."""
    if dataset == "mnist":
      imagem = "image"
    elif dataset == "cifar10":
      imagem = "img"
    
    net.to(device)  # move model to GPU if available
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[imagem]
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.05 * kld_loss
            loss.backward()
            optimizer.step()


def test(net, testloader, device, dataset="mnist"):
    """Validate the network on the entire test set."""
    if dataset == "mnist":
      imagem = "image"
    elif dataset == "cifar10":
      imagem = "img"
    total, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[imagem].to(device)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss
            total += len(images)
    return loss / total


def generate(net, image):
    """Reproduce the input with trained VAE."""
    with torch.no_grad():
        return net.forward(image)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
