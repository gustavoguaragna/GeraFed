"""fedvae: A Flower app for Federated Variational Autoencoder."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Flatten(nn.Module):
    """Flattens input by reshaping it into a one-dimensional tensor."""

    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """Unflattens a tensor converting it to a desired shape."""

    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, input):
        return input.view(*self.target_shape)


class Net(nn.Module):
    def __init__(self, dataset="mnist", z_dim=10) -> None:
        super().__init__()
        if dataset == "mnist":
            in_channels = 1
            out_channels = 1
            self.encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=1, out_channels=6, kernel_size=4, stride=2
                ),  # [batch,6,13,13]
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=6, out_channels=16, kernel_size=5, stride=2
                ),  # [batch,16,5,5]
                nn.ReLU(),
                Flatten(),
            )
            h_dim = 16 * 5 * 5  # 400
            self.fc1 = nn.Linear(h_dim, z_dim)
            self.fc2 = nn.Linear(h_dim, z_dim)
            self.fc3 = nn.Linear(z_dim, h_dim)
            self.decoder = nn.Sequential(
                UnFlatten((-1, 16, 5, 5)),  # [batch,16,5,5]
                nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=2),  # [batch,6,15,15]
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=4, stride=2),  # [batch,1,28,28]
                nn.Tanh(),
            )
        elif dataset == "cifar10":
            in_channels = 3
            out_channels = 3
            self.encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=3, out_channels=6, kernel_size=4, stride=2
                ),  # [batch,6,15,15]
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=6, out_channels=16, kernel_size=5, stride=2
                ),  # [batch,16,6,6]
                nn.ReLU(),
                Flatten(),
            )
            h_dim = 16 * 6 * 6  # 576
            self.fc1 = nn.Linear(h_dim, z_dim)
            self.fc2 = nn.Linear(h_dim, z_dim)
            self.fc3 = nn.Linear(z_dim, h_dim)
            self.decoder = nn.Sequential(
                UnFlatten((-1, 16, 6, 6)),  # [batch,16,6,6]
                nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=2),  # [batch,6,15,15]
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2),  # [batch,3,32,32]
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Dataset {dataset} not supported")

    def reparametrize(self, h):
        """Reparametrization layer of VAE."""
        mu, logvar = self.fc1(h), self.fc2(h)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def encode(self, x):
        """Encoder of the VAE."""
        h = self.encoder(x)
        z, mu, logvar = self.reparametrize(h)
        return z, mu, logvar

    def decode(self, z):
        """Decoder of the VAE."""
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z_decode = self.decode(z)
        return z_decode, mu, logvar


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions, dataset="mnist"):
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
            [ToTensor(), Normalize((0.5,), (0.5,))]  # MNIST has 1 channel
        )
    elif dataset == "cifar10":
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # CIFAR-10 has 3 channels
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
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, learning_rate, device, dataset="mnist"):
    """Train the network on the training set."""
    if dataset == "mnist":
      imagem = "image"
    elif dataset == "cifar10":
      imagem = "img"
    net.to(device)  # move model to GPU if available
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
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
