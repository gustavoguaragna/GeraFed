"""Shared models and helpers for the Flower CVAE simulation."""

from __future__ import annotations

import io
import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor

from Simulation.FLEG.task import (
    ClassPartitioner,
    ClassifierHead1,
    ClassifierHead2,
    ClassifierHead3,
    ClassifierHead4,
    ClassifierHead1_Cifar,
    ClassifierHead2_Cifar,
    ClassifierHead3_Cifar,
    ClassifierHead4_Cifar,
    DictTensorDataset,
    EmbeddingPairDataset,
    FeatureExtractor1,
    FeatureExtractor2,
    FeatureExtractor3,
    FeatureExtractor4,
    FeatureExtractor1_Cifar,
    FeatureExtractor2_Cifar,
    FeatureExtractor3_Cifar,
    FeatureExtractor4_Cifar,
    Net,
    Net_Cifar,
    augment_client_with_generated,
    get_label_counts,
    get_model_size_mb,
    get_weights,
    local_test,
    set_weights,
)


DATASET_ALIASES = {
    "mnist": "mnist",
    "ylecun/mnist": "mnist",
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "uoft-cs/cifar10": "cifar10",
}

DATASET_CONFIG = {
    "mnist": {
        "hf_name": "mnist",
        "image_key": "image",
        "input_shape": (1, 28, 28),
        "mean": (0.5,),
        "std": (0.5,),
    },
    "cifar10": {
        "hf_name": "uoft-cs/cifar10",
        "image_key": "img",
        "input_shape": (3, 32, 32),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
}

NET_COMPONENTS = {
    "mnist": {
        0: (Net, None),
        1: (ClassifierHead1, FeatureExtractor1),
        2: (ClassifierHead2, FeatureExtractor2),
        3: (ClassifierHead3, FeatureExtractor3),
        4: (ClassifierHead4, FeatureExtractor4),
    },
    "cifar10": {
        0: (Net_Cifar, None),
        1: (ClassifierHead1_Cifar, FeatureExtractor1_Cifar),
        2: (ClassifierHead2_Cifar, FeatureExtractor2_Cifar),
        3: (ClassifierHead3_Cifar, FeatureExtractor3_Cifar),
        4: (ClassifierHead4_Cifar, FeatureExtractor4_Cifar),
    },
}


def normalize_dataset_name(dataset: str) -> str:
    dataset_key = str(dataset).lower()
    try:
        return DATASET_ALIASES[dataset_key]
    except KeyError as exc:
        raise ValueError(
            f"Dataset {dataset} nao identificado. Deveria ser 'mnist' ou 'cifar10'"
        ) from exc


def get_image_key(dataset: str) -> str:
    return DATASET_CONFIG[normalize_dataset_name(dataset)]["image_key"]


_fds_cache = {}


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    dataset: str = "mnist",
    teste: bool = False,
    partitioner_type: str = "IID",
    alpha_dir: Optional[float] = None,
    seed: int = 42,
)-> tuple[DataLoader, DataLoader, DataLoader]:
    
    """Carrega dataset com splits de treino e teste separados para o cliente específico."""

    dataset = normalize_dataset_name(dataset)
    dataset_config = DATASET_CONFIG[dataset]
    cache_key = (dataset, num_partitions, partitioner_type, alpha_dir, seed)

    if cache_key not in _fds_cache:
        if "Dir" in partitioner_type:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha_dir if alpha_dir is not None else 0.1,
                min_partition_size=0,
                self_balancing=False,
            )
        elif "Class" in partitioner_type:
            partitioner = ClassPartitioner(num_partitions=num_partitions, seed=seed)
        else:
            partitioner = IidPartitioner(num_partitions=num_partitions)

        _fds_cache[cache_key] = FederatedDataset(
            dataset=dataset_config["hf_name"],
            partitioners={"train": partitioner},
        )

    fds = _fds_cache[cache_key]
    train_partition = fds.load_partition(partition_id, split="train")
    test_partition = fds.load_split("test")

    if teste:
        num_samples = max(1, int(len(train_partition) / 10))
        train_partition = train_partition.select(range(num_samples))

    transforms = Compose(
        [ToTensor(), Normalize(dataset_config["mean"], dataset_config["std"])]
    )
    image_key = dataset_config["image_key"]

    def apply_transforms(batch):
        batch[image_key] = [transforms(img) for img in batch[image_key]]
        return batch

    train_partition = train_partition.with_transform(apply_transforms)
    test_partition = test_partition.with_transform(apply_transforms)

    test_size = int(len(train_partition) * 0.2)
    train_size = len(train_partition) - test_size
    client_train, client_test = random_split(
        train_partition,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    trainloader = DataLoader(client_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=64, shuffle=False)
    testloader_local = DataLoader(client_test, batch_size=64, shuffle=False)

    return trainloader, testloader, testloader_local


def create_full_model(dataset: str, seed: Optional[int] = None) -> nn.Module:
    dataset = normalize_dataset_name(dataset)
    if dataset == "mnist":
        return Net(seed=seed)
    if dataset == "cifar10":
        return Net_Cifar(seed=seed)
    raise ValueError(f"Dataset {dataset} nao identificado")


def get_component_classes(dataset: str, level: int):
    dataset = normalize_dataset_name(dataset)
    try:
        return NET_COMPONENTS[dataset][level]
    except KeyError as exc:
        raise ValueError(
            f"Nível {level} inválido para {dataset}. Use níveis entre 0 e 4."
        ) from exc


def create_classifier(dataset: str, level: int, seed: Optional[int] = None) -> nn.Module:
    classifier_class, _ = get_component_classes(dataset, level)
    return classifier_class(seed=seed)


def create_feature_extractor(
    dataset: str, level: int, seed: Optional[int] = None
) -> Optional[nn.Module]:
    _, feature_extractor_class = get_component_classes(dataset, level)
    if feature_extractor_class is None:
        return None
    return feature_extractor_class(seed=seed)

def state_dict_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in state_dict.items()}, buffer)
    return buffer.getvalue()


def state_dict_from_bytes(payload: bytes, device: torch.device | str = "cpu") -> dict:
    return torch.load(io.BytesIO(payload), map_location=device)


def unpack_batch(batch, image_key="image", label_key="label"):
    if isinstance(batch, dict):
        return batch[image_key], batch[label_key]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(f"Unsupported batch type: {type(batch)}")


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, minmax: bool = True):
        super().__init__()
        self.minmax = minmax
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        h = self.fc1(x)
        if not self.minmax:
            h = self.ln1(h)
        h = self.act(h)
        h = self.fc2(h)
        if not self.minmax:
            h = self.ln2(h)
        return self.act(h + residual)


class CVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int,
        device,
        beta: float = 0.1,
        resblock: bool = False,
        minmax: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.device = device
        self.beta = beta

        encoder_layers = [nn.Linear(input_dim + condition_dim, hidden_dim), nn.ReLU()]
        if resblock:
            encoder_layers.extend(
                [ResidualBlock(hidden_dim, minmax=minmax), ResidualBlock(hidden_dim, minmax=minmax)]
            )
        else:
            encoder_layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            )
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        decoder_layers = [nn.Linear(latent_dim + condition_dim, hidden_dim), nn.ReLU()]
        if resblock:
            decoder_layers.extend(
                [ResidualBlock(hidden_dim, minmax=minmax), ResidualBlock(hidden_dim, minmax=minmax)]
            )
        else:
            decoder_layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            )
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        if minmax:
            decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, y):
        h = self.encoder(torch.cat([x, y], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        return self.decoder(torch.cat([z, y], dim=-1))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kld_loss


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int,
        input_dim: int,
        resblock: bool = False,
        minmax: bool = True,
    ):
        super().__init__()
        decoder_layers = [nn.Linear(latent_dim + condition_dim, hidden_dim), nn.ReLU()]
        if resblock:
            decoder_layers.extend(
                [ResidualBlock(hidden_dim, minmax=minmax), ResidualBlock(hidden_dim, minmax=minmax)]
            )
        else:
            decoder_layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            )
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        if minmax:
            decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def decode(self, z, y):
        return self.decoder(torch.cat([z, y], dim=-1))

    def forward(self, z, y):
        return self.decode(z, y)


def infer_feature_dim(feature_extractor, sample_input, device):
    feature_extractor.to(device)
    feature_extractor.eval()
    with torch.no_grad():
        output = feature_extractor(sample_input.unsqueeze(0).to(device))
    feature_shape = tuple(output.shape[1:])
    input_dim = output.view(1, -1).size(1)
    return input_dim, feature_shape
