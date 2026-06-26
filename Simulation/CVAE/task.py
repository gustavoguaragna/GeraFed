"""Shared models and helpers for the Flower CVAE simulation."""

from __future__ import annotations

import io
import math
import random
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset as TorchDataset,
    Subset,
    random_split,
)
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


DATASET_ALIASES = {
    "mnist": "mnist",
    "ylecun/mnist": "mnist",
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "uoft-cs/cifar10": "cifar10",
    "organsmnist": "organsmnist",
    "organ_s_mnist": "organsmnist",
    "organ-smnist": "organsmnist",
    "camelyon17": "camelyon17",
    "camelyon": "camelyon17",
    "octdl": "octdl",
    "skinl_derm": "skinl_derm",
    "skinl-derm": "skinl_derm",
    "skin_lesion_derm": "skinl_derm",
    "organs_axial": "organs_axial",
    "organs-axial": "organs_axial",
    "breastmnist": "breastmnist",
    "breast-mnist": "breastmnist",
}

DATASET_CONFIG = {
    "mnist": {
        "source": "hf",
        "hf_name": "mnist",
        "image_key": "image",
        "input_shape": (1, 28, 28),
        "mean": (0.5,),
        "std": (0.5,),
        "num_classes": 10,
    },
    "cifar10": {
        "source": "hf",
        "hf_name": "uoft-cs/cifar10",
        "image_key": "img",
        "input_shape": (3, 32, 32),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "num_classes": 10,
    },
    "organsmnist": {
        "source": "medmnist",
        "medmnist_name": "organsmnist",
        "image_key": "image",
        "input_shape": (3, 224, 224),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "num_classes": 11,
    },
    "breastmnist": {
        "source": "medmnist",
        "medmnist_name": "breastmnist",
        "image_key": "image",
        "input_shape": (3, 224, 224),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "num_classes": 2,
    },
    "camelyon17": {
        "source": "wilds",
        "wilds_name": "camelyon17",
        "image_key": "image",
        "input_shape": (3, 96, 96),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "num_classes": 2,
    },
    "octdl": {
        "source": "imagefolder",
        "imagefolder_root": ("octdl", "dataset_1"),
        "image_key": "image",
        "input_shape": (3, 224, 224),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "num_classes": 7,
    },
    "skinl_derm": {
        "source": "medimeta",
        "medimeta_name": "skinl_derm",
        "medimeta_target": "Diagnosis grouped",
        "image_key": "image",
        "input_shape": (3, 224, 224),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "num_classes": 5,
    },
    "organs_axial": {
        "source": "medimeta",
        "medimeta_name": "organs_axial",
        "medimeta_target": "organ label",
        "image_key": "image",
        "input_shape": (3, 224, 224),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "num_classes": 11,
    },
}

TORCH_DATASET_SOURCES = {"medmnist", "medimeta", "wilds", "imagefolder"}
_runtime_num_classes: dict[str, int] = {}

STOP_CRITERION_ALIASES = {
    "global_test_acc": "global_test_acc",
    "global_test_accuracy": "global_test_acc",
    "global_accuracy": "global_test_acc",
    "server_global_acc": "global_test_acc",
    "server_global_accuracy": "global_test_acc",
    "test_acc": "global_test_acc",
    "client_val_loss": "client_val_loss",
    "client_validation_loss": "client_val_loss",
    "val_loss": "client_val_loss",
    "validation_loss": "client_val_loss",
    "client_val_acc": "client_val_acc",
    "client_val_accuracy": "client_val_acc",
    "client_validation_acc": "client_val_acc",
    "client_validation_accuracy": "client_val_acc",
    "val_acc": "client_val_acc",
    "val_accuracy": "client_val_acc",
    "validation_acc": "client_val_acc",
    "validation_accuracy": "client_val_acc",
    "fixed_rounds": "fixed_rounds",
    "fixed": "fixed_rounds",
    "num_rounds": "fixed_rounds",
    "rodadas_fixas": "fixed_rounds",
}

CLIENT_VALIDATION_STOP_CRITERIA = {"client_val_loss", "client_val_acc"}


def normalize_stop_criterion(stop_criterion: str) -> str:
    key = str(stop_criterion).strip().lower()
    try:
        return STOP_CRITERION_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(
            ["global_test_acc", "client_val_loss", "client_val_acc", "fixed_rounds"]
        )
        raise ValueError(
            f"criterio_parada invalido: {stop_criterion}. Use um de: {valid}"
        ) from exc


def uses_client_validation_criterion(stop_criterion: str) -> bool:
    return normalize_stop_criterion(stop_criterion) in CLIENT_VALIDATION_STOP_CRITERIA


class Net(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
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


class FeatureExtractor1(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(F.relu(self.conv1(x)))


class ClassifierHead1(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 6, 12, 12)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FeatureExtractor2(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(-1, 16 * 4 * 4)


class ClassifierHead2(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FeatureExtractor3(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return F.relu(self.fc1(x))


class ClassifierHead3(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FeatureExtractor4(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class ClassifierHead4(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        return self.fc3(x)


class Net_Cifar(nn.Module):
    def __init__(self, seed=None):
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
        return self.fc3(x)


class FeatureExtractor1_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(F.relu(self.conv1(x)))


class ClassifierHead1_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 6, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FeatureExtractor2_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return torch.flatten(x, 1)


class ClassifierHead2_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FeatureExtractor3_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return F.relu(self.fc1(x))


class ClassifierHead3_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FeatureExtractor4_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
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
        return F.relu(self.fc2(x))


class ClassifierHead4_Cifar(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        return self.fc3(x)


class GenericAdaptiveNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GenericFeatureExtractor1(nn.Module):
    def __init__(self, in_channels: int, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(F.relu(self.conv1(x)))


class GenericClassifierHead1(nn.Module):
    def __init__(self, num_classes: int, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GenericFeatureExtractor2(nn.Module):
    def __init__(self, in_channels: int, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        return torch.flatten(x, 1)


class GenericClassifierHead2(nn.Module):
    def __init__(self, num_classes: int, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GenericFeatureExtractor3(nn.Module):
    def __init__(self, in_channels: int, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 256)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return F.relu(self.fc1(x))


class GenericClassifierHead3(nn.Module):
    def __init__(self, num_classes: int, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GenericFeatureExtractor4(nn.Module):
    def __init__(self, in_channels: int, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class GenericClassifierHead4(nn.Module):
    def __init__(self, num_classes: int, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc3(x)


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
    dataset_key = str(dataset).strip().lower()
    try:
        return DATASET_ALIASES[dataset_key]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_CONFIG))
        raise ValueError(
            f"Dataset {dataset} nao identificado. Use um de: {valid}"
        ) from exc


def get_dataset_config(dataset: str) -> dict[str, Any]:
    return DATASET_CONFIG[normalize_dataset_name(dataset)]


def get_image_key(dataset: str) -> str:
    return get_dataset_config(dataset)["image_key"]


def get_input_shape(dataset: str) -> tuple[int, int, int]:
    return tuple(get_dataset_config(dataset)["input_shape"])


def get_num_channels(dataset: str) -> int:
    return int(get_input_shape(dataset)[0])


def _set_runtime_num_classes(dataset: str, num_classes: int | None) -> None:
    dataset = normalize_dataset_name(dataset)
    if num_classes is not None:
        _runtime_num_classes[dataset] = int(num_classes)


def get_num_classes(dataset: str) -> int:
    dataset = normalize_dataset_name(dataset)
    config_num_classes = DATASET_CONFIG[dataset].get("num_classes")
    if config_num_classes is not None:
        return int(config_num_classes)
    if dataset in _runtime_num_classes:
        return _runtime_num_classes[dataset]
    raise ValueError(
        f"Number of classes for dataset '{dataset}' is not known before loading it. "
        "Load the dataset first so labels can be inspected."
    )


_fds_cache = {}
_torch_dataset_cache = {}


def _coerce_label(label) -> int:
    if isinstance(label, torch.Tensor):
        return int(label.detach().cpu().view(-1)[0].item())
    if isinstance(label, np.ndarray):
        return int(label.reshape(-1)[0].item())
    if isinstance(label, (list, tuple)):
        return _coerce_label(label[0])
    return int(label)


class _TorchImageDataset(TorchDataset):
    def __init__(
        self,
        base_dataset,
        image_key: str = "image",
        label_mapping: Optional[dict[Any, int]] = None,
    ):
        self.base_dataset = base_dataset
        self.image_key = image_key
        self.label_mapping = label_mapping if label_mapping is not None else {}

    def __len__(self):
        return len(self.base_dataset)

    def _normalize_label(self, label) -> int:
        try:
            return _coerce_label(label)
        except (TypeError, ValueError):
            label_key = str(label)
            if label_key not in self.label_mapping:
                self.label_mapping[label_key] = len(self.label_mapping)
            return self.label_mapping[label_key]

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if isinstance(item, dict):
            image = item.get(self.image_key, item.get("image", item.get("img")))
            label = item.get("label", item.get("target", item.get("y")))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            image, label = item[0], item[1]
        else:
            raise ValueError(f"Unsupported dataset item type: {type(item)}")
        return {self.image_key: image, "label": self._normalize_label(label)}


def _build_torch_transform(dataset: str):
    config = get_dataset_config(dataset)
    _, height, width = config["input_shape"]
    return Compose(
        [
            Resize((height, width)),
            ToTensor(),
            Normalize(config["mean"], config["std"]),
        ]
    )


def _missing_dependency(dataset: str, package: str, exc: ImportError) -> ImportError:
    error = ImportError(
        f"Dataset '{dataset}' requires the optional dependency '{package}'. "
        "Install project dependencies before loading it."
    )
    error.__cause__ = exc
    return error


def _load_medmnist_datasets(dataset: str, data_root: str, download_datasets: bool):
    config = get_dataset_config(dataset)
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:
        raise _missing_dependency(dataset, "medmnist", exc)

    medmnist_name = config["medmnist_name"]
    info = INFO[medmnist_name]
    data_class = getattr(medmnist, info["python_class"])
    transform = _build_torch_transform(dataset)
    root = Path(data_root).expanduser() / "medmnist"
    size = int(config["input_shape"][1])

    common_kwargs = {
        "transform": transform,
        "download": download_datasets,
        "as_rgb": True,
        "root": str(root),
        "size": size,
    }
    train_ds = data_class(split="train", **common_kwargs)
    val_ds = data_class(split="val", **common_kwargs)
    test_ds = data_class(split="test", **common_kwargs)
    label_mapping: dict[Any, int] = {}
    return (
        ConcatDataset(
            [
                _TorchImageDataset(train_ds, config["image_key"], label_mapping),
                _TorchImageDataset(val_ds, config["image_key"], label_mapping),
            ]
        ),
        _TorchImageDataset(test_ds, config["image_key"], label_mapping),
    )


def _load_medimeta_datasets(dataset: str, data_root: str, download_datasets: bool):
    del download_datasets
    config = get_dataset_config(dataset)
    try:
        from medimeta import MedIMeta
    except ImportError as exc:
        raise _missing_dependency(dataset, "medimeta", exc)

    transform = _build_torch_transform(dataset)
    root = Path(data_root).expanduser() / "medimeta"
    label_mapping: dict[Any, int] = {}

    def make_split(split: str):
        base = MedIMeta(
            str(root),
            config["medimeta_name"],
            config["medimeta_target"],
            split=split,
            transform=transform,
        )
        return _TorchImageDataset(base, config["image_key"], label_mapping)

    return ConcatDataset([make_split("train"), make_split("val")]), make_split("test")


def _load_octdl_datasets(dataset: str, data_root: str, download_datasets: bool):
    del download_datasets
    config = get_dataset_config(dataset)
    transform = _build_torch_transform(dataset)
    root = Path(data_root).expanduser().joinpath(*config["imagefolder_root"])
    required = [root / split for split in ("train", "val", "test")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "OCTDL must be available as an ImageFolder tree with "
            f"train/val/test splits under '{root}'. Missing: {missing}"
        )

    train_ds = ImageFolder(root / "train", transform=transform)
    val_ds = ImageFolder(root / "val", transform=transform)
    test_ds = ImageFolder(root / "test", transform=transform)
    _set_runtime_num_classes(dataset, len(train_ds.classes))
    return (
        ConcatDataset(
            [
                _TorchImageDataset(train_ds, config["image_key"]),
                _TorchImageDataset(val_ds, config["image_key"]),
            ]
        ),
        _TorchImageDataset(test_ds, config["image_key"]),
    )


def _wilds_subset(dataset_obj, names: tuple[str, ...], transform):
    last_error = None
    for name in names:
        try:
            return dataset_obj.get_subset(name, transform=transform)
        except Exception as exc:  # WILDS raises different errors across versions.
            last_error = exc
    raise ValueError(f"Could not load any WILDS subset from {names}") from last_error


def _load_wilds_datasets(dataset: str, data_root: str, download_datasets: bool):
    config = get_dataset_config(dataset)
    try:
        from wilds import get_dataset as wilds_get_dataset
    except ImportError as exc:
        raise _missing_dependency(dataset, "wilds", exc)

    transform = _build_torch_transform(dataset)
    root = Path(data_root).expanduser() / "wilds"
    wilds_ds = wilds_get_dataset(
        dataset=config["wilds_name"],
        root_dir=str(root),
        download=download_datasets,
    )
    label_mapping: dict[Any, int] = {}
    train_ds = _wilds_subset(wilds_ds, ("train",), transform)
    val_ds = _wilds_subset(wilds_ds, ("id_val", "val"), transform)
    test_ds = _wilds_subset(wilds_ds, ("test", "id_test"), transform)
    return (
        ConcatDataset(
            [
                _TorchImageDataset(train_ds, config["image_key"], label_mapping),
                _TorchImageDataset(val_ds, config["image_key"], label_mapping),
            ]
        ),
        _TorchImageDataset(test_ds, config["image_key"], label_mapping),
    )


def _load_torch_source_datasets(
    dataset: str,
    data_root: str,
    download_datasets: bool,
) -> tuple[TorchDataset, TorchDataset]:
    source = get_dataset_config(dataset)["source"]
    if source == "medmnist":
        return _load_medmnist_datasets(dataset, data_root, download_datasets)
    if source == "medimeta":
        return _load_medimeta_datasets(dataset, data_root, download_datasets)
    if source == "wilds":
        return _load_wilds_datasets(dataset, data_root, download_datasets)
    if source == "imagefolder":
        return _load_octdl_datasets(dataset, data_root, download_datasets)
    raise ValueError(f"Unsupported torch dataset source: {source}")


def _labels_from_array(values) -> list:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if values.ndim > 1:
        values = values.reshape(values.shape[0], -1)[:, 0]
    return values.tolist()


def _raw_dataset_labels(dataset: TorchDataset) -> Optional[list]:
    if hasattr(dataset, "targets"):
        return _labels_from_array(dataset.targets)
    if hasattr(dataset, "labels"):
        return _labels_from_array(dataset.labels)
    if hasattr(dataset, "y_array"):
        labels = _labels_from_array(dataset.y_array)
        if len(labels) == len(dataset):
            return labels
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        parent = dataset.dataset
        parent_labels = None
        if hasattr(parent, "y_array"):
            parent_labels = _labels_from_array(parent.y_array)
        elif hasattr(parent, "targets"):
            parent_labels = _labels_from_array(parent.targets)
        elif hasattr(parent, "labels"):
            parent_labels = _labels_from_array(parent.labels)
        if parent_labels is not None:
            return [parent_labels[int(idx)] for idx in dataset.indices]
    if hasattr(dataset, "samples"):
        return [label for _, label in dataset.samples]
    return None


def _dataset_labels(dataset: TorchDataset) -> list[int]:
    if isinstance(dataset, _TorchImageDataset):
        raw_labels = _raw_dataset_labels(dataset.base_dataset)
        if raw_labels is not None:
            return [dataset._normalize_label(label) for label in raw_labels]
    if isinstance(dataset, ConcatDataset):
        labels = []
        for child in dataset.datasets:
            labels.extend(_dataset_labels(child))
        return labels
    if isinstance(dataset, Subset):
        parent_labels = _dataset_labels(dataset.dataset)
        return [parent_labels[int(idx)] for idx in dataset.indices]

    raw_labels = _raw_dataset_labels(dataset)
    if raw_labels is not None:
        return [_coerce_label(label) for label in raw_labels]

    labels = []
    for idx in range(len(dataset)):
        labels.append(_get_item_label(dataset[idx]))
    return labels


def _partition_torch_indices(
    labels: list[int],
    num_partitions: int,
    partitioner_type: str,
    alpha_dir: Optional[float],
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    classes = np.unique(labels_array)

    if "Class" in partitioner_type:
        if num_partitions > len(classes):
            raise ValueError(
                f"Cannot create {num_partitions} partitions with only "
                f"{len(classes)} classes. Reduce partitions to <= {len(classes)}."
            )
        shuffled_classes = list(classes)
        rng.shuffle(shuffled_classes)
        class_groups = np.array_split(shuffled_classes, num_partitions)
        partitions = []
        for class_group in class_groups:
            indices = []
            for cls in class_group:
                indices.extend(np.flatnonzero(labels_array == cls).tolist())
            rng.shuffle(indices)
            partitions.append(indices)
        return partitions

    if "Dir" in partitioner_type:
        alpha = alpha_dir if alpha_dir is not None else 0.1
        partitions = [[] for _ in range(num_partitions)]
        for cls in classes:
            class_indices = np.flatnonzero(labels_array == cls)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_partitions, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for partition, split_indices in zip(
                partitions,
                np.split(class_indices, split_points),
            ):
                partition.extend(split_indices.tolist())
        for partition in partitions:
            rng.shuffle(partition)
        return partitions

    all_indices = np.arange(len(labels))
    rng.shuffle(all_indices)
    return [split.tolist() for split in np.array_split(all_indices, num_partitions)]


def _load_torch_partitioned_data(
    partition_id: int,
    num_partitions: int,
    dataset: str,
    teste: bool,
    partitioner_type: str,
    alpha_dir: Optional[float],
    seed: int,
    data_root: str,
    download_datasets: bool,
) -> tuple[TorchDataset, TorchDataset]:
    cache_key = (
        dataset,
        str(Path(data_root).expanduser()),
        bool(download_datasets),
        num_partitions,
        partitioner_type,
        alpha_dir,
        seed,
    )
    if cache_key not in _torch_dataset_cache:
        train_pool, global_test = _load_torch_source_datasets(
            dataset=dataset,
            data_root=data_root,
            download_datasets=download_datasets,
        )
        labels = _dataset_labels(train_pool)
        _set_runtime_num_classes(dataset, len(set(labels)))
        partitions = _partition_torch_indices(
            labels=labels,
            num_partitions=num_partitions,
            partitioner_type=partitioner_type,
            alpha_dir=alpha_dir,
            seed=seed,
        )
        _torch_dataset_cache[cache_key] = {
            "train_pool": train_pool,
            "global_test": global_test,
            "partitions": partitions,
        }

    cached = _torch_dataset_cache[cache_key]
    partition_indices = list(cached["partitions"][partition_id])
    if teste:
        num_samples = max(1, int(len(partition_indices) / 10))
        partition_indices = partition_indices[:num_samples]
    return Subset(cached["train_pool"], partition_indices), cached["global_test"]


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    dataset: str = "mnist",
    teste: bool = False,
    partitioner_type: str = "IID",
    alpha_dir: Optional[float] = None,
    seed: int = 42,
    client_validation: bool = False,
    data_root: str = "data",
    download_datasets: bool = True,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader], DataLoader]:
    """Carrega dataset com splits globais e locais para o cliente especifico."""

    dataset = normalize_dataset_name(dataset)
    dataset_config = DATASET_CONFIG[dataset]
    _set_runtime_num_classes(dataset, dataset_config.get("num_classes"))

    if dataset_config["source"] in TORCH_DATASET_SOURCES:
        train_partition, test_partition = _load_torch_partitioned_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
            dataset=dataset,
            teste=teste,
            partitioner_type=partitioner_type,
            alpha_dir=alpha_dir,
            seed=seed,
            data_root=data_root,
            download_datasets=download_datasets,
        )
    else:
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

    num_client_samples = len(train_partition)
    split_generator = torch.Generator().manual_seed(seed)
    if client_validation:
        train_size = int(num_client_samples * 0.70)
        val_size = int(num_client_samples * 0.15)
        test_size = num_client_samples - train_size - val_size
        if num_client_samples >= 3:
            if val_size == 0:
                val_size = 1
                train_size = max(1, train_size - 1)
            if test_size == 0:
                test_size = 1
                train_size = max(1, train_size - 1)
        client_train, client_val, client_test = random_split(
            train_partition,
            [train_size, val_size, test_size],
            generator=split_generator,
        )
        valloader_local = DataLoader(client_val, batch_size=64, shuffle=False)
    else:
        test_size = int(num_client_samples * 0.2)
        train_size = num_client_samples - test_size
        client_train, client_test = random_split(
            train_partition,
            [train_size, test_size],
            generator=split_generator,
        )
        valloader_local = None

    trainloader = DataLoader(client_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=64, shuffle=False)
    testloader_local = DataLoader(client_test, batch_size=64, shuffle=False)

    return trainloader, testloader, valloader_local, testloader_local


def create_full_model(dataset: str, seed: Optional[int] = None) -> nn.Module:
    dataset = normalize_dataset_name(dataset)
    if dataset == "mnist":
        return Net(seed=seed)
    if dataset == "cifar10":
        return Net_Cifar(seed=seed)
    if DATASET_CONFIG[dataset]["source"] in TORCH_DATASET_SOURCES:
        return GenericAdaptiveNet(
            in_channels=get_num_channels(dataset),
            num_classes=get_num_classes(dataset),
            seed=seed,
        )
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
    dataset = normalize_dataset_name(dataset)
    if dataset in NET_COMPONENTS:
        classifier_class, _ = get_component_classes(dataset, level)
        return classifier_class(seed=seed)
    if DATASET_CONFIG[dataset]["source"] in TORCH_DATASET_SOURCES:
        if level == 0:
            return create_full_model(dataset, seed=seed)
        num_classes = get_num_classes(dataset)
        classifier_heads = {
            1: GenericClassifierHead1,
            2: GenericClassifierHead2,
            3: GenericClassifierHead3,
            4: GenericClassifierHead4,
        }
        try:
            return classifier_heads[level](num_classes=num_classes, seed=seed)
        except KeyError as exc:
            raise ValueError(
                f"Nível {level} inválido para {dataset}. Use níveis entre 0 e 4."
            ) from exc
    raise ValueError(f"Dataset {dataset} nao identificado")


def create_feature_extractor(
    dataset: str, level: int, seed: Optional[int] = None
) -> Optional[nn.Module]:
    dataset = normalize_dataset_name(dataset)
    if dataset in NET_COMPONENTS:
        _, feature_extractor_class = get_component_classes(dataset, level)
        if feature_extractor_class is None:
            return None
        return feature_extractor_class(seed=seed)
    if DATASET_CONFIG[dataset]["source"] in TORCH_DATASET_SOURCES:
        if level == 0:
            return None
        feature_extractors = {
            1: GenericFeatureExtractor1,
            2: GenericFeatureExtractor2,
            3: GenericFeatureExtractor3,
            4: GenericFeatureExtractor4,
        }
        try:
            return feature_extractors[level](
                in_channels=get_num_channels(dataset),
                seed=seed,
            )
        except KeyError as exc:
            raise ValueError(
                f"Nível {level} inválido para {dataset}. Use níveis entre 0 e 4."
            ) from exc
    raise ValueError(f"Dataset {dataset} nao identificado")

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


class DictTensorDataset(TorchDataset):
    def __init__(self, assets, labels, asset_col_name="image", label_col_name="label"):
        self.assets = assets
        self.labels = labels
        self.asset_col_name = asset_col_name
        self.label_col_name = label_col_name

    def __len__(self):
        return self.assets.shape[0]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Dataset index out of range")
        return {
            self.asset_col_name: self.assets[idx],
            self.label_col_name: int(self.labels[idx]),
        }


_ClassPartitionerImpl = None


def _get_class_partitioner_impl():
    global _ClassPartitionerImpl
    if _ClassPartitionerImpl is not None:
        return _ClassPartitionerImpl

    from flwr_datasets.partitioner import Partitioner

    class ClassPartitionerImpl(Partitioner):
        """Partitions a dataset by class, with each class in exactly one partition."""

        def __init__(
            self,
            num_partitions: int,
            seed: Optional[int] = None,
            label_column: str = "label",
        ) -> None:
            super().__init__()
            self._num_partitions = num_partitions
            self._seed = seed
            self._label_column = label_column
            self._partition_indices: Optional[List[List[int]]] = None

        def _create_partitions(self) -> None:
            labels = self.dataset[self._label_column]
            class_indices = defaultdict(list)
            for idx, label in enumerate(labels):
                class_indices[label].append(idx)

            classes = list(class_indices.keys())
            num_classes = len(classes)
            if self._num_partitions > num_classes:
                raise ValueError(
                    f"Cannot create {self._num_partitions} partitions with only "
                    f"{num_classes} classes. Reduce partitions to <= {num_classes}."
                )

            rng = random.Random(self._seed)
            rng.shuffle(classes)
            partition_classes = np.array_split(classes, self._num_partitions)

            self._partition_indices = []
            for class_group in partition_classes:
                indices = []
                for cls in class_group:
                    indices.extend(class_indices[cls])
                self._partition_indices.append(indices)

        @property
        def dataset(self):
            return super().dataset

        @dataset.setter
        def dataset(self, value) -> None:
            super(ClassPartitionerImpl, ClassPartitionerImpl).dataset.fset(self, value)
            self._create_partitions()

        def load_partition(self, partition_id: int):
            if not self.is_dataset_assigned():
                raise RuntimeError("Dataset must be assigned before loading partitions")
            if partition_id < 0 or partition_id >= self.num_partitions:
                raise ValueError(f"Invalid partition ID: {partition_id}")
            return self.dataset.select(self._partition_indices[partition_id])

        @property
        def num_partitions(self) -> int:
            return self._num_partitions

        def __repr__(self) -> str:
            return (
                f"ClassPartitioner(num_partitions={self._num_partitions}, "
                f"seed={self._seed}, label_column='{self._label_column}')"
            )

    ClassPartitionerImpl.__name__ = "ClassPartitioner"
    _ClassPartitionerImpl = ClassPartitionerImpl
    return _ClassPartitionerImpl


class ClassPartitioner:
    def __new__(cls, *args, **kwargs):
        return _get_class_partitioner_impl()(*args, **kwargs)


def _get_item_label(item, label_key="label"):
    if isinstance(item, dict):
        return _coerce_label(item[label_key])
    if isinstance(item, (list, tuple)):
        maybe_label = item[1] if len(item) > 1 else item[0]
        return _coerce_label(maybe_label)
    return _coerce_label(item)


def get_label_counts(dataset, label_key="label", max_samples=None) -> Counter:
    indices = None
    if isinstance(dataset, Subset):
        indices = dataset.indices
        underlying = dataset.dataset
    else:
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
        for item in dataset:
            counts[_get_item_label(item, label_key)] += 1
            n += 1
            if max_samples and n >= max_samples:
                break
    return counts


def choose_minority_labels(
    counts: Counter,
    total_num_classes: Optional[int] = None,
    method: str = "topk",
    k: Optional[int] = None,
    threshold: Optional[int] = None,
    ratio: Optional[float] = None,
) -> List[int]:
    if total_num_classes is None:
        total_num_classes = max(counts.keys(), default=-1) + 1
    full_counts = {lbl: counts.get(lbl, 0) for lbl in range(total_num_classes)}

    if method == "topk":
        if k is None:
            raise ValueError("k must be provided for topk method")
        return [label for label, _ in sorted(full_counts.items(), key=lambda kv: kv[1])][:k]
    if method == "threshold":
        if threshold is None:
            raise ValueError("threshold must be provided for threshold method")
        return [label for label, count in full_counts.items() if count <= threshold]
    if method == "ratio":
        if ratio is None:
            raise ValueError("ratio must be provided for ratio method")
        max_count = max(full_counts.values()) if full_counts else 0
        return [label for label, count in full_counts.items() if count <= ratio * max_count]
    raise ValueError(f"Unknown method '{method}'.")


def build_label_index_map(dataset, label_key="label") -> Dict[int, List[int]]:
    label_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        label = _get_item_label(item, label_key)
        label_to_indices[int(label)].append(idx)
    return label_to_indices


def sample_generated_indices_for_labels(
    gen_dataset,
    desired_labels: List[int],
    num_per_label: Dict[int, int],
    label_key="label",
    rng_seed=None,
) -> List[int]:
    if rng_seed is not None:
        random.seed(rng_seed)

    label_map = build_label_index_map(gen_dataset, label_key=label_key)
    chosen = []
    for label in desired_labels:
        available = label_map.get(label, [])
        need = num_per_label.get(label, 0)
        if need <= 0:
            continue
        if len(available) == 0:
            print(f"Warning: no generated samples for label {label}")
            continue
        if len(available) < need:
            print(
                f"Warning: requested {need} for label {label} but only "
                f"{len(available)} available; taking all."
            )
            chosen.extend(available)
        else:
            chosen.extend(random.sample(available, need))
    return chosen


class _IndexWrappingDataset(TorchDataset):
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


class EmbeddingPairDataset(TorchDataset):
    def __init__(self, embeddings, labels, asset_col_name="image", label_col_name="label"):
        self.emb = embeddings
        self.lbl = labels
        self.asset_col_name = asset_col_name
        self.label_col_name = label_col_name

    def __len__(self):
        return self.emb.size(0)

    def __getitem__(self, idx):
        return {
            self.asset_col_name: self.emb[idx],
            self.label_col_name: int(self.lbl[idx]),
        }


def augment_client_with_generated(
    client_train,
    gen_dataset,
    counts,
    label_key_client="label",
    label_key_gen="label",
    strategy="fill_to_max",
    fill_to: Optional[int] = None,
    k: Optional[int] = None,
    threshold: Optional[int] = None,
    ratio: Optional[float] = None,
    rng_seed: Optional[int] = None,
) -> Tuple[TorchDataset, Dict]:
    if len(counts) == 0:
        print("Warning: client_train has zero samples.")
    max_count = max(counts.values()) if counts else 0

    if strategy == "fill_to_max":
        desired_labels = [label for label, count in counts.items() if count < max_count]
        per_label_target = {label: max_count for label in desired_labels}
    elif strategy == "fill_to":
        if fill_to is None:
            raise ValueError("fill_to must be provided for 'fill_to' strategy")
        desired_labels = [label for label, count in counts.items() if count < fill_to]
        per_label_target = {label: fill_to for label in desired_labels}
    elif strategy == "topk":
        if k is None or fill_to is None:
            raise ValueError("k and fill_to required for topk")
        desired_labels = choose_minority_labels(counts, method="topk", k=k)
        per_label_target = {label: fill_to for label in desired_labels}
    elif strategy == "threshold":
        if threshold is None or fill_to is None:
            raise ValueError("threshold and fill_to required for threshold")
        desired_labels = choose_minority_labels(
            counts, method="threshold", threshold=threshold
        )
        per_label_target = {label: fill_to for label in desired_labels}
    elif strategy == "ratio":
        if ratio is None or fill_to is None:
            raise ValueError("ratio and fill_to required for ratio")
        desired_labels = choose_minority_labels(counts, method="ratio", ratio=ratio)
        per_label_target = {label: fill_to for label in desired_labels}
    else:
        raise ValueError("Unknown strategy")

    need_per_label = {}
    for label in desired_labels:
        need = per_label_target[label] - counts.get(label, 0)
        if need > 0:
            need_per_label[label] = need

    chosen_gen_indices = sample_generated_indices_for_labels(
        gen_dataset,
        desired_labels=desired_labels,
        num_per_label=need_per_label,
        label_key=label_key_gen,
        rng_seed=rng_seed,
    )
    gen_subset = (
        _IndexWrappingDataset(gen_dataset, chosen_gen_indices)
        if chosen_gen_indices
        else None
    )

    if gen_subset is None or len(gen_subset) == 0:
        combined = client_train
    else:
        combined = ConcatDataset([client_train, gen_subset])

    stats = {
        "client_counts": counts,
        "desired_labels": desired_labels,
        "need_per_label": need_per_label,
        "gen_selected_count": len(chosen_gen_indices),
    }
    return combined, stats


def local_test(
    net: nn.Module,
    testloader: DataLoader,
    device: str,
    acc_filepath: str,
    epoch: int,
    cliente: str,
    num_classes: Optional[int] = None,
    continue_epoch: int = 0,
    dataset: str = "mnist",
    return_loss: bool = False,
):
    client_eval_time = time.time()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    predictions_counter = defaultdict(int)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_total = 0.0

    net.eval()
    net.to(device)
    image = get_image_key(dataset)
    if num_classes is None:
        num_classes = get_num_classes(dataset)

    with torch.no_grad():
        for batch in testloader:
            images, labels = batch[image].to(device), batch["label"].to(device)
            outputs = net(images)
            loss_total += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)

            for true_label, pred_label in zip(labels, predicted):
                true_idx = true_label.item()
                pred_idx = pred_label.item()
                class_total[true_idx] += 1
                predictions_counter[pred_idx] += 1
                if true_idx == pred_idx:
                    class_correct[true_idx] += 1

        total_samples = sum(class_total.values())
        overall_accuracy = (
            sum(class_correct.values()) / total_samples if total_samples > 0 else 0.0
        )
        overall_loss = loss_total / total_samples if total_samples > 0 else 0.0
        results_metrics = {
            "class_metrics": {},
            "overall_accuracy": overall_accuracy,
            "overall_loss": overall_loss,
            "prediction_distribution": dict(predictions_counter),
        }

        for cls in range(num_classes):
            accuracy = (
                class_correct[cls] / class_total[cls]
                if class_total[cls] > 0
                else "N/A"
            )
            results_metrics["class_metrics"][f"class_{cls}"] = {
                "samples": class_total[cls],
                "predictions": predictions_counter[cls],
                "accuracy": accuracy,
            }

        with open(acc_filepath, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch + continue_epoch} - Client {cliente}\n")
            f.write("{:<10} {:<10} {:<10} {:<10}\n".format(
                "Class", "Accuracy", "Samples", "Predictions"
            ))
            f.write("-" * 45 + "\n")
            for cls in range(num_classes):
                metrics = results_metrics["class_metrics"][f"class_{cls}"]
                accuracy = (
                    f"{metrics['accuracy']:.4f}"
                    if isinstance(metrics["accuracy"], float)
                    else "  N/A  "
                )
                f.write("{:<10} {:<10} {:<10} {:<10}\n".format(
                    f"Class {cls}",
                    accuracy,
                    metrics["samples"],
                    metrics["predictions"],
                ))
            f.write(
                "\n{:<20} {:.4f}".format(
                    "Overall Accuracy:", results_metrics["overall_accuracy"]
                )
            )
            f.write(
                "\n{:<20} {:.4f}".format(
                    "Overall Loss:", results_metrics["overall_loss"]
                )
            )
            f.write("\n{:<20} {}".format("Total Samples:", total_samples))
            f.write(
                "\n{:<20} {}".format(
                    "Total Predictions:", sum(predictions_counter.values())
                )
            )
            f.write(
                "\n{:<20} {:.4f}".format(
                    "Client Evaluation Time:", time.time() - client_eval_time
                )
            )
            f.write("\n\n")

    if return_loss:
        return results_metrics["overall_accuracy"], results_metrics["overall_loss"]
    return results_metrics["overall_accuracy"]


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    device = next(net.parameters()).device
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({key: torch.tensor(value).to(device) for key, value in params_dict})
    net.load_state_dict(state_dict, strict=False)


def get_model_size_mb(params, divisor=10**6):
    buffer = io.BytesIO()
    np.savez(buffer, *params)
    return len(buffer.getvalue()) / divisor


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
