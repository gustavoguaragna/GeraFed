"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader, Subset, random_split
import math
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets.partitioner import Partitioner
from typing import Optional, List, Union
import random
import numpy as np
import datasets



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
    
class Net_CIFAR(nn.Module):
    def __init__(self,seed=42):
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
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

def load_data(partition_id: int, 
              num_partitions: int,
              batch_size: int,
              partitioner_type: str = "IID",
              dataset: str = "mnist",
              teste: bool = False,
              alpha_dir: Optional[float] = None,
              num_chunks: int = 1) -> tuple[Union[DataLoader, List[DataLoader]], DataLoader, DataLoader]:
    
    """Carrega MNIST com splits de treino e teste separados. Se examples_per_class > 0, inclui dados gerados."""
   
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
            chunks_real.append(Subset(client_dataset["train"], chunk_indices))

        trainloader_real = [DataLoader(chunk, batch_size=batch_size, shuffle=True) for chunk in chunks_real if len(chunk) > 0]

    else:
        trainloader_real = DataLoader(client_dataset["train"], batch_size=batch_size, shuffle=True)
    
    testloader = DataLoader(test_partition, batch_size=batch_size, shuffle=True)
    testloader_local = DataLoader(client_dataset["test"], batch_size=batch_size, shuffle=True)

    return trainloader_real, testloader, testloader_local


def train(net, trainloader, epochs, device, dataset, lr):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    if dataset == "mnist":
        image = "image"
    elif dataset == "cifar10":
        image = "img"
    else:
        raise ValueError(f"Dataset {dataset} nao identificado. Deveria ser 'mnist' ou 'cifar10'")
    
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[image]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss


def test(net, testloader, device, dataset):
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
