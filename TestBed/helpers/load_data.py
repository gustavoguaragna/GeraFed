import numpy as np
#import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import logging
#import keras

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


def load_data(data_sampling_percentage=0.8, client_id=1, total_clients=2, partitioner_type="PARTITIONER", dataset="mnist"):
    """Load federated dataset partition based on client ID.

    Args:
        data_sampling_percentage (float): Percentage of the dataset to use for training.
        client_id (int): Unique ID for the client.
        total_clients (int): Total number of clients.

    Returns:
        Tuple of arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    # Download and partition dataset
    if partitioner_type == "DIRICHLET":
        partitioner = DirichletPartitioner(num_partitions=total_clients, partition_by="label",
                                           alpha=0.5, min_partition_size=10, self_balancing=True)
        fds = FederatedDataset(dataset=dataset, partitioners={"train": partitioner})
    elif partitioner_type == "PARTITIONER":
        fds = FederatedDataset(dataset=dataset, partitioners={"train": total_clients})
    else:
        raise ValueError(f"Partitioner {partitioner_type} is not supported.")

    partition = fds.load_partition(client_id - 1, "train")
    partition.set_format("numpy")

    # Divide data on each client: 80% train, 20% test
    if dataset == "mnist":
        img = "image"
    else:
        img = "img"
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train = partition["train"][img] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"][img] / 255.0, partition["test"]["label"]

    # Apply data sampling
    num_samples = int(data_sampling_percentage * len(x_train))
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train, y_train = x_train[indices], y_train[indices]
    # x_train = np.expand_dims(x_train, 1)
    # x_test = np.expand_dims(x_test, 1)

    return (x_train, y_train), (x_test, y_test)