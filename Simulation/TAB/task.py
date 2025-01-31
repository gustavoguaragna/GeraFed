"""xgboost_quickstart: A Flower / XGBoost app."""

from logging import INFO

import xgboost as xgb
from flwr.common import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.preprocessing import LabelEncoder


def train_test_split(partition, test_fraction, seed):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]
    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost."""
    x = data.drop(columns=["income"]).values
    y = data["income"].values
    new_data = xgb.DMatrix(x, label=y)
    return new_data


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_clients: int):
    """Load partition HIGGS data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_clients)
        fds = FederatedDataset(
            dataset="scikit-learn/adult-census-income",
            partitioners={"train": partitioner},
        )

    # Load the partition for this `partition_id`
    partition = fds.load_partition(partition_id, split="train")

    # Train/test splitting
    train_data, valid_data, num_train, num_val = train_test_split(
        partition, test_fraction=0.2, seed=42
    )

    # Reformat data to DMatrix for xgboost
    log(INFO, "Reformatting data...")
    train_data_df = train_data.to_pandas()
    valid_data_df = valid_data.to_pandas()
    categorical_columns = train_data_df.select_dtypes(include=["object"]).columns
    
    # Label encoding for categorical columns
    for col in categorical_columns:
        le = LabelEncoder()
        train_data_df[col] = le.fit_transform(train_data_df[col])
        valid_data_df[col] = le.transform(valid_data_df[col])

    train_dmatrix = transform_dataset_to_dmatrix(train_data_df)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data_df)

    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
