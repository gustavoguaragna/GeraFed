"""xgboost_quickstart: A Flower / XGBoost app."""

from logging import INFO

import xgboost as xgb
from flwr.common import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


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


def load_data(partition_id: int, 
              num_clients: int,
              niid: bool = False,
              alpha_dir: float = 1.0):
    """Load partition Adult data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        if niid:
            partitioner = DirichletPartitioner(
                num_partitions=num_clients,
                partition_by="income",
                alpha=alpha_dir,
                min_partition_size=30,
                self_balancing=False
                )
        else:
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
    
    combined_data_df = pd.concat([train_data_df, valid_data_df])
    print(f"Class distribution: {combined_data_df['income'].value_counts()}")
    categorical_cols = combined_data_df.select_dtypes(include=["object"]).columns
    #categorical_columns = train_data_df.select_dtypes(include=["object"]).columns

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    combined_data_df[categorical_cols] = ordinal_encoder.fit_transform(combined_data_df[categorical_cols])
    train_data_df[categorical_cols] = ordinal_encoder.transform(train_data_df[categorical_cols])
    valid_data_df[categorical_cols] = ordinal_encoder.transform(valid_data_df[categorical_cols])
    # Label encoding for categorical columns
    # for col in categorical_columns:
    #     le = LabelEncoder()
    #     train_data_df[col] = le.fit_transform(train_data_df[col])
    #     valid_data_df[col] = le.transform(valid_data_df[col])

    train_dmatrix = transform_dataset_to_dmatrix(train_data_df)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data_df)

    return train_dmatrix, valid_dmatrix, num_train, num_val, ordinal_encoder


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

def test_global(model, encoder):
    """Evaluate model in global test_data"""

    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    # Column names for the dataset
    columns = [
        "age", "workclass", "fnlwgt", "education", "education.num",
        "marital.status", "occupation", "relationship", "race", "sex",
        "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"
    ]

    test_data_g = pd.read_csv(test_url, header=None, names=columns, skipinitialspace=True, skiprows=1)
    test_data_g["income"] = test_data_g["income"].str.strip(".")

    categorical_cols = test_data_g.select_dtypes(include=["object"]).columns
    
    # encode no teste global
    test_data_g[categorical_cols] = encoder.transform(test_data_g[categorical_cols])

    # dmatrix para teste global
    X_test_g = test_data_g.drop(columns=["income"]).values
    y_test_g = test_data_g["income"].values
    test_dmatrix_g = xgb.DMatrix(data=X_test_g, label=y_test_g)

    y_pred_g = model.predict(test_dmatrix_g)
    y_pred_g_binary = np.where(y_pred_g > 0.5, 1, 0)
    y_test_g = test_dmatrix_g.get_label()

    f1 = f1_score(y_test_g, y_pred_g_binary, average='macro')
    acc = accuracy_score(y_test_g, y_pred_g_binary)
    auc = roc_auc_score(y_test_g, y_pred_g)
    
    return f1, acc, auc
