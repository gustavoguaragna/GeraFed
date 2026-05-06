"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

import os
import time
import torch
from torch.utils.data import DataLoader, ConcatDataset
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from Simulation.CNN.task import (
    create_model,
    get_weights, 
    load_data, 
    local_test,
    normalize_dataset_name,
    set_weights, 
    train
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self,
                 cid,
                 net,
                 trainloader,
                 testloader_local,
                 local_epochs,
                 dataset,
                 lr,
                 batch_size,
                 folder,
                 continue_epoch):
        self.cid = cid
        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.trainloader = trainloader
        self.testloader_local = testloader_local
        self.local_epochs = local_epochs
        self.dataset = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.folder = folder
        self.continue_epoch = continue_epoch

    def fit(self, parameters, config):

        # Atualiza pesos do modelo classificador
        set_weights(self.net, parameters)

        # Define o dataloader como no FLEG: o classificador usa todos os chunks locais.
        if isinstance(self.trainloader, list):
            trainloader = DataLoader(
                ConcatDataset([dl.dataset for dl in self.trainloader]),
                batch_size=self.batch_size,
                shuffle=True
            )
        else:
            trainloader = self.trainloader

        train_start_time = time.time()
        train_loss = train(
            net=self.net,
            trainloader=trainloader,
            epochs=self.local_epochs,
            device=self.device,
            dataset=self.dataset,
            lr=self.lr
        )
        train_time = time.time() - train_start_time

        return (
            get_weights(self.net),
            len(trainloader.dataset),
            {
                "train_loss": train_loss,
                "tempo_treino_alvo": train_time,
            },
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        local_test_start_time = time.time()
        local_acc = local_test(
            net=self.net,
            testloader=self.testloader_local,
            device=self.device,
            acc_filepath=f"{self.folder}/accuracy_report.txt",
            epoch=int(config["round"]),
            cliente=self.cid,
            continue_epoch=self.continue_epoch,
            dataset=self.dataset
        )
        local_test_time = time.time() - local_test_start_time

        return (
            0.0,
            len(self.testloader_local.dataset),
            {
                "local_test_time": local_test_time,
                "local_accuracy": local_acc,
            },
        )


def client_fn(context: Context):
    # Load model and data
    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partitioner    = context.run_config["partitioner"]
    if partitioner == "Dir01":
        alpha_dir       = 0.1
    elif partitioner == "Dir05":
        alpha_dir       = 0.5
    else:
        alpha_dir       = None
    batch_size     = context.run_config["tam_batch"]
    local_epochs   = context.run_config["epocas_alvo"]
    dataset        = normalize_dataset_name(context.run_config["dataset"])
    seed           = context.run_config["seed"]
    teste          = context.run_config["teste"]
    num_chunks     = context.run_config["num_chunks"]
    lr             = context.run_config["learn_rate_alvo"]
    strategy       = context.run_config["strategy"]
    num_clients    = context.run_config["num_clients"]
    continue_epoch = context.run_config["continue_epoch"]
    folder         = f"{context.run_config['Exp_name_folder']}CNN/{dataset}/{partitioner}/{strategy}/{num_clients}_clients"
    os.makedirs(folder, exist_ok=True)

    net = create_model(dataset, seed=seed)

    trainloader, _, testloader_local = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        dataset=dataset,
        teste=teste,
        partitioner_type=partitioner,
        num_chunks=num_chunks,
        batch_size=batch_size,
        alpha_dir=alpha_dir
    )

    # Return Client instance
    return FlowerClient(cid=partition_id,
                        net=net,
                        trainloader=trainloader,
                        testloader_local=testloader_local,
                        local_epochs=local_epochs,
                        dataset=dataset,
                        lr=lr,
                        batch_size=batch_size,
                        folder=folder,
                        continue_epoch=continue_epoch).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
