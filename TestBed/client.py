#import os
import argparse
import flwr as fl
#import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
from helpers.load_data import load_data
from model.model import Model, Net2
import time ##
from sklearn.metrics import f1_score ##
#from model.model import get_model

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Make TensorFlow log less verbose
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower client")

parser.add_argument(
    "--server_address", type=str, default="server:8080", help="Address of the server"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer"
)
parser.add_argument("--client_id", type=int, default=1, help="Unique ID for the client")
parser.add_argument(
    "--total_clients", type=int, default=5, help="Total number of clients"
)
parser.add_argument(
    "--data_percentage", type=float, default=1, help="Portion of client data to use"
)
# parser.add_argument(
#     "--dataset", type=str, default="mnist", help="Dataset to use (MNIST or CIFAR-10)"
# )
## Non-IID
parser.add_argument(
    "--partitioner_type", type=str, default="DIRICHLET", help="Type of partitioner to use ('PARTITIONER' or 'DIRICHLET')"
)

args = parser.parse_args()

# Create an instance of the model and pass the learning rate as an argument
#model = Model(learning_rate=args.learning_rate)
model = Net2(learning_rate=args.learning_rate)
# Compile the model
#model.compile()


class Client(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args

        logger.info("Preparing data...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        (x_train, y_train), (x_test, y_test) = load_data(
            data_sampling_percentage=self.args.data_percentage,
            client_id=self.args.client_id,
            total_clients=self.args.total_clients,
            partitioner_type=self.args.partitioner_type,  ## Pass the partitioner type
        )

        # self.x_train = x_train
        # self.y_train = y_train
        # self.x_test = x_test
        # self.y_test = y_test

        # self.x_train = torch.tensor(x_train, dtype=torch.float32)
        # self.y_train = torch.tensor(y_train, dtype=torch.long)
        # self.x_test = torch.tensor(x_test, dtype=torch.float32)
        # self.y_test = torch.tensor(y_test, dtype=torch.long)

        self.x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        self.x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)

        self.train_loader = DataLoader(list(zip(self.x_train, self.y_train)), batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = DataLoader(list(zip(self.x_test, self.y_test)), batch_size=self.args.batch_size, shuffle=False)
        self.model = model.to(self.device)

    def get_parameters(self, config):
        # Return the parameters of the model
        #return model.get_model().get_weights()
        #return [val.cpu().numpy() for val in model.get_model().state_dict().values()]
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def fit(self, parameters, config):
        start_time = time.time() ## Start computation time
        # Set the weights of the model
        #model.get_model().set_weights(parameters)

        # params_dict = zip(model.get_model().state_dict().keys(), parameters)
        # state_dict = {k: torch.tensor(v) for k, v in params_dict}
        # model.get_model().load_state_dict(state_dict, strict=True)

        # Train the model
        # history = model.get_model().fit(
        #     self.x_train, self.y_train, batch_size=self.args.batch_size, epochs = 2
        # )
        # model.get_model().train()
        # train_loader = torch.utils.data.DataLoader(
        #     list(zip(self.x_train, self.y_train)), batch_size=self.args.batch_size, shuffle=True
        # )
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.get_model().parameters(), lr=self.args.learning_rate)

        # for epoch in range(2):  # number of epochs can be adjusted
        #     for x_batch, y_batch in train_loader:
        #         optimizer.zero_grad()
        #         outputs = model.get_model()(x_batch)
        #         loss = criterion(outputs, y_batch)
        #         loss.backward()
        #         optimizer.step()

        #self.model.train()
        self.model.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        for epoch in range(2):  # Train for 2 epochs
            for x_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()


        end_time = time.time() ## End computation time

        # Calculate evaluation metric
        accuracy = self.evaluate_model()
        results = {
            "accuracy": accuracy,
            "start_time": start_time, ##
            "end_time": end_time ##
        }

        # Get the parameters after training
        #parameters_prime = model.get_model().get_weights()
        #parameters_prime = [val.cpu().numpy() for val in model.get_model().state_dict().values()]
        parameters_prime = self.get_parameters(config)

        # Directly return the parameters and the number of examples trained on
        return parameters_prime, len(self.x_train), results

    def evaluate(self, parameters, config):
        # Set the weights of the model
        #model.get_model().set_weights(parameters)
        # params_dict = zip(model.get_model().state_dict().keys(), parameters)
        # state_dict = {k: torch.tensor(v) for k, v in params_dict}
        # model.get_model().load_state_dict(state_dict, strict=True)

        # # Evaluate the model and get the loss and accuracy
        # # loss, accuracy = model.get_model().evaluate(
        # #     self.x_test, self.y_test, batch_size=self.args.batch_size
        # # )

        # model.get_model().eval()
        # test_loader = torch.utils.data.DataLoader(
        #     list(zip(self.x_test, self.y_test)), batch_size=self.args.batch_size, shuffle=False
        # )
        # criterion = nn.CrossEntropyLoss()
        # correct = 0
        # total = 0
        

        # with torch.no_grad():
        #     for x_batch, y_batch in test_loader:
        #         outputs = model.get_model()(x_batch)
        #         loss = criterion(outputs, y_batch)
        #         test_loss += loss.item()
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += y_batch.size(0)
        #         correct += (predicted == y_batch).sum().item()
        #         all_preds.extend(predicted.cpu().numpy())
        #         all_labels.extend(y_batch.cpu().numpy())
            
        self.model.eval()
        self.model.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()  # Convert tensor to float
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        # Ensure the loss is averaged over the number of batches
        avg_test_loss = float(test_loss / len(self.test_loader))

        # ## Calculate F1 score
        # y_pred = model.get_model().predict(self.x_test)
        # y_pred_classes = tf.argmax(y_pred, axis=1)
        # f1 = f1_score(self.y_test, y_pred_classes, average='weighted')

        # Return the loss, the number of examples evaluated on and the accuracy
        #return float(loss), len(self.x_test), {"loss": float(loss), "accuracy": float(accuracy), "f1": float(f1)} ## Add loss and f1
        return (avg_test_loss, len(self.x_test), {"loss": avg_test_loss, "accuracy": accuracy, "f1": f1})


    def evaluate_model(self):
        # model.get_model().eval()
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for x_batch, y_batch in torch.utils.data.DataLoader(
        #         list(zip(self.x_test, self.y_test)), batch_size=self.args.batch_size, shuffle=False
        #     ):
        #         outputs = model.get_model()(x_batch)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += y_batch.size(0)
        #         correct += (predicted == y_batch).sum().item()
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                outputs = self.model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        return correct / total
    
# Function to Start the Client
def start_fl_client():
    try:
        client = Client(args).to_client()
        fl.client.start_client(server_address="127.0.0.1:8080", client=client) ## server_address=args.server_address para docker  127.0.0.1:8080 sem docker 172.18.255.255:8080 com raspberries
    except Exception as e:
        logger.error("Error starting FL client: %s", e)
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Call the function to start the client
    start_fl_client()