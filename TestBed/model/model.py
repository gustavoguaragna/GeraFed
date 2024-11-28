#import tensorflow as tf
#from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#TF
# class Net2:
#     def __init__(self, learning_rate):
#         self.learning_rate = learning_rate
#         self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
#         self.model = models.Sequential(
#             [
#                 layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"),
#                 layers.MaxPooling2D(pool_size=(2, 2)),
#                 layers.MaxPooling2D(pool_size=(2, 2)),
#                 layers.Flatten(),
#                 layers.Dropout(0.5),
#                 layers.Dense(10, activation="softmax")
#             ]
#         )   
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

#     def compile(self):
#         self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=["accuracy"])

#     def get_model(self):
#         return self.model

#Pytorch
class Net2(nn.Module):
    def __init__(self, learning_rate):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 6 * 6, 10)
        self.dropout = nn.Dropout(0.5)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = x.view(-1, 32 * 6 * 6)
        x = self.dropout(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

    def set_parameters(self, parameters):
        for param, new_param in zip(self.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.data.dtype)

    def get_parameters(self):
        return [param.data.detach().cpu().numpy() for param in self.parameters()]

    def train_model(self, train_loader, epochs=2):
        self.train()
        for epoch in range(epochs):
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self(x_batch)
                loss = self.loss_function(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

    def get_model(self):
        return self


# Clase para el modelo LeNet-5 em TF
# class Model:
#     def __init__(self, learning_rate):
#         self.learning_rate = learning_rate
#         self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
#         self.model = self.build_model()
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

#     def build_model(self):
#         model = models.Sequential()
#         model.add(layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(28, 28, 1), padding='same'))
#         model.add(layers.AveragePooling2D())
#         model.add(layers.Conv2D(16, (5, 5), activation='tanh', padding='valid'))
#         model.add(layers.AveragePooling2D())
#         model.add(layers.Flatten())
#         model.add(layers.Dense(120, activation='tanh'))
#         model.add(layers.Dense(84, activation='tanh'))
#         model.add(layers.Dense(10, activation='softmax'))
#         return model

#     def compile(self):
#         self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=["accuracy"])

#     def get_model(self):
#         return self.model
    
#LeNet5 Pytorch
class Model(nn.Module):
    def __init__(self, learning_rate):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool(x)
        x = F.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    # def compile(self):
    #     pass
    def set_parameters(self, parameters):
        for param, new_param in zip(self.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.data.dtype)

    def get_parameters(self):
        return [param.data.detach().cpu().numpy() for param in self.parameters()]

    def train_model(self, train_loader, epochs=2):
        self.train()
        for epoch in range(epochs):
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self(x_batch)
                loss = self.loss_function(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

    def get_model(self):
        return self
    