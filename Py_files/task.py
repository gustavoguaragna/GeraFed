import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from collections import defaultdict
from flwr_datasets.partitioner.partitioner import Partitioner
import matplotlib.pyplot as plt
from datasets import Dataset
import numpy as np
import random

class Net(nn.Module):
    def __init__(self, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Net_Cifar(nn.Module):
    def __init__(self,seed=None):
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

class F2U_GAN(nn.Module):
    def __init__(self, dataset="mnist", img_size=28, latent_dim=128, condition=True, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(F2U_GAN, self).__init__()
        if dataset == "mnist":
            self.classes = 10
            self.channels = 1
        else:
            raise NotImplementedError("Only MNIST is supported")

        self.condition = condition
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None
        #self.label_embedding_disc = nn.Embedding(self.classes, self.img_size*self.img_size) if condition else None
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.input_shape_gen = self.latent_dim + self.label_embedding.embedding_dim if condition else self.latent_dim
        self.input_shape_disc = self.channels + self.classes if condition else self.channels

        self.adv_loss = torch.nn.BCEWithLogitsLoss()

        # Generator (unchanged) To calculate output shape of convtranspose layers, we can use the formula:
        # output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding (or dilation * (kernel_size - 1) + 1 inplace of kernel_size if using dilation)
        self.generator = nn.Sequential(
            nn.Linear(self.input_shape_gen, 256 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (256,7,7) -> (128,14,14)
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (128,14,14) -> (64,28,28)
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.channels, kernel_size=3, stride=1, padding=1), # (64,28,28) -> (1,28,28)
            nn.Tanh()
        )

        # Discriminator (corrected) To calculate output shape of conv layers, we can use the formula:
        # output_shape = ⌊(input_shape - kernel_size + 2 * padding) / stride + 1⌋ (or (dilation * (kernel_size - 1) - 1) inplace of kernel_size if using dilation)
        self.discriminator = nn.Sequential(
        # Camada 1: (1,28,28) -> (32,13,13)
        nn.utils.spectral_norm(nn.Conv2d(self.input_shape_disc, 32, kernel_size=3, stride=2, padding=0)),
        nn.LeakyReLU(0.2, inplace=True),

        # Camada 2: (32,14,14) -> (64,7,7)
        nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),

        # Camada 3: (64,7,7) -> (128,3,3)
        nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)),
        nn.LeakyReLU(0.2, inplace=True),

        # Camada 4: (128,3,3) -> (256,1,1)
        nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)),  # Padding 0 aqui!
        nn.LeakyReLU(0.2, inplace=True),

        # Achata e concatena com as labels
        nn.Flatten(), # (256,1,1) -> (256*1*1,)
        nn.utils.spectral_norm(nn.Linear(256 * 1 * 1, 1))  # 256 (features)
        )

    def forward(self, input, labels=None):
        if input.dim() == 2:
            # Generator forward pass (unchanged)
            if self.condition:
                embedded_labels = self.label_embedding(labels)
                gen_input = torch.cat((input, embedded_labels), dim=1)
                x = self.generator(gen_input)
            else:
                x = self.generator(input)
            return x.view(-1, *self.img_shape)

        elif input.dim() == 4:
            # Discriminator forward pass
            if self.condition:
                embedded_labels = self.label_embedding(labels)
                image_labels = embedded_labels.view(embedded_labels.size(0), self.label_embedding.embedding_dim, 1, 1).expand(-1, -1, self.img_size, self.img_size)
                x = torch.cat((input, image_labels), dim=1)
            else:
                x = input
            return self.discriminator(x)

    def loss(self, output, label):
        return self.adv_loss(output, label)


class F2U_GAN_SlowDisc(nn.Module):
    def __init__(self, dataset="mnist", img_size=28, latent_dim=128, condition=True, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(F2U_GAN_SlowDisc, self).__init__()
        if dataset == "mnist":
            self.classes = 10
            self.channels = 1
        else:
            raise NotImplementedError("Only MNIST is supported")

        self.condition = condition
        self.label_embedding = nn.Embedding(self.classes, self.classes) if condition else None
        #self.label_embedding_disc = nn.Embedding(self.classes, self.img_size*self.img_size) if condition else None
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.input_shape_gen = self.latent_dim + self.label_embedding.embedding_dim if condition else self.latent_dim
        self.input_shape_disc = self.channels + self.classes if condition else self.channels

        self.adv_loss = torch.nn.BCEWithLogitsLoss()

        # Generator (unchanged) To calculate output shape of convtranspose layers, we can use the formula:
        # output_shape = (input_shape - 1) * stride - 2 * padding + kernel_size + output_padding (or dilation * (kernel_size - 1) + 1 inplace of kernel_size if using dilation)
        self.generator = nn.Sequential(
            nn.Linear(self.input_shape_gen, 256 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (256,7,7) -> (128,14,14)
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (128,14,14) -> (64,28,28)
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.channels, kernel_size=3, stride=1, padding=1), # (64,28,28) -> (1,28,28)
            nn.Tanh()
        )

        # Discriminator (corrected) To calculate output shape of conv layers, we can use the formula:
        # output_shape = ⌊(input_shape - kernel_size + 2 * padding) / stride + 1⌋ (or (dilation * (kernel_size - 1) - 1) inplace of kernel_size if using dilation)
        self.discriminator = nn.Sequential(
        # Camada 1: (1,28,28) -> (32,13,13)
        nn.utils.spectral_norm(nn.Conv2d(self.input_shape_disc, 32, kernel_size=3, stride=2, padding=0)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout2d(0.3),

        # Camada 2: (32,14,14) -> (64,7,7)
        nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout2d(0.3),

        # Camada 3: (64,7,7) -> (128,3,3)
        nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout2d(0.3),

        # Camada 4: (128,3,3) -> (256,1,1)
        nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)),  # Padding 0 aqui!
        nn.LeakyReLU(0.2, inplace=True),

        # Achata e concatena com as labels
        nn.Flatten(), # (256,1,1) -> (256*1*1,)
        nn.utils.spectral_norm(nn.Linear(256 * 1 * 1, 1))  # 256 (features)
        )

    def forward(self, input, labels=None):
        if input.dim() == 2:
            # Generator forward pass (unchanged)
            if self.condition:
                embedded_labels = self.label_embedding(labels)
                gen_input = torch.cat((input, embedded_labels), dim=1)
                x = self.generator(gen_input)
            else:
                x = self.generator(input)
            return x.view(-1, *self.img_shape)

        elif input.dim() == 4:
            # Discriminator forward pass
            if self.condition:
                embedded_labels = self.label_embedding(labels)
                image_labels = embedded_labels.view(embedded_labels.size(0), self.label_embedding.embedding_dim, 1, 1).expand(-1, -1, self.img_size, self.img_size)
                #input = input + torch.randn_like(input) * 0.1
                x = torch.cat((input, image_labels), dim=1)
            else:
                x = input
            return self.discriminator(x)

    def loss(self, output, label):
        return self.adv_loss(output, label)
    
# ==========================
# GENERATOR
# ==========================
class F2U_Generator(nn.Module):
    def __init__(self, img_size=28, latent_dim=128, classes=10, condition=True, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.condition = condition
        self.classes = classes
        self.channels = 1
        self.img_size = img_size
        self.latent_dim = latent_dim

        if self.condition:
            self.label_embedding = nn.Embedding(self.classes, self.classes)
            input_dim = self.latent_dim + self.classes
        else:
            self.label_embedding = None
            input_dim = self.latent_dim

        self.img_shape = (self.channels, self.img_size, self.img_size)

        # Generator network
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (256,7,7)->(128,14,14)
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (128,14,14)->(64,28,28)
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.channels, kernel_size=3, stride=1, padding=1),  # (64,28,28)->(1,28,28)
            nn.Tanh()
        )

    def forward(self, z, labels=None):
        if self.condition:
            embedded = self.label_embedding(labels)
            z = torch.cat((z, embedded), dim=1)
        out = self.model(z)
        return out.view(-1, *self.img_shape)


# ==========================
# DISCRIMINATOR
# ==========================
class F2U_Discriminator(nn.Module):
    def __init__(self, img_size=28, classes=10, condition=True, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.condition = condition
        self.classes = classes
        self.channels = 1
        self.img_size = img_size

        if self.condition:
            self.label_embedding = nn.Embedding(self.classes, self.classes)
            input_channels = self.channels + self.classes
        else:
            self.label_embedding = None
            input_channels = self.channels

        # Discriminator network
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=0)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(256 * 1 * 1, 1))
        )

    def forward(self, x, labels=None):
        if self.condition:
            embedded_labels = self.label_embedding(labels)
            label_map = embedded_labels.view(
                embedded_labels.size(0), self.classes, 1, 1
            ).expand(-1, -1, self.img_size, self.img_size)
            x = torch.cat((x, label_map), dim=1)
        return self.model(x)


# ==========================
# OPTIONAL COMBINED WRAPPER
# ==========================
class F2U_GAN_Wrapper:
    """Utility class to hold both networks and the BCE loss."""
    def __init__(self, img_size=28, latent_dim=128, condition=True, seed=None):
        self.generator = F2U_Generator(
            img_size=img_size, latent_dim=latent_dim, condition=condition, seed=seed
        )
        self.discriminator = F2U_Discriminator(
            img_size=img_size, condition=condition, seed=seed
        )
        self.adv_loss = nn.BCEWithLogitsLoss()

    def loss(self, preds, labels):
        return self.adv_loss(preds, labels)

    
class F2U_GAN_CIFAR(nn.Module):
    def __init__(self, img_size=32, latent_dim=128, condition=True, seed=None):
        if seed is not None:
          torch.manual_seed(seed)
        super(F2U_GAN_CIFAR, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.classes = 10
        self.channels = 3
        self.condition = condition

        # Embedding para condicionamento
        self.label_embedding = nn.Embedding(self.classes, self.classes) if self.condition else None

        # Shapes de entrada
        self.input_shape_gen = self.latent_dim + (self.classes if self.condition else 0)
        self.input_shape_disc = self.channels + (self.classes if self.condition else 0)

        # -----------------
        #  Generator
        # -----------------
        self.generator = nn.Sequential(
            nn.Linear(self.input_shape_gen, 512 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 4, 4)),                  # → (512,4,4)

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # → (256,8,8)
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # → (128,16,16)
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1),  # → ( 64,32,32)
            nn.BatchNorm2d(64,  momentum=0.1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d( 64,   self.channels, kernel_size=3, stride=1, padding=1),  # → (3,32,32)
            nn.Tanh()
        )

        # -----------------
        #  Discriminator
        # -----------------
        layers = []
        in_ch = self.input_shape_disc
        cfg = [
            ( 64, 3, 1),  # → spatial stays 32
            ( 64, 4, 2),  # → 16
            (128, 3, 1),  # → 16
            (128, 4, 2),  # → 8
            (256, 4, 2),  # → 4
        ]
        for out_ch, k, s in cfg:
            layers += [
                nn.utils.spectral_norm(
                    nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=1)
                ),
                nn.LeakyReLU(0.1, inplace=True)
            ]
            in_ch = out_ch

        layers += [
            nn.Flatten(),
            nn.utils.spectral_norm(
                nn.Linear(256 * 4 * 4, 1)
            )
        ]
        self.discriminator = nn.Sequential(*layers)

        # adversarial loss
        self.adv_loss = nn.BCEWithLogitsLoss()

    def forward(self, input, labels=None):
        # Generator pass
        if input.dim() == 2 and input.size(1) == self.latent_dim:
            if self.condition:
                if labels is None:
                    raise ValueError("Labels must be provided for conditional generation")
                embedded = self.label_embedding(labels)
                gen_input = torch.cat((input, embedded), dim=1)
            else:
                gen_input = input
            img = self.generator(gen_input)
            return img

        # Discriminator pass
        elif input.dim() == 4 and input.size(1) == self.channels:
            x = input
            if self.condition:
                if labels is None:
                    raise ValueError("Labels must be provided for conditional discrimination")
                embedded = self.label_embedding(labels)
                # criar mapa de labels e concatenar
                lbl_map = embedded.view(-1, self.classes, 1, 1).expand(-1, self.classes, self.img_size, self.img_size)
                x = torch.cat((x, lbl_map), dim=1)
            return self.discriminator(x)

        else:
            raise ValueError("Input shape not recognized")

    def loss(self, logits, targets):
        return self.adv_loss(logits.view(-1), targets.float().view(-1))

import torch
import torch.nn as nn


# -------------------------
#   Generator
# -------------------------
class F2U_Generator_CIFAR(nn.Module):
    def __init__(self, img_size=32, latent_dim=128, num_classes=10, channels=3, condition=True, seed=None):
        super(F2U_Generator_CIFAR, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.channels = channels
        self.condition = condition

        # Label embedding for conditioning
        self.label_embedding = nn.Embedding(num_classes, num_classes) if condition else None

        # Input dimension for generator
        input_dim = latent_dim + (num_classes if condition else 0)

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 4, 4)),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # → (256,8,8)
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # → (128,16,16)
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # → (64,32,32)
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=1, padding=1),  # → (3,32,32)
            nn.Tanh()
        )

    def forward(self, z, labels=None):
        if self.condition and labels is not None:
            embedded = self.label_embedding(labels)
            z = torch.cat((z, embedded), dim=1)
        return self.model(z)


# -------------------------
#   Discriminator
# -------------------------
class F2U_Discriminator_CIFAR(nn.Module):
    def __init__(self, img_size=32, num_classes=10, channels=3, condition=True, seed=None):
        super(F2U_Discriminator_CIFAR, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.img_size = img_size
        self.num_classes = num_classes
        self.channels = channels
        self.condition = condition

        # Label embedding for conditioning
        self.label_embedding = nn.Embedding(num_classes, num_classes) if condition else None

        # Input channels (image + label map if conditional)
        input_channels = channels + (num_classes if condition else 0)

        layers = []
        cfg = [
            (64, 3, 1),  # stays 32×32
            (64, 4, 2),  # → 16×16
            (128, 3, 1),
            (128, 4, 2),  # → 8×8
            (256, 4, 2),  # → 4×4
        ]
        in_ch = input_channels
        for out_ch, k, s in cfg:
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=1)),
                nn.LeakyReLU(0.1, inplace=True)
            ]
            in_ch = out_ch

        layers += [
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(256 * 4 * 4, 1))
        ]

        self.model = nn.Sequential(*layers)
        self.adv_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, labels=None):
        if self.condition and labels is not None:
            embedded = self.label_embedding(labels)
            lbl_map = embedded.view(-1, self.num_classes, 1, 1).expand(-1, self.num_classes, self.img_size, self.img_size)
            x = torch.cat((x, lbl_map), dim=1)
        return self.model(x)

    def loss(self, logits, targets):
        return self.adv_loss(logits.view(-1), targets.float().view(-1))


class WGAN(nn.Module):
    def __init__(self, num_channels=3, latent_dim=128, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super(WGAN, self).__init__()
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.num_classes = 10

        # Camada de Convolução para o Discriminador
        def conv_block(in_channels, out_channels, kernel_size=5, stride=2, padding=2, use_bn=False):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        # Discriminator
        self.discriminator = nn.Sequential(
            # Camada 1: (batch, num_channels, 32, 32) -> (batch, 64, 16, 16)
            conv_block(self.num_channels + self.num_classes, 64),
            # Camada 2: (batch, 64, 16, 16) -> (batch, 128, 8, 8)
            conv_block(64, 128, use_bn=True),
            # Camada 3: (batch, 128, 8, 8) -> (batch, 256, 4, 4)
            conv_block(128, 256, use_bn=True),
            # Camada 4: (batch, 256, 4, 4) -> (batch, 512, 2, 2)
            conv_block(256, 512, use_bn=True),
            nn.Flatten(),
            # Camada Final: (batch, 512*2*2) -> (batch, 1)
            nn.Linear(512 * 2 * 2, 1)
        ) 

        # Camada de upsample para o Gerador
        def upsample_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
            layers = [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            return nn.Sequential(*layers)

        # Generator
        self.generator = nn.Sequential(
            # Camada Inicial: (batch, latent_dim + num_classes) -> (batch, 4*4*256)
            nn.Linear(self.latent_dim + self.num_classes, 4*4*256),
            nn.BatchNorm1d(4 * 4 * 256),
            nn.LeakyReLU(0.2, inplace=True),
            # Reshape para (batch, 256, 4, 4)
            nn.Unflatten(1, (256, 4, 4)),
            # Camada 1: (batch, 256, 4, 4) -> (batch, 128, 8, 8)
            upsample_block(256, 128),
            # Camada 2: (batch, 128, 8, 8) -> (batch, 64, 16, 16)
            upsample_block(128, 64),
            # Camada 3: (batch, 64, 16, 16) -> (batch, 32, 32, 32)
            upsample_block(64, 32),
            # Camada Final: (batch, 32, 32, 32) -> (batch, num_channels, 32, 32)
            nn.Conv2d(32, self.num_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()

        )  

    def forward(self, x, labels):
        if x.dim() == 2:
            # Generator forward pass
            labels = torch.nn.functional.one_hot(labels, self.num_classes).float()
            input = torch.cat((x, labels), dim =1)
            img = self.generator(input)
            return img
        
        elif x.dim() == 4:
            # Discriminator forward pass
            labels = torch.nn.functional.one_hot(labels, self.num_classes).float()
            # criar mapa de labels e concatenar
            lbl_map = labels.view(-1, self.num_classes, 1, 1).expand(-1, self.num_classes, x.size(2), x.size(3))
            x = torch.cat((x, lbl_map), dim=1)
            logits = self.discriminator(x)
            return logits
    
    
 # Função de perda Wasserstein
def discriminator_loss(real_output, fake_output):
    return fake_output.mean() - real_output.mean()

def generator_loss(fake_output):
    return -fake_output.mean()

# Função para calcular Gradient Penalty
def gradient_penalty(D, real_samples, fake_samples, labels, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolated = D(interpolated, labels)
    gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones_like(d_interpolated),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 generator,
                 num_samples,
                 latent_dim=100,
                 num_classes=10, # Total classes the generator model knows
                 desired_classes=None, # Optional: List of specific class indices to generate
                 device="cpu",
                 image_col_name="image",
                 label_col_name="label"):
        """
        Generates a dataset using a conditional generative model, potentially
        focusing on a subset of classes.

        Args:
            generator: The pre-trained generative model.
            num_samples (int): Total number of images to generate across the desired classes.
            latent_dim (int): Dimension of the latent space vector (z).
            num_classes (int): The total number of classes the generator was trained on.
                               This is crucial for correct label conditioning (e.g., one-hot dim).
            desired_classes (list[int], optional): A list of integer class indices to generate.
                                                  If None or empty, images for all classes
                                                  (from 0 to num_classes-1) will be generated,
                                                  distributed as evenly as possible.
                                                  Defaults to None.
            device (str): Device to run generation on ('cpu' or 'cuda').
            image_col_name (str): Name for the image column in the output dictionary.
            label_col_name (str): Name for the label column in the output dictionary.
        """
        self.generator = generator
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        # Store the total number of classes the generator understands
        self.total_num_classes = num_classes
        self.device = device
        self.model_type = type(self.generator).__name__ # Get generator class name
        self.image_col_name = image_col_name
        self.label_col_name = label_col_name

        # Determine the actual classes to generate based on desired_classes
        if desired_classes is not None and len(desired_classes) > 0:
            # Validate that desired classes are within the generator's known range
            if not all(0 <= c < self.total_num_classes for c in desired_classes):
                raise ValueError(f"All desired classes must be integers between 0 and {self.total_num_classes - 1}")
            # Use only the unique desired classes, sorted for consistency
            self._actual_classes_to_generate = sorted(list(set(desired_classes)))
        else:
            # If no specific classes desired, generate all classes
            self._actual_classes_to_generate = list(range(self.total_num_classes))

        # The 'classes' attribute of the dataset reflects only those generated
        self.classes = self._actual_classes_to_generate
        self.num_generated_classes = len(self.classes) # Number of classes being generated

        if self.num_generated_classes == 0 and self.num_samples > 0:
             raise ValueError("Cannot generate samples with an empty list of desired classes.")
        elif self.num_samples == 0:
             print("Warning: num_samples is 0. Dataset will be empty.")
             self.images = torch.empty(0) # Adjust shape if known
             self.labels = torch.empty(0, dtype=torch.long)
        else:
             # Generate the data only if needed
             self.images, self.labels = self.generate_data()


    def generate_data(self):
        """Generates images and corresponding labels for the specified classes."""
        self.generator.eval()
        self.generator.to(self.device)

        # --- Create Labels ---
        generated_labels_list = []
        if self.num_generated_classes > 0:
            # Distribute samples as evenly as possible among the desired classes
            samples_per_class = self.num_samples // self.num_generated_classes
            for cls in self._actual_classes_to_generate:
                generated_labels_list.extend([cls] * samples_per_class)

            # Handle remaining samples if num_samples is not perfectly divisible
            num_remaining = self.num_samples - len(generated_labels_list)
            if num_remaining > 0:
                # Add remaining samples by randomly choosing from the desired classes
                remainder_labels = random.choices(self._actual_classes_to_generate, k=num_remaining)
                generated_labels_list.extend(remainder_labels)

            # Shuffle labels for better distribution in batches later
            random.shuffle(generated_labels_list)

        # Convert labels list to tensor
        labels = torch.tensor(generated_labels_list, dtype=torch.long, device=self.device)

        # Double check label count (should match num_samples due to logic above)
        if len(labels) != self.num_samples:
             # This indicates an unexpected issue, potentially if num_generated_classes was 0 initially
             # but num_samples > 0. Raise error or adjust. Let's adjust defensively.
             print(f"Warning: Label count mismatch. Expected {self.num_samples}, got {len(labels)}. Adjusting size.")
             if len(labels) > self.num_samples:
                 labels = labels[:self.num_samples]
             else:
                 # Pad if too few (less likely with current logic unless num_generated_classes=0)
                 num_needed = self.num_samples - len(labels)
                 if self.num_generated_classes > 0:
                      padding = torch.tensor(random.choices(self._actual_classes_to_generate, k=num_needed), dtype=torch.long, device=self.device)
                      labels = torch.cat((labels, padding))
                 # If no classes to generate from, labels tensor might remain smaller

        # --- Create Latent Noise ---
        z = torch.randn(self.num_samples, self.latent_dim, device=self.device)

        # --- Generate Images in Batches ---
        generated_images_list = []
        # Consider making batch_size configurable
        batch_size = min(1024, self.num_samples) if self.num_samples > 0 else 1

        with torch.no_grad():
            for i in range(0, self.num_samples, batch_size):
                z_batch = z[i : min(i + batch_size, self.num_samples)]
                labels_batch = labels[i : min(i + batch_size, self.num_samples)]

                # Skip if batch is empty (can happen if num_samples = 0)
                if z_batch.shape[0] == 0:
                    continue

                gen_imgs = self.generator(z_batch, labels_batch)

                generated_images_list.append(gen_imgs.cpu()) # Move generated images to CPU

        self.generator.cpu() # Move generator back to CPU after generation

        # Concatenate all generated image batches
        if generated_images_list:
            all_gen_imgs = torch.cat(generated_images_list, dim=0)
        else:
            # If no images were generated (e.g., num_samples = 0)
            # Create an empty tensor. Shape needs care - determine from generator or use placeholder.
            # Let's attempt a placeholder [0, C, H, W] - requires knowing C, H, W.
            # For now, a simple empty tensor. User might need to handle this downstream.
            print("Warning: No images generated. Returning empty tensor for images.")
            all_gen_imgs = torch.empty(0)

        return all_gen_imgs, labels.cpu() # Return images and labels (on CPU)

    def __len__(self):
        # Return the actual number of samples generated
        return self.images.shape[0]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Dataset index out of range")
        return {
            self.image_col_name: self.images[idx],
            self.label_col_name: int(self.labels[idx]) # Return label as standard Python int
        }
    
def generate_plot(net, 
                  round_number, 
                  device: str="cpu", 
                  client_id = None, 
                  examples_per_class: int=5, 
                  classes: int=10, 
                  latent_dim: int=128, 
                  folder: str="."):
    """Gera plot de imagens de cada classe"""

    net_type = type(net).__name__
    net.to(device)
    net.eval()
    batch_size = examples_per_class * classes
    dataset = "cifar10" if "CIFAR" in net_type else "mnist"

    latent_vectors = torch.randn(batch_size, latent_dim, device=device)
    labels = torch.tensor([i for i in range(classes) for _ in range(examples_per_class)], device=device)

    with torch.no_grad():
        generated_images = net(latent_vectors, labels)

    # Criar uma figura com 10 linhas e 5 colunas de subplots
    fig, axes = plt.subplots(classes, examples_per_class, figsize=(5, 9))

    # Adiciona título no topo da figura
    if isinstance(client_id, int):
        fig.text(0.5, 0.98, f"Round: {round_number} | Client: {client_id}", ha="center", fontsize=12)
    else:
        fig.text(0.5, 0.98, f"Round: {round_number}", ha="center", fontsize=12)

    # Exibir as imagens nos subplots
    for i, ax in enumerate(axes.flat):
        if dataset == "mnist":
            ax.imshow(generated_images[i, 0, :, :], cmap='gray')
        else:
            images = (generated_images[i] + 1)/2
            ax.imshow(images.permute(1, 2, 0).clamp(0,1))
        ax.set_xticks([])
        ax.set_yticks([])

    # Ajustar o layout antes de calcular as posições
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])

    # Reduzir espaço entre colunas
    # plt.subplots_adjust(wspace=0.05)

    # Adicionar os rótulos das classes corretamente alinhados
    fig.canvas.draw()  # Atualiza a renderização para obter posições corretas
    for row in range(classes):
        # Obter posição do subplot em coordenadas da figura
        bbox = axes[row, 0].get_window_extent(fig.canvas.get_renderer())
        pos = fig.transFigure.inverted().transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        center_y = (pos[0, 1] + pos[1, 1]) / 2  # Centro exato da linha

        # Adicionar o rótulo
        fig.text(0.04, center_y, str(row), va='center', fontsize=12, color='black')

    if isinstance(client_id, int):
        fig.savefig(f"{folder}/{dataset}{net_type}_r{round_number}_c{client_id}.png")
        print("Imagem do cliente salva")
    else:
        fig.savefig(f"{folder}/{dataset}{net_type}_r{round_number}.png")
        print("Imagem do servidor salva")
    plt.close(fig)
    return
    
# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Class-based partitioner for Hugging Face Datasets."""

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
    def dataset(self) -> Dataset:
        return super().dataset

    @dataset.setter
    def dataset(self, value: Dataset) -> None:
        # Use parent setter for basic validation
        super(ClassPartitioner, ClassPartitioner).dataset.fset(self, value)

        # Create partitions once dataset is set
        self._create_partitions()

    def load_partition(self, partition_id: int) -> Dataset:
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
