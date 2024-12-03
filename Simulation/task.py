"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from collections import OrderedDict

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate

def define_discriminator(img_size, classes, batch_size):
    image_input = Input(shape=(img_size*img_size,))
    label_input = Input(shape=(classes,))
    disc_input = Concatenate()([image_input, label_input])

    x = layers.Dense(batch_size*2*2*2, use_bias=True)(disc_input)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size*2*2, use_bias=True)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size*2, use_bias=True)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size, use_bias=True)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    discriminator = Model([image_input, label_input], output)
    return discriminator

def define_generator(noise_dim, classes, batch_size, img_size):
    noise_input = Input(shape=(noise_dim,))
    label_input = Input(shape=(classes,))
    gen_input = Concatenate()([noise_input, label_input])

    x = layers.Dense(batch_size, use_bias=True)(gen_input)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size*2, use_bias=True)(x)
    x = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size*2*2, use_bias=True)(x)
    x = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size*2*2*2, use_bias=True)(x)
    x = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)(x)
    x = layers.LeakyReLU(0.2)(x)

    output = layers.Dense(img_size*img_size, activation='tanh')(x)

    generator = Model([noise_input, label_input], output)
    return generator


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions, dataset="mnist", tam_batch=32):
    """Load partition dataset (MNIST or CIFAR10)."""
    # Only initialize FederatedDataset once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        if dataset == "mnist":
            fds = FederatedDataset(
                dataset="mnist",
                partitioners={"train": partitioner},
            )
        elif dataset == "cifar10":
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        else:
            raise ValueError(f"Dataset {dataset} not supported")
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    if dataset == "mnist":
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5,), (0.5,))]  # MNIST has 1 channel
        )
    elif dataset == "cifar10":
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # CIFAR-10 has 3 channels
        )

    def apply_transforms(batch, dataset=dataset):
        if dataset == "mnist":
          imagem = "image"
        elif dataset == "cifar10":
          imagem = "img"
        """Apply transforms to the partition from FederatedDataset."""
        batch[imagem] = [pytorch_transforms(img) for img in batch[imagem]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=tam_batch, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=tam_batch)
    return trainloader, testloader


def train(net, trainloader, epochs, learning_rate, device, dataset="mnist"):
    """Train the network on the training set."""
    if dataset == "mnist":
      imagem = "image"
    elif dataset == "cifar10":
      imagem = "img"
    net.to(device)  # move model to GPU if available
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[imagem]
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.05 * kld_loss
            loss.backward()
            optimizer.step()


def test(net, testloader, device, dataset="mnist"):
    """Validate the network on the entire test set."""
    if dataset == "mnist":
      imagem = "image"
    elif dataset == "cifar10":
      imagem = "img"
    total, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[imagem].to(device)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss
            total += len(images)
    return loss / total


def generate(net, image):
    """Reproduce the input with trained VAE."""
    with torch.no_grad():
        return net.forward(image)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
