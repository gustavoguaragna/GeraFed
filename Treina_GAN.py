import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import logging
import time

class Net(nn.Module):
    def __init__(self):
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
    
# Configurações
BATCH_SIZE = 128
LATENT_DIM = 128
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.9
GP_SCALE = 10
NUM_CHANNELS = 1
NUM_CLASSES = 10
EPOCHS = 50

    # Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the training and test datasets
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainset_reduzido = torch.utils.data.random_split(trainset, [1000, len(trainset) - 1000])[0]
# Create data loaders
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
trainloader_reduzido = DataLoader(trainset_reduzido, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE)

class GeneratedDataset(Dataset):
    def __init__(self, generator, num_samples, latent_dim, num_classes, device):
        self.generator = generator
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device
        self.images, self.labels = self.generate_data()

    def generate_data(self):
        self.generator.eval()
        labels = torch.tensor([i for i in range(self.num_classes) for _ in range(self.num_samples // self.num_classes)], device=self.device)
        labels_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device) #
        z = torch.randn(self.num_samples, self.latent_dim, device=self.device)
        with torch.no_grad():
            gen_imgs = self.generator(torch.cat([z, labels_one_hot], dim=1))
            #gen_imgs = self.generator(z, labels)
            
        return gen_imgs.cpu(), labels.cpu()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class CGAN(nn.Module):
    def __init__(self, dataset="mnist", img_size=28, latent_dim=100, batch_size=64):
        super(CGAN, self).__init__()
        if dataset == "mnist":
            self.classes = 10
            self.channels = 1
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.label_embedding = nn.Embedding(self.classes, self.classes)
        self.adv_loss = torch.nn.BCELoss()


        self.generator = nn.Sequential(
            *self._create_layer_gen(self.latent_dim + self.classes, 128, False),
            *self._create_layer_gen(128, 256),
            *self._create_layer_gen(256, 512),
            *self._create_layer_gen(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            *self._create_layer_disc(self.classes + int(np.prod(self.img_shape)), 1024, False, True),
            *self._create_layer_disc(1024, 512, True, True),
            *self._create_layer_disc(512, 256, True, True),
            *self._create_layer_disc(256, 128, False, False),
            *self._create_layer_disc(128, 1, False, False),
            nn.Sigmoid()
        )

    def _create_layer_gen(self, size_in, size_out, normalize=True):
        layers = [nn.Linear(size_in, size_out)]
        if normalize:
            layers.append(nn.BatchNorm1d(size_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def _create_layer_disc(self, size_in, size_out, drop_out=True, act_func=True):
        layers = [nn.Linear(size_in, size_out)]
        if drop_out:
            layers.append(nn.Dropout(0.4))
        if act_func:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, input, labels):
        if input.dim() == 2:
            z = torch.cat((self.label_embedding(labels), input), -1)
            x = self.generator(z)
            x = x.view(x.size(0), *self.img_shape) #Em
            return x
        elif input.dim() == 4:
            x = torch.cat((input.view(input.size(0), -1), self.label_embedding(labels)), -1)
            return self.discriminator(x)

    def loss(self, output, label):
        return self.adv_loss(output, label)


def train(net, trainloader, epochs, learning_rate, device, dataset="mnist", latent_dim=100):
    """Train the network on the training set."""
    if dataset == "mnist":
      imagem = "image"
    elif dataset == "cifar10":
      imagem = "img"

    net.to(device)  # move model to GPU if available
    optim_G = torch.optim.Adam(net.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(net.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []


    for epoch in range(epochs):
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch[0].to(device), batch[1].to(device)
            batch_size = images.size(0)
            real_ident = torch.full((batch_size, 1), 1., device=device)
            fake_ident = torch.full((batch_size, 1), 0., device=device)

            # Train G
            net.zero_grad()
            z_noise = torch.randn(batch_size, latent_dim, device=device)
            x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
            x_fake = net(z_noise, x_fake_labels)
            y_fake_g = net(x_fake, x_fake_labels)
            g_loss = net.loss(y_fake_g, real_ident)
            g_loss.backward()
            optim_G.step()

            # Train D
            net.zero_grad()
            y_real = net(images, labels)
            d_real_loss = net.loss(y_real, real_ident)
            y_fake_d = net(x_fake.detach(), x_fake_labels)
            d_fake_loss = net.loss(y_fake_d, fake_ident)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optim_D.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            if batch_idx % 100 == 0 and batch_idx > 0:
                print('Epoch {} [{}/{}] loss_D_treino: {:.4f} loss_G_treino: {:.4f}'.format(
                            epoch, batch_idx, len(trainloader),
                            d_loss.mean().item(),
                            g_loss.mean().item()))



def test(net, testloader, device, dataset="mnist", latent_dim=100):
    """Validate the network on the entire test set."""
    if dataset == "mnist":
      imagem = "image"
    elif dataset == "cifar10":
      imagem = "img"
    g_losses = []
    d_losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            images, labels = batch[imagem].to(device), batch["label"].to(device)
            batch_size = images.size(0)
            real_ident = torch.full((batch_size, 1), 1., device=device)
            fake_ident = torch.full((batch_size, 1), 0., device=device)

            #Gen loss
            z_noise = torch.randn(batch_size, latent_dim, device=device)
            x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
            x_fake = net(z_noise, x_fake_labels)
            y_fake_g = net(x_fake, x_fake_labels)
            g_loss = net.loss(y_fake_g, real_ident)

            #Disc loss
            y_real = net(images, labels)
            d_real_loss = net.loss(y_real, real_ident)
            y_fake_d = net(x_fake.detach(), x_fake_labels)
            d_fake_loss = net.loss(y_fake_d, fake_ident)
            d_loss = (d_real_loss + d_fake_loss) / 2

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            if batch_idx % 100 == 0 and batch_idx > 0:
                print('[{}/{}] loss_D_teste: {:.4f} loss_G_teste: {:.4f}'.format(
                            batch_idx, len(testloader),
                            d_loss.mean().item(),
                            g_loss.mean().item()))
    return np.mean(g_losses), np.mean(d_losses)

# Camada de Convolução para o Discriminador
def conv_block(in_channels, out_channels, kernel_size=5, stride=2, padding=2, use_bn=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

# Discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            conv_block(NUM_CHANNELS + NUM_CLASSES, 64, use_bn=False),
            conv_block(64, 128, use_bn=True),
            conv_block(128, 256, use_bn=True),
            conv_block(256, 512, use_bn=True),
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1)
        )

    def forward(self, x):
        return self.model(x)

# Camada de upsample para o Gerador
def upsample_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
    layers = [
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
        nn.LeakyReLU(0.2, inplace=True)
    ]
    return nn.Sequential(*layers)

# Gerador
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + NUM_CLASSES, 4 * 4 * 256),
            nn.BatchNorm1d(4 * 4 * 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 4, 4)),
            upsample_block(256, 128),
            upsample_block(128, 64),
            upsample_block(64, 32),
            nn.Conv2d(32, NUM_CHANNELS, kernel_size=5, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializar modelos
D = Discriminator().to(device)
G = Generator(latent_dim=LATENT_DIM).to(device)
# Otimizadores
optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
 
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.9)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.9)

 # Função de perda Wasserstein
def discriminator_loss(real_output, fake_output):
    return fake_output.mean() - real_output.mean()

def generator_loss(fake_output):
    return -fake_output.mean()

# Função para calcular Gradient Penalty
def gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolated = D(interpolated)
    gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones_like(d_interpolated),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# Treinamento
historico_metricas = []
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
epoch_bar = tqdm(range(EPOCHS), desc="Treinamento")
for epoch in epoch_bar:

    print(f"\n🔹 Epoch {epoch+1}/{EPOCHS}")
    G_loss = 0
    D_loss = 0
    batches = 0

    batch_bar = tqdm(trainloader_reduzido, desc="Batches")

    start_time = time.time()

    for real_images, labels in batch_bar:
        real_images = real_images.to(device)
        labels = torch.nn.functional.one_hot(labels, NUM_CLASSES).float().to(device)
        fake_labels = torch.randint(0, NUM_CLASSES, (real_images.size(0),), device=device)
        fake_labels = torch.nn.functional.one_hot(fake_labels, NUM_CLASSES).float()

        # Adicionar labels ao real_images para treinamento do Discriminador
        image_labels = labels.view(labels.size(0), NUM_CLASSES, 1, 1).expand(-1, -1, 28, 28)
        image_fake_labels = fake_labels.view(fake_labels.size(0), NUM_CLASSES, 1, 1).expand(-1, -1, 28, 28)
        
        real_images = torch.cat([real_images, image_labels], dim=1)

        # Treinar Discriminador
        z = torch.randn(real_images.size(0), LATENT_DIM).to(device)
        z = torch.cat([z, fake_labels], dim=1)
        fake_images = G(z).detach()
        fake_images = torch.cat([fake_images, image_fake_labels], dim=1)

        optimizer_D.zero_grad() 
        D(real_images)
        loss_D = discriminator_loss(D(real_images), D(fake_images)) + GP_SCALE * gradient_penalty(D, real_images, fake_images)
        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        z = torch.randn(real_images.size(0), LATENT_DIM).to(device)
        z = torch.cat([z, fake_labels], dim=1)
        fake_images = G(z)
        loss_G = generator_loss(D(torch.cat([fake_images, image_fake_labels], dim=1)))
        loss_G.backward()
        optimizer_G.step()

        G_loss += loss_G.item()
        D_loss += loss_D.item()
        batches += BATCH_SIZE
    
    avg_epoch_G_loss = G_loss/batches
    avg_epoch_D_loss = D_loss/batches
    # Create the dataset and dataloader
    generated_dataset = GeneratedDataset(generator=G, num_samples=10000, latent_dim=128, num_classes=10, device=device)
    generated_dataloader = DataLoader(generated_dataset, batch_size=64, shuffle=True)

    net = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    for _ in range(5):
        for data in generated_dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    net.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[0]
            labels = batch[1]
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)

    end_time = time.time()
    total_time = end_time - start_time

    epoch_bar.set_postfix({
        "D_loss": f"{avg_epoch_D_loss:.4f}",
        "G_loss": f"{avg_epoch_G_loss:.4f}",
        "Acc": f"{accuracy:.4f}"
    })

    logging.info(f"✅ Epoch {epoch+1}: D_loss={avg_epoch_D_loss:.4f}, G_loss={avg_epoch_G_loss:.4f}, Acc={accuracy:.4f}")

    with open("Treino_GAN.txt", "a") as f:
            f.write(f"Epoca: {epoch+1}, D_loss: {avg_epoch_D_loss:.4f}, G_loss: {avg_epoch_G_loss:.4f}, Acc: {accuracy:.4f}, Tempo: {total_time:.4f}\n")


    # Salvar modelo a cada época
    torch.save({"generator": G.state_dict(), "discriminator": D.state_dict()}, f"wgan_{epoch+1}e_{BATCH_SIZE}b_{LEARNING_RATE}lr.pth")
    
    #Atualiza o learning_rate
    scheduler_G.step()
    scheduler_D.step()
    print(f"Após Epoch {epoch+1}, LR_G: {optimizer_G.param_groups[0]['lr']:.6f}, LR_D: {optimizer_D.param_groups[0]['lr']:.6f}")
    
print("✅ Treinamento Concluído!")