import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configurações
BATCH_SIZE = 32
NUM_CHANNELS = 1
NUM_CLASSES = 10
LATENT_DIM = 128
LEARNING_RATE = 0.002
BETA1, BETA2 = 0.5, 0.9
EPOCHS = 5
CHECKPOINT_PATH = "wgan_checkpoint.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformação para normalizar imagens entre [-1,1]
transform = transforms.Compose([  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carregar dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Classes do dataset
class_names = train_dataset.classes
print(f"Classes: {class_names}")

# Função para visualizar imagens
def plot_random_images(data_loader, class_names, num_images=5):
    images, labels = next(iter(data_loader))
    images, labels = images[:num_images], labels[:num_images]

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()

# Visualizar imagens do dataset
plot_random_images(train_loader, class_names)

# -----------------------
# Modelos do WGAN-GP
# -----------------------

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
            nn.Linear(512 * 4 * 4, 1)
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
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + NUM_CLASSES, 4 * 4 * 256),
            nn.BatchNorm1d(4 * 4 * 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 4, 4)),
            upsample_block(256, 128),
            upsample_block(128, 64),
            upsample_block(64, 32),
            nn.Conv2d(32, NUM_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Inicializar modelos
D = Discriminator().to(device)
G = Generator().to(device)

# Otimizadores
optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

# Função de perda Wasserstein
def discriminator_loss(real_output, fake_output):
    return fake_output.mean() - real_output.mean()

def generator_loss(fake_output):
    return -fake_output.mean()

# -----------------------
# Treinamento do WGAN-GP
# -----------------------

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
for epoch in range(EPOCHS):
    print(f"\n🔹 Epoch {epoch+1}/{EPOCHS}")
    progress_bar = tqdm(train_loader, desc=f"Treinando {epoch+1}/{EPOCHS}", leave=True)

    for real_images, labels in progress_bar:
        real_images = real_images.to(device)
        labels = torch.nn.functional.one_hot(labels, NUM_CLASSES).float().to(device)

        # Adicionar labels ao real_images para treinamento do Discriminador
        image_labels = labels.view(labels.size(0), NUM_CLASSES, 1, 1).expand(-1, -1, IMG_SIZE, IMG_SIZE)
        real_images = torch.cat([real_images, image_labels], dim=1)

        # Treinar Discriminador
        z = torch.randn(real_images.size(0), LATENT_DIM).to(device)
        z = torch.cat([z, labels], dim=1)
        fake_images = G(z).detach()
        fake_images = torch.cat([fake_images, image_labels], dim=1)

        optimizer_D.zero_grad()
        loss_D = discriminator_loss(D(real_images), D(fake_images)) + 10 * gradient_penalty(D, real_images, fake_images)
        loss_D.backward()
        optimizer_D.step()

        # Treinar Gerador
        if epoch % 3 == 0:
            optimizer_G.zero_grad()
            z = torch.randn(real_images.size(0), LATENT_DIM).to(device)
            z = torch.cat([z, labels], dim=1)
            fake_images = G(z)
            loss_G = generator_loss(D(torch.cat([fake_images, image_labels], dim=1)))
            loss_G.backward()
            optimizer_G.step()

        progress_bar.set_postfix(d_loss=loss_D.item(), g_loss=loss_G.item())

    # Salvar modelo a cada época
    torch.save({"generator": G.state_dict(), "discriminator": D.state_dict()}, CHECKPOINT_PATH)

print("✅ Treinamento Concluído!")
