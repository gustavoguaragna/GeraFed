import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm
from optuna.importance import get_param_importances
import numpy as np
import torch.nn as nn


EPOCHS = 5

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the training and test datasets
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainset_reduzido = torch.utils.data.random_split(trainset, [1000, len(trainset) - 1000])[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model="cgan"
EPOCHS = 20
NUM_CLASSES = 10

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
    
# Função Objetiva (a ser otimizada pelo Optuna)
def objective(trial):
    # Escolher os hiperparâmetros dentro de um intervalo
    batch_size = trial.suggest_int("batch_size", 16, 1024)
    latent_dim = trial.suggest_int("latent_dim", 10, 1000)
    lr = trial.suggest_float("learning_rate", 0.0001, 0.05, log=True)
    beta1 = trial.suggest_float("beta1", 0.0, 0.9)
    beta2 = trial.suggest_float("beta2", 0.8, 0.999)
    global model
    model = model.lower()
    if model=="wgan":
        gp_scale = trial.suggest_int("gp_scale", 0, 100)

    # Criar DataLoader com batch_size otimizado
    trainloader = DataLoader(trainset_reduzido, batch_size=batch_size, shuffle=True)

    # Criar novos modelos e otimizadores
    if model == "wgan":
        D = Discriminator().to(device)
        G = Generator(latent_dim=latent_dim).to(device)
        optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
        optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

        G.train()
        D.train()
    elif model == "cgan":
        gan = CGAN(latent_dim=latent_dim).to(device)
        optimizer_D = torch.optim.Adam(gan.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        optimizer_G = torch.optim.Adam(gan.generator.parameters(), lr=lr, betas=(beta1, beta2))
    
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.9)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.9)

    total_loss = 0.0
    total_batches = 0

    # Treinar por algumas épocas
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_batches = 0
        progress_bar = tqdm.tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for real_images, labels in progress_bar:
            real_images = real_images.to(device)
            labels = labels.to(device)
            fake_labels = torch.randint(0, NUM_CLASSES, (real_images.size(0),), device=device)
            z = torch.randn(real_images.size(0), latent_dim).to(device)
            optimizer_D.zero_grad()
            if model=="wgan":
                labels = torch.nn.functional.one_hot(labels, NUM_CLASSES).float().to(device)
                fake_labels = torch.nn.functional.one_hot(fake_labels, NUM_CLASSES).float()

                # Adicionar labels ao real_images para treinamento do Discriminador
                image_labels = labels.view(labels.size(0), NUM_CLASSES, 1, 1).expand(-1, -1, 28, 28)
                image_fake_labels = fake_labels.view(fake_labels.size(0), NUM_CLASSES, 1, 1).expand(-1, -1, 28, 28)
        
                real_images = torch.cat([real_images, image_labels], dim=1)

                # Treinar Discriminador
                z = torch.cat([z, labels], dim=1)
                fake_images = G(z).detach()
                fake_images = torch.cat([fake_images, image_labels], dim=1)

                loss_D = discriminator_loss(D(real_images), D(fake_images)) + gp_scale * gradient_penalty(D, real_images, fake_images)
           
           
            else:
                real_ident = torch.full((real_images.size(0), 1), 1., device=device)
                fake_ident = torch.full((real_images.size(0), 1), 0., device=device)
                x_fake = gan(z, fake_labels)

                y_real = gan(real_images, labels)
                d_real_loss = gan.loss(y_real, real_ident)
                y_fake_d = gan(x_fake.detach(), fake_labels)
                d_fake_loss = gan.loss(y_fake_d, fake_ident)
                loss_D = (d_real_loss + d_fake_loss) / 2          
           
            loss_D.backward()
            optimizer_D.step()
            

            # Treinar Gerador
            optimizer_G.zero_grad()
  
            if model=="wgan":
                fake_images = G(z)
                loss_G = generator_loss(D(torch.cat([fake_images, image_fake_labels], dim=1)))
            else:
                y_fake_g = gan(x_fake, fake_labels)
                loss_G = gan.loss(y_fake_g, real_ident)
            
            loss_G.backward()
            optimizer_G.step()

            epoch_loss += loss_G.item()
            total_loss += loss_G.item()
            total_batches += 1
            epoch_batches += 1

            progress_bar.set_postfix(d_loss=loss_D.item(), g_loss=loss_G.item())

        # Calcular a loss média dessa época
        epoch_avg_loss = epoch_loss / epoch_batches
        # Reporta a loss média da época para pruning
        trial.report(epoch_avg_loss, epoch)
        if trial.should_prune() and epoch >=3:
            raise optuna.exceptions.TrialPruned()
        
        scheduler_G.step()
        scheduler_D.step()
        print(f"Após Epoch {epoch+1}, LR_G: {optimizer_G.param_groups[0]['lr']:.6f}, LR_D: {optimizer_D.param_groups[0]['lr']:.6f}")
   
            

    avg_loss = total_loss / total_batches

    return avg_loss  # Optuna tentará minimizar essa métrica

# Criar estudo do Optuna e otimizar hiperparâmetros
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Exibir os melhores hiperparâmetros encontrados
print("\n🔹 Melhores Hiperparâmetros Encontrados:")
print(study.best_params)

importance = get_param_importances(study)
print("Hyperparameter Importances:")
for param, imp in importance.items():
    print(f"{param}: {imp:.4f}")