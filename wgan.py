import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configuração global
IMG_SIZE = 64
NUM_CHANNELS = 1
NUM_CLASSES = 5
LATENT_DIM = 128
EPOCHS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função Objetiva (a ser otimizada pelo Optuna)
def objective(trial):
    # Escolher os hiperparâmetros dentro de um intervalo
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lr = trial.suggest_loguniform("learning_rate", 0.0001, 0.0005)
    beta1 = trial.suggest_float("beta1", 0.0, 0.9)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)

    # Criar DataLoader com batch_size otimizado
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Criar novos modelos e otimizadores
    D = Discriminator().to(device)
    G = Generator().to(device)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

    best_loss = float("inf")

    # Treinar por algumas épocas
    for epoch in range(EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for real_images, labels in progress_bar:
            real_images = real_images.to(device)
            labels = torch.nn.functional.one_hot(labels, NUM_CLASSES).float().to(device)

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
            optimizer_G.zero_grad()
            z = torch.randn(real_images.size(0), LATENT_DIM).to(device)
            z = torch.cat([z, labels], dim=1)
            fake_images = G(z)
            loss_G = generator_loss(D(torch.cat([fake_images, image_labels], dim=1)))
            loss_G.backward()
            optimizer_G.step()

            progress_bar.set_postfix(d_loss=loss_D.item(), g_loss=loss_G.item())

        # Salvar melhor loss do Gerador
        best_loss = min(best_loss, loss_G.item())

    return best_loss  # Optuna tentará minimizar essa métrica

# Criar estudo do Optuna e otimizar hiperparâmetros
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Exibir os melhores hiperparâmetros encontrados
print("\n🔹 Melhores Hiperparâmetros Encontrados:")
print(study.best_params)
