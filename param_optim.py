import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
from optuna.importance import get_param_importances

EPOCHS = 5

# Função Objetiva (a ser otimizada pelo Optuna)
def objective(trial, model):
    # Escolher os hiperparâmetros dentro de um intervalo
    batch_size = trial.suggest_int("batch_size", 16, 1024)
    latent_dim = trial.suggest_categorical("latent_dim", [10, 100, 128, 256])
    lr = trial.suggest_float("learning_rate", 0.0001, 0.05, log=True)
    beta1 = trial.suggest_float("beta1", 0.0, 0.9)
    beta2 = trial.suggest_float("beta2", 0.8, 0.999)
    model = model.lower()
    if model=="wgan":
        gp_scale = trial.suggest_int("gp_scale", 0, 100)

    # Criar DataLoader com batch_size otimizado
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

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

    total_loss = 0.0
    total_batches = 0

    # Treinar por algumas épocas
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_batches = 0
        progress_bar = tqdm.TQDM(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for real_images, labels in progress_bar:
            real_images = real_images.to(device)
            labels = torch.nn.functional.one_hot(labels, NUM_CLASSES).float().to(device)

            image_labels = labels.view(labels.size(0), NUM_CLASSES, 1, 1).expand(-1, -1, 28, 28)
            real_images = torch.cat([real_images, image_labels], dim=1)

            # Treinar Discriminador
            z = torch.randn(real_images.size(0), latent_dim).to(device)
            z = torch.cat([z, labels], dim=1)
            fake_images = G(z).detach()
            fake_images = torch.cat([fake_images, image_labels], dim=1)

            optimizer_D.zero_grad()
            loss_D = discriminator_loss(D(real_images), D(fake_images)) + gp_scale * gradient_penalty(D, real_images, fake_images)
            loss_D.backward()
            optimizer_D.step()

            # Treinar Gerador
            optimizer_G.zero_grad()
            z = torch.randn(real_images.size(0), latent_dim).to(device)
            z = torch.cat([z, labels], dim=1)
            fake_images = G(z)
            loss_G = generator_loss(D(torch.cat([fake_images, image_labels], dim=1)))
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
            

    avg_loss = total_loss / total_batches

    return avg_loss  # Optuna tentará minimizar essa métrica

# Criar estudo do Optuna e otimizar hiperparâmetros
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Exibir os melhores hiperparâmetros encontrados
print("\n🔹 Melhores Hiperparâmetros Encontrados:")
print(study.best_params)

importance = get_param_importances(study)
print("Hyperparameter Importances:")
for param, imp in importance.items():
    print(f"{param}: {imp:.4f}")