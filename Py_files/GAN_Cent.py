import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from Py_files.task import F2U_GAN_CIFAR, WGAN, generate_plot, discriminator_loss, generator_loss, gradient_penalty
from tqdm import tqdm
import argparse
import json
import time

def main(test_mode: bool, gan_arq: str, freq_save: int):
    epochs = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder = f"{gan_arq}"
    os.makedirs(folder, exist_ok=True)
    metrics_filename = os.path.join(folder, 'metrics.json')

    # Start model
    if gan_arq == 'F2U_CIFAR':
        gan = F2U_GAN_CIFAR().to(device)
    elif gan_arq == 'WGAN':
        gan = WGAN().to(device)

    optim_G = torch.optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)

    if test_mode:
        dataset, _ = random_split(dataset, [1000, len(dataset) - 1000])
        epochs = 2
        freq_save = 1

    trainloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True) 

    metrics_dict = {
        'D_loss': [],
        'G_loss': [],
        'epoch_time': []
    }

    # Training loop
    initial_time = time.time()
    for epoch in range(epochs):
        gan.train()
        D_epoch_loss = 0.0
        G_epoch_loss = 0.0

        start_time = time.time()
        for data in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = data[0].to(device), data[1].to(device)
            batch_size = images.size(0)

            real_targets = torch.full((batch_size, 1), 1.0, device=device)
            fake_targets = torch.full((batch_size, 1), 0.0, device=device)

            # Train Discriminator
            optim_D.zero_grad()

            real_logits = gan(images, labels)

            z = torch.randn(batch_size, 128).to(device)
            fake_images = gan(z, labels)
            fake_logits = gan(fake_images.detach(), labels)

            if gan_arq == 'F2U_CIFAR':
                real_loss = gan.loss(real_logits, real_targets)
                fake_loss = gan.loss(fake_logits, fake_targets)
                D_loss = (real_loss + fake_loss) / 2
            elif gan_arq == 'WGAN':
                D_loss = discriminator_loss(real_logits, fake_logits) + 10 * gradient_penalty(gan, images, fake_images, labels)

            D_loss.backward()
            optim_D.step()

            # Train Generator
            optim_G.zero_grad()

            z = torch.randn(batch_size, 128).to(device)
            fake_images = gan(z, labels)
            fake_logits = gan(fake_images, labels)

            if gan_arq == 'F2U_CIFAR':
                G_loss = gan.loss(fake_logits, real_targets)
            elif gan_arq == 'WGAN':
                G_loss = generator_loss(fake_logits)

            G_loss.backward()
            optim_G.step()

            D_epoch_loss += D_loss.item() * batch_size
            G_epoch_loss += G_loss.item() * batch_size

        D_epoch_loss /= len(trainloader.dataset)
        G_epoch_loss /= len(trainloader.dataset)

        metrics_dict['D_loss'].append(D_epoch_loss)
        metrics_dict['G_loss'].append(G_epoch_loss)
        metrics_dict['epoch_time'].append(time.time() - start_time)
        
        if (epoch + 1) % freq_save == 0 or (epoch + 1) == epochs or epoch == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'gan_state_dict': gan.state_dict(),
                'optim_G_state_dict': optim_G.state_dict(),
                'optim_D_state_dict': optim_D.state_dict(),
            }
            torch.save(checkpoint, os.path.join(folder, f'checkpoint_epoch{epoch+1}.pth'))
            generate_plot(net=gan, device=device, round_number=epoch+1, folder=folder)

            try:
                with open(metrics_filename, 'w', encoding='utf-8') as f:
                    json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"Erro ao salvar métricas: {e}")

        print(f"Época {epoch+1} completa!")

    metrics_dict['total_time'] = time.time() - initial_time
    try:
        with open(metrics_filename, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Erro ao salvar métricas: {e}")

    print("Treinamento completo!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_mode', action='store_true', help='Run in test mode')
    parser.add_argument('--gan_arq', type=str, default='F2U_CIFAR', choices=['F2U_CIFAR', 'WGAN'], help='Choose GAN architecture')
    parser.add_argument('--freq_save', type=int, default=10, help='Frequency of saving model checkpoints')

    args = parser.parse_args()

    main()
