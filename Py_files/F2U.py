import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import Compose, ToTensor, Normalize
from task import F2U_GAN, generate_plot, ClassPartitioner
from flwr_datasets import FederatedDataset
from collections import Counter
from tqdm import tqdm
import random
import math
import time



num_partitions = 4
alpha_dir = 0.1

partitioner = ClassPartitioner(num_partitions=num_partitions, seed=42, label_column="label")

fds = FederatedDataset(
    dataset="mnist",
    partitioners={"train": partitioner}
)

train_partitions = [fds.load_partition(i, split="train") for i in range(num_partitions)]

# num_samples = [int(len(train_partition)/100) for train_partition in train_partitions]
# train_partitions = [train_partition.select(range(n)) for train_partition, n in zip(train_partitions, num_samples)]

min_lbl_count = 0.05
class_labels = train_partitions[0].info.features["label"]
labels_str = class_labels.names
label_to_client = {lbl: [] for lbl in labels_str}
for idx, ds in enumerate(train_partitions):
    counts = Counter(ds['label'])
    for label, cnt in counts.items():
        if cnt / len(ds) >= min_lbl_count:
            label_to_client[class_labels.int2str(label)].append(idx)

pytorch_transforms = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

def apply_transforms(batch):
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

train_partitions = [train_partition.with_transform(apply_transforms) for train_partition in train_partitions]

testpartition = fds.load_split("test")
testpartition = testpartition.with_transform(apply_transforms)
testloader = DataLoader(testpartition, batch_size=64)

test_frac = 0.2
client_datasets = []

for train_part in train_partitions:
    total     = len(train_part)
    test_size = int(total * test_frac)
    train_size = total - test_size

    client_train, client_test = random_split(
        train_part,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    client_datasets.append({
        "train": client_train,
        "test":  client_test,
    })

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for num_chunks in [1, 10, 50, 100, 500, 1000, 5000]:
    seed = 42 

    models = [F2U_GAN(condition=True, seed=seed) for i in range(num_partitions)]
    gen = F2U_GAN(condition=True, seed=seed).to(device)
    optim_G = torch.optim.Adam(gen.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    optim_Ds = [
        torch.optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        for model in models
    ]
    
    client_chunks = []

    for train_partition in client_datasets:
        dataset = train_partition["train"]
        n = len(dataset)

        # 1) embaralha os índices com seed fixa
        indices = list(range(n))
        random.seed(seed)
        random.shuffle(indices)

        # 2) calcula tamanho aproximado de cada chunk
        chunk_size = math.ceil(n / num_chunks)

        # 3) divide em chunks usando fatias dos índices embaralhados
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            chunk_indices = indices[start:end]
            chunks.append(Subset(dataset, chunk_indices))

        client_chunks.append(chunks)

    batch_size = 64
    client_test_loaders = [DataLoader(dataset=ds["test"], batch_size=batch_size, shuffle=True) for ds in client_datasets]

    wgan = False
    f2a = False
    epochs = 100
    losses_dict = {"g_losses_chunk": [],
                "d_losses_chunk": [],
                "g_losses_round": [],
                "d_losses_round": [],
                "time_chunk": [],
                "time_round": [],
                "disc_time": [],
                "gen_time": [],
                "track_mismatch_time": []
                }

    epoch_bar = tqdm(range(0, epochs), desc="Treinamento", leave=True, position=0)

    batch_size_gen = 1
    batch_tam = 32
    extra_g_e = 20
    latent_dim = 128
    num_classes = 10
 
    folder = f"num_chunks{num_chunks}"
    os.makedirs(folder, exist_ok=True)
    loss_filename = f"{folder}/losses.json"
    dmax_mismatch_log = f"{folder}/dmax_mismatch.txt"
    lambda_log = "lambda_log.txt"

    for epoch in epoch_bar:
        epoch_start_time = time.time()
        mismatch_count = 0
        total_checked = 0
        g_loss_c = 0.0
        d_loss_c = 0.0
        total_d_samples = 0  # Amostras totais processadas pelos discriminadores
        total_g_samples = 0  # Amostras totais processadas pelo gerador

        chunk_bar = tqdm(range(num_chunks), desc="Chunks", leave=True, position=1)

        for chunk_idx in chunk_bar:
            chunk_start_time = time.time()
            # ====================================================================
            # Treino dos Discriminadores (clientes) no bloco atual
            # ====================================================================
            d_loss_b = 0
            total_chunk_samples = 0


            client_bar = tqdm(enumerate(zip(models, client_chunks)), desc="Clients", leave=True, position=2)

            for cliente, (disc, chunks) in client_bar:
                # Carregar o bloco atual do cliente
                chunk_dataset = chunks[chunk_idx]
                if len(chunk_dataset) == 0:
                    tqdm.write(f"Chunk {chunk_idx} for client {cliente} is empty, skipping.")
                    continue
                chunk_loader = DataLoader(chunk_dataset, batch_size=batch_tam, shuffle=True)

                # Treinar o discriminador no bloco
                disc.to(device)
                optim_D = optim_Ds[cliente]

                batch_bar = tqdm(chunk_loader, desc="Batches", leave=True, position=4)

                start_disc_time = time.time()
                for batch in chunk_loader:
                    images, labels = batch["image"].to(device), batch["label"].to(device)
                    batch_size = images.size(0)
                    if batch_size == 1:
                        tqdm.write("Batch size is 1, skipping batch")
                        continue

                    real_ident = torch.full((batch_size, 1), 1., device=device)
                    fake_ident = torch.full((batch_size, 1), 0., device=device)

                    z_noise = torch.randn(batch_size, latent_dim, device=device)
                    x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)

                    # Train D
                    optim_D.zero_grad()

                    if wgan:
                        labels = torch.nn.functional.one_hot(labels, 10).float().to(device)
                        x_fake_l = torch.nn.functional.one_hot(x_fake_labels, 10).float()

                        # Adicionar labels ao images para treinamento do Discriminador
                        image_labels = labels.view(labels.size(0), 10, 1, 1).expand(-1, -1, 28, 28)
                        image_fake_labels = x_fake_l.view(x_fake_l.size(0), 10, 1, 1).expand(-1, -1, 28, 28)

                        images = torch.cat([images, image_labels], dim=1)

                        # Treinar Discriminador
                        z = torch.cat([z_noise, x_fake_l], dim=1)
                        fake_images = gen(z).detach()
                        fake_images = torch.cat([fake_images, image_fake_labels], dim=1)

                        d_loss = discriminator_loss(disc(images), disc(fake_images)) + 10 * gradient_penalty(disc, images, fake_images)

                    else:
                        # Dados Reais
                        y_real = disc(images, labels)
                        d_real_loss = disc.loss(y_real, real_ident)

                        # Dados Falsos
                        x_fake = gen(z_noise, x_fake_labels).detach()
                        y_fake_d = disc(x_fake, x_fake_labels)
                        d_fake_loss = disc.loss(y_fake_d, fake_ident)

                        # Loss total e backprop
                        d_loss = (d_real_loss + d_fake_loss) / 2

                    d_loss.backward()
                    #torch.nn.utils.clip_grad_norm_(disc.discriminator.parameters(), max_norm=1.0)
                    optim_D.step()
                    d_loss_b += d_loss.item()
                    total_chunk_samples += 1
                disc_time = time.time() - start_disc_time  


            # Média da perda dos discriminadores neste chunk
            avg_d_loss_chunk = d_loss_b / total_chunk_samples if total_chunk_samples > 0 else 0.0
            losses_dict["d_losses_chunk"].append(avg_d_loss_chunk)
            d_loss_c += avg_d_loss_chunk * total_chunk_samples
            total_d_samples += total_chunk_samples

            chunk_g_loss = 0.0

            epoch_gen_bar = tqdm(range(extra_g_e), desc="Gerador", leave=True, position=2)

            start_gen_time = time.time()
            for g_epoch in epoch_gen_bar:
                # Train G
                optim_G.zero_grad()

                # Gera dados falsos
                z_noise = torch.randn(batch_size_gen, latent_dim, device=device)
                x_fake_labels = torch.randint(0, 10, (batch_size_gen,), device=device)
                label = int(x_fake_labels.item())

                if wgan:
                    x_fake_labels = torch.nn.functional.one_hot(x_fake_labels, 10).float()
                    z_noise = torch.cat([z_noise, x_fake_labels], dim=1)
                    fake_images = gen(z_noise)

                    # Seleciona o melhor discriminador (Dmax)
                    image_fake_labels = x_fake_labels.view(x_fake_labels.size(0), 10, 1, 1).expand(-1, -1, 28, 28)
                    fake_images = torch.cat([fake_images, image_fake_labels], dim=1)

                    y_fake_gs = [model(fake_images.detach()) for model in models]

                else:
                    x_fake = gen(z_noise, x_fake_labels)

                    if f2a:
                        y_fakes = []
                        for D in models:
                            D = D.to(device)
                            y_fakes.append(D(x_fake, x_fake_labels))  # each is [B,1]
                        # stack into [N_discriminators, B, 1]
                        y_stack = torch.stack(y_fakes, dim=0)

                        # 4) Compute λ = ReLU(lambda_star) to enforce λ ≥ 0
                        lam = relu(lambda_star)

                        # 5) Soft‐max weights across the 0th dim (discriminators)
                        #    we want S_i = exp(λ D_i) / sum_j exp(λ D_j)
                        #    shape remains [N, B, 1]
                        S = torch.softmax(lam * y_stack, dim=0)

                        # 6) Weighted sum: D_agg shape [B,1]
                        D_agg = (S * y_stack).sum(dim=0)

                        # 7) Compute your generator loss + β λ² regularizer
                        real_ident = torch.full((batch_size_gen, 1), 1., device=device)
                        adv_loss   = gen.loss(D_agg, real_ident)       # BCEWithLogitsLoss or whatever
                        reg_loss   = beta * lam.pow(2)                 # β λ²
                        g_loss     = adv_loss + reg_loss

                    else:
                        # Seleciona o melhor discriminador (Dmax)
                        y_fake_gs = [model(x_fake.detach(), x_fake_labels) for model in models]
                        y_fake_g_means = [torch.mean(y).item() for y in y_fake_gs]
                        dmax_index = y_fake_g_means.index(max(y_fake_g_means))
                        Dmax = models[dmax_index]

                        start_track_mismatch_time = time.time()
                        #Track mismatches
                        expected_indexes = label_to_client[class_labels.int2str(x_fake_labels.item())] ##PEGA SOMENTE A PRIMEIRA LABEL, SE BATCH_SIZE_GEN FOR DIFERENTE DE 1 VAI DAR ERRO
                        if dmax_index not in expected_indexes:
                            mismatch_count += 1
                            total_checked +=1
                            percent_mismatch =  mismatch_count / total_checked
                            with open(dmax_mismatch_log, "a") as mismatch_file:
                                mismatch_file.write(f"{epoch+1} {x_fake_labels.item()} {expected_indexes} {dmax_index} {percent_mismatch:.2f}\n")
                        else:
                            total_checked += 1
                            if g_epoch == extra_g_e - 1 and chunk_idx == num_chunks - 1:
                                percent_mismatch =  mismatch_count / total_checked
                                with open(dmax_mismatch_log, "a") as mismatch_file:
                                    mismatch_file.write(f"{epoch+1} {x_fake_labels.item()} {expected_indexes} {dmax_index} {percent_mismatch:.2f}\n")
                        track_mismatch_time = time.time() - start_track_mismatch_time

                        # Calcula a perda do gerador
                        real_ident = torch.full((batch_size_gen, 1), 1., device=device)
                        if wgan:
                            y_fake_g = Dmax(fake_images)
                            g_loss = generator_loss(y_fake_g)

                        else:
                            y_fake_g = Dmax(x_fake, x_fake_labels)  # Detach explícito
                            g_loss = gen.loss(y_fake_g, real_ident)

                g_loss.backward()
                #torch.nn.utils.clip_grad_norm_(gen.generator.parameters(), max_norm=1.0)
                optim_G.step()
                gen.to(device)
                chunk_g_loss += g_loss.item()
            gen_time = time.time() - start_gen_time

            losses_dict["g_losses_chunk"].append(chunk_g_loss / extra_g_e)
            g_loss_c += chunk_g_loss /extra_g_e

            losses_dict["time_chunk"].append(time.time() - chunk_start_time)
            losses_dict["disc_time"].append(disc_time)
            losses_dict["gen_time"].append(gen_time)
            losses_dict["track_mismatch_time"].append(track_mismatch_time)


        g_loss_e = g_loss_c/num_chunks
        d_loss_e = d_loss_c / total_d_samples if total_d_samples > 0 else 0.0

        losses_dict["g_losses_round"].append(g_loss_e)
        losses_dict["d_losses_round"].append(d_loss_e)

        if (epoch+1)%1==0:
            checkpoint = {
                    'epoch': epoch+1,  # número da última época concluída
                    'gen_state_dict': gen.state_dict(),
                    'optim_G_state_dict': optim_G.state_dict(),
                    'discs_state_dict': [model.state_dict() for model in models],
                    'optim_Ds_state_dict:': [optim_d.state_dict() for optim_d in optim_Ds]
                }
            checkpoint_file = f"{folder}/checkpoint_epoch{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_file)
            tqdm.write(f"Global net saved to {checkpoint_file}")

            if f2a:
                current_lambda_star = lambda_star.item()
                current_lam         = F.relu(lambda_star).item()

                with open(lambda_log, "a") as f:
                 f.write(f"{current_lambda_star},{current_lam}\n")

        tqdm.write(f"Época {epoch+1} completa")
        generate_plot(gen, "cpu", epoch+1, latent_dim=128, folder=folder)
        gen.to(device)

        losses_dict["time_round"].append(time.time() - epoch_start_time)

        try:
            with open(loss_filename, 'w', encoding='utf-8') as f:
                json.dump(losses_dict, f, ensure_ascii=False, indent=4) # indent makes it readable
            tqdm.write(f"Losses dict successfully saved to {loss_filename}")
        except Exception as e:
            tqdm.write(f"Error saving losses dict to JSON: {e}")
