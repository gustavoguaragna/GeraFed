import os
import json
import time
from collections import OrderedDict, defaultdict
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from aux import F2U_GAN, F2U_GAN_CIFAR, ClassPartitioner, GeneratedDataset, generate_plot, Net, Net_Cifar
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate_inplace
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset
from tqdm import tqdm
import argparse
import random
import math

parser = argparse.ArgumentParser(description="F2U Federated GAN Training")

parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist")
parser.add_argument("--local_test_frac", type=float, default=0.2)
parser.add_argument("--num_chunks", type=int, default=100)
parser.add_argument("--num_partitions", type=int, default=4)
parser.add_argument("--partitioner", type=str, choices=["ClassPartitioner", "Dirichlet"], default="ClassPartitioner")

parser.add_argument("--beta1_disc", type=float, default=0.5)
parser.add_argument("--beta1_gen", type=float, default=0.5)
parser.add_argument("--beta2_disc", type=float, default=0.999)
parser.add_argument("--beta2_gen", type=float, default=0.999)
parser.add_argument("--lr_disc", type=float, default=0.0002)
parser.add_argument("--lr_gen", type=float, default=0.0002)

parser.add_argument("--checkpoint_epoch", type=int, default=None)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--extra_g_e", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test_mode", action="store_true")

args = parser.parse_args()

alpha_dir = args.alpha
batch_size = args.batch_size
dataset = args.dataset
local_test_frac = args.local_test_frac
num_chunks = args.num_chunks
num_partitions = args.num_partitions
partitioner = args.partitioner

beta1_disc = args.beta1_disc
beta1_gen = args.beta1_gen
beta2_disc = args.beta2_disc
beta2_gen = args.beta2_gen
lr_disc = args.lr_disc
lr_gen = args.lr_gen

checkpoint_epoch = args.checkpoint_epoch
epochs = args.epochs
extra_g_e = args.extra_g_e
seed = args.seed
start_epoch = 0

print(f"""
Epochs: {epochs}
Checkpoint Epoch: {checkpoint_epoch}
Dataset: {dataset}
Num Chunks: {num_chunks}
Num Partitions: {num_partitions}
Partitioner: {partitioner}
""")
if partitioner == "Dirichlet":
    print(f"Alpha (for Dirichlet): {alpha_dir}")
if args.test_mode:
    print("Test Mode")

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if partitioner == "Dirichlet":
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=alpha_dir,
    min_partition_size=0,
    self_balancing=False
)
else:
    partitioner = ClassPartitioner(num_partitions=num_partitions, seed=seed, label_column="label")

if dataset == "mnist":
    image = "image"
    
    pytorch_transforms = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])
    models = [F2U_GAN(condition=True, seed=seed) for _ in range(num_partitions)]
    gen = F2U_GAN(condition=True, seed=seed).to(device)

    nets = [Net(seed).to(device) for _ in range(num_partitions)]
    global_net = Net(seed).to(device)

elif dataset == "cifar10":
    image = "img"

    pytorch_transforms = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    models = [F2U_GAN_CIFAR(condition=True, seed=seed) for _ in range(num_partitions)]
    gen = F2U_GAN_CIFAR(condition=True, seed=seed).to(device)

    nets = [Net_Cifar(seed).to(device) for _ in range(num_partitions)]
    global_net = Net_Cifar(seed).to(device)

optim_G = torch.optim.Adam(gen.generator.parameters(), lr=lr_gen, betas=(beta1_gen, beta2_gen))
optim_Ds = [
    torch.optim.Adam(model.discriminator.parameters(), lr=lr_disc, betas=(beta1_disc, beta2_disc))
    for model in models
]

optims = [torch.optim.Adam(net.parameters(), lr=0.01) for net in nets]
criterion = torch.nn.CrossEntropyLoss()


fds = FederatedDataset(
    dataset=dataset,
    partitioners={"train": partitioner}
)

train_partitions = [fds.load_partition(i, split="train") for i in range(num_partitions)]

# Test Mode
if args.test_mode:
    num_samples = [int(len(train_partition)/100) for train_partition in train_partitions]
    train_partitions = [train_partition.select(range(n)) for train_partition, n in zip(train_partitions, num_samples)]
    epochs = 2
    extra_g_e = 2
    num_chunks_list = [1, 10]

# min_lbl_count = 0.05
# class_labels = train_partitions[0].info.features["label"]
# labels_str = class_labels.names
# label_to_client = {lbl: [] for lbl in labels_str}
# for idx, ds in enumerate(train_partitions):
#     counts = Counter(ds['label'])
#     for label, cnt in counts.items():
#         if cnt / len(ds) >= min_lbl_count:
#             label_to_client[class_labels.int2str(label)].append(idx)

def apply_transforms(batch):
    batch[image] = [pytorch_transforms(img) for img in batch[image]]
    return batch

train_partitions = [train_partition.with_transform(apply_transforms) for train_partition in train_partitions]

testpartition = fds.load_split("test")
testpartition = testpartition.with_transform(apply_transforms)
testloader = DataLoader(testpartition, batch_size=batch_size)

client_datasets = []

for train_part in train_partitions:
    total     = len(train_part)
    test_size = int(total * local_test_frac)
    train_size = total - test_size

    client_train, client_test = random_split(
        train_part,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    client_datasets.append({
        "train": client_train,
        "test":  client_test,
    })

if checkpoint_epoch:
    checkpoint_loaded = torch.load(f"chunk_analysis/{dataset}/num_chunks{num_chunks}/checkpoint_epoch{checkpoint_epoch}.pth")

    global_net.load_state_dict(checkpoint_loaded['alvo_state_dict'])
    global_net.to(device)
    for optim, state in zip(optims, checkpoint_loaded['optimizer_alvo_state_dict']):
        optim.load_state_dict(state)

    gen.load_state_dict(checkpoint_loaded["gen_state_dict"])
    gen.to(device)
    optim_G.load_state_dict(checkpoint_loaded["optim_G_state_dict"])

    for model, optim_d, state_model, state_optim in zip(models, optim_Ds, checkpoint_loaded["discs_state_dict"], checkpoint_loaded["optim_Ds_state_dict:"]):
        model.load_state_dict(state_model)
        model.to(device)
        optim_d.load_state_dict(state_optim)
    start_epoch = checkpoint_epoch

seed = seed 
client_chunks = []
for train_partition in client_datasets:
    dts = train_partition["train"]
    n = len(dts)

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    chunk_size = math.ceil(n / num_chunks)

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        chunk_indices = indices[start:end]
        chunks.append(Subset(dts, chunk_indices))

    client_chunks.append(chunks)

client_test_loaders = [DataLoader(dataset=ds["test"], batch_size=batch_size, shuffle=True) for ds in client_datasets]

losses_dict = {
            "g_losses_chunk": [],
            "d_losses_chunk": [],
            "g_losses_round": [],
            "d_losses_round": [],
            "net_loss_chunk": [],
            "net_acc_chunk": [],
            "net_loss_round": [],
            "net_acc_round": [],
            "time_chunk": [],
            "time_round": [],
            "net_time": [],
            "disc_time": [],
            "gen_time": [],
            "img_syn_time": [],
        }

epoch_bar = tqdm(range(start_epoch, epochs), desc="Training", leave=True, position=0)

batch_size_gen = 1
batch_tam = 32
latent_dim = 128
num_classes = 10

folder = f"{dataset}_numchunks{num_chunks}"
os.makedirs(folder, exist_ok=True)
loss_filename = f"{folder}/losses.json"
acc_filename = f"{folder}/accuracy_report.txt"
# dmax_mismatch_log = f"{folder}/dmax_mismatch.txt"
# lambda_log = "lambda_log.txt"

for epoch in epoch_bar:
    epoch_start_time = time.time()
    # mismatch_count = 0
    # total_checked = 0
    g_loss_c = 0.0
    d_loss_c = 0.0
    total_d_samples = 0
    total_g_samples = 0 
    params = []
    results = []

    chunk_bar = tqdm(range(num_chunks), desc="Chunks", leave=True, position=1)

    for chunk_idx in chunk_bar:
        chunk_start_time = time.time()

        d_loss_b = 0
        total_chunk_samples = 0


        client_bar = tqdm(enumerate(zip(nets, models, client_chunks)), desc="Clients", leave=True, position=2)

        for cliente, (net, disc, chunks) in client_bar:

            chunk_dataset = chunks[chunk_idx]
            if len(chunk_dataset) == 0:
                print(f"Chunk {chunk_idx} for client {cliente} is empty, skipping.")
                continue
            chunk_loader = DataLoader(chunk_dataset, batch_size=batch_tam, shuffle=True)
            
            if chunk_idx == 0:
                client_eval_time = time.time()
                # Evaluation in client test
                # Initialize counters
                class_correct = defaultdict(int)
                class_total = defaultdict(int)
                predictions_counter = defaultdict(int)

                global_net.eval()
                with torch.no_grad():
                    for batch in client_test_loaders[cliente]:
                        images, labels = batch[image].to(device), batch["label"].to(device)
                        outputs = global_net(images)
                        _, predicted = torch.max(outputs, 1)

                        # Update counts for each sample in batch
                        for true_label, pred_label in zip(labels, predicted):
                            true_idx = true_label.item()
                            pred_idx = pred_label.item()

                            class_total[true_idx] += 1
                            predictions_counter[pred_idx] += 1

                            if true_idx == pred_idx:
                                class_correct[true_idx] += 1

                    # Create results dictionary
                    results_metrics = {
                        "class_metrics": {},
                        "overall_accuracy": None,
                        "prediction_distribution": dict(predictions_counter)
                    }

                    # Calculate class-wise metrics
                    for i in range(num_classes):
                        metrics = {
                            "samples": class_total[i],
                            "predictions": predictions_counter[i],
                            "accuracy": class_correct[i] / class_total[i] if class_total[i] > 0 else "N/A"
                        }
                        results_metrics["class_metrics"][f"class_{i}"] = metrics

                    # Calculate overall accuracy
                    total_samples = sum(class_total.values())
                    results_metrics["overall_accuracy"] = sum(class_correct.values()) / total_samples

                    # Save to txt file
                    with open(acc_filename, "a") as f:
                        f.write(f"Epoch {epoch + 1} - Client {cliente}\n")
                        # Header with fixed widths
                        f.write("{:<10} {:<10} {:<10} {:<10}\n".format(
                            "Class", "Accuracy", "Samples", "Predictions"))
                        f.write("-"*45 + "\n")

                        # Class rows with consistent formatting
                        for cls in range(num_classes):
                            metrics = results_metrics["class_metrics"][f"class_{cls}"]

                            # Format accuracy (handle "N/A" case)
                            accuracy = (f"{metrics['accuracy']:.4f}"
                                    if isinstance(metrics['accuracy'], float)
                                    else "  N/A  ")

                            f.write("{:<10} {:<10} {:<10} {:<10}\n".format(
                                f"Class {cls}",
                                accuracy,
                                metrics['samples'],
                                metrics['predictions']
                            ))

                        # Footer with alignment
                        f.write("\n{:<20} {:.4f}".format("Overall Accuracy:", results_metrics["overall_accuracy"]))
                        f.write("\n{:<20} {}".format("Total Samples:", total_samples))
                        f.write("\n{:<20} {}".format("Total Predictions:", sum(predictions_counter.values())))
                        f.write("\n{:<20} {:.4f}".format("Client Evaluation Time:", time.time() - client_eval_time))
                        f.write("\n")
                        f.write("\n")

            net.load_state_dict(global_net.state_dict(), strict=True)
            net.to(device)
            net.train()
            optim = optims[cliente]
            disc.to(device)
            optim_D = optim_Ds[cliente]

            start_img_syn_time = time.time()
            num_samples = int(13 * (math.exp(0.01*epoch) - 1) / (math.exp(0.01*50) - 1)) * 10
            generated_dataset = GeneratedDataset(generator=gen.to("cpu"), num_samples=num_samples, latent_dim=latent_dim, num_classes=10, device="cpu", image_col_name=image)
            gen.to(device)
            cmb_ds = ConcatDataset([chunk_dataset, generated_dataset])
            combined_dataloader= DataLoader(cmb_ds, batch_size=batch_tam, shuffle=True)
            img_syn_time = time.time() - start_img_syn_time

            batch_bar_net = tqdm(combined_dataloader, desc="Batches", leave=True, position=3)
            start_net_time = time.time()
            for batch in batch_bar_net:
                images, labels = batch[image].to(device), batch["label"].to(device)
                batch_size = images.size(0)
                if batch_size == 1:
                    print("Batch size is 1, skipping batch")
                    continue
                optim.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optim.step()
            net_time = time.time() - start_net_time

            batch_bar = tqdm(chunk_loader, desc="Batches", leave=True, position=4)

            start_disc_time = time.time()
            for batch in chunk_loader:
                images, labels = batch[image].to(device), batch["label"].to(device)
                batch_size = images.size(0)
                if batch_size == 1:
                    print("Batch size is 1, skipping batch")
                    continue

                real_ident = torch.full((batch_size, 1), 1., device=device)
                fake_ident = torch.full((batch_size, 1), 0., device=device)

                z_noise = torch.randn(batch_size, latent_dim, device=device)
                x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)

                optim_D.zero_grad()

                # if wgan:
                #     labels = torch.nn.functional.one_hot(labels, 10).float().to(device)
                #     x_fake_l = torch.nn.functional.one_hot(x_fake_labels, 10).float()

                #     # Adicionar labels ao images para treinamento do Discriminador
                #     image_labels = labels.view(labels.size(0), 10, 1, 1).expand(-1, -1, 28, 28)
                #     image_fake_labels = x_fake_l.view(x_fake_l.size(0), 10, 1, 1).expand(-1, -1, 28, 28)

                #     images = torch.cat([images, image_labels], dim=1)

                #     # Treinar Discriminador
                #     z = torch.cat([z_noise, x_fake_l], dim=1)
                #     fake_images = gen(z).detach()
                #     fake_images = torch.cat([fake_images, image_fake_labels], dim=1)

                #     d_loss = discriminator_loss(disc(images), disc(fake_images)) + 10 * gradient_penalty(disc, images, fake_images)

                # else:

                y_real = disc(images, labels)
                d_real_loss = disc.loss(y_real, real_ident)

                x_fake = gen(z_noise, x_fake_labels).detach()
                y_fake_d = disc(x_fake, x_fake_labels)
                d_fake_loss = disc.loss(y_fake_d, fake_ident)

                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                #torch.nn.utils.clip_grad_norm_(disc.discriminator.parameters(), max_norm=1.0)
                optim_D.step()
                d_loss_b += d_loss.item()
                total_chunk_samples += 1
            disc_time = time.time() - start_disc_time  

            params.append(ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()]))
            results.append((cliente, FitRes(status=Status(code=Code.OK, message="Success"), parameters=params[cliente], num_examples=len(chunk_loader.dataset), metrics={})))

        aggregated_ndarrays = aggregate_inplace(results)

        params_dict = zip(global_net.state_dict().keys(), aggregated_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
        global_net.load_state_dict(state_dict, strict=True)

        # Evaluation
        if chunk_idx % 10 == 0:
            global_net.eval()
            correct, loss = 0, 0.0
            with torch.no_grad():
                for batch in testloader:
                    images = batch[image].to(device)
                    labels = batch["label"].to(device)
                    outputs = global_net(images)
                    loss += criterion(outputs, labels).item()
                    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            accuracy = correct / len(testloader.dataset)
            losses_dict["net_loss_chunk"].append(loss / len(testloader))
            losses_dict["net_acc_chunk"].append(accuracy)


        avg_d_loss_chunk = d_loss_b / total_chunk_samples if total_chunk_samples > 0 else 0.0
        losses_dict["d_losses_chunk"].append(avg_d_loss_chunk)
        d_loss_c += avg_d_loss_chunk * total_chunk_samples
        total_d_samples += total_chunk_samples

        chunk_g_loss = 0.0

        epoch_gen_bar = tqdm(range(extra_g_e), desc="Generator", leave=True, position=2)

        start_gen_time = time.time()
        for g_epoch in epoch_gen_bar:

            optim_G.zero_grad()

            z_noise = torch.randn(batch_size_gen, latent_dim, device=device)
            x_fake_labels = torch.randint(0, 10, (batch_size_gen,), device=device)
            label = int(x_fake_labels.item())

            # if wgan:
            #     x_fake_labels = torch.nn.functional.one_hot(x_fake_labels, 10).float()
            #     z_noise = torch.cat([z_noise, x_fake_labels], dim=1)
            #     fake_images = gen(z_noise)

            #     # Seleciona o melhor discriminador (Dmax)
            #     image_fake_labels = x_fake_labels.view(x_fake_labels.size(0), 10, 1, 1).expand(-1, -1, 28, 28)
            #     fake_images = torch.cat([fake_images, image_fake_labels], dim=1)

            #     y_fake_gs = [model(fake_images.detach()) for model in models]

            # else:
            x_fake = gen(z_noise, x_fake_labels)

            # if f2a:
            #     y_fakes = []
            #     for D in models:
            #         D = D.to(device)
            #         y_fakes.append(D(x_fake, x_fake_labels))  # each is [B,1]
            #     # stack into [N_discriminators, B, 1]
            #     y_stack = torch.stack(y_fakes, dim=0)

            #     # 4) Compute λ = ReLU(lambda_star) to enforce λ ≥ 0
            #     lam = relu(lambda_star)

            #     # 5) Soft‐max weights across the 0th dim (discriminators)
            #     #    we want S_i = exp(λ D_i) / sum_j exp(λ D_j)
            #     #    shape remains [N, B, 1]
            #     S = torch.softmax(lam * y_stack, dim=0)

            #     # 6) Weighted sum: D_agg shape [B,1]
            #     D_agg = (S * y_stack).sum(dim=0)

            #     # 7) Compute your generator loss + β λ² regularizer
            #     real_ident = torch.full((batch_size_gen, 1), 1., device=device)
            #     adv_loss   = gen.loss(D_agg, real_ident)       # BCEWithLogitsLoss or whatever
            #     reg_loss   = beta * lam.pow(2)                 # β λ²
            #     g_loss     = adv_loss + reg_loss

            # else:

            y_fake_gs = [model(x_fake.detach(), x_fake_labels) for model in models]
            y_fake_g_means = [torch.mean(y).item() for y in y_fake_gs]
            dmax_index = y_fake_g_means.index(max(y_fake_g_means))
            Dmax = models[dmax_index]

            # start_track_mismatch_time = time.time()

            # expected_indexes = label_to_client[class_labels.int2str(x_fake_labels.item())] ##PEGA SOMENTE A PRIMEIRA LABEL, SE BATCH_SIZE_GEN FOR DIFERENTE DE 1 VAI DAR ERRO
            # if dmax_index not in expected_indexes:
            #     mismatch_count += 1
            #     total_checked +=1
            #     percent_mismatch =  mismatch_count / total_checked
            #     with open(dmax_mismatch_log, "a") as mismatch_file:
            #         mismatch_file.write(f"{epoch+1} {x_fake_labels.item()} {expected_indexes} {dmax_index} {percent_mismatch:.2f}\n")
            # else:
            #     total_checked += 1
            #     if g_epoch == extra_g_e - 1 and chunk_idx == num_chunks - 1:
            #         percent_mismatch =  mismatch_count / total_checked
            #         with open(dmax_mismatch_log, "a") as mismatch_file:
            #             mismatch_file.write(f"{epoch+1} {x_fake_labels.item()} {expected_indexes} {dmax_index} {percent_mismatch:.2f}\n")
            # track_mismatch_time = time.time() - start_track_mismatch_time

            real_ident = torch.full((batch_size_gen, 1), 1., device=device)
            # if wgan:
            #     y_fake_g = Dmax(fake_images)
            #     g_loss = generator_loss(y_fake_g)

            # else:
            y_fake_g = Dmax(x_fake, x_fake_labels)
            g_loss = gen.loss(y_fake_g, real_ident)

            g_loss.backward()
            # torch.nn.utils.clip_grad_norm_(gen.generator.parameters(), max_norm=1.0)
            optim_G.step()
            gen.to(device)
            chunk_g_loss += g_loss.item()
        gen_time = time.time() - start_gen_time

        losses_dict["g_losses_chunk"].append(chunk_g_loss / extra_g_e)
        g_loss_c += chunk_g_loss /extra_g_e

        losses_dict["time_chunk"].append(time.time() - chunk_start_time)
        losses_dict["net_time"].append(net_time)
        losses_dict["disc_time"].append(disc_time)
        losses_dict["gen_time"].append(gen_time)
        losses_dict["img_syn_time"].append(img_syn_time)
        # losses_dict["track_mismatch_time"].append(track_mismatch_time)


    g_loss_e = g_loss_c/num_chunks
    d_loss_e = d_loss_c / total_d_samples if total_d_samples > 0 else 0.0

    losses_dict["g_losses_round"].append(g_loss_e)
    losses_dict["d_losses_round"].append(d_loss_e)

    if (epoch+1)%1==0:
        checkpoint = {
                'epoch': epoch+1,
                'alvo_state_dict': global_net.state_dict(),
                'optimizer_alvo_state_dict': [optim.state_dict() for optim in optims],
                'gen_state_dict': gen.state_dict(),
                'optim_G_state_dict': optim_G.state_dict(),
                'discs_state_dict': [model.state_dict() for model in models],
                'optim_Ds_state_dict:': [optim_d.state_dict() for optim_d in optim_Ds]
            }
        checkpoint_file = f"{folder}/checkpoint_epoch{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_file)
        print(f"Global net saved to {checkpoint_file}")

        # if f2a:
        #     current_lambda_star = lambda_star.item()
        #     current_lam         = F.relu(lambda_star).item()

        #     with open(lambda_log, "a") as f:
        #      f.write(f"{current_lambda_star},{current_lam}\n")

    correct, loss = 0, 0.0
    global_net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch[image].to(device)
            labels = batch["label"].to(device)
            outputs = global_net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    losses_dict["net_loss_round"].append(loss / len(testloader))
    losses_dict["net_acc_round"].append(accuracy)

    print(f"Época {epoch+1} completa")
    generate_plot(gen, "cpu", epoch+1, latent_dim=128, folder=folder)
    gen.to(device)

    losses_dict["time_round"].append(time.time() - epoch_start_time)

    try:
        with open(loss_filename, 'w', encoding='utf-8') as f:
            json.dump(losses_dict, f, ensure_ascii=False, indent=4)
        print(f"Losses dict successfully saved to {loss_filename}")
    except Exception as e:
        print(f"Error saving losses dict to JSON: {e}")
