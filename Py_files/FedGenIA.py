import os
import json
import time
from collections import OrderedDict, defaultdict
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from task import F2U_GAN, F2U_GAN_CIFAR, ClassPartitioner, GeneratedDataset, generate_plot, Net, Net_Cifar, aggregate_scaffold
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate_inplace
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset
from tqdm import tqdm
import argparse
import random
import math

def state_dict_to_vector(sd):
    """Flatten state_dict tensors (torch) to a single vector list (list of tensors)."""
    # Keep as list of tensors in same order as state_dict().items()
    return [v.clone().detach() for _, v in sd.items()]

def vector_subtract(vecA, vecB):
    return [a - b for a, b in zip(vecA, vecB)]

def vector_scale(vec, scalar):
    return [v * scalar for v in vec]

def main():

        parser = argparse.ArgumentParser(description="FedGenIA")

        parser.add_argument("--alpha", type=float, default=0.1)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist")
        parser.add_argument("--local_test_frac", type=float, default=0.2)
        parser.add_argument("--num_chunks", type=int, default=100)
        parser.add_argument("--num_partitions", type=int, default=4)
        parser.add_argument("--partitioner", type=str, choices=["ClassPartitioner", "Dirichlet"], default="ClassPartitioner")
        parser.add_argument("--strategy", type=str, choices=["fedavg", "fedprox", "scaffold"], default="fedavg")
        parser.add_argument("--mu", type=float, default=0.5, help="fedprox parameter")

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
        parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait for improvement before stopping")
        parser.add_argument("--test_mode", action="store_true")

        args = parser.parse_args()

        # Set hyperparameters
        alpha_dir = args.alpha
        batch_size = args.batch_size
        dataset = args.dataset
        local_test_frac = args.local_test_frac
        num_chunks = args.num_chunks
        num_partitions = args.num_partitions
        partition_dist = args.partitioner
        strategy = args.strategy
        mu = args.mu

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
        patience = args.patience

        test_mode = args.test_mode

        if test_mode:
            print("Test Mode")
            num_chunks = 10
            epochs = 2
            extra_g_e = 2

        IN_COLAB = False
        try:
            # Tenta importar um módulo específico do Colab
            from google.colab import drive

            save_dir = "/content/drive/MyDrive/GAN_Training_Results" # Ajuste o caminho como desejar
            os.makedirs(save_dir, exist_ok=True)
            IN_COLAB = True
            print("✅ Ambiente Google Colab detectado. Downloads automáticos (a cada 2 épocas) ativados.")
        except ImportError:
            print("✅ Ambiente local detectado. Downloads automáticos desativados.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"""
        Epochs: {epochs}
        Checkpoint Epoch: {checkpoint_epoch}
        Dataset: {dataset}
        Num Chunks: {num_chunks}
        Num Partitions: {num_partitions}
        Partitioner: {partition_dist}
        Strategy: {strategy}
        Device: {device}
        """)

        if partition_dist == "Dirichlet":
            print(f"Alpha (for Dirichlet): {alpha_dir}")
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha_dir,
                min_partition_size=0,
                self_balancing=False
        )
        else:
            partitioner = ClassPartitioner(num_partitions=num_partitions, seed=seed, label_column="label")

        # Initialize models, optimizers, and set loss function
        if dataset == "mnist":
            image = "image"
            
            pytorch_transforms = Compose([
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
            models = [F2U_GAN(condition=True, seed=seed).to(device) for _ in range(num_partitions)]
            gen = F2U_GAN(condition=True, seed=seed).to(device)

            nets = [Net(seed).to(device) for _ in range(num_partitions)]
            global_net = Net(seed).to(device)

        elif dataset == "cifar10":
            image = "img"

            pytorch_transforms = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
            models = [F2U_GAN_CIFAR(condition=True, seed=seed).to(device) for _ in range(num_partitions)]
            gen = F2U_GAN_CIFAR(condition=True, seed=seed).to(device)

            nets = [Net_Cifar(seed).to(device) for _ in range(num_partitions)]
            global_net = Net_Cifar(seed).to(device)

        optim_G = torch.optim.Adam(list(gen.generator.parameters())+list(gen.label_embedding.parameters()), lr=lr_gen, betas=(beta1_gen, beta2_gen))
        optim_Ds = [
            torch.optim.Adam(list(model.discriminator.parameters())+list(model.label_embedding.parameters()), lr=lr_disc, betas=(beta1_disc, beta2_disc))
            for model in models
        ]

        optims = [torch.optim.Adam(net.parameters(), lr=0.01) for net in nets]
        criterion = torch.nn.CrossEntropyLoss()

        if IN_COLAB:
            folder = f"{save_dir}/{dataset}_{partition_dist}_{alpha_dir}_{strategy}_fedgenia" if partition_dist == "Dirichlet" else f"{save_dir}/{dataset}_{partition_dist}_{strategy}_fedgenia"
        else:   
            folder = f"{dataset}_{partition_dist}_{alpha_dir}_{strategy}_fedgenia" if partition_dist == "Dirichlet" else f"{dataset}_{partition_dist}_{strategy}_fedgenia"
        
        os.makedirs(folder, exist_ok=True)

        if checkpoint_epoch:
            checkpoint_loaded = torch.load(f"{folder}/checkpoint_epoch{checkpoint_epoch}.pth")

            global_net.load_state_dict(checkpoint_loaded['alvo_state_dict'])
            global_net.to(device)
            for optim, state in zip(optims, checkpoint_loaded['optimizer_alvo_state_dict']):
                optim.load_state_dict(state)

            gen.load_state_dict(checkpoint_loaded["gen_state_dict"])
            gen.to(device)
            optim_G.load_state_dict(checkpoint_loaded["optim_G_state_dict"])

            for model, optim_d, state_model, state_optim in zip(models, optim_Ds, checkpoint_loaded["discs_state_dict"], checkpoint_loaded["optim_Ds_state_dict"]):
                model.load_state_dict(state_model)
                model.to(device)
                optim_d.load_state_dict(state_optim)
            start_epoch = checkpoint_epoch
    
        # Load and partition dataset
        fds = FederatedDataset(
            dataset=dataset,
            partitioners={"train": partitioner}
        )

        train_partitions = [fds.load_partition(i, split="train") for i in range(num_partitions)]

        if test_mode:
            num_samples = [int(len(train_partition)/100) for train_partition in train_partitions]
            train_partitions = [train_partition.select(range(n)) for train_partition, n in zip(train_partitions, num_samples)]

        def apply_transforms(batch):
            batch[image] = [pytorch_transforms(img) for img in batch[image]]
            return batch

        train_partitions = [train_partition.with_transform(apply_transforms) for train_partition in train_partitions]

        testpartition = fds.load_split("test")
        testpartition = testpartition.with_transform(apply_transforms)
        testloader = DataLoader(testpartition, batch_size=64)

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


        client_test_loaders = [DataLoader(dataset=ds["test"], batch_size=batch_size, shuffle=True) for ds in client_datasets]

        # --- SCAFFOLD state initialization ---
        c = None
        client_cs = None
        if strategy == "scaffold":
            global_sd = global_net.state_dict()
            zero_vec = [torch.zeros_like(v).to(device) for v in global_sd.values()]
            c = [v.clone().detach() for v in zero_vec]  # list of tensors
            client_cs = [[v.clone().detach() for v in zero_vec] for _ in range(num_partitions)]

        # Training loop
        metrics_dict = {
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
        
        # Early stopping variables
        best_accuracy = 0.0
        epochs_no_improve = 0
        early_stop = False

        epoch_bar = tqdm(range(start_epoch, epochs), desc="Training", leave=True, position=0)

        batch_size_gen = 1
        latent_dim = 128
        num_classes = 10


        metrics_filename = f"{folder}/losses.json"
        acc_filename = f"{folder}/local_accuracy_report.txt"
        # dmax_mismatch_log = f"{folder}/dmax_mismatch.txt"
        # lambda_log = "lambda_log.txt"

        first_stop = True

        # Epoch loop
        for epoch in epoch_bar:
            epoch_start_time = time.time()

            client_chunks = []
            if not test_mode:
                if epoch < 10:
                    num_chunks = 100
                elif epoch < 20:
                    num_chunks = 50
                elif epoch < 50:
                    num_chunks = 10
                else:
                    num_chunks = 1
                
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

            # mismatch_count = 0
            # total_checked = 0
            g_loss_c = 0.0
            d_loss_c = 0.0
            total_d_samples = 0

            chunk_bar = tqdm(range(num_chunks), desc="Chunks", leave=True, position=1)

            for chunk_idx in chunk_bar:
                chunk_start_time = time.time()

                params = []
                results = []

                d_loss_b = 0
                total_chunk_samples = 0

                sum_delta_ci = None

                client_bar = tqdm(enumerate(zip(nets, models, client_chunks)), desc="Clients", leave=True, position=2)

                for cliente, (net, disc, chunks) in client_bar:

                    chunk_dataset = chunks[chunk_idx]
                    if len(chunk_dataset) == 0:
                        print(f"Chunk {chunk_idx} for client {cliente} is empty, skipping.")
                        continue

                    chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)
                    
                    # Evaluate global model on local test set once per epoch
                    if chunk_idx == 0:
                        client_eval_time = time.time()

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

                    # Load global model weights
                    net.load_state_dict(global_net.state_dict(), strict=True)
                    net.to(device)
                    net.train()
                    optim = optims[cliente]
                    disc.to(device)
                    optim_D = optim_Ds[cliente]
                    if strategy == "scaffold":
                        ci = client_cs[cliente]
                    K_local = 0

                    # Create combined dataloader with real and synthetic data
                    start_img_syn_time = time.time()
                    num_samples = int(13 * (math.exp(0.01*epoch) - 1) / (math.exp(0.01*50) - 1)) * 10
                    generated_dataset = GeneratedDataset(generator=gen.to("cpu"), num_samples=num_samples, latent_dim=latent_dim, num_classes=10, device="cpu", image_col_name=image)
                    gen.to(device)
                    cmb_ds = ConcatDataset([chunk_dataset, generated_dataset])
                    combined_dataloader= DataLoader(cmb_ds, batch_size=batch_size, shuffle=True)
                    img_syn_time = time.time() - start_img_syn_time

                    # Train classifier on combined dataset
                    batch_bar_net = tqdm(combined_dataloader, desc="Batches", leave=True, position=3)
                    start_net_time = time.time()
                    for batch in batch_bar_net:
                        images, labels = batch[image].to(device), batch["label"].to(device)
                        real_batch_size = images.size(0)
                        if real_batch_size == 1:
                            print("Batch size is 1, skipping batch")
                            continue
                        optim.zero_grad()
                        outputs = net(images)
                        if strategy == "fedprox":
                            proximal_term = 0
                            for local_weights, global_weights in zip(net.parameters(), global_net.parameters()):
                                proximal_term += (local_weights - global_weights).norm(2)
                            loss = criterion(net(images), labels) + (mu / 2) * proximal_term
                        else:
                            loss = criterion(outputs, labels)
                        loss.backward()
                        if strategy == "scaffold":
                            sd = net.state_dict()
                            sd_keys = list(sd.keys())
                            name_to_param = dict(net.named_parameters())
                            for idx, key in enumerate(sd_keys):
                                if key in name_to_param:
                                    param = name_to_param[key]
                                    if param.grad is None:
                                        continue
                                    corr = (c[idx] - ci[idx])
                                    if corr.shape != param.grad.shape:
                                        try:
                                            corr = corr.view(param.grad.shape)
                                        except Exception:
                                            continue
                                    param.grad.data.add_(corr)
                                else:
                                    continue
                        optim.step()
                        K_local += 1
                    net_time = time.time() - start_net_time

                    params.append(ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()]))
                    if strategy == "scaffold":
                        num_examples = 1
                    else:
                        num_examples = len(chunk_loader.dataset)
                    results.append((cliente, FitRes(status=Status(code=Code.OK, message="Success"), parameters=params[cliente], num_examples=num_examples, metrics={})))
                    
                    # --- SCAFFOLD control variate update ---
                    if strategy == "scaffold":
                        if K_local == 0:
                            ci_plus = ci
                            delta_ci = [torch.zeros_like(t).to(device) for t in ci]
                        else:
                            try:
                                eta_l = optim.param_groups[0]['lr']
                            except Exception:
                                eta_l = 0.01
                            factor = 1.0 / (max(K_local, 1) * eta_l)
                            local_vec = state_dict_to_vector(net.state_dict())
                            global_vec = state_dict_to_vector(global_net.state_dict())
                            x_minus_y = vector_subtract(global_vec, local_vec)
                            scaled = vector_scale(x_minus_y, factor)
                            ci_plus = [ci_j - c_j + s_j for (ci_j, c_j, s_j) in zip(ci, c, scaled)]
                            delta_ci = [ci_p - ci_j for (ci_p, ci_j) in zip(ci_plus, ci)]

                        client_cs[cliente] = [t.clone().detach() for t in ci_plus]

                        if sum_delta_ci is None:
                            sum_delta_ci = [d.clone().detach() for d in delta_ci]
                        else:
                            sum_delta_ci = [s + d for s, d in zip(sum_delta_ci, delta_ci)]

                    # Train discriminator
                    if early_stop:
                        if first_stop:
                            print(f"Early stopping GAN triggered at epoch {epoch+1}")
                            first_stop = False
                    else:
                        start_disc_time = time.time()
                        for batch in chunk_loader:
                            images, labels = batch[image].to(device), batch["label"].to(device)
                            real_batch_size = images.size(0)
                            if real_batch_size == 1:
                                print("Batch size is 1, skipping batch")
                                continue

                            real_ident = torch.full((real_batch_size, 1), 1., device=device)
                            fake_ident = torch.full((real_batch_size, 1), 0., device=device)

                            z_noise = torch.randn(real_batch_size, latent_dim, device=device)
                            x_fake_labels = torch.randint(0, 10, (real_batch_size,), device=device)

                            optim_D.zero_grad()

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
                            total_chunk_samples += real_batch_size
                        disc_time = time.time() - start_disc_time  

                # Aggregate updated classifier weights
                if strategy == "scaffold":    
                    aggregated_ndarrays = aggregate_scaffold(results, [p.cpu().numpy() for p in global_net.state_dict().values()], eta_g=1.0)
                else:
                    aggregated_ndarrays = aggregate_inplace(results)

                params_dict = zip(global_net.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
                global_net.load_state_dict(state_dict, strict=True)

                # --- SCAFFOLD: update server control variate c using sum_delta_ci ---
                if strategy == "scaffold" and sum_delta_ci is not None:
                    add_term = vector_scale(sum_delta_ci, 1.0 / float(num_partitions))
                    c= [c_j + a_j for c_j, a_j in zip(c, add_term)]

                # Evaluation globally each 10 chunks
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
                    metrics_dict["net_loss_chunk"].append(loss / len(testloader))
                    metrics_dict["net_acc_chunk"].append(accuracy)

                if not early_stop:
                    # Log metrics
                    avg_d_loss_chunk = d_loss_b / total_chunk_samples if total_chunk_samples > 0 else 0.0
                    metrics_dict["d_losses_chunk"].append(avg_d_loss_chunk)
                    d_loss_c += avg_d_loss_chunk * total_chunk_samples
                    total_d_samples += total_chunk_samples

                    # Train generator
                    chunk_g_loss = 0.0
                    epoch_gen_bar = tqdm(range(extra_g_e), desc="Generator", leave=True, position=2)
                    start_gen_time = time.time()
                    for g_epoch in epoch_gen_bar:

                        optim_G.zero_grad()

                        z_noise = torch.randn(batch_size_gen, latent_dim, device=device)
                        x_fake_labels = torch.randint(0, 10, (batch_size_gen,), device=device)
                        #label = int(x_fake_labels.item())

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

                        y_fake_g = Dmax(x_fake, x_fake_labels)
                        g_loss = gen.loss(y_fake_g, real_ident)

                        g_loss.backward()
                        # torch.nn.utils.clip_grad_norm_(gen.generator.parameters(), max_norm=1.0)
                        optim_G.step()
                        gen.to(device)
                        chunk_g_loss += g_loss.item()
                    gen_time = time.time() - start_gen_time

                    metrics_dict["g_losses_chunk"].append(chunk_g_loss / extra_g_e)
                    g_loss_c += chunk_g_loss /extra_g_e
                    metrics_dict["disc_time"].append(disc_time)
                    metrics_dict["gen_time"].append(gen_time)
                    # metrics_dict["track_mismatch_time"].append(track_mismatch_time)

                metrics_dict["time_chunk"].append(time.time() - chunk_start_time)
                metrics_dict["net_time"].append(net_time)
                metrics_dict["img_syn_time"].append(img_syn_time)

            if not early_stop:
                g_loss_e = g_loss_c/num_chunks
                d_loss_e = d_loss_c / total_d_samples if total_d_samples > 0 else 0.0

                metrics_dict["g_losses_round"].append(g_loss_e)
                metrics_dict["d_losses_round"].append(d_loss_e)

                # if f2a:
                #     current_lambda_star = lambda_star.item()
                #     current_lam         = F.relu(lambda_star).item()

                #     with open(lambda_log, "a") as f:
                #      f.write(f"{current_lambda_star},{current_lam}\n")

            # Evaluate globally each epoch
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
            metrics_dict["net_loss_round"].append(loss / len(testloader))
            metrics_dict["net_acc_round"].append(accuracy)

            if early_stop:
                epochs_since_stop_trigger += 1
                if epochs_since_stop_trigger >= patience/2:
                    print("Early stopping: Ending training")
                    break

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_no_improve = 0

                # Save checkpoint
                checkpoint = {
                        'epoch': epoch+1,
                        'alvo_state_dict': global_net.state_dict(),
                        'optim_alvo_state_dict': [optim.state_dict() for optim in optims],
                        'gen_state_dict': gen.state_dict(),
                        'optim_G_state_dict': optim_G.state_dict(),
                        'discs_state_dict': [model.state_dict() for model in models],
                        'optim_Ds_state_dict': [optim_d.state_dict() for optim_d in optim_Ds]
                    }
                if strategy == "scaffold":
                    checkpoint['c'] = c
                    checkpoint['client_cs'] = client_cs
                
                checkpoint_file = f"{folder}/checkpoint_epoch{epoch+1}.pth"
                torch.save(checkpoint, checkpoint_file)
                print(f"New best accuracy: {best_accuracy:.4f} - Global net saved to {checkpoint_file}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s). Best accuracy: {best_accuracy:.4f}")

                if epochs_no_improve >= patience and not early_stop:
                    early_stop = True
                    epochs_since_stop_trigger = 0
                    print(f"Early stopping: No improvement for {patience} epochs. Will stop after {round(patience/2)} more epochs.")

            print(f"Época {epoch+1} completa")
            generate_plot(net=gen, device="cpu", round_number=epoch+1, latent_dim=128, folder=folder)
            gen.to(device)

            metrics_dict["time_round"].append(time.time() - epoch_start_time)

            try:
                with open(metrics_filename, 'w', encoding='utf-8') as f:
                    json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
                print(f"Losses dict successfully saved to {metrics_filename}")
            except Exception as e:
                print(f"Error saving losses dict to JSON: {e}")
            

if __name__ == "__main__":
    main()