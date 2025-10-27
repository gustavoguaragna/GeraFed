import os
import json
import time
from collections import OrderedDict, defaultdict
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split, Subset
from task import ClassPartitioner, Net, Net_Cifar, aggregate_scaffold
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

    parser = argparse.ArgumentParser(description="Classifier Model Federated")

    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet parameter")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--local_test_frac", type=float, default=0.2)
    parser.add_argument("--num_chunks", type=int, default=100)
    parser.add_argument("--num_partitions", type=int, default=4)
    parser.add_argument("--partitioner", type=str, choices=["ClassPartitioner", "Dirichlet"], default="ClassPartitioner")
    parser.add_argument("--strategy", type=str, choices=["fedavg", "fedprox", "scaffold"], default="fedavg")
    parser.add_argument("--mu", type=float, default=0.5, help="fedprox parameter")

    parser.add_argument("--checkpoint_epoch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial", type=int, default=1)
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

    checkpoint_epoch = args.checkpoint_epoch
    epochs = args.epochs
    seed = args.seed
    trial = args.trial
    start_epoch = 0

    if args.test_mode:
        print("Test Mode")
        num_chunks = 10
        epochs = 2

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

        nets = [Net(seed).to(device) for _ in range(num_partitions)]
        global_net = Net(seed).to(device)

    elif dataset == "cifar10":
        image = "img"

        pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

        nets = [Net_Cifar(seed).to(device) for _ in range(num_partitions)]
        global_net = Net_Cifar(seed).to(device)

    optims = [torch.optim.Adam(net.parameters(), lr=0.01) for net in nets]
    criterion = torch.nn.CrossEntropyLoss()

    folder = f"{dataset}_{partition_dist}_{alpha_dir}_{strategy}_trial{trial}" if partition_dist == "Dirichlet" else f"{dataset}_{partition_dist}_{strategy}_trial{trial}"
    os.makedirs(folder, exist_ok=True)

    if checkpoint_epoch:
        checkpoint_loaded = torch.load(f"{folder}/checkpoint_epoch{checkpoint_epoch}.pth")

        global_net.load_state_dict(checkpoint_loaded['alvo_state_dict'])
        global_net.to(device)
        for optim, state in zip(optims, checkpoint_loaded['optimizer_alvo_state_dict']):
            optim.load_state_dict(state)

        start_epoch = checkpoint_epoch

    # Load and partition dataset
    fds = FederatedDataset(
        dataset=dataset,
        partitioners={"train": partitioner}
    )

    train_partitions = [fds.load_partition(i, split="train") for i in range(num_partitions)]

    if args.test_mode:
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

    client_test_loaders = [DataLoader(dataset=ds["test"], batch_size=64, shuffle=True) for ds in client_datasets]

    # --- SCAFFOLD state initialization ---
    # server control variate c (list of tensors matching global_net.state_dict())
    c = None
    client_cs = None
    if strategy == "scaffold":
        # initialize c and ci to zeros (same shapes as model parameters)
        global_sd = global_net.state_dict()
        zero_vec = [torch.zeros_like(v).to(device) for v in global_sd.values()]
        c = [v.clone().detach() for v in zero_vec]  # list of tensors
        # one ci per client
        client_cs = [[v.clone().detach() for v in zero_vec] for _ in range(num_partitions)]
        # Note: warm-start (computing gi(x0)) can be added if desired (paper suggests it for theory).

    # Training loop
    metrics_dict = {
                "net_loss_chunk": [],
                "net_acc_chunk": [],
                "net_loss_round": [],
                "net_acc_round": [],
                "time_chunk": [],
                "time_round": [],
                "net_time": [],
            }

    epoch_bar = tqdm(range(start_epoch, epochs), desc="Training", leave=True, position=0)

    num_classes = 10

    metrics_filename = f"{folder}/metrics.json"
    acc_filename = f"{folder}/local_accuracy_report.txt"
    
    # Epoch loop
    for epoch in epoch_bar:
        epoch_start_time = time.time()

        chunk_bar = tqdm(range(num_chunks), desc="Chunks", leave=True, position=1)

        # Chunk/round loop
        for chunk_idx in chunk_bar:
            chunk_start_time = time.time()

            params = []
            results = []

            sum_delta_ci = None  # will be accumulated as list of tensors

            client_bar = tqdm(enumerate(zip(nets, client_chunks)), desc="Clients", leave=True, position=2)

            # Client loop (Parallelizable)
            for cliente, (net, chunks) in client_bar:

                chunk_dataset = chunks[chunk_idx]
                if len(chunk_dataset) == 0:
                    print(f"Chunk {chunk_idx} for client {cliente} is empty, skipping.")
                    continue

                chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)
                
                # Evaluate global model on local test set once per epoch
                if chunk_idx == 0:
                    client_eval_time = time.time()

                    # Start counters
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

                # --- SCAFFOLD: get local control variate ci (if strategy==scaffold) ---
                if strategy == "scaffold":
                    ci = client_cs[cliente]  # list of tensors on device

                # Train classifier on dataset
                batch_bar_net = tqdm(chunk_loader, desc="Batches", leave=True, position=3)
                start_net_time = time.time()

                # Count of local update steps K_local for Option II formula (K)
                K_local = 0

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

                    # If scaffold: modify gradients by adding (c - ci) to each param grad
                    if strategy == "scaffold":
                        # c and ci are lists of tensors in same order as state_dict().items()
                        # map model.parameters() with the state_dict ordering
                        # We'll add the respective correction term to each param.grad in-place.
                        # Build an iterator over state_dict keys to match shapes
                        sd = net.state_dict()
                        sd_keys = list(sd.keys())
                        # create mapping: parameter tensor names -> param objects (by traversal)
                        # We will iterate param tensors in the same order as state_dict
                        # Danger: param ordering between state_dict and net.parameters() may differ;
                        # safest is to iterate over state_dict keys and find corresponding param in net by name.
                        # We'll use named_parameters() for reliable name mapping.
                        name_to_param = dict(net.named_parameters())
                        for idx, key in enumerate(sd_keys):
                            # state_dict key may be like 'layer.weight' or 'layer.bias' and maps to parameter name
                            # only apply to keys that correspond to parameters (i.e., in name_to_param)
                            # in many models state_dict keys equal named_parameters keys; if not, skip gracefully
                            if key in name_to_param:
                                param = name_to_param[key]
                                if param.grad is None:
                                    # if no grad (e.g., buffer or something) skip
                                    continue
                                # add correction (c - ci) for this param
                                corr = (c[idx] - ci[idx])
                                # ensure shapes match; if not, try to reshape
                                if corr.shape != param.grad.shape:
                                    # try broadcasting safely
                                    try:
                                        corr = corr.view(param.grad.shape)
                                    except Exception:
                                        # shapes mismatch, skip this key
                                        continue
                                param.grad.data.add_(corr)
                            else:
                                # not a parameter (maybe a buffer) -> skip
                                continue

                    optim.step()
                    K_local += 1

                net_time = time.time() - start_net_time

                # Append updated parameters (for aggregate_inplace) - same behavior as before:
                params.append(ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()]))
                # num_examples:
                if strategy == "scaffold":
                    num_examples = 1
                else:
                    num_examples = len(chunk_loader.dataset)

                results.append((cliente, FitRes(status=Status(code=Code.OK, message="Success"), parameters=params[cliente], num_examples=num_examples, metrics={})))

                # --- SCAFFOLD: compute updated client control variate ci^+ (Option II) ---
                if strategy == "scaffold":
                    # If no local steps happened (K_local==0) we skip update
                    if K_local == 0:
                        ci_plus = ci
                        delta_ci = [torch.zeros_like(t).to(device) for t in ci]
                    else:
                        # local learning rate (eta_l) — assume from optimizer
                        try:
                            eta_l = optim.param_groups[0]['lr']
                        except Exception:
                            eta_l = 0.01  # fallback

                        # ci_plus = ci - c + (1/(K * eta_l)) * (x - y_i)
                        # note (x - y_i) = - delta_y
                        factor = 1.0 / (max(K_local, 1) * eta_l)
                        # compute (x - y_i)
                        local_vec = state_dict_to_vector(net.state_dict())
                        global_vec = state_dict_to_vector(global_net.state_dict())
                        x_minus_y = vector_subtract(global_vec, local_vec)
                        scaled = vector_scale(x_minus_y, factor)
                        # ci_plus = ci - c + scaled
                        ci_plus = [ci_j - c_j + s_j for (ci_j, c_j, s_j) in zip(ci, c, scaled)]
                        delta_ci = [ci_p - ci_j for (ci_p, ci_j) in zip(ci_plus, ci)]

                    # Update local storage of client control variate
                    client_cs[cliente] = [t.clone().detach() for t in ci_plus]

                    # accumulate sum_delta_ci for server update
                    if sum_delta_ci is None:
                        sum_delta_ci = [d.clone().detach() for d in delta_ci]
                    else:
                        sum_delta_ci = [s + d for s, d in zip(sum_delta_ci, delta_ci)]

            if strategy == "scaffold":    
                aggregated_ndarrays = aggregate_scaffold(results, [p.cpu().numpy() for p in global_net.state_dict().values()], eta_g=1.0)
            else:
                aggregated_ndarrays = aggregate_inplace(results)

            params_dict = zip(global_net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
            global_net.load_state_dict(state_dict, strict=True)

            # --- SCAFFOLD: update server control variate c using sum_delta_ci ---
            if strategy == "scaffold" and sum_delta_ci is not None:
                # c <- c + (1/N) * sum_{i in S} (c_i^+ - c_i)
                # paper uses division by N (total clients), not |S|, see Eq (5).
                add_term = vector_scale(sum_delta_ci, 1.0 / float(num_partitions))
                c= [c_j + a_j for c_j, a_j in zip(c, add_term)]
                # Note: client_cs were already updated per-client above (ci <- ci^+)

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

            metrics_dict["time_chunk"].append(time.time() - chunk_start_time)
            metrics_dict["net_time"].append(net_time)

        # Save checkpoint
        if (epoch+1)%10==0:
            checkpoint = {
                    'epoch': epoch+1,
                    'alvo_state_dict': global_net.state_dict(),
                    'optimizer_alvo_state_dict': [optim.state_dict() for optim in optims]
                }
            # Also save scaffold state if used
            if strategy == "scaffold":
                checkpoint['c'] = c
                checkpoint['client_cs'] = client_cs
            checkpoint_file = f"{folder}/checkpoint_epoch{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_file)
            print(f"Global net saved to {checkpoint_file}")

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

        # Log metrics
        print(f"Época {epoch+1} completa")

        metrics_dict["time_round"].append(time.time() - epoch_start_time)

        try:
            with open(metrics_filename, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, ensure_ascii=False, indent=4)
            print(f"Metrics dict successfully saved to {metrics_filename}")
        except Exception as e:
            print(f"Error saving metrics dict to JSON: {e}")

if __name__ == "__main__":
    main()
