import optuna
import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize
from task import F2U_GAN, ClassPartitioner
from flwr_datasets import FederatedDataset

class GANConvergenceMetric:
    """
    Evaluates GAN convergence based on multiple criteria.
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        
    def compute_score(self, g_losses, d_losses, epoch):
        """
        Lower score is better. Combines multiple metrics:
        1. Generator loss stability (lower variance is better)
        2. Discriminator not too weak (d_loss not too low)
        3. Balance between G and D losses
        4. Recent trend in losses
        """
        if len(g_losses) < self.window_size or len(d_losses) < self.window_size:
            return float('inf')  # Not enough data
        
        # Get recent losses
        recent_g = g_losses[-self.window_size:]
        recent_d = d_losses[-self.window_size:]
        
        # 1. Generator loss should be reasonable and stable
        g_mean = np.mean(recent_g)
        g_std = np.std(recent_g)
        
        # 2. Discriminator should not be too confident (loss too low)
        d_mean = np.mean(recent_d)
        
        # 3. Balance metric - they should be in similar range
        # Ideal ratio is around 1-2 (G slightly higher than D)
        if d_mean < 0.1:  # D too strong
            balance_penalty = 100.0
        else:
            loss_ratio = g_mean / (d_mean + 1e-8)
            # Penalize if ratio is too far from ideal range [0.8, 3.0]
            if loss_ratio < 0.8:
                balance_penalty = (0.8 - loss_ratio) * 10
            elif loss_ratio > 3.0:
                balance_penalty = (loss_ratio - 3.0) * 10
            else:
                balance_penalty = 0.0
        
        # 4. Trend - prefer decreasing or stable G loss
        if len(recent_g) >= 5:
            first_half_g = np.mean(recent_g[:5])
            second_half_g = np.mean(recent_g[-5:])
            trend_penalty = max(0, second_half_g - first_half_g) * 2
        else:
            trend_penalty = 0
        
        # 5. Penalize if D loss is too low (discriminator too strong)
        d_strength_penalty = max(0, 0.2 - d_mean) * 20

        # 6. Penalize if G loss is too high (generator too weak)
        g_strength_penalty = max(0, g_mean - 0.7) * 20

        # Combine all metrics
        score = (
            g_mean * 1.0 +           # Lower G loss is better
            g_std * 2.0 +            # Lower variance is better
            balance_penalty +         # Penalty for imbalance
            trend_penalty +           # Penalty for increasing G loss
            d_strength_penalty +      # Penalty for too-strong D
            g_strength_penalty        # Penalty for too-weak G
        )
        
        return score


def run_trial_training(trial, dataset="mnist", epochs=20, device="cuda"):
    """
    Train GAN with hyperparameters suggested by Optuna trial.
    Returns a score indicating convergence quality.
    """
    
    # Hyperparameter suggestions
    d_lr = trial.suggest_float("d_lr", 1e-8, 1e-3, log=True)
    g_lr = trial.suggest_float("g_lr", 1e-5, 1e-3, log=True)
    beta1_d = trial.suggest_float("beta1_d", 0.2, 0.9)
    beta2_d = trial.suggest_float("beta2_d", 0.7, 0.999)
    wd_d = trial.suggest_float("wd_d", 0.0, 1e-1)
    batch_tam = trial.suggest_categorical("batch_tam", [16, 32, 64, 128, 256])
    
    # Optional: add more hyperparameters
    beta1_g = trial.suggest_float("beta1_g", 0.4, 0.6)
    beta2_g = trial.suggest_float("beta2_g", 0.9, 0.999)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"d_lr={d_lr:.6f}, beta1_d={beta1_d:.3f}, beta2_d={beta2_d:.4f}")
    print(f"wd_d={wd_d:.5f}, batch_tam={batch_tam}")
    print(f"{'='*60}\n")
    
    # Setup
    num_partitions = 4
    num_chunks = 1  # No chunking!
    extra_g_e = 20
    latent_dim = 128
    
    # Data preparation
    partitioner = ClassPartitioner(num_partitions=num_partitions, seed=42, label_column="label")
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    
    models = [F2U_GAN(condition=True, seed=42) for _ in range(num_partitions)]
    gen = F2U_GAN(condition=True, seed=42).to(device)
    
    optim_G = torch.optim.Adam(gen.generator.parameters(), lr=g_lr, betas=(beta1_g, beta2_g))
    optim_Ds = [
        torch.optim.Adam(model.discriminator.parameters(), lr=d_lr, betas=(beta1_d, beta2_d), weight_decay=wd_d)
        for model in models
    ]
    # optim_G = torch.optim.Adam(list(gen.generator.parameters())+list(gen.label_embedding.parameters()), lr=g_lr, betas=(beta1_g, beta2_g))
    # optim_Ds = [
    #     torch.optim.Adam(list(model.discriminator.parameters())+list(model.label_embedding.parameters()), lr=d_lr, betas=(beta1_d, beta2_d), weight_decay=wd_d)
    #     for model in models
    # ]

    fds = FederatedDataset(dataset=dataset, partitioners={"train": partitioner})
    train_partitions = [fds.load_partition(i, split="train") for i in range(num_partitions)]
    
    # Use smaller subset for faster trials
    num_samples = [min(5000, len(tp)) for tp in train_partitions]
    train_partitions = [tp.select(range(n)) for tp, n in zip(train_partitions, num_samples)]
    
    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch
    
    train_partitions = [tp.with_transform(apply_transforms) for tp in train_partitions]
    
    # Split into train/test
    test_frac = 0.2
    client_datasets = []
    for train_part in train_partitions:
        total = len(train_part)
        test_size = int(total * test_frac)
        train_size = total - test_size
        client_train, client_test = random_split(
            train_part, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        client_datasets.append({"train": client_train, "test": client_test})
    
    # Training loop
    g_losses_round = []
    d_losses_round = []
    metric_evaluator = GANConvergenceMetric(window_size=5)
    
    batch_size_gen = 1
    
    for epoch in range(epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        total_d_samples = 0
        
        # Train Discriminators
        for cliente, disc in enumerate(models):
            chunk_dataset = client_datasets[cliente]["train"]
            chunk_loader = DataLoader(chunk_dataset, batch_size=batch_tam, shuffle=True)
            
            disc.to(device)
            optim_D = optim_Ds[cliente]
            
            for batch in chunk_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                batch_size = images.size(0)
                if batch_size == 1:
                    continue
                
                real_ident = torch.full((batch_size, 1), 1., device=device)
                fake_ident = torch.full((batch_size, 1), 0., device=device)
                
                z_noise = torch.randn(batch_size, latent_dim, device=device)
                x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
                
                # Train D
                optim_D.zero_grad()
                
                y_real = disc(images, labels)
                d_real_loss = disc.loss(y_real, real_ident)
                
                x_fake = gen(z_noise, x_fake_labels).detach()
                y_fake_d = disc(x_fake, x_fake_labels)
                d_fake_loss = disc.loss(y_fake_d, fake_ident)
                
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optim_D.step()
                
                d_loss_epoch += d_loss.item()
                total_d_samples += 1
        
        avg_d_loss = d_loss_epoch / total_d_samples if total_d_samples > 0 else 0.0
        
        # Train Generator
        chunk_g_loss = 0.0
        for g_epoch in range(extra_g_e):
            optim_G.zero_grad()
            
            z_noise = torch.randn(batch_size_gen, latent_dim, device=device)
            x_fake_labels = torch.randint(0, 10, (batch_size_gen,), device=device)
            x_fake = gen(z_noise, x_fake_labels)
            
            # Select Dmax
            y_fake_gs = [model(x_fake.detach(), x_fake_labels) for model in models]
            y_fake_g_means = [torch.mean(y).item() for y in y_fake_gs]
            dmax_index = y_fake_g_means.index(max(y_fake_g_means))
            Dmax = models[dmax_index]
            
            real_ident = torch.full((batch_size_gen, 1), 1., device=device)
            y_fake_g = Dmax(x_fake, x_fake_labels)
            g_loss = gen.loss(y_fake_g, real_ident)
            
            g_loss.backward()
            optim_G.step()
            gen.to(device)
            chunk_g_loss += g_loss.item()
        
        avg_g_loss = chunk_g_loss / extra_g_e
        
        g_losses_round.append(avg_g_loss)
        d_losses_round.append(avg_d_loss)
        
        # Report intermediate values
        trial.report(avg_g_loss, epoch)
        
        # Check for pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        print(f"Epoch {epoch+1}/{epochs} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
    
    # Compute final score
    final_score = metric_evaluator.compute_score(g_losses_round, d_losses_round, epochs)
    
    # Save losses for this trial
    trial.set_user_attr("g_losses", g_losses_round)
    trial.set_user_attr("d_losses", d_losses_round)
    
    return final_score


def optimize_hyperparameters(n_trials=50, dataset="mnist", epochs=20):
    """
    Run Optuna optimization to find best hyperparameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: run_trial_training(trial, dataset=dataset, epochs=epochs, device=device),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best score: {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "best_params": study.best_trial.params,
        "best_score": study.best_trial.value,
        "best_trial_number": study.best_trial.number,
        "n_trials": len(study.trials)
    }
    
    output_dir = Path("optuna_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Save study
    import joblib
    joblib.dump(study, output_dir / "study.pkl")
    
    # Plot optimization history
    try:
        import optuna.visualization as vis
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html(str(output_dir / "optimization_history.html"))
        
        fig2 = vis.plot_param_importances(study)
        fig2.write_html(str(output_dir / "param_importances.html"))
        
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_html(str(output_dir / "parallel_coordinate.html"))
        
        print(f"\nResults saved to {output_dir}/")
        print("- best_params.json: Best hyperparameters")
        print("- study.pkl: Full study object")
        print("- *.html: Visualization plots")
    except ImportError:
        print("\nInstall plotly for visualizations: pip install plotly")
    
    return study


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize GAN Hyperparameters")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per trial")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    
    args = parser.parse_args()
    
    study = optimize_hyperparameters(
        n_trials=args.n_trials,
        dataset=args.dataset,
        epochs=args.epochs
    )