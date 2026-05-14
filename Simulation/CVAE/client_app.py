"""Flower client app for the CVAE version of FLEG."""

from __future__ import annotations

import math
import pickle
import time
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from Simulation.CVAE.task import (
    CVAE,
    DictTensorDataset,
    EmbeddingPairDataset,
    augment_client_with_generated,
    create_classifier,
    create_feature_extractor,
    create_full_model,
    get_image_key,
    get_label_counts,
    get_target_classifier_level,
    get_weights,
    infer_feature_dim,
    load_data,
    local_test,
    normalize_dataset_name,
    set_weights,
    state_dict_from_bytes,
    unpack_batch,
)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        cid: int,
        dataset: str,
        batch_size: int,
        trainloader: Union[DataLoader, list],
        testloader_local: DataLoader,
        context: Context,
        folder: str,
        seed: int = 42,
        lr_alvo: float = 0.01,
        cvae_lr: float = 0.001,
        local_epochs_alvo: int = 1,
        cvae_local_epochs: int = 1,
        continue_epoch: int = 0,
    ):
        self.cid = cid
        self.dataset = normalize_dataset_name(dataset)
        self.batch_size = batch_size
        self.trainloader = trainloader
        self.testloader_local = testloader_local
        self.client_state = context.state
        self.folder = folder
        self.seed = seed
        self.lr_alvo = lr_alvo
        self.cvae_lr = cvae_lr
        self.local_epochs_alvo = local_epochs_alvo
        self.cvae_local_epochs = cvae_local_epochs
        self.continue_epoch = continue_epoch
        self.num_classes = 10
        self.image_key = get_image_key(self.dataset)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _full_trainloader(self) -> DataLoader:
        if isinstance(self.trainloader, list):
            return DataLoader(
                ConcatDataset([dl.dataset for dl in self.trainloader]),
                batch_size=self.batch_size,
                shuffle=True,
            )
        return self.trainloader

    def _feature_extractor(self, level: int, global_state_bytes: bytes):
        feature_extractor = create_feature_extractor(
            self.dataset, level=level, seed=self.seed
        ).to(self.device)
        global_state = state_dict_from_bytes(global_state_bytes, self.device)
        feature_extractor.load_state_dict(global_state, strict=False)
        return feature_extractor

    def _extract_embeddings(self, feature_extractor, trainloader):
        all_embeddings = []
        all_labels = []
        feature_extractor.eval()

        with torch.no_grad():
            for batch in trainloader:
                images, labels = unpack_batch(batch, image_key=self.image_key)
                embeddings = feature_extractor(images.to(self.device))
                embeddings = embeddings.view(embeddings.size(0), -1)
                all_embeddings.append(embeddings)
                all_labels.append(labels.to(self.device))

        final_embeddings = torch.cat(all_embeddings, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        return final_embeddings, final_labels

    def _normalized_embedding_loader(self, feature_extractor, trainloader, normalization):
        embeddings, labels = self._extract_embeddings(feature_extractor, trainloader)

        if normalization == "minmax":
            min_ = embeddings.min(dim=0).values
            max_ = embeddings.max(dim=0).values
            scale = torch.clamp(max_ - min_, min=1e-8)
            normalized = (embeddings - min_) / scale
            norm_stats = {"min": min_, "scale": scale}
        elif normalization == "z":
            mean = embeddings.mean(dim=0)
            std = torch.clamp(embeddings.std(dim=0), min=1e-8)
            normalized = (embeddings - mean) / std
            norm_stats = {"mean": mean, "std": std}
        else:
            raise ValueError(f"normalization should be 'minmax' or 'z', got {normalization}")

        loader = DataLoader(
            TensorDataset(normalized, labels),
            batch_size=self.batch_size,
            shuffle=True,
        )
        return loader, norm_stats

    def _train_classifier(self, classifier, trainloader, level, strategy, mu, mixup_type):
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.lr_alvo)
        global_ref = create_classifier(self.dataset, level=level, seed=self.seed).to(self.device)
        global_ref.load_state_dict(classifier.state_dict())
        for param in global_ref.parameters():
            param.requires_grad = False

        classifier.train()
        running_loss = 0.0
        batches_seen = 0

        for _ in range(self.local_epochs_alvo):
            for batch in trainloader:
                images, labels = unpack_batch(batch, image_key=self.image_key)
                images = images.to(self.device)
                labels = labels.to(self.device)
                if images.size(0) == 1:
                    continue

                optimizer.zero_grad()
                apply_mixup = mixup_type != "none" and level > 0 and np.random.rand() < 0.5

                if apply_mixup:
                    lam = np.random.beta(0.2, 0.2)
                    if mixup_type == "intraclass":
                        rand_index = torch.arange(images.size(0), device=self.device)
                        for label in torch.unique(labels):
                            idx = (labels == label).nonzero(as_tuple=True)[0]
                            if len(idx) > 1:
                                rand_index[idx] = idx[torch.randperm(len(idx))]
                    elif mixup_type == "manifold":
                        rand_index = torch.randperm(images.size(0), device=self.device)
                    else:
                        raise ValueError(f"mixup_type invalido: {mixup_type}")

                    mixed_images = lam * images + (1 - lam) * images[rand_index]
                    outputs = classifier(mixed_images)
                    loss_clf = lam * criterion(outputs, labels) + (1 - lam) * criterion(
                        outputs, labels[rand_index]
                    )
                else:
                    outputs = classifier(images)
                    loss_clf = criterion(outputs, labels)

                if strategy == "fedprox":
                    proximal_term = torch.tensor(0.0, device=self.device)
                    for local_weights, global_weights in zip(
                        classifier.parameters(), global_ref.parameters()
                    ):
                        proximal_term += torch.square((local_weights - global_weights).norm(2))
                    loss = loss_clf + (mu / 2) * proximal_term
                else:
                    loss = loss_clf

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batches_seen += 1

        return running_loss / max(batches_seen, 1)

    def _build_augmented_loader(self, trainloader, feature_extractor, generated_payload):
        embeddings, labels = self._extract_embeddings(feature_extractor, trainloader)
        embedding_dataset = EmbeddingPairDataset(
            embeddings=embeddings,
            labels=labels,
            asset_col_name=self.image_key,
            label_col_name="label",
        )

        if not generated_payload:
            return DataLoader(embedding_dataset, batch_size=self.batch_size, shuffle=True)

        received = pickle.loads(generated_payload)
        if len(received["assets"]) == 0:
            return DataLoader(embedding_dataset, batch_size=self.batch_size, shuffle=True)

        gen_assets = torch.from_numpy(received["assets"]).float().to(self.device)
        gen_labels = torch.from_numpy(received["labels"]).long().to(self.device)
        gen_dataset = DictTensorDataset(
            assets=gen_assets,
            labels=gen_labels,
            asset_col_name=self.image_key,
            label_col_name="label",
        )

        feature_shape = tuple(gen_assets.shape[1:])
        embeddings = embeddings.view(-1, *feature_shape)
        embedding_dataset = EmbeddingPairDataset(
            embeddings=embeddings,
            labels=labels,
            asset_col_name=gen_dataset.asset_col_name,
            label_col_name=gen_dataset.label_col_name,
        )
        counts = get_label_counts(trainloader.dataset)
        fill_to = max(1, int(len(embedding_dataset) / 10))
        combined_ds, stats = augment_client_with_generated(
            client_train=embedding_dataset,
            gen_dataset=gen_dataset,
            counts=counts,
            strategy="threshold",
            fill_to=fill_to,
            threshold=fill_to,
            rng_seed=self.seed,
        )
        print(
            f"Cliente {self.cid}: adicionou {stats['gen_selected_count']} "
            f"amostras CVAE para as classes {stats['desired_labels']}"
        )
        return DataLoader(combined_ds, batch_size=self.batch_size, shuffle=True)

    def _fit_classifier(self, parameters, config):
        level = int(config["classifier_level"])
        trainloader = self._full_trainloader()
        classifier = create_classifier(self.dataset, level=level, seed=self.seed).to(self.device)
        set_weights(classifier, parameters)

        if level > 0:
            feature_extractor = self._feature_extractor(level, config["global_net_state"])
            trainloader = self._build_augmented_loader(
                trainloader=trainloader,
                feature_extractor=feature_extractor,
                generated_payload=config.get("generated", b""),
            )

        train_start = time.time()
        train_loss = self._train_classifier(
            classifier=classifier,
            trainloader=trainloader,
            level=level,
            strategy=config["strategy"],
            mu=float(config["mu"]),
            mixup_type=config["mixup_type"],
        )
        train_time = time.time() - train_start
        return (
            get_weights(classifier),
            len(self._full_trainloader().dataset),
            {
                "cid": self.cid,
                "train_loss": train_loss,
                "tempo_treino_alvo": train_time,
            },
        )

    def _fit_cvae(self, parameters, config, generate_only=False):
        level = int(config["cvae_level"])
        trainloader = self._full_trainloader()
        feature_extractor = self._feature_extractor(level, config["global_net_state"])
        embedding_loader, norm_stats = self._normalized_embedding_loader(
            feature_extractor=feature_extractor,
            trainloader=trainloader,
            normalization=config["normalization"],
        )

        input_dim = int(config["input_dim"])
        latent_dim = int(config["latent_dim"])
        hidden_dim = int(config["hidden_dim"])
        minmax = config["normalization"] == "minmax"
        cvae = CVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            condition_dim=self.num_classes,
            device=self.device,
            beta=float(config["beta"]),
            resblock=bool(config["resblock"]),
            minmax=minmax,
        ).to(self.device)
        set_weights(cvae.decoder, parameters)

        cvae_time = 0.0
        avg_loss = 0.0
        if not generate_only:
            optimizer = torch.optim.Adam(cvae.parameters(), lr=self.cvae_lr)
            cvae.train()
            start = time.time()
            for _ in range(self.cvae_local_epochs):
                local_loss = 0.0
                batches = 0
                for embeddings, labels in embedding_loader:
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)
                    labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
                    optimizer.zero_grad()
                    recon_x, z_mu, z_logvar = cvae(embeddings, labels_one_hot)
                    loss = cvae.loss_function(recon_x, embeddings, z_mu, z_logvar)
                    loss.backward()
                    optimizer.step()
                    local_loss += loss.item()
                    batches += 1
                avg_loss = local_loss / max(batches, 1)
            cvae_time = time.time() - start

        metrics = {
            "cid": self.cid,
            "cvae_loss": float(avg_loss),
            "cvae_time": float(cvae_time),
        }

        if generate_only:
            metrics["generated"] = self._generate_embeddings(
                cvae=cvae,
                norm_stats=norm_stats,
                feature_shape=pickle.loads(config["feature_shape"]),
                normalization=config["normalization"],
                latent_dim=latent_dim,
                num_samples=int(config["num_samples"]),
            )

        return get_weights(cvae.decoder), len(trainloader.dataset), metrics

    def _generate_embeddings(
        self,
        cvae,
        norm_stats,
        feature_shape,
        normalization,
        latent_dim,
        num_samples,
    ) -> bytes:
        start = time.time()
        random_labels = torch.randint(0, self.num_classes, (num_samples,), device=self.device)
        samples_per_class = torch.bincount(random_labels, minlength=self.num_classes)

        all_labels = []
        all_latents = []
        cvae.eval()
        with torch.no_grad():
            for label in range(self.num_classes):
                n_samples = samples_per_class[label].item()
                if n_samples <= 0:
                    continue
                all_labels.append(
                    torch.full((n_samples,), label, dtype=torch.long, device=self.device)
                )
                all_latents.append(torch.randn((n_samples, latent_dim), device=self.device))

            if not all_labels:
                payload = {"assets": np.array([]), "labels": np.array([]), "time": time.time() - start}
                return pickle.dumps(payload)

            labels = torch.cat(all_labels, dim=0)
            latents = torch.cat(all_latents, dim=0)
            one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            generated = cvae.decode(latents, one_hot)

            if normalization == "minmax":
                generated = generated * norm_stats["scale"] + norm_stats["min"]
            else:
                generated = generated * norm_stats["std"] + norm_stats["mean"]

            generated = generated.view(-1, *feature_shape)
            payload = {
                "assets": generated.detach().cpu().numpy(),
                "labels": labels.detach().cpu().numpy(),
                "time": time.time() - start,
            }
            return pickle.dumps(payload)

    def fit(self, parameters, config):
        if config["model"] == "classifier":
            return self._fit_classifier(parameters, config)
        if config["model"] == "cvae":
            return self._fit_cvae(parameters, config, generate_only=False)
        if config["model"] == "cvae_generate":
            return self._fit_cvae(parameters, config, generate_only=True)
        raise ValueError(f"Modelo desconhecido em config['model']: {config['model']}")

    def evaluate(self, parameters, config):
        level = int(config["classifier_level"])
        if level == 0:
            net = create_full_model(self.dataset, seed=self.seed).to(self.device)
            set_weights(net, parameters)
        else:
            net = create_full_model(self.dataset, seed=self.seed).to(self.device)
            net.load_state_dict(state_dict_from_bytes(config["global_net_state"], self.device))
            classifier = create_classifier(self.dataset, level=level, seed=self.seed).to(self.device)
            set_weights(classifier, parameters)
            net.load_state_dict(classifier.state_dict(), strict=False)

        local_test_start = time.time()
        local_acc = local_test(
            net=net,
            testloader=self.testloader_local,
            device=self.device,
            acc_filepath=f"{self.folder}/accuracy_report.txt",
            epoch=int(config["round"]),
            cliente=self.cid,
            continue_epoch=self.continue_epoch,
            dataset=self.dataset,
        )
        local_test_time = time.time() - local_test_start
        return (
            0.0,
            len(self.testloader_local.dataset),
            {"local_test_time": local_test_time, "local_accuracy": local_acc},
        )


def _alpha_from_partitioner(partitioner: str):
    if partitioner in {"Dir01", "Dirichlet01"}:
        return 0.1
    if partitioner in {"Dir05", "Dirichlet05"}:
        return 0.5
    return None


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    run_config = context.run_config

    dataset = normalize_dataset_name(run_config["dataset"])
    partitioner = run_config["partitioner"]
    batch_size = run_config["tam_batch"]
    seed = run_config["seed"]
    cvae_epochs = run_config.get("cvae_epochs", run_config.get("epocas_gen", 25))
    syn_input = run_config.get("num_syn", run_config.get("syn_input", "dynamic"))
    trial = run_config.get("trial", seed)
    alpha = run_config.get("alpha", _alpha_from_partitioner(partitioner))
    partitioner_for_data = "Dir01" if partitioner == "Dirichlet" else partitioner

    folder = (
        f"{run_config['Exp_name_folder']}CVAE/"
        f"{dataset}_{partitioner}_{run_config['strategy']}_"
        f"cvaeepochs{cvae_epochs}_{syn_input}_fleg_trial{trial}"
    )

    trainloader, _, testloader_local = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        dataset=dataset,
        teste=run_config.get("teste", run_config.get("test_mode", False)),
        partitioner_type=partitioner_for_data,
        num_chunks=run_config.get("num_chunks", 1),
        alpha_dir=alpha,
        seed=seed,
    )

    return FlowerClient(
        cid=partition_id,
        dataset=dataset,
        batch_size=batch_size,
        trainloader=trainloader,
        testloader_local=testloader_local,
        context=context,
        folder=folder,
        seed=seed,
        lr_alvo=run_config.get("learn_rate_alvo", 0.01),
        cvae_lr=run_config.get("cvae_lr", 0.001),
        local_epochs_alvo=run_config.get("epocas_alvo", 1),
        cvae_local_epochs=run_config.get("cvae_local_epochs", 1),
        continue_epoch=run_config.get("continue_epoch", 0),
    ).to_client()


app = ClientApp(client_fn=client_fn)
