"""Flower client app for the CVAE version of FLEG."""

from __future__ import annotations

import gc
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, array_from_numpy
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from Simulation.CVAE.task import (
    CVAE,
    Decoder,
    DictTensorDataset,
    EmbeddingPairDataset,
    choose_minority_labels,
    create_classifier,
    create_feature_extractor,
    create_full_model,
    get_image_key,
    get_label_counts,
    get_num_classes,
    get_weights,
    load_data,
    log_memory_event,
    local_test,
    normalize_dataset_name,
    object_tensor_size_mb,
    set_weights,
    state_dict_from_bytes,
    tensor_size_mb,
    uses_client_validation_criterion,
    unpack_batch,
)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        cid: int,
        dataset: str,
        batch_size: int,
        trainloader: DataLoader,
        valloader_local: DataLoader | None,
        testloader_local: DataLoader,
        context: Context,
        folder: str,
        seed: int = 42,
        lr_alvo: float = 0.01,
        cvae_lr: float = 0.001,
        local_epochs_alvo: int = 1,
        cvae_local_epochs: int = 1,
        continue_epoch: int = 0,
        medmnist_size: int = 28,
        memory_logging: bool = False,
        run_id: int = 0,
    ):
        self.cid = cid
        self.run_id = int(run_id)
        self.dataset = normalize_dataset_name(dataset)
        self.batch_size = batch_size
        self.trainloader = trainloader
        self.valloader_local = valloader_local
        self.testloader_local = testloader_local
        self.client_state = context.state
        self.folder = folder
        self.client_cache_dir = (
            Path(self.folder)
            / "client_disk_cache"
            / f"run_{self.run_id}"
            / f"client_{self.cid}"
        )
        self.client_cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.lr_alvo = lr_alvo
        self.cvae_lr = cvae_lr
        self.local_epochs_alvo = local_epochs_alvo
        self.cvae_local_epochs = cvae_local_epochs
        self.continue_epoch = continue_epoch
        self.medmnist_size = int(medmnist_size)
        self.memory_logging = memory_logging
        self.memory_log_path = (
            f"{self.folder}/memory_client_{self.cid}.jsonl"
            if self.memory_logging
            else None
        )
        self.num_classes = get_num_classes(self.dataset)
        self.image_key = get_image_key(self.dataset)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._log_memory(
            "client_init",
            dataset=self.dataset,
            medmnist_size=self.medmnist_size,
            batch_size=self.batch_size,
            train_examples=len(self.trainloader.dataset),
            local_test_examples=len(self.testloader_local.dataset),
            has_validation_loader=self.valloader_local is not None,
            run_id=self.run_id,
            disk_cache_dir=str(self.client_cache_dir),
        )

    def _log_memory(self, event: str, **fields) -> None:
        payload = {"cid": self.cid, "dataset": self.dataset}
        payload.update(fields)
        log_memory_event(
            self.memory_log_path,
            event,
            **payload,
        )

    def _record_bytes(self, key: str) -> bytes | None:
        if key not in self.client_state.parameters_records:
            return None
        record = self.client_state.parameters_records[key]
        return record["state_bytes"].numpy().tobytes()

    def _record_level(self, key: str) -> int | None:
        if key not in self.client_state.parameters_records:
            return None
        record = self.client_state.parameters_records[key]
        if "level" not in record:
            return None
        return int(record["level"].numpy()[0])

    def _save_bytes_record(self, key: str, payload: bytes, level: int) -> None:
        record = ParametersRecord()
        record["state_bytes"] = array_from_numpy(
            np.frombuffer(payload, dtype=np.uint8)
        )
        record["level"] = array_from_numpy(np.array([level], dtype=np.int64))
        self.client_state.parameters_records[key] = record

    def _save_feature_extractor_state(self, level: int, payload: bytes) -> None:
        if payload:
            self._save_bytes_record("feature_extractor_state", payload, level)

    def _cached_feature_extractor_state(self, level: int) -> bytes | None:
        if self._record_level("feature_extractor_state") != level:
            return None
        return self._record_bytes("feature_extractor_state")

    def _generated_embeddings_path(self, level: int) -> Path:
        return self.client_cache_dir / f"generated_embeddings_level{level}.npz"

    def _cvae_training_state_path(self, level: int) -> Path:
        return self.client_cache_dir / f"cvae_training_state_level{level}.pth"

    def _classifier_optimizer_state_path(
        self,
        level: int,
        optimizer_name: str,
    ) -> Path:
        return (
            self.client_cache_dir
            / f"classifier_optimizer_{optimizer_name}_level{level}.pth"
        )

    def _feature_extractor(self, level: int, config):
        feature_extractor = create_feature_extractor(
            self.dataset,
            level=level,
            seed=self.seed,
            medmnist_size=self.medmnist_size,
        ).to(self.device)
        self._save_feature_extractor_state(
            level, config.get("feature_extractor_state", b"")
        )
        feature_state_bytes = self._cached_feature_extractor_state(level)
        if feature_state_bytes is None:
            raise RuntimeError(
                f"Client {self.cid} does not have a cached feature extractor "
                f"for level {level}."
            )
        feature_state = state_dict_from_bytes(feature_state_bytes, self.device)
        feature_extractor.load_state_dict(feature_state, strict=False)
        return feature_extractor

    def _decoder(self, level: int, config):
        decoder_state_bytes = config.get("decoder_state", b"")
        if not decoder_state_bytes:
            raise RuntimeError(
                f"Client {self.cid} did not receive a decoder to generate "
                f"embeddings at level {level}."
            )

        decoder = Decoder(
            input_dim=int(config["input_dim"]),
            latent_dim=int(config["latent_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            condition_dim=self.num_classes,
            resblock=bool(config["resblock"]),
            minmax=config["normalization"] == "minmax",
            depth=int(config.get("cvae_depth", 2)),
        ).to(self.device)
        decoder.load_state_dict(state_dict_from_bytes(decoder_state_bytes, self.device))
        decoder.eval()
        return decoder

    def _save_generated_embeddings(
        self,
        level: int,
        generated_data: dict,
    ) -> None:
        path = self._generated_embeddings_path(level)
        np.savez(
            path,
            assets=generated_data["assets"],
            labels=generated_data["labels"],
            level=np.array([level], dtype=np.int64),
        )

    def _cached_generated_embeddings(self, level: int):
        path = self._generated_embeddings_path(level)
        if not path.exists():
            return None
        with np.load(path) as data:
            cached_level = int(data["level"][0]) if "level" in data.files else level
            if cached_level != level:
                return None
            assets = data["assets"].copy()
            labels = data["labels"].copy()
        return {
            "assets": assets,
            "labels": labels,
        }

    def _cvae_state_matches(self, state: dict, config: dict) -> bool:
        expected = {
            "level": int(config["level"]),
            "input_dim": int(config["input_dim"]),
            "latent_dim": int(config["latent_dim"]),
            "hidden_dim": int(config["hidden_dim"]),
            "normalization": config["normalization"],
            "resblock": bool(config["resblock"]),
            "cvae_depth": int(config.get("cvae_depth", 2)),
        }
        return all(state.get(key) == value for key, value in expected.items())

    def _load_cvae_training_state(self, cvae, optimizer, config: dict) -> None:
        level = int(config["level"])
        path = self._cvae_training_state_path(level)
        if not path.exists():
            return

        state = torch.load(path, map_location=self.device)
        if not self._cvae_state_matches(state, config):
            return

        cvae.encoder.load_state_dict(state["encoder_state_dict"])
        cvae.fc_mu.load_state_dict(state["fc_mu_state_dict"])
        cvae.fc_logvar.load_state_dict(state["fc_logvar_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        self._log_memory(
            "cvae_state_loaded_from_disk",
            level=level,
            state_file_mb=path.stat().st_size / 10**6,
        )

    def _save_cvae_training_state(self, cvae, optimizer, config: dict) -> None:
        state = {
            "level": int(config["level"]),
            "input_dim": int(config["input_dim"]),
            "latent_dim": int(config["latent_dim"]),
            "hidden_dim": int(config["hidden_dim"]),
            "normalization": config["normalization"],
            "resblock": bool(config["resblock"]),
            "cvae_depth": int(config.get("cvae_depth", 2)),
            "encoder_state_dict": cvae.encoder.state_dict(),
            "fc_mu_state_dict": cvae.fc_mu.state_dict(),
            "fc_logvar_state_dict": cvae.fc_logvar.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        self._log_memory(
            "cvae_state_before_disk_save",
            level=int(config["level"]),
            cvae_state_tensor_mb=object_tensor_size_mb(state),
        )
        path = self._cvae_training_state_path(int(config["level"]))
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp_path)
        tmp_path.replace(path)
        self._log_memory(
            "cvae_state_saved_to_disk",
            level=int(config["level"]),
            state_file_mb=path.stat().st_size / 10**6,
        )

    def _classifier_optimizer_state_matches(
        self,
        state: dict,
        level: int,
        optimizer_name: str,
    ) -> bool:
        expected = {
            "level": int(level),
            "optimizer": str(optimizer_name).lower(),
            "lr_alvo": float(self.lr_alvo),
        }
        return all(state.get(key) == value for key, value in expected.items())

    def _load_classifier_optimizer_state(
        self,
        optimizer,
        level: int,
        optimizer_name: str,
    ) -> None:
        optimizer_name = str(optimizer_name).lower()
        if optimizer_name != "adam":
            return

        path = self._classifier_optimizer_state_path(level, optimizer_name)
        if not path.exists():
            return

        state = torch.load(path, map_location=self.device)
        if not self._classifier_optimizer_state_matches(state, level, optimizer_name):
            return

        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        except ValueError as exc:
            self._log_memory(
                "classifier_optimizer_state_load_skipped",
                level=level,
                classifier_optimizer=optimizer_name,
                reason=str(exc),
            )
            return

        self._log_memory(
            "classifier_optimizer_state_loaded_from_disk",
            level=level,
            classifier_optimizer=optimizer_name,
            optimizer_state_mb=object_tensor_size_mb(optimizer.state_dict()),
            state_file_mb=path.stat().st_size / 10**6,
        )

    def _save_classifier_optimizer_state(
        self,
        optimizer,
        level: int,
        optimizer_name: str,
    ) -> None:
        optimizer_name = str(optimizer_name).lower()
        if optimizer_name != "adam":
            return

        state = {
            "level": int(level),
            "optimizer": optimizer_name,
            "lr_alvo": float(self.lr_alvo),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        self._log_memory(
            "classifier_optimizer_state_before_disk_save",
            level=level,
            classifier_optimizer=optimizer_name,
            optimizer_state_mb=object_tensor_size_mb(state),
        )
        path = self._classifier_optimizer_state_path(level, optimizer_name)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp_path)
        tmp_path.replace(path)
        self._log_memory(
            "classifier_optimizer_state_saved_to_disk",
            level=level,
            classifier_optimizer=optimizer_name,
            state_file_mb=path.stat().st_size / 10**6,
        )

    def _extract_embeddings(self, feature_extractor, trainloader):
        all_embeddings = []
        all_labels = []
        feature_extractor.eval()
        self._log_memory(
            "extract_embeddings_start",
            train_examples=len(trainloader.dataset),
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(trainloader):
                images, labels = unpack_batch(batch, image_key=self.image_key)
                embeddings = feature_extractor(images.to(self.device))
                embeddings = embeddings.view(embeddings.size(0), -1)
                all_embeddings.append(embeddings)
                all_labels.append(labels.to(self.device))
                if batch_idx == 0 or (batch_idx + 1) % 25 == 0:
                    self._log_memory(
                        "extract_embeddings_batch",
                        batch=batch_idx + 1,
                        batches_cached=len(all_embeddings),
                        cached_embeddings_mb=sum(
                            tensor_size_mb(item) for item in all_embeddings
                        ),
                        cached_labels_mb=sum(
                            tensor_size_mb(item) for item in all_labels
                        ),
                        current_batch_embeddings_mb=tensor_size_mb(embeddings),
                    )

        self._log_memory(
            "extract_embeddings_before_cat",
            batches_cached=len(all_embeddings),
            cached_embeddings_mb=sum(tensor_size_mb(item) for item in all_embeddings),
            cached_labels_mb=sum(tensor_size_mb(item) for item in all_labels),
        )
        final_embeddings = torch.cat(all_embeddings, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        self._log_memory(
            "extract_embeddings_after_cat",
            embeddings_shape=tuple(final_embeddings.shape),
            labels_shape=tuple(final_labels.shape),
            final_embeddings_mb=tensor_size_mb(final_embeddings),
            final_labels_mb=tensor_size_mb(final_labels),
        )
        return final_embeddings, final_labels

    def _normalized_embedding_loader(self, feature_extractor, trainloader, normalization):
        embeddings, labels = self._extract_embeddings(feature_extractor, trainloader)
        self._log_memory(
            "normalize_embeddings_start",
            normalization=normalization,
            embeddings_mb=tensor_size_mb(embeddings),
            labels_mb=tensor_size_mb(labels),
        )

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

        self._log_memory(
            "normalize_embeddings_done",
            normalized_mb=tensor_size_mb(normalized),
            labels_mb=tensor_size_mb(labels),
            norm_stats_mb=object_tensor_size_mb(norm_stats),
        )
        loader = DataLoader(
            TensorDataset(normalized, labels),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self._log_memory(
            "embedding_loader_created",
            normalized_mb=tensor_size_mb(normalized),
            labels_mb=tensor_size_mb(labels),
        )
        return loader, norm_stats

    def _embedding_norm_stats(self, embeddings, normalization):
        if normalization == "minmax":
            min_ = embeddings.min(dim=0).values
            max_ = embeddings.max(dim=0).values
            scale = torch.clamp(max_ - min_, min=1e-8)
            return {"min": min_, "scale": scale}
        if normalization == "z":
            mean = embeddings.mean(dim=0)
            std = torch.clamp(embeddings.std(dim=0), min=1e-8)
            return {"mean": mean, "std": std}
        raise ValueError(f"normalization should be 'minmax' or 'z', got {normalization}")

    def _local_generation_plan(
        self,
        trainloader,
        num_real_embeddings: int,
        num_samples: int | None,
    ):
        counts = get_label_counts(trainloader.dataset)
        fill_to = max(1, int(num_real_embeddings / self.num_classes))
        desired_labels = choose_minority_labels(
            counts,
            total_num_classes=self.num_classes,
            method="threshold",
            threshold=fill_to,
        )
        need_per_label = {
            label: fill_to - counts.get(label, 0)
            for label in desired_labels
            if fill_to - counts.get(label, 0) > 0
        }
        requested_total = sum(need_per_label.values())
        if num_samples is not None:
            requested_total = max(0, min(int(num_samples), requested_total))
            total_needed = sum(need_per_label.values())
            if requested_total == 0:
                need_per_label = {}
            elif requested_total < total_needed:
                scaled_need = {}
                allocated = 0
                fractions = []
                for label, need in need_per_label.items():
                    exact = requested_total * need / total_needed
                    base = int(exact)
                    scaled_need[label] = base
                    allocated += base
                    fractions.append((exact - base, label))
                remaining = requested_total - allocated
                for _, label in sorted(fractions, reverse=True):
                    if remaining <= 0:
                        break
                    if scaled_need[label] < need_per_label[label]:
                        scaled_need[label] += 1
                        remaining -= 1
                need_per_label = {
                    label: need for label, need in scaled_need.items() if need > 0
                }
            desired_labels = [
                label for label in desired_labels if need_per_label.get(label, 0) > 0
            ]
        return counts, fill_to, desired_labels, need_per_label

    def _generate_embeddings_with_decoder(
        self,
        decoder,
        embeddings,
        trainloader,
        config,
    ):
        feature_shape = tuple(pickle.loads(config["feature_shape"]))
        num_samples = int(config["num_samples"]) if "num_samples" in config else None
        counts, fill_to, desired_labels, need_per_label = self._local_generation_plan(
            trainloader=trainloader,
            num_real_embeddings=embeddings.size(0),
            num_samples=num_samples,
        )

        total_needed = sum(need_per_label.values())
        if total_needed == 0:
            return {
                "assets": np.empty((0, *feature_shape), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int64),
                "stats": {
                    "client_counts": counts,
                    "desired_labels": desired_labels,
                    "need_per_label": need_per_label,
                    "fill_to": fill_to,
                    "requested_num_samples": num_samples,
                    "gen_selected_count": 0,
                },
            }

        labels = torch.cat(
            [
                torch.full((need,), label, dtype=torch.long, device=self.device)
                for label, need in need_per_label.items()
            ]
        )
        labels = labels[torch.randperm(labels.numel(), device=self.device)]
        latents = torch.randn(
            (labels.numel(), int(config["latent_dim"])),
            device=self.device,
        )
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        norm_stats = self._embedding_norm_stats(embeddings, config["normalization"])

        with torch.no_grad():
            generated = decoder.decode(latents, one_hot)
            if config["normalization"] == "minmax":
                generated = generated * norm_stats["scale"] + norm_stats["min"]
            else:
                generated = generated * norm_stats["std"] + norm_stats["mean"]
            generated = generated.view(-1, *feature_shape)

        return {
            "assets": generated.detach().cpu().numpy().astype(np.float32, copy=False),
            "labels": labels.detach().cpu().numpy().astype(np.int64, copy=False),
            "stats": {
                "client_counts": counts,
                "desired_labels": desired_labels,
                "need_per_label": need_per_label,
                "fill_to": fill_to,
                "requested_num_samples": num_samples,
                "gen_selected_count": int(labels.numel()),
            },
        }

    def _generated_embeddings(self, level: int, embeddings, trainloader, config):
        cached = self._cached_generated_embeddings(level)
        if cached is not None:
            self._log_memory(
                "generated_embeddings_cache_hit",
                level=level,
                cached_assets_mb=tensor_size_mb(cached["assets"]),
                cached_labels_mb=tensor_size_mb(cached["labels"]),
            )
            return cached, 0.0, None

        start = time.time()
        self._log_memory(
            "generated_embeddings_start",
            level=level,
            real_embeddings_mb=tensor_size_mb(embeddings),
        )
        decoder = self._decoder(level, config)
        self._log_memory(
            "generated_embeddings_decoder_loaded",
            level=level,
            decoder_state_mb=object_tensor_size_mb(decoder.state_dict()),
        )
        generated_data = self._generate_embeddings_with_decoder(
            decoder=decoder,
            embeddings=embeddings,
            trainloader=trainloader,
            config=config,
        )
        generation_time = time.time() - start
        self._log_memory(
            "generated_embeddings_created",
            level=level,
            generated_assets_mb=tensor_size_mb(generated_data["assets"]),
            generated_labels_mb=tensor_size_mb(generated_data["labels"]),
            generation_time=float(generation_time),
            gen_selected_count=generated_data["stats"].get("gen_selected_count", 0),
        )
        self._save_generated_embeddings(level, generated_data)
        embeddings_path = self._generated_embeddings_path(level)
        self._log_memory(
            "generated_embeddings_saved_to_disk",
            level=level,
            generated_assets_mb=tensor_size_mb(generated_data["assets"]),
            generated_labels_mb=tensor_size_mb(generated_data["labels"]),
            embeddings_file_mb=embeddings_path.stat().st_size / 10**6,
        )
        return generated_data, generation_time, generated_data["stats"]

    def _classifier_optimizer(self, classifier, optimizer_name: str):
        optimizer_name = str(optimizer_name).lower()
        if optimizer_name == "sgd":
            return torch.optim.SGD(classifier.parameters(), lr=self.lr_alvo)
        if optimizer_name == "adam":
            return torch.optim.Adam(classifier.parameters(), lr=self.lr_alvo)
        raise ValueError(
            "classifier_optimizer must be 'sgd' or 'adam', "
            f"got {optimizer_name}"
        )

    def _train_classifier(
        self,
        classifier,
        trainloader,
        level,
        strategy,
        mu,
        mixup_type,
        optimizer_name,
    ):
        optimizer_name = str(optimizer_name).lower()
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = self._classifier_optimizer(classifier, optimizer_name)
        self._load_classifier_optimizer_state(optimizer, level, optimizer_name)
        global_ref = create_classifier(
            self.dataset,
            level=level,
            seed=self.seed,
            medmnist_size=self.medmnist_size,
        ).to(self.device)
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

        self._save_classifier_optimizer_state(optimizer, level, optimizer_name)
        return running_loss / max(batches_seen, 1)

    def _build_augmented_loader(self, trainloader, feature_extractor, level, config):
        self._log_memory(
            "build_augmented_loader_start",
            level=level,
            train_examples=len(trainloader.dataset),
        )
        embeddings, labels = self._extract_embeddings(feature_extractor, trainloader)
        feature_shape = tuple(pickle.loads(config["feature_shape"]))
        embeddings = embeddings.view(-1, *feature_shape)
        self._log_memory(
            "build_augmented_loader_real_embeddings_ready",
            level=level,
            embeddings_shape=tuple(embeddings.shape),
            embeddings_mb=tensor_size_mb(embeddings),
            labels_mb=tensor_size_mb(labels),
        )
        embedding_dataset = EmbeddingPairDataset(
            embeddings=embeddings,
            labels=labels,
            asset_col_name=self.image_key,
            label_col_name="label",
        )

        generated_data, generation_time, stats = self._generated_embeddings(
            level=level,
            embeddings=embeddings.view(embeddings.size(0), -1),
            trainloader=trainloader,
            config=config,
        )
        if not generated_data:
            self._log_memory(
                "build_augmented_loader_no_generated_data",
                level=level,
                embedding_dataset_examples=len(embedding_dataset),
            )
            return (
                DataLoader(embedding_dataset, batch_size=self.batch_size, shuffle=True),
                generation_time,
            )

        if len(generated_data["assets"]) == 0:
            self._log_memory(
                "build_augmented_loader_empty_generated_data",
                level=level,
                embedding_dataset_examples=len(embedding_dataset),
            )
            return (
                DataLoader(embedding_dataset, batch_size=self.batch_size, shuffle=True),
                generation_time,
            )

        gen_assets_np = generated_data["assets"]
        gen_labels_np = generated_data["labels"]
        gen_assets = torch.from_numpy(gen_assets_np).float().to(self.device)
        gen_labels = torch.from_numpy(gen_labels_np).long().to(self.device)
        del generated_data
        del gen_assets_np
        del gen_labels_np
        self._log_memory(
            "build_augmented_loader_generated_tensors_ready",
            level=level,
            gen_assets_mb=tensor_size_mb(gen_assets),
            gen_labels_mb=tensor_size_mb(gen_labels),
        )
        gen_dataset = DictTensorDataset(
            assets=gen_assets,
            labels=gen_labels,
            asset_col_name=self.image_key,
            label_col_name="label",
            )

        combined_ds = ConcatDataset([embedding_dataset, gen_dataset])
        if stats is None:
            print(
                f"Client {self.cid}: using {len(gen_dataset)} "
                "cached synthetic embeddings."
            )
        else:
            print(
                f"Client {self.cid}: added {stats['gen_selected_count']} CVAE samples "
                f"for classes {stats['desired_labels']} "
                f"(requested target: {stats.get('requested_num_samples', 'cached')})."
            )
        self._log_memory(
            "build_augmented_loader_done",
            level=level,
            real_examples=len(embedding_dataset),
            generated_examples=len(gen_dataset),
        )
        return DataLoader(combined_ds, batch_size=self.batch_size, shuffle=True), generation_time

    def _fit_classifier(self, parameters, config):
        level = int(config["level"])
        trainloader = self.trainloader
        self._log_memory(
            "fit_classifier_start",
            level=level,
            train_examples=len(trainloader.dataset),
            classifier_optimizer=config.get("classifier_optimizer", "sgd"),
        )
        classifier = create_classifier(
            self.dataset,
            level=level,
            seed=self.seed,
            medmnist_size=self.medmnist_size,
        ).to(self.device)
        set_weights(classifier, parameters)
        self._log_memory(
            "fit_classifier_model_loaded",
            level=level,
            classifier_state_mb=object_tensor_size_mb(classifier.state_dict()),
        )

        if level > 0:
            feature_extractor = self._feature_extractor(level, config)
            trainloader, img_syn_time = self._build_augmented_loader(
                trainloader=trainloader,
                feature_extractor=feature_extractor,
                level=level,
                config=config,
            )
        else:
            img_syn_time = 0.0

        train_start = time.time()
        train_loss = self._train_classifier(
            classifier=classifier,
            trainloader=trainloader,
            level=level,
            strategy=config["strategy"],
            mu=float(config["mu"]),
            mixup_type=config["mixup_type"],
            optimizer_name=config.get("classifier_optimizer", "sgd"),
        )
        train_time = time.time() - train_start
        classifier_weights = get_weights(classifier)
        self._log_memory(
            "fit_classifier_done",
            level=level,
            train_loss=float(train_loss),
            train_time=float(train_time),
            classifier_weights_mb=object_tensor_size_mb(classifier_weights),
        )
        del classifier
        if level > 0:
            del feature_extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log_memory("fit_classifier_after_cleanup", level=level)
        return (
            classifier_weights,
            len(self.trainloader.dataset),
            {
                "cid": self.cid,
                "train_loss": train_loss,
                "tempo_treino_alvo": train_time,
                "img_syn_time": float(img_syn_time),
            },
        )

    def _fit_cvae(self, parameters, config):
        level = int(config["level"])
        trainloader = self.trainloader
        self._log_memory(
            "fit_cvae_start",
            level=level,
            input_dim=int(config["input_dim"]),
            latent_dim=int(config["latent_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            cvae_depth=int(config.get("cvae_depth", 2)),
            local_epochs=int(config.get("local_epochs", self.cvae_local_epochs)),
            cvae_epoch_start=int(config.get("cvae_epoch_start", 0)),
            cvae_epochs_total=int(config.get("cvae_epochs_total", 0)),
        )
        feature_extractor = self._feature_extractor(level, config)
        self._log_memory(
            "fit_cvae_feature_extractor_loaded",
            level=level,
            feature_extractor_mb=object_tensor_size_mb(feature_extractor.state_dict()),
        )
        embedding_loader, _ = self._normalized_embedding_loader(
            feature_extractor=feature_extractor,
            trainloader=trainloader,
            normalization=config["normalization"],
        )
        embedding_dataset = embedding_loader.dataset
        if isinstance(embedding_dataset, TensorDataset):
            embedding_loader_tensor_mb = sum(
                tensor_size_mb(tensor) for tensor in embedding_dataset.tensors
            )
        else:
            embedding_loader_tensor_mb = 0.0
        self._log_memory(
            "fit_cvae_embedding_loader_ready",
            level=level,
            embedding_loader_tensor_mb=embedding_loader_tensor_mb,
            embedding_loader_examples=len(embedding_dataset),
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
            depth=int(config.get("cvae_depth", 2)),
        ).to(self.device)
        set_weights(cvae.decoder, parameters)
        self._log_memory(
            "fit_cvae_model_created",
            level=level,
            cvae_depth=int(config.get("cvae_depth", 2)),
            cvae_state_mb=object_tensor_size_mb(cvae.state_dict()),
            decoder_state_mb=object_tensor_size_mb(cvae.decoder.state_dict()),
        )

        optimizer = torch.optim.Adam(cvae.parameters(), lr=self.cvae_lr)
        self._log_memory(
            "fit_cvae_optimizer_created",
            level=level,
            optimizer_state_mb=object_tensor_size_mb(optimizer.state_dict()),
        )
        self._load_cvae_training_state(cvae, optimizer, config)
        self._log_memory(
            "fit_cvae_state_loaded",
            level=level,
            cvae_state_mb=object_tensor_size_mb(cvae.state_dict()),
            optimizer_state_mb=object_tensor_size_mb(optimizer.state_dict()),
        )
        cvae.train()
        start = time.time()
        avg_loss = 0.0
        local_epochs = int(config.get("local_epochs", self.cvae_local_epochs))
        epoch_start = int(config.get("cvae_epoch_start", 0))
        epoch_total = int(config.get("cvae_epochs_total", local_epochs))
        for local_epoch in range(local_epochs):
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
            self._log_memory(
                "fit_cvae_epoch_done",
                level=level,
                local_epoch=local_epoch + 1,
                global_cvae_epoch=epoch_start + local_epoch + 1,
                cvae_epochs_total=epoch_total,
                avg_loss=float(avg_loss),
                optimizer_state_mb=object_tensor_size_mb(optimizer.state_dict()),
            )
            
        cvae_time = time.time() - start
        self._save_cvae_training_state(cvae, optimizer, config)
        decoder_weights = get_weights(cvae.decoder)
        self._log_memory(
            "fit_cvae_decoder_weights_ready",
            level=level,
            decoder_weights_mb=object_tensor_size_mb(decoder_weights),
            cvae_time=float(cvae_time),
        )

        metrics = {
            "cid": self.cid,
            "cvae_loss": float(avg_loss),
            "cvae_time": float(cvae_time)
        }

        self._log_memory("fit_cvae_before_cleanup", level=level)
        del cvae
        del optimizer
        del embedding_loader
        del embedding_dataset
        del feature_extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log_memory("fit_cvae_after_cleanup", level=level)

        return decoder_weights, len(trainloader.dataset), metrics

    def fit(self, parameters, config):
        if config["model"] == "classifier":
            return self._fit_classifier(parameters, config)
        if config["model"] == "cvae":
            return self._fit_cvae(parameters, config)
        raise ValueError(f"Unknown model in config['model']: {config['model']}")

    def evaluate(self, parameters, config):
        level = int(config["level"])
        if level == 0:
            net = create_full_model(
                self.dataset,
                seed=self.seed,
                medmnist_size=self.medmnist_size,
            ).to(self.device)
            set_weights(net, parameters)
        else:
            net = create_full_model(
                self.dataset,
                seed=self.seed,
                medmnist_size=self.medmnist_size,
            ).to(self.device)
            self._save_feature_extractor_state(
                level, config.get("feature_extractor_state", b"")
            )
            feature_state_bytes = self._cached_feature_extractor_state(level)
            if feature_state_bytes is None:
                raise RuntimeError(
                    f"Client {self.cid} does not have a cached feature extractor "
                    f"to evaluate level {level}."
                )
            feature_state = state_dict_from_bytes(feature_state_bytes, self.device)
            net.load_state_dict(feature_state, strict=False)
            classifier = create_classifier(
                self.dataset,
                level=level,
                seed=self.seed,
                medmnist_size=self.medmnist_size,
            ).to(self.device)
            set_weights(classifier, parameters)
            net.load_state_dict(classifier.state_dict(), strict=False)

        evaluation_split = str(config.get("evaluation_split", "test"))
        if evaluation_split == "validation":
            if self.valloader_local is None:
                raise RuntimeError(
                    f"Client {self.cid} does not have a local validation loader."
                )
            eval_loader = self.valloader_local
            report_filename = "validation_report.txt"
        elif evaluation_split == "test":
            eval_loader = self.testloader_local
            report_filename = "accuracy_report.txt"
        else:
            raise ValueError(
                f"evaluation_split must be 'test' or 'validation', got {evaluation_split}"
            )

        eval_start = time.time()
        local_acc, local_loss = local_test(
            net=net,
            testloader=eval_loader,
            device=self.device,
            acc_filepath=f"{self.folder}/{report_filename}",
            epoch=int(config["round"]),
            cliente=self.cid,
            num_classes=self.num_classes,
            continue_epoch=self.continue_epoch,
            dataset=self.dataset,
            return_loss=True,
        )
        eval_time = time.time() - eval_start

        local_test_acc = local_acc
        local_test_loss = local_loss
        local_test_time = eval_time
        total_eval_time = eval_time
        if evaluation_split == "validation":
            local_test_start = time.time()
            local_test_acc, local_test_loss = local_test(
                net=net,
                testloader=self.testloader_local,
                device=self.device,
                acc_filepath=f"{self.folder}/accuracy_report.txt",
                epoch=int(config["round"]),
                cliente=self.cid,
                num_classes=self.num_classes,
                continue_epoch=self.continue_epoch,
                dataset=self.dataset,
                return_loss=True,
            )
            local_test_time = time.time() - local_test_start
            total_eval_time += local_test_time

        return (
            float(local_loss),
            len(eval_loader.dataset),
            {
                "local_test_time": float(local_test_time),
                "local_eval_time": float(total_eval_time),
                "local_accuracy": float(local_acc),
                "local_loss": float(local_loss),
                "local_val_accuracy": float(local_acc),
                "local_val_loss": float(local_loss),
                "local_val_num_examples": len(eval_loader.dataset),
                "local_val_time": float(eval_time),
                "local_test_accuracy": float(local_test_acc),
                "local_test_loss": float(local_test_loss),
                "local_test_num_examples": len(self.testloader_local.dataset),
                "evaluation_split": evaluation_split,
            },
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
    cvae_epochs = run_config.get("epocas_gen", 25)
    cvae_local_epochs = run_config.get("epocas_locais_gen", 1)
    syn_input = run_config.get("num_syn", run_config.get("syn_input", "dynamic"))
    medmnist_size = int(run_config.get("medmnist_size", 224))
    cvae_depth = int(run_config.get("cvae_depth", 2))
    levels = int(run_config.get("levels", 4))
    alpha = run_config.get("alpha", _alpha_from_partitioner(partitioner))
    partitioner_for_data = f"Dir{int(alpha * 10):02d}" if partitioner == "Dirichlet" else partitioner
    stop_criterion = run_config.get("criterio_parada", "global_test_acc")
    use_client_validation = uses_client_validation_criterion(stop_criterion)
    baseline = run_config.get("baseline", False)

    if seed == 42:
        trial = 1
    elif seed == 30:
        trial = 2
    elif seed == 20:
        trial = 3
    else:
        trial = seed

    dataset_folder_name = dataset
    if dataset.endswith("mnist") and dataset not in {"mnist"}:
        dataset_folder_name = f"{dataset}_size{medmnist_size}"


    method = "baseline" if baseline else "fleg"
    classifier_optimizer = str(run_config.get("classifier_optimizer", "sgd")).lower()
    folder_parts = [
        dataset_folder_name,
        partitioner_for_data,
        run_config["strategy"],
        f"netoptim{classifier_optimizer}",
    ]
    if not baseline:
        folder_parts.extend(
            [
                f"cvaeepochs{cvae_epochs}",
                f"depth_{cvae_depth}",
                str(syn_input),
            ]
        )
    folder_parts.append(method)
    if not baseline:
        folder_parts.append(f"levels{levels}")
    folder_parts.append(f"trial{trial}")

    folder = (
        f"{run_config['Exp_name_folder']}CVAE/"
        f"{'_'.join(folder_parts)}"
    )


    trainloader, _, valloader_local, testloader_local = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        dataset=dataset,
        teste=run_config.get("teste", run_config.get("test_mode", False)),
        partitioner_type=partitioner_for_data,
        alpha_dir=alpha,
        seed=seed,
        client_validation=use_client_validation,
        data_root=run_config.get("data_root", "data"),
        download_datasets=run_config.get("download_datasets", True),
        medmnist_size=medmnist_size,
    )

    return FlowerClient(
        cid=partition_id,
        dataset=dataset,
        batch_size=batch_size,
        trainloader=trainloader,
        valloader_local=valloader_local,
        testloader_local=testloader_local,
        context=context,
        folder=folder,
        seed=seed,
        lr_alvo=run_config.get("learn_rate_alvo", 0.01),
        cvae_lr=run_config.get("learn_rate_gen", run_config.get("cvae_lr", 0.001)),
        local_epochs_alvo=run_config.get("epocas_alvo", 1),
        cvae_local_epochs=cvae_local_epochs,
        continue_epoch=run_config.get("continue_epoch", 0),
        medmnist_size=medmnist_size,
        memory_logging=run_config.get("memory_logging", False),
        run_id=context.run_id,
    ).to_client()


app = ClientApp(client_fn=client_fn)
