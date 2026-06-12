"""Custom Flower strategy for the CVAE version of FLEG."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
import time
from logging import WARNING
from typing import Callable, Optional, Union

import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg

from Simulation.CVAE.task import (
    Decoder,
    create_classifier,
    create_feature_extractor,
    create_full_model,
    get_image_key,
    get_model_size_mb,
    get_weights,
    infer_feature_dim,
    normalize_dataset_name,
    set_weights,
    state_dict_to_bytes,
)


class FLEG_CVAE(Strategy):
    def __init__(
        self,
        *,
        fraction_fit_alvo: float = 1.0,
        fraction_fit_cvae: float = 1.0,
        fraction_evaluate_alvo: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        dataset: str = "mnist",
        folder: str = ".",
        strategy_name: str = "fedavg",
        mu: float = 0.5,
        seed: int = 42,
        patience: int = 10,
        levels: int = 4,
        lesslvl: bool = False,
        baseline: bool = False,
        cvae_epochs: int = 25,
        cvae_local_epochs: int = 1,
        cvae_beta: float = 1.0,
        cvae_lr: float = 0.001,
        normalization: str = "minmax",
        resblock: bool = False,
        anealing: bool = False,
        latent_dim_mode: str = "fixed",
        latent_dim: int = 100,
        num_syn: str = "dynamic",
        mixup_type: str = "none",
        valloader=None,
        num_clients: int = 4,
        resume_from_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.fraction_fit_alvo = fraction_fit_alvo
        self.fraction_fit_cvae = fraction_fit_cvae
        self.fraction_evaluate_alvo = fraction_evaluate_alvo
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.parameters_classifier = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace

        self.dataset = normalize_dataset_name(dataset)
        self.image_key = get_image_key(self.dataset)
        self.folder = folder
        self.strategy_name = strategy_name
        self.mu = mu
        self.seed = seed
        self.patience = patience
        self.levels = levels
        self.lesslvl = lesslvl
        self.baseline = baseline
        self.cvae_epochs_target = cvae_epochs
        self.cvae_local_epochs = cvae_local_epochs
        self.cvae_beta = cvae_beta
        self.cvae_lr = cvae_lr
        self.normalization = normalization
        self.resblock = resblock
        self.anealing = anealing
        self.latent_dim_mode = latent_dim_mode
        self.fixed_latent_dim = latent_dim
        self.num_syn = num_syn
        self.mixup_type = mixup_type
        self.valloader = valloader
        self.num_clients = num_clients
        self.resume_from_checkpoint = resume_from_checkpoint

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.phase = "classifier"
        self.lvl = 0
        self.net_epochs = 0
        self.best_accuracy = 0.0
        self.epochs_no_improve = 0
        self.finished = False
        self.generated_payload: bytes = b""
        self.generated_sent_to_cids: set[str] = set()
        self.feature_extractor_payload: bytes = b""
        self.feature_extractor_sent_to_cids: set[str] = set()
        self._skip_next_client_evaluation = False
        self._last_total_examples = num_clients
        self._level_transmission_mb = 0.0
        self._level_classifier_traffic_mb = 0.0
        self._level_cvae_traffic_mb = 0.0

        self.global_net = create_full_model(self.dataset, seed=self.seed).to(self.device)
        self.best_model = create_full_model(self.dataset, seed=self.seed).to(self.device)
        self.classifier = create_classifier(self.dataset, level=0, seed=self.seed).to(self.device)

        self.decoder = None
        self.parameters_cvae = None
        self.cvae_trained_epochs = 0
        self.cvae_level = None
        self.input_dim = None
        self.hidden_dim = None
        self.latent_dim = None
        self.feature_shape = None

        self.metrics_dict = {
            "avg_cvae_loss": [],
            "avg_net_loss": [],
            "global_net_acc": [],
            "local_acc_epoch": [],
            "time_epoch_classifier": [],
            "time_epoch_cvae": [],
            "level_time": [],
            "max_net_time": [],
            "max_cvae_time": [],
            "net_global_eval_time": [],
            "img_syn_time": [],
            "MB_transmission": [],
            "traffic_cost_classifier": [],
            "traffic_cost_cvae": [],
            "epoch_transition": [],
            "max_local_test_time": [],
        }
        self.init_level_time = time.time()
        os.makedirs(self.folder, exist_ok=True)
        if self.resume_from_checkpoint:
            self._load_latest_checkpoint()

    def __repr__(self) -> str:
        return "FLEG_CVAE()"

    def _classifier_level(self) -> int:
        return 0 if self.lvl == 0 else self.lvl + (1 if self.lesslvl else 0)

    def _is_final_level(self) -> bool:
        return self.lvl + (1 if self.lesslvl else 0) >= self.levels

    def _save_metrics(self) -> None:
        metrics_filename = f"{self.folder}/metrics.json"
        try:
            with open(metrics_filename, "w", encoding="utf-8") as f:
                json.dump(self.metrics_dict, f, ensure_ascii=False, indent=4)
            print(f"Metrics dict successfully saved to {metrics_filename}")
        except Exception as exc:
            print(f"Error saving metrics dict to JSON: {exc}")

    def _checkpoint_files(self) -> list[Path]:
        folder = Path(self.folder)
        candidates = list(folder.glob("checkpoint_level*.pth"))
        candidates.extend(folder.glob("checkpoint.pth"))
        return [path for path in candidates if path.is_file()]

    def _latest_checkpoint_path(self) -> Optional[Path]:
        checkpoints = self._checkpoint_files()
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda path: path.stat().st_mtime)

    def _save_checkpoint(
        self,
        filename: str,
        *,
        level: Optional[int] = None
    ) -> None:
        checkpoint = {
            "classifier_state_dict": self.global_net.state_dict(),
            "level": self.lvl if level is None else level,
            "net_epochs": self.net_epochs,
            "best_accuracy": self.best_accuracy,
            "metrics_dict": self.metrics_dict,
            "generated_payload": self.generated_payload,
        }
        torch.save(checkpoint, f"{self.folder}/{filename}")

    def _load_latest_checkpoint(self) -> None:
        checkpoint_path = self._latest_checkpoint_path()
        if checkpoint_path is None:
            print("Nenhum checkpoint CVAE encontrado para retomada.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.global_net.load_state_dict(checkpoint["classifier_state_dict"])
        self.best_model.load_state_dict(checkpoint["classifier_state_dict"])
        self.lvl = int(checkpoint.get("level", self.lvl))
        self.net_epochs = int(checkpoint.get("net_epochs", self.net_epochs))
        self.best_accuracy = float(checkpoint.get("best_accuracy", self.best_accuracy))
        self.metrics_dict = checkpoint.get("metrics_dict", self.metrics_dict)
        self.generated_payload = checkpoint.get("generated_payload", b"")

        self._setup_classifier_for_current_level()

        self.init_level_time = time.time()
        print(f"Retomando treinamento a partir de {checkpoint_path}.")

    @staticmethod
    def _bytes_size_mb(payload) -> float:
        if payload is None:
            return 0.0
        if isinstance(payload, (bytes, bytearray)):
            return len(payload) / 10**6
        return 0.0

    def _parameters_size_mb(self, parameters: Optional[Parameters]) -> float:
        if parameters is None:
            return 0.0
        return get_model_size_mb(parameters_to_ndarrays(parameters))

    def _config_payload_size_mb(self, config: dict) -> float:
        return sum(self._bytes_size_mb(value) for value in config.values())

    def _add_level_traffic(self, amount_mb: float, bucket: str) -> None:
        self._level_transmission_mb += amount_mb
        if bucket == "classifier":
            self._level_classifier_traffic_mb += amount_mb
        elif bucket == "cvae":
            self._level_cvae_traffic_mb += amount_mb

    def _record_level_transmission(self) -> None:
        self.metrics_dict["traffic_cost_classifier"].append(
            self._level_classifier_traffic_mb
        )
        self.metrics_dict["traffic_cost_cvae"].append(self._level_cvae_traffic_mb)
        self.metrics_dict["MB_transmission"].append(self._level_transmission_mb)
        self._level_transmission_mb = 0.0
        self._level_classifier_traffic_mb = 0.0
        self._level_cvae_traffic_mb = 0.0

    def _setup_classifier_for_current_level(self) -> None:
        level = self._classifier_level()
        self.classifier = create_classifier(self.dataset, level=level, seed=self.seed).to(
            self.device
        )
        self.classifier.load_state_dict(self.global_net.state_dict(), strict=False)
        self.parameters_classifier = ndarrays_to_parameters(get_weights(self.classifier))

    def _first_validation_image(self):
        batch = next(iter(self.valloader))
        return batch[self.image_key][0]

    def _setup_cvae_for_current_level(self) -> None:
        self.cvae_level = self.lvl + (2 if self.lesslvl else 1)
        feature_extractor = create_feature_extractor(
            self.dataset, level=self.cvae_level, seed=self.seed
        ).to(self.device)
        feature_extractor.load_state_dict(self.global_net.state_dict(), strict=False)

        self.input_dim, self.feature_shape = infer_feature_dim(
            feature_extractor=feature_extractor,
            sample_input=self._first_validation_image(),
            device=self.device,
        )
        self.feature_extractor_payload = state_dict_to_bytes(
            feature_extractor.state_dict()
        )
        self.feature_extractor_sent_to_cids = set()
        self.hidden_dim = max(1, self.input_dim // 2)
        self.latent_dim = (
            max(1, self.input_dim // 4)
            if self.latent_dim_mode == "dynamic"
            else self.fixed_latent_dim
        )
        self.decoder = Decoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            condition_dim=10,
            resblock=self.resblock,
            minmax=self.normalization == "minmax",
        ).to(self.device)
        self.parameters_cvae = ndarrays_to_parameters(get_weights(self.decoder.decoder))
        self.cvae_trained_epochs = 0

    def _num_generated_samples(self) -> int:
        total_examples = self._last_total_examples or self.num_clients
        if self.num_syn == "dynamic":
            num_samples = int(
                max(1, torch.ceil(torch.tensor(total_examples / self.num_clients)).item())
                / self.levels
                * (self.lvl + 1)
            )
            num_samples = max(1, num_samples)
        elif self.num_syn == "fixed":
            num_samples = 48000 if self.dataset == "mnist" else 40000
        else:
            raise ValueError(f"num_syn deve ser 'dynamic' ou 'fixed', recebeu {self.num_syn}")

        return num_samples

    def _cvae_config(self, model: str) -> dict[str, Scalar]:
        warmup_epochs = self.cvae_epochs_target // 3
        beta = self.cvae_beta
        if self.anealing:
            beta = min(
                1.0,
                self.cvae_trained_epochs / warmup_epochs if warmup_epochs > 0 else 0.0,
            )
        remaining_epochs = max(0, self.cvae_epochs_target - self.cvae_trained_epochs)
        self.local_epochs = min(self.cvae_local_epochs, remaining_epochs)

        return {
            "model": model,
            "cvae_level": self.cvae_level,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "feature_shape": pickle.dumps(self.feature_shape),
            "normalization": self.normalization,
            "beta": beta,
            "resblock": self.resblock,
            "num_samples": self._num_generated_samples(),
            "local_epochs": self.local_epochs,
        }

    def _server_embedding_norm_stats(self) -> dict[str, torch.Tensor]:
        feature_extractor = create_feature_extractor(
            self.dataset, level=self.cvae_level, seed=self.seed
        ).to(self.device)
        feature_extractor.load_state_dict(self.global_net.state_dict(), strict=False)
        feature_extractor.eval()

        all_embeddings = []
        with torch.no_grad():
            for batch in self.valloader:
                images = batch[self.image_key].to(self.device)
                embeddings = feature_extractor(images)
                embeddings = embeddings.view(embeddings.size(0), -1)
                all_embeddings.append(embeddings)

        if not all_embeddings:
            raise RuntimeError("Nao foi possivel calcular estatisticas do servidor.")

        embeddings = torch.cat(all_embeddings, dim=0)
        if self.normalization == "minmax":
            min_ = embeddings.min(dim=0).values
            max_ = embeddings.max(dim=0).values
            scale = torch.clamp(max_ - min_, min=1e-8)
            return {"min": min_, "scale": scale}
        if self.normalization == "z":
            mean = embeddings.mean(dim=0)
            std = torch.clamp(embeddings.std(dim=0), min=1e-8)
            return {"mean": mean, "std": std}
        raise ValueError(
            f"normalization deve ser 'minmax' ou 'z', recebeu {self.normalization}"
        )

    def _sample_generated_labels(self, num_samples: int) -> torch.Tensor:
        labels = torch.arange(10, device=self.device).repeat(num_samples // 10)
        remainder = num_samples % 10
        if remainder:
            labels = torch.cat(
                [
                    labels,
                    torch.randint(0, 10, (remainder,), device=self.device),
                ],
                dim=0,
            )
        if labels.numel() == 0:
            labels = torch.randint(0, 10, (num_samples,), device=self.device)
        return labels[torch.randperm(labels.numel(), device=self.device)]

    def _generate_embeddings_on_server(self) -> bytes:
        if self.decoder is None:
            return b""

        start = time.time()
        num_samples = self._num_generated_samples()
        norm_stats = self._server_embedding_norm_stats()
        labels = self._sample_generated_labels(num_samples)
        latents = torch.randn((num_samples, self.latent_dim), device=self.device)

        self.decoder.eval()
        with torch.no_grad():
            one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float()
            generated = self.decoder.decode(latents, one_hot)
            if self.normalization == "minmax":
                generated = generated * norm_stats["scale"] + norm_stats["min"]
            else:
                generated = generated * norm_stats["std"] + norm_stats["mean"]

            generated = generated.view(-1, *self.feature_shape)
            payload = {
                "assets": generated.detach().cpu().numpy(),
                "labels": labels.detach().cpu().numpy(),
                "time": time.time() - start,
            }
        return pickle.dumps(payload)

    def _feature_payload_for_client(self, cid: str, level: int) -> bytes:
        if level <= 0 or cid in self.feature_extractor_sent_to_cids:
            return b""
        if not self.feature_extractor_payload:
            feature_extractor = create_feature_extractor(
                self.dataset, level=level, seed=self.seed
            ).to(self.device)
            feature_extractor.load_state_dict(self.global_net.state_dict(), strict=False)
            self.feature_extractor_payload = state_dict_to_bytes(
                feature_extractor.state_dict()
            )
        self.feature_extractor_sent_to_cids.add(cid)
        return self.feature_extractor_payload

    def _generated_payload_for_client(self, cid: str, level: int) -> bytes:
        if level <= 0 or cid in self.generated_sent_to_cids:
            return b""
        self.generated_sent_to_cids.add(cid)
        return self.generated_payload

    def _global_eval(self) -> float:
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        start = time.time()
        self.global_net.eval()
        with torch.no_grad():
            for batch in self.valloader:
                images = batch[self.image_key].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.global_net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(self.valloader.dataset)
        self.metrics_dict["net_global_eval_time"].append(time.time() - start)
        self.metrics_dict["global_net_acc"].append(accuracy)
        return accuracy

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        fraction = self.fraction_fit_cvae if self.phase.startswith("cvae") else self.fraction_fit_alvo
        num_clients = int(num_available_clients * fraction)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate_alvo)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.parameters_classifier

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            return None
        eval_res = self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
        if eval_res is None:
            return None
        return eval_res

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        if self.finished:
            return []

        self.init_round_time = time.time()
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if self.phase == "classifier":
            classifier_level = self._classifier_level()
            classifier_parameters_mb = self._parameters_size_mb(self.parameters_classifier)
            fit_instructions = []
            for client in clients:
                cid = str(client.cid)
                feature_payload = self._feature_payload_for_client(
                    cid, classifier_level
                )
                generated_payload = self._generated_payload_for_client(
                    cid, classifier_level
                )
                config = {
                    "model": "classifier",
                    "classifier_level": classifier_level,
                    "strategy": self.strategy_name,
                    "mu": self.mu,
                    "mixup_type": self.mixup_type,
                    "feature_extractor_state": feature_payload,
                    "generated": generated_payload,
                }
                self._add_level_traffic(
                    classifier_parameters_mb/len(clients)
                    + self._bytes_size_mb(feature_payload)/len(clients)
                    + self._bytes_size_mb(generated_payload)/len(clients),
                    bucket="classifier",
                )
                fit_instructions.append(
                    (client, FitIns(parameters=self.parameters_classifier, config=config))
                )
            return fit_instructions

        if self.phase == "cvae":
            if self.decoder is None:
                self._setup_cvae_for_current_level()
            base_config = self._cvae_config(model="cvae")
            parameters_cvae_mb = self._parameters_size_mb(self.parameters_cvae)
            fit_instructions = []
            for client in clients:
                config = dict(base_config)
                feature_payload = self._feature_payload_for_client(
                    str(client.cid), self.cvae_level
                )
                config["feature_extractor_state"] = feature_payload
                self._add_level_traffic(
                    (
                        parameters_cvae_mb/len(clients)
                        + self._config_payload_size_mb(config)/len(clients)
                    ),
                    bucket="cvae",
                )
                fit_instructions.append(
                    (client, FitIns(parameters=self.parameters_cvae, config=config))
                )
            return fit_instructions

        raise ValueError(f"Fase desconhecida: {self.phase}")

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        if self._skip_next_client_evaluation:
            self._skip_next_client_evaluation = False
            return []
        if (
            self.finished
            or self.phase != "classifier"
            or self.fraction_evaluate_alvo == 0.0
        ):
            return []

        classifier_level = self._classifier_level()
        base_config = {
            "round": self.net_epochs,
            "classifier_level": classifier_level,
        }
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        evaluate_instructions = []
        parameters_classifier_mb = self._parameters_size_mb(self.parameters_classifier)
        for client in clients:
            config = dict(base_config)
            feature_payload = self._feature_payload_for_client(
                str(client.cid), classifier_level
            )
            config["feature_extractor_state"] = feature_payload
            self._add_level_traffic(
                (
                    parameters_classifier_mb/len(clients)
                    + self._config_payload_size_mb(config)/len(clients)
                ),
                bucket="classifier",
            )
            evaluate_instructions.append(
                (client, EvaluateIns(self.parameters_classifier, config))
            )
        return evaluate_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results or self.finished:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        self._last_total_examples = sum(fit_res.num_examples for _, fit_res in results)

        if self.phase == "classifier":
            if self.inplace:
                aggregated_ndarrays = aggregate_inplace(results)
            else:
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results
                ]
                aggregated_ndarrays = aggregate(weights_results)

            self.parameters_classifier = ndarrays_to_parameters(aggregated_ndarrays)
            set_weights(self.classifier, aggregated_ndarrays)
            self.global_net.load_state_dict(self.classifier.state_dict(), strict=False)

            avg_loss = weighted_loss_avg(
                [(fit_res.num_examples, fit_res.metrics["train_loss"]) for _, fit_res in results]
            )
            self.metrics_dict["avg_net_loss"].append(avg_loss)
            self.metrics_dict["max_net_time"].append(
                max(fit_res.metrics["tempo_treino_alvo"] for _, fit_res in results)
            )
            self.metrics_dict["time_epoch_classifier"].append(
                time.time() - self.init_round_time
            )
            classifier_upload_mb = sum(
                get_model_size_mb(parameters_to_ndarrays(fit_res.parameters))
                for _, fit_res in results
            )
            self._add_level_traffic(
                classifier_upload_mb/len(results),
                bucket="classifier",
            )

            accuracy = self._global_eval()
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.epochs_no_improve = 0
                self.best_model.load_state_dict(self.global_net.state_dict())
            else:
                self.epochs_no_improve += 1
            print(
                f"Sem melhorias por {self.epochs_no_improve} épocas. "
                f"Melhor acurácia: {self.best_accuracy:.4f} x Acurácia atual: {accuracy:.4f}."
            )
            self.net_epochs += 1
            return self.parameters_classifier, {"avg_net_loss": avg_loss}

        if self.phase == "cvae":
            aggregated_ndarrays = aggregate_inplace(results)
            set_weights(self.decoder.decoder, aggregated_ndarrays)
            self.parameters_cvae = ndarrays_to_parameters(aggregated_ndarrays)

            avg_cvae_loss = weighted_loss_avg(
                [(fit_res.num_examples, fit_res.metrics["cvae_loss"]) for _, fit_res in results]
            )
            self.metrics_dict["avg_cvae_loss"].append(avg_cvae_loss)
            self.metrics_dict["max_cvae_time"].append(
                max(fit_res.metrics["cvae_time"] for _, fit_res in results)
            )
            self.metrics_dict["time_epoch_cvae"].append(time.time() - self.init_round_time)
            cvae_upload_mb = sum(
                get_model_size_mb(parameters_to_ndarrays(fit_res.parameters))
                for _, fit_res in results
            )
            self._add_level_traffic(
                cvae_upload_mb/len(results),
                bucket="cvae",
            )

            self.cvae_trained_epochs += self.local_epochs
            if self.cvae_trained_epochs >= self.cvae_epochs_target:
                self.generated_payload = self._generate_embeddings_on_server()
                self.generated_sent_to_cids = set()
                generation_time = 0.0
                if self.generated_payload:
                    generation_time = pickle.loads(self.generated_payload).get("time", 0.0)
                self.metrics_dict["img_syn_time"].append(generation_time)
                self.metrics_dict["level_time"].append(time.time() - self.init_level_time)
                self._record_level_transmission()
                self._save_checkpoint(
                    f"checkpoint_level{self.lvl + 1}.pth",
                    level=self.lvl + 1,
                )
                self._save_metrics()

                self.lvl += 1
                self.epochs_no_improve = 0
                self.phase = "classifier"
                self._skip_next_client_evaluation = True
                self.decoder = None
                self.parameters_cvae = None
                self.cvae_trained_epochs = 0
                self.init_level_time = time.time()
                self._setup_classifier_for_current_level()
                print(f"Avançando para o nível {self.lvl}.")
                return self.parameters_classifier, {"avg_cvae_loss": avg_cvae_loss}
            return self.parameters_cvae, {"avg_cvae_loss": avg_cvae_loss}

        return None, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        if not results or self.finished:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )
        accuracy_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.metrics["local_accuracy"])
                for _, evaluate_res in results
            ]
        )
        self.metrics_dict["local_acc_epoch"].append(accuracy_aggregated)
        self.metrics_dict["max_local_test_time"].append(
            max(evaluate_res.metrics["local_test_time"] for _, evaluate_res in results)
        )

        if self.epochs_no_improve > self.patience:
            self.global_net.load_state_dict(self.best_model.state_dict())
            self.metrics_dict["epoch_transition"].append(self.net_epochs)
            if self.baseline or self._is_final_level():
                self.finished = True
                self.metrics_dict["level_time"].append(time.time() - self.init_level_time)
                self._record_level_transmission()
                self._save_checkpoint("checkpoint_end.pth")
                self._save_metrics()
                print("Treinamento finalizado.")
            else:
                self.phase = "cvae"
                self.parameters_cvae = None
                self.generated_payload = b""
                self.generated_sent_to_cids = set()
                self.feature_extractor_payload = b""
                self.feature_extractor_sent_to_cids = set()
                print("Alternando para treinamento do CVAE.")

        metrics = {"loss": loss_aggregated, "accuracy": accuracy_aggregated}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics
