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
    get_num_classes,
    get_weights,
    infer_feature_dim,
    log_memory_event,
    normalize_dataset_name,
    normalize_stop_criterion,
    object_tensor_size_mb,
    set_weights,
    state_dict_to_bytes,
    uses_client_validation_criterion,
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
        stop_criterion: str = "global_test_acc",
        fixed_classifier_rounds: int = 1,
        reset_best_metric_per_level: bool = True,
        levels: int = 4,
        lesslvl: bool = False,
        baseline: bool = False,
        cvae_epochs: int = 25,
        cvae_local_epochs: int = 1,
        cvae_beta: float = 1.0,
        cvae_lr: float = 0.001,
        normalization: str = "minmax",
        resblock: bool = False,
        cvae_depth: int = 2,
        annealing: bool = False,
        latent_dim_mode: str = "fixed",
        latent_dim: int = 100,
        num_syn: str = "dynamic",
        mixup_type: str = "none",
        valloader=None,
        num_clients: int = 4,
        resume_from_checkpoint: bool = False,
        memory_logging: bool = False,
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
        self.num_classes = get_num_classes(self.dataset)
        self.folder = folder
        self.strategy_name = strategy_name
        self.mu = mu
        self.seed = seed
        self.patience = patience
        self.stop_criterion = normalize_stop_criterion(stop_criterion)
        self.fixed_classifier_rounds = max(1, int(fixed_classifier_rounds))
        self.reset_best_metric_per_level = reset_best_metric_per_level
        self.levels = levels
        self.lesslvl = lesslvl
        self.baseline = baseline
        self.cvae_epochs_target = cvae_epochs
        self.cvae_local_epochs = cvae_local_epochs
        self.cvae_beta = cvae_beta
        self.cvae_lr = cvae_lr
        self.normalization = normalization
        self.resblock = resblock
        self.cvae_depth = max(0, int(cvae_depth))
        self.annealing = annealing
        self.latent_dim_mode = latent_dim_mode
        self.fixed_latent_dim = latent_dim
        self.num_syn = num_syn
        self.mixup_type = mixup_type
        self.valloader = valloader
        self.num_clients = num_clients
        self.resume_from_checkpoint = resume_from_checkpoint
        self.memory_logging = memory_logging
        self.memory_log_path = (
            f"{self.folder}/memory_server.jsonl" if self.memory_logging else None
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.phase = "classifier"
        self.lvl = 0
        self.net_epochs = 0
        self.global_accuracy = 0.0
        self.best_metric = self._initial_best_metric()
        self.epochs_no_improve = 0
        self.classifier_rounds_current_level = 0
        self.finished = False
        self.decoder_payload: bytes = b""
        self.decoder_sent_to_cids: set[str] = set()
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
        self.input_dim = None
        self.hidden_dim = None
        self.latent_dim = None
        self.feature_shape = None

        self.metrics_dict = {
            "avg_cvae_loss": [],
            "avg_net_loss": [],
            "global_net_acc": [],
            "local_acc_epoch": [],
            "local_val_loss_epoch": [],
            "local_val_acc_epoch": [],
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
        self._log_memory(
            "strategy_init",
            dataset=self.dataset,
            num_clients=self.num_clients,
            levels=self.levels,
            baseline=self.baseline,
        )
        loaded_checkpoint = False
        if self.resume_from_checkpoint:
            loaded_checkpoint = self._load_latest_checkpoint()
        self._ensure_metric_keys()
        if self.phase == "classifier" and self.lvl == 0 and not loaded_checkpoint:
            print("Starting warmup classifier level 0.")

    def __repr__(self) -> str:
        return "FLEG_CVAE()"

    def _log_memory(self, event: str, **fields) -> None:
        payload = {
            "dataset": self.dataset,
            "phase": self.phase,
            "level": self.lvl,
        }
        payload.update(fields)
        log_memory_event(
            self.memory_log_path,
            event,
            **payload,
        )

    def _ensure_metric_keys(self) -> None:
        for key in (
            "avg_cvae_loss",
            "avg_net_loss",
            "global_net_acc",
            "local_acc_epoch",
            "local_val_loss_epoch",
            "local_val_acc_epoch",
            "time_epoch_classifier",
            "time_epoch_cvae",
            "level_time",
            "max_net_time",
            "max_cvae_time",
            "net_global_eval_time",
            "img_syn_time",
            "MB_transmission",
            "traffic_cost_classifier",
            "traffic_cost_cvae",
            "epoch_transition",
            "max_local_test_time",
        ):
            self.metrics_dict.setdefault(key, [])

    def _uses_client_validation(self) -> bool:
        return uses_client_validation_criterion(self.stop_criterion)

    def _metric_is_loss(self) -> bool:
        return self.stop_criterion == "client_val_loss"

    def _metric_is_accuracy(self) -> bool:
        return self.stop_criterion in {"global_test_acc", "client_val_acc"}

    def _initial_best_metric(self) -> float:
        if self._metric_is_loss():
            return float("inf")
        if self._metric_is_accuracy():
            return float("-inf")
        return 0.0

    def _reset_classifier_stop_state(self) -> None:
        self.epochs_no_improve = 0
        self.classifier_rounds_current_level = 0
        if self.reset_best_metric_per_level:
            self.best_metric = self._initial_best_metric()
            self.best_model.load_state_dict(self.global_net.state_dict())

    @staticmethod
    def _weighted_metric(
        results: list[tuple[ClientProxy, EvaluateRes]],
        metric_key: str,
        examples_key: str,
    ) -> float:
        return weighted_loss_avg(
            [
                (
                    int(evaluate_res.metrics[examples_key]),
                    float(evaluate_res.metrics[metric_key]),
                )
                for _, evaluate_res in results
            ]
        )

    def _record_stop_metric(self, current_metric: float, label: str) -> None:
        improved = (
            current_metric < self.best_metric
            if self._metric_is_loss()
            else current_metric > self.best_metric
        )
        if improved:
            self.best_metric = current_metric
            self.epochs_no_improve = 0
            self.best_model.load_state_dict(self.global_net.state_dict())
        else:
            self.epochs_no_improve += 1

        print(
            f"Current {label}: {current_metric:.4f} | Best: "
            f"{self.best_metric:.4f} | No improvement: "
            f"{self.epochs_no_improve}/{self.patience}"
        )

    def _classifier_stop_reached(self) -> bool:
        if self.stop_criterion == "fixed_rounds":
            return self.classifier_rounds_current_level >= self.fixed_classifier_rounds
        return self.epochs_no_improve > self.patience

    def _next_level_after_classifier(self) -> int:
        if self.lesslvl and self.lvl == 0:
            print("lesslvl is enabled. Level 1 will be skipped after warmup.")
            return 2
        return self.lvl + 1

    def _is_final_level(self) -> bool:
        return self.lvl >= self.levels

    def _save_completed_level_state(self, final_checkpoint: bool = False) -> None:
        self.metrics_dict["level_time"].append(time.time() - self.init_level_time)
        self._record_level_transmission()
        checkpoint_filename = (
            "checkpoint_end.pth"
            if final_checkpoint
            else f"checkpoint_level{self.lvl}.pth"
        )
        self._save_checkpoint(checkpoint_filename, level=self.lvl)
        self._save_metrics()

    def _switch_to_next_cvae_phase(self) -> None:
        self.lvl = self._next_level_after_classifier()

        self.phase = "cvae"
        self.decoder = None
        self.parameters_cvae = None
        self.cvae_trained_epochs = 0
        self.decoder_payload = b""
        self.decoder_sent_to_cids = set()
        self.feature_extractor_payload = b""
        self.feature_extractor_sent_to_cids = set()
        self.init_level_time = time.time()
        print(f"Switching to CVAE level {self.lvl}.")

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
        return [
            path for path in folder.glob("checkpoint_level*.pth") if path.is_file()
        ]

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
            "best_metric": self.best_metric,
            "metrics_dict": self.metrics_dict,
        }
        torch.save(checkpoint, f"{self.folder}/{filename}")

    def _load_latest_checkpoint(self) -> bool:
        checkpoint_path = self._latest_checkpoint_path()
        if checkpoint_path is None:
            print("No CVAE checkpoint found for resume.")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.global_net.load_state_dict(checkpoint["classifier_state_dict"])
        self.best_model.load_state_dict(checkpoint["classifier_state_dict"])
        self.lvl = int(checkpoint.get("level", self.lvl))
        self.net_epochs = int(checkpoint.get("net_epochs", self.net_epochs))
        self.best_metric = float(checkpoint["best_metric"])
        self.metrics_dict = checkpoint.get("metrics_dict", self.metrics_dict)
        self._ensure_metric_keys()

        print(f"Resuming training from {checkpoint_path}.")
        if self.baseline or self._is_final_level():
            self.finished = True
            print("Loaded checkpoint is already at the final level.")
        else:
            self._switch_to_next_cvae_phase()
        return True

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
        self.classifier = create_classifier(
            self.dataset, level=self.lvl, seed=self.seed
        ).to(self.device)
        self.classifier.load_state_dict(self.global_net.state_dict(), strict=False)
        self.parameters_classifier = ndarrays_to_parameters(get_weights(self.classifier))

    def _first_validation_image(self):
        batch = next(iter(self.valloader))
        return batch[self.image_key][0]

    def _setup_cvae_for_current_level(self) -> None:
        if self.lvl <= 0:
            raise RuntimeError("CVAE cannot be trained at warmup level 0.")
        self._log_memory("setup_cvae_start")
        feature_extractor = create_feature_extractor(
            self.dataset, level=self.lvl, seed=self.seed
        ).to(self.device)
        feature_extractor.load_state_dict(self.global_net.state_dict(), strict=False)
        self._log_memory(
            "setup_cvae_feature_extractor_loaded",
            feature_extractor_mb=object_tensor_size_mb(feature_extractor.state_dict()),
        )

        self.input_dim, self.feature_shape = infer_feature_dim(
            feature_extractor=feature_extractor,
            sample_input=self._first_validation_image(),
            device=self.device,
        )
        self._log_memory(
            "setup_cvae_feature_dim_inferred",
            input_dim=self.input_dim,
            feature_shape=self.feature_shape,
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
            condition_dim=self.num_classes,
            resblock=self.resblock,
            minmax=self.normalization == "minmax",
            depth=self.cvae_depth,
        ).to(self.device)
        self.parameters_cvae = ndarrays_to_parameters(get_weights(self.decoder.decoder))
        self.cvae_trained_epochs = 0
        self._log_memory(
            "setup_cvae_done",
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            cvae_depth=self.cvae_depth,
            feature_extractor_payload_mb=len(self.feature_extractor_payload) / 10**6,
            decoder_state_mb=object_tensor_size_mb(self.decoder.state_dict()),
        )

    def _num_generated_samples(self) -> int:
        if self.lvl <= 0:
            return 0
        total_examples = self._last_total_examples or self.num_clients
        if self.num_syn == "dynamic":
            num_samples = int(
                max(1, torch.ceil(torch.tensor(total_examples / self.num_clients)).item())
                / self.levels
                * self.lvl
            )
            num_samples = max(1, num_samples)
        elif self.num_syn == "fixed":
            num_samples = max(1, total_examples)
        else:
            raise ValueError(f"num_syn must be 'dynamic' or 'fixed', got {self.num_syn}")

        return num_samples

    def _cvae_config(self, model: str) -> dict[str, Scalar]:
        warmup_epochs = self.cvae_epochs_target // 3
        beta = self.cvae_beta
        if self.annealing:
            beta = min(
                1.0,
                self.cvae_trained_epochs / warmup_epochs if warmup_epochs > 0 else 1.0,
            )
        remaining_epochs = max(0, self.cvae_epochs_target - self.cvae_trained_epochs)
        self.local_epochs = min(self.cvae_local_epochs, remaining_epochs)

        return {
            "model": model,
            "level": self.lvl,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "feature_shape": pickle.dumps(self.feature_shape),
            "normalization": self.normalization,
            "beta": beta,
            "resblock": self.resblock,
            "cvae_depth": self.cvae_depth,
            "num_samples": self._num_generated_samples(),
            "local_epochs": self.local_epochs,
            "cvae_epoch_start": self.cvae_trained_epochs,
            "cvae_epochs_total": self.cvae_epochs_target,
        }

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

    def _decoder_payload_for_client(self, cid: str, level: int) -> bytes:
        if level <= 0 or cid in self.decoder_sent_to_cids:
            return b""
        self.decoder_sent_to_cids.add(cid)
        return self.decoder_payload

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
            num_samples = self._num_generated_samples()
            classifier_parameters_mb = self._parameters_size_mb(self.parameters_classifier)

            fit_instructions = []
            for client in clients:
                cid = str(client.cid)
                decoder_payload = self._decoder_payload_for_client(
                    cid, self.lvl
                )
                config = {
                    "model": "classifier",
                    "level": self.lvl,
                    "strategy": self.strategy_name,
                    "mu": self.mu,
                    "mixup_type": self.mixup_type,
                    "decoder_state": decoder_payload,
                    "input_dim": self.input_dim or 0,
                    "hidden_dim": self.hidden_dim or 0,
                    "latent_dim": self.latent_dim or 0,
                    "feature_shape": pickle.dumps(self.feature_shape),
                    "normalization": self.normalization,
                    "resblock": self.resblock,
                    "cvae_depth": self.cvae_depth,
                    "num_samples": num_samples,
                }
                self._add_level_traffic(
                    classifier_parameters_mb/len(clients),
                    bucket="classifier",
                )
                self._add_level_traffic(
                    self._bytes_size_mb(decoder_payload)/len(clients),
                    bucket="cvae",
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
                    str(client.cid), self.lvl
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

        raise ValueError(f"Unknown phase: {self.phase}")

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

        base_config = {
            "round": self.net_epochs,
            "level": self.lvl,
            "evaluation_split": "validation" if self._uses_client_validation() else "test",
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
        self._log_memory(
            "aggregate_fit_start",
            server_round=server_round,
            results=len(results),
            total_examples=self._last_total_examples,
        )

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
            self._log_memory(
                "aggregate_classifier_weights_done",
                server_round=server_round,
                aggregated_weights_mb=object_tensor_size_mb(aggregated_ndarrays),
                classifier_state_mb=object_tensor_size_mb(self.classifier.state_dict()),
                global_state_mb=object_tensor_size_mb(self.global_net.state_dict()),
            )

            avg_loss = weighted_loss_avg(
                [(fit_res.num_examples, fit_res.metrics["train_loss"]) for _, fit_res in results]
            )
            self.metrics_dict["avg_net_loss"].append(avg_loss)
            self.metrics_dict["max_net_time"].append(
                max(fit_res.metrics["tempo_treino_alvo"] for _, fit_res in results)
            )
            max_generation_time = max(
                float(fit_res.metrics.get("img_syn_time", 0.0))
                for _, fit_res in results
            )
            if max_generation_time > 0.0:
                self.metrics_dict["img_syn_time"].append(max_generation_time)
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

            self.global_accuracy = self._global_eval()
            print(
                f"Server: classifier level {self.lvl} "
                f"round {self.classifier_rounds_current_level + 1} "
                f"- avg train loss: {avg_loss:.6f} "
                f"- global accuracy: {self.global_accuracy:.4f}"
            )
            if self.stop_criterion == "global_test_acc":
                self._record_stop_metric(self.global_accuracy, "global accuracy")
            elif self.stop_criterion == "fixed_rounds":
                self.best_model.load_state_dict(self.global_net.state_dict())

            self.net_epochs += 1
            self.classifier_rounds_current_level += 1
            if self.stop_criterion == "fixed_rounds":
                print(
                    "Classifier fixed round: "
                    f"{self.classifier_rounds_current_level}/"
                    f"{self.fixed_classifier_rounds}"
                )
            self._log_memory(
                "aggregate_classifier_done",
                server_round=server_round,
                global_accuracy=float(self.global_accuracy),
            )
            return self.parameters_classifier, {"avg_net_loss": avg_loss}

        if self.phase == "cvae":
            aggregated_ndarrays = aggregate_inplace(results)
            set_weights(self.decoder.decoder, aggregated_ndarrays)
            self.parameters_cvae = ndarrays_to_parameters(aggregated_ndarrays)
            self._log_memory(
                "aggregate_cvae_weights_done",
                server_round=server_round,
                aggregated_weights_mb=object_tensor_size_mb(aggregated_ndarrays),
                decoder_state_mb=object_tensor_size_mb(self.decoder.state_dict()),
            )

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

            next_cvae_epoch = min(
                self.cvae_trained_epochs + self.local_epochs,
                self.cvae_epochs_target,
            )
            print(
                f"Server: CVAE level {self.lvl} epoch "
                f"{next_cvae_epoch}/{self.cvae_epochs_target} "
                f"- avg loss: {avg_cvae_loss:.6f}"
            )
            self.cvae_trained_epochs += self.local_epochs
            if self.cvae_trained_epochs >= self.cvae_epochs_target:
                self.decoder_payload = state_dict_to_bytes(self.decoder.state_dict())
                self.decoder_sent_to_cids = set()
                self._log_memory(
                    "aggregate_cvae_decoder_payload_ready",
                    server_round=server_round,
                    decoder_payload_mb=len(self.decoder_payload) / 10**6,
                )

                self.phase = "classifier"
                self._skip_next_client_evaluation = True
                self.decoder = None
                self.parameters_cvae = None
                self.cvae_trained_epochs = 0
                self._log_memory("aggregate_cvae_before_classifier_setup")
                self._setup_classifier_for_current_level()
                self._reset_classifier_stop_state()
                self._log_memory("aggregate_cvae_switched_to_classifier")
                print(f"Switching to classifier level {self.lvl}.")
                return self.parameters_classifier, {"avg_cvae_loss": avg_cvae_loss}
            self._log_memory(
                "aggregate_cvae_done",
                server_round=server_round,
                cvae_trained_epochs=self.cvae_trained_epochs,
            )
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
        local_test_accuracy_aggregated = self._weighted_metric(
            results,
            metric_key="local_test_accuracy",
            examples_key="local_test_num_examples",
        )
        if self._uses_client_validation():
            self.metrics_dict["local_val_loss_epoch"].append(loss_aggregated)
            self.metrics_dict["local_val_acc_epoch"].append(accuracy_aggregated)
            self.metrics_dict["local_acc_epoch"].append(local_test_accuracy_aggregated)
            if self.stop_criterion == "client_val_loss":
                self._record_stop_metric(loss_aggregated, "validation loss")
            else:
                self._record_stop_metric(accuracy_aggregated, "validation accuracy")
        else:
            self.metrics_dict["local_acc_epoch"].append(local_test_accuracy_aggregated)
        self.metrics_dict["max_local_test_time"].append(
            max(evaluate_res.metrics["local_test_time"] for _, evaluate_res in results)
        )

        if self._classifier_stop_reached():
            self.global_net.load_state_dict(self.best_model.state_dict())
            self.metrics_dict["epoch_transition"].append(self.net_epochs)
            final_checkpoint = self.baseline or self._is_final_level()
            self._save_completed_level_state(final_checkpoint=final_checkpoint)
            print(f"Completed level {self.lvl}.")
            if final_checkpoint:
                self.finished = True
                print("Training finished.")
            else:
                self._switch_to_next_cvae_phase()

        metrics = {
            "loss": loss_aggregated,
            "global_accuracy": self.global_accuracy,
            "local_test_accuracy": local_test_accuracy_aggregated,
        }
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics.update(self.evaluate_metrics_aggregation_fn(eval_metrics))
            metrics["global_accuracy"] = self.global_accuracy
            metrics.pop("accuracy", None)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics
