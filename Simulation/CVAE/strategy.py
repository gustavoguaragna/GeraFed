"""Custom Flower strategy for the CVAE version of FLEG."""

from __future__ import annotations

import json
import os
import pickle
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
    get_target_classifier_level,
    get_target_cvae_level,
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
        cvae_beta: float = 1.0,
        cvae_lr: float = 0.001,
        cvae_local_epochs: int = 1,
        normalization: str = "minmax",
        resblock: bool = False,
        anealing: bool = False,
        latent_dim_mode: str = "fixed",
        latent_dim: int = 100,
        num_syn: str = "dynamic",
        mixup_type: str = "none",
        valloader=None,
        num_clients: int = 4,
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
        self.cvae_epochs = cvae_epochs
        self.cvae_beta = cvae_beta
        self.cvae_lr = cvae_lr
        self.cvae_local_epochs = cvae_local_epochs
        self.normalization = normalization
        self.resblock = resblock
        self.anealing = anealing
        self.latent_dim_mode = latent_dim_mode
        self.fixed_latent_dim = latent_dim
        self.num_syn = num_syn
        self.mixup_type = mixup_type
        self.valloader = valloader
        self.num_clients = num_clients

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.phase = "classifier"
        self.lvl = 0
        self.net_epochs = 0
        self.best_accuracy = 0.0
        self.epochs_no_improve = 0
        self.finished = False
        self.generated_by_cid: dict[str, bytes] = {}
        self._last_total_examples = num_clients

        self.global_net = create_full_model(self.dataset, seed=self.seed).to(self.device)
        self.best_model = create_full_model(self.dataset, seed=self.seed).to(self.device)
        self.classifier = create_classifier(self.dataset, level=0, seed=self.seed).to(self.device)

        self.decoder = None
        self.parameters_cvae = None
        self.cvae_round = 0
        self.cvae_level = None
        self.input_dim = None
        self.hidden_dim = None
        self.latent_dim = None
        self.feature_shape = None

        self.metrics_dict = {
            "cvae_loss": [],
            "net_loss": [],
            "net_acc": [],
            "local_acc_epoch": [],
            "time_epoch_classifier": [],
            "time_epoch_cvae": [],
            "time_level": [],
            "net_time": [],
            "cvae_time": [],
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

    def __repr__(self) -> str:
        return "FLEG_CVAE()"

    def _classifier_level(self) -> int:
        return get_target_classifier_level(self.lvl, self.lesslvl)

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

    def _save_checkpoint(self, filename: str) -> None:
        checkpoint = {
            "classifier_state_dict": self.global_net.state_dict(),
            "level": self.lvl,
        }
        if self.decoder is not None:
            checkpoint["decoder_state_dict"] = self.decoder.state_dict()
        torch.save(checkpoint, f"{self.folder}/{filename}")

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
        self.cvae_level = get_target_cvae_level(self.lvl, self.lesslvl)
        feature_extractor = create_feature_extractor(
            self.dataset, level=self.cvae_level, seed=self.seed
        ).to(self.device)
        feature_extractor.load_state_dict(self.global_net.state_dict(), strict=False)

        self.input_dim, self.feature_shape = infer_feature_dim(
            feature_extractor=feature_extractor,
            sample_input=self._first_validation_image(),
            device=self.device,
        )
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
        self.cvae_round = 0

    def _cvae_config(self, model: str) -> dict[str, Scalar]:
        warmup_epochs = self.cvae_epochs // 3
        beta = self.cvae_beta
        if self.anealing:
            beta = min(1.0, self.cvae_round / warmup_epochs if warmup_epochs > 0 else 1.0)

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

        return {
            "model": model,
            "cvae_level": self.cvae_level,
            "global_net_state": state_dict_to_bytes(self.global_net.state_dict()),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "feature_shape": pickle.dumps(self.feature_shape),
            "normalization": self.normalization,
            "beta": beta,
            "resblock": self.resblock,
            "num_samples": num_samples,
        }

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
        self.metrics_dict["net_acc"].append(accuracy)
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
            fit_instructions = []
            for client in clients:
                config = {
                    "model": "classifier",
                    "round": self.net_epochs,
                    "logical_level": self.lvl,
                    "classifier_level": classifier_level,
                    "strategy": self.strategy_name,
                    "mu": self.mu,
                    "mixup_type": self.mixup_type,
                    "global_net_state": state_dict_to_bytes(self.global_net.state_dict()),
                    "generated": self.generated_by_cid.get(str(client.cid), b""),
                }
                fit_instructions.append(
                    (client, FitIns(parameters=self.parameters_classifier, config=config))
                )
            return fit_instructions

        if self.phase == "cvae":
            if self.decoder is None:
                self._setup_cvae_for_current_level()
            fit_ins = FitIns(
                parameters=self.parameters_cvae,
                config=self._cvae_config(model="cvae"),
            )
            return [(client, fit_ins) for client in clients]

        if self.phase == "cvae_generate":
            fit_ins = FitIns(
                parameters=self.parameters_cvae,
                config=self._cvae_config(model="cvae_generate"),
            )
            return [(client, fit_ins) for client in clients]

        raise ValueError(f"Fase desconhecida: {self.phase}")

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        if self.finished or self.phase != "classifier" or self.fraction_evaluate_alvo == 0.0:
            return []

        config = {
            "round": self.net_epochs,
            "classifier_level": self._classifier_level(),
            "global_net_state": state_dict_to_bytes(self.global_net.state_dict()),
        }
        evaluate_ins = EvaluateIns(self.parameters_classifier, config)
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, evaluate_ins) for client in clients]

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
            level = self._classifier_level()
            if level == 0:
                set_weights(self.global_net, aggregated_ndarrays)
            else:
                set_weights(self.classifier, aggregated_ndarrays)
                self.global_net.load_state_dict(self.classifier.state_dict(), strict=False)

            avg_loss = weighted_loss_avg(
                [(fit_res.num_examples, fit_res.metrics["train_loss"]) for _, fit_res in results]
            )
            self.metrics_dict["net_loss"].append(avg_loss)
            self.metrics_dict["net_time"].append(
                max(fit_res.metrics["tempo_treino_alvo"] for _, fit_res in results)
            )
            self.metrics_dict["time_epoch_classifier"].append(
                time.time() - self.init_round_time
            )
            self.metrics_dict["traffic_cost_classifier"].append(
                sum(
                    get_model_size_mb(parameters_to_ndarrays(fit_res.parameters))
                    for _, fit_res in results
                )
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
                    f"Melhor acurácia: {self.best_accuracy:.4f}"
                )
            self.net_epochs += 1
            return self.parameters_classifier, {"net_loss": avg_loss}

        if self.phase == "cvae":
            aggregated_ndarrays = aggregate_inplace(results)
            set_weights(self.decoder.decoder, aggregated_ndarrays)
            self.parameters_cvae = ndarrays_to_parameters(aggregated_ndarrays)

            avg_cvae_loss = weighted_loss_avg(
                [(fit_res.num_examples, fit_res.metrics["cvae_loss"]) for _, fit_res in results]
            )
            self.metrics_dict["cvae_loss"].append(avg_cvae_loss)
            self.metrics_dict["cvae_time"].append(
                max(fit_res.metrics["cvae_time"] for _, fit_res in results)
            )
            self.metrics_dict["time_epoch_cvae"].append(time.time() - self.init_round_time)
            self.metrics_dict["traffic_cost_cvae"].append(
                get_model_size_mb(aggregated_ndarrays) * 2
            )
            self.cvae_round += 1
            if self.cvae_round >= self.cvae_epochs:
                self.phase = "cvae_generate"
            return self.parameters_cvae, {"cvae_loss": avg_cvae_loss}

        if self.phase == "cvae_generate":
            generation_times = []
            self.generated_by_cid = {}
            for client, fit_res in results:
                cid = str(fit_res.metrics.get("cid"))
                payload = fit_res.metrics.get("generated", b"")
                self.generated_by_cid[cid] = payload
                self.generated_by_cid[str(client.cid)] = payload
                if payload:
                    generation_times.append(pickle.loads(payload).get("time", 0.0))
            self.metrics_dict["img_syn_time"].append(
                sum(generation_times) / len(generation_times) if generation_times else 0.0
            )
            self.metrics_dict["time_level"].append(time.time() - self.init_level_time)
            self._save_checkpoint(f"checkpoint_level{self.lvl + 1}.pth")
            self._save_metrics()

            self.global_net.load_state_dict(self.best_model.state_dict())
            self.lvl += 1
            self.epochs_no_improve = 0
            self.phase = "classifier"
            self.decoder = None
            self.parameters_cvae = None
            self.init_level_time = time.time()
            self._setup_classifier_for_current_level()
            print(f"Avançando para o nível {self.lvl}.")
            return self.parameters_classifier, {}

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
                self.metrics_dict["time_level"].append(time.time() - self.init_level_time)
                self._save_checkpoint("checkpoint_end.pth")
                self._save_metrics()
                print("Treinamento CVAE finalizado.")
            else:
                self.phase = "cvae"
                self.decoder = None
                self.parameters_cvae = None
                self.generated_by_cid = {}
                print("Alternando para treinamento do CVAE.")

        metrics = {"loss": loss_aggregated, "accuracy": accuracy_aggregated}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics
