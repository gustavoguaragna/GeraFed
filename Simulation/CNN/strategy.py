import json
import time
from flwr.server.strategy import Strategy
from logging import WARNING
from typing import Callable, Optional, Union
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
import torch

# from collections import Counter

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
# import random

from Simulation.CNN.task import Net, Net_CIFAR, set_weights
# import torch


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
    Setting `min_available_clients` lower than `min_fit_clients` or
    `min_evaluate_clients` can cause the server to fail when there are too few clients
    connected to the server. `min_available_clients` must be set to a value larger
    than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
    """
class GeraFed(Strategy):

    """GeraFed Strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global net parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of net updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
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
        img_size: int = 28,
        latent_dim: int = 100,
        agg: str = "full",
        num_chunks: int = 1,
        folder: str = ".",
        valloader = None,
        model_save_interval: int = 10,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.dataset = dataset
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.agg = agg
        self.num_chunks = num_chunks
        self.folder = folder
        self.valloader = valloader
        self.model_save_interval = model_save_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dataset == "mnist":
            self.image = "image"
        elif self.dataset == "cifar10":
            self.image = "img"
        else:
            raise ValueError(f"Dataset {self.dataset} nao identificado. Deveria ser 'mnist' ou 'cifar10'")
        self.metrics_dict = {
            "time_round": [],
            "net_loss_epoch": [],
            "local_acc_epoch": [],
            "val_acc": [],
            "global_net_eval_time": [],
            "max_net_time": [],
            "max_local_test_time": [],
        }
        self._saved_metric_counts = {key: 0 for key in self.metrics_dict}

    def _save_metrics(self) -> None:
        metrics_filename = f"{self.folder}/metrics.json"

        try:
            with open(metrics_filename, 'r', encoding='utf-8') as f:
                existing_metrics = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_metrics = {}
            print("Metrics file not found or invalid. A new one will be created.")
            print(metrics_filename)

        for key, values in self.metrics_dict.items():
            saved_count = self._saved_metric_counts.get(key, 0)
            new_values = values[saved_count:]
            if not new_values:
                continue
            existing_list = existing_metrics.get(key, [])
            existing_list.extend(new_values)
            existing_metrics[key] = existing_list
            self._saved_metric_counts[key] = len(values)

        try:
            with open(metrics_filename, 'w', encoding='utf-8') as f:
                json.dump(existing_metrics, f, ensure_ascii=False, indent=4)
            print(f"Metrics dict successfully saved to {metrics_filename}")
        except Exception as e:
            print(f"Error saving metrics dict to JSON: {e}")

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"GeraFed(accept_failures={self.accept_failures})"
        return rep



    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients


    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global net parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters


    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate net parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.init_round_time = time.time()
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        #fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = list(client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        ))
    
        # sorted_clients = sorted(clients, key=lambda c: self.client_counter[c])
        # metade = len(clients) // 2
        # conjunto_gen = sorted_clients[:metade]
        # conjunto_alvo = sorted_clients[metade:]

        # self.client_counter.update(conjunto_gen)

        fit_instructions = []
        config = {"round": server_round}
        # config_alvo = {"modelo": "alvo"}
        # config_gen = {"modelo": "gen", "round": server_round}

        for c in clients:
            fit_ins = FitIns(parameters=parameters, config=config)
            fit_instructions.append((c, fit_ins))

        # for c in conjunto_alvo:
        #     fit_ins_alvo = FitIns(parameters=self.parameters_alvo, config=config_alvo)
        #     fit_instructions.append((c, fit_ins_alvo))
        
        # for c in conjunto_gen:
        #     fit_ins_gen = FitIns(parameters=self.parameters_gen, config=config_gen)
        #     fit_instructions.append((c, fit_ins_gen))

        # Return client/config pairs
        return fit_instructions


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []
        
        # Parameters and config
        config = {"round": server_round}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]


    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # results_alvo = [res for res in results if res[1].metrics["modelo"] == "alvo"]
        # results_gen = [res for res in results if res[1].metrics["modelo"] == "gen"]

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
            # aggregated_ndarrays_alvo = aggregate_inplace(results_alvo)
            # aggregated_ndarrays_gen = aggregate_inplace(results_gen)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        # parameters_aggregated_alvo = ndarrays_to_parameters(aggregated_ndarrays_alvo)
        # parameters_aggregated_gen = ndarrays_to_parameters(aggregated_ndarrays_gen)

        self.parameters = parameters_aggregated
        # self.parameters_alvo = parameters_aggregated_alvo
        # self.parameters_gen = parameters_aggregated_gen

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        avg_net_loss = weighted_loss_avg(
            [
                (fit_res.num_examples, fit_res.metrics["train_loss"])
                for _, fit_res in results
            ]
        )
        self.metrics_dict["net_loss_epoch"].append(avg_net_loss)

        net_times = [
            fit_res.metrics["tempo_treino_alvo"]
            for _, fit_res in results
            if "tempo_treino_alvo" in fit_res.metrics
        ]
        if net_times:
            self.metrics_dict["max_net_time"].append(max(net_times))

        if parameters_aggregated is not None:
            if self.dataset == "mnist":
                net = Net().to(self.device)
            elif self.dataset == "cifar10":
                net = Net_CIFAR().to(self.device)
            else:
                raise ValueError(f"Dataset {self.dataset} nao identificado. Deveria ser 'mnist' ou 'cifar10'")

            set_weights(net, aggregated_ndarrays)

            if self.valloader is not None:
                criterion = torch.nn.CrossEntropyLoss()
                correct, loss = 0, 0.0
                net_global_eval_start_time = time.time()
                net.eval()
                with torch.no_grad():
                    for batch in self.valloader:
                        images = batch[self.image].to(self.device)
                        labels = batch["label"].to(self.device)
                        outputs = net(images)
                        loss += criterion(outputs, labels).item()
                        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                accuracy = correct / len(self.valloader.dataset)
                self.metrics_dict["global_net_eval_time"].append(time.time() - net_global_eval_start_time)
                self.metrics_dict["val_acc"].append(accuracy)
                print(f"Acurácia global no test set: {accuracy:.4f}")

            if server_round % self.model_save_interval == 0:
                net_path = f"{self.folder}/modelo_epoch_{server_round}.pth"
                torch.save(net.state_dict(), net_path)
                print(f"Modelo salvo em {net_path}")

        # if parameters_aggregated_gen is not None and self.agg == "full":
        #     ndarrays = parameters_to_ndarrays(parameters_aggregated_gen)
        #     # Cria uma instância do modelo
        #     net = CGAN(dataset=self.dataset,
        #                  img_size=self.img_size,
        #                  latent_dim=self.latent_dim)
        #     # Define os pesos do modelo
        #     set_weights(net, ndarrays)
        #     # Salva o modelo no disco com o nome específico do dataset
        #     model_path = f"modelo_gen_round_{server_round}_mnist.pt"
        #     torch.save(net.state_dict(), model_path)
        #     print(f"Modelo salvo em {model_path}")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        local_test_times = [evaluate_res.metrics["local_test_time"] for _, evaluate_res in results]
        self.metrics_dict["max_local_test_time"].append(max(local_test_times))

        accuracy_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.metrics["local_accuracy"])
                for _, evaluate_res in results
            ]
        )

        self.metrics_dict["local_acc_epoch"].append(accuracy_aggregated)

        loss_file = f"{self.folder}/losses.txt"
        with open(loss_file, "a") as f:
            f.write(f"Rodada {server_round}, Perda: {loss_aggregated}, Acuracia: {accuracy_aggregated}\n")
        print(f"Perda da rodada {server_round} salva em {loss_file}")

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {"loss": loss_aggregated, "accuracy": accuracy_aggregated}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        round_time = time.time() - self.init_round_time
        self.metrics_dict["time_round"].append(round_time)

        self._save_metrics()

        return loss_aggregated, metrics_aggregated
