from flwr.server.strategy import Strategy
from logging import WARNING
from typing import Callable, Optional, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArray,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
# from collections import Counter
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import numpy as np
# import random
from functools import partial, reduce
#from Simulation.task import Net, set_weights
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
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
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
        agg: str = "full"
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
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters




    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
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

        fit_instructions = []
        config = {}

        for c in clients:
            fit_ins = FitIns(parameters=parameters, config=config)
            fit_instructions.append((c, fit_ins))

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
        config = {}
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

        def aggregate_inplace(results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
            """Compute in-place weighted average where each parameter is weighted by its normalized absolute proportion and the number of examples from each client."""
            # Converter os parâmetros de cada cliente para ndarrays
            all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            
            # Calcular a soma total do valor absoluto de cada parâmetro entre os clientes
            total_sums = [
                sum(np.abs(layer_params)) for layer_params in zip(*all_params)
            ]

            # Calcular o total de exemplos
            num_examples_total = sum(fit_res.num_examples for _, fit_res in results)

            def _try_inplace(
                x: NDArray, y: Union[NDArray, np.float64], np_binary_op: np.ufunc
            ) -> NDArray:
                return (
                    np_binary_op(x, y, out=x)
                    if np.can_cast(y, x.dtype, casting="same_kind")
                    else np_binary_op(x, np.array(y, x.dtype), out=x)
                )

            # Inicializar os parâmetros agregados com zeros
            params = [np.zeros_like(layer) for layer in all_params[0]]

            # Agregar ponderando pela proporção normalizada do valor absoluto de cada parâmetro e pelo número de exemplos
            for i, (_, fit_res) in enumerate(results):
                client_params = parameters_to_ndarrays(fit_res.parameters)
                example_weight = fit_res.num_examples / num_examples_total  # Ponderação pelo número de exemplos
                for j, (param, total_sum) in enumerate(zip(client_params, total_sums)):
                    # Evitar divisão por zero
                    proportion = np.abs(param) / total_sum if np.all(total_sum != 0) else np.zeros_like(param)
                    # Normalizar a proporção com softmax para suavizar a influência
                    # Somar ponderadamente com normalização
                    weighted_param = param * example_weight  # Combina ambas as ponderações com suavização
                    params[j] = _try_inplace(params[j], weighted_param, np.add)

            return params


        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        results = [res for res in results]

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        
        param_ndarray = parameters_to_ndarrays(self.parameters)
        #print(f'SELF.PARAM: {param_ndarray[9][0]} GRAD ACU: {aggregated_ndarrays[9][0]}')
    
        aggregated_ndarrays = [w - 0.001*g for w, g in zip(param_ndarray, aggregated_ndarrays)]
        #print(f'PESO ATUALIZADO: {aggregated_ndarrays[9][0]}')

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        # parameters_aggregated_ndarray = parameters_to_ndarrays(parameters_aggregated)
        # print(f'PARAMETRO: {parameters_aggregated_ndarray[9][0]}')

        self.parameters = parameters_aggregated

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # if parameters_aggregated is not None:
        #     # Salva o modelo após a agregação
        #     ndarrays = parameters_to_ndarrays(parameters_aggregated)
        #     # Cria uma instância do modelo
        #     model = Net()
        #     # Define os pesos do modelo
        #     set_weights(model, ndarrays)
        #     # Salva o modelo no disco com o nome específico do dataset
        #     model_path = f"modelo_alvo_round_{server_round}_mnist.pt"
        #     torch.save(model.state_dict(), model_path)
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

        accuracies = [
            evaluate_res.metrics["accuracy"] * evaluate_res.num_examples
            for _, evaluate_res in results
        ]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        accuracy_aggregated = (
            sum(accuracies) / sum(examples) if sum(examples) != 0 else 0
        )

        loss_file = f"losses.txt"
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

        return loss_aggregated, metrics_aggregated

