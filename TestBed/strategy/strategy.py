from typing import Dict, List, Optional, Tuple, Union
from flwr.common import Scalar, EvaluateRes, Parameters, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import flwr as fl
import logging
from prometheus_client import Gauge
import time 

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


class FedCustom(fl.server.strategy.FedAvg):
    def __init__(
        self,
        accuracy_gauge: Gauge = None,
        loss_gauge: Gauge = None, 
        latency_gauge: Gauge = None, 
        log_file: str = "metrics.log", 
        patience: int = 3, ## Add patience for early stopping 
        min_available_clients: int = 5, ## Add min clients to start round
        *args, **kwargs ## Add log_file
    ):
        super().__init__(*args, **kwargs)

        self.accuracy_gauge = accuracy_gauge
        self.loss_gauge = loss_gauge
        self.latency_gauge = latency_gauge
        ## Add log_file and early stopping
        self.log_file = log_file
        self.patience = patience 
        self.min_available_clients = min_available_clients
        self.no_improvement_rounds = 0
        self.best_loss = float("inf")
        self.early_stop = False

        ## Create the log file and write the header
        with open(self.log_file, "w") as f:
            f.write("Round, Loss, Accuracy, F1, Computation Time, Communication Time Upstream, Total Time\n")

    ## configure_fit
    # def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager):
    #     return super().configure_fit(server_round, parameters, client_manager)

    #def __repr__(self) -> str:
    #    return "FedCustom"

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        self.upstream_end_time = time.time()
        
        # Collect min start training time and max end training time
        start_times = [res.metrics["start_time"] for _, res in results]
        end_times = [res.metrics["end_time"] for _, res in results]
        self.computation_start_time = min(start_times)
        self.computation_end_time = max(end_times)
        self.upstream_start_time = self.computation_end_time

        # Add mean computation time to fit metrics
        self.computation_time = self.computation_end_time - self.computation_start_time
        aggregated_params, fit_metrics = super().aggregate_fit(server_round, results, failures)
        fit_metrics["computation time"] = self.computation_time
        
        return aggregated_params, fit_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses and accuracy using weighted average."""

        if not results:
            return None, {}

        # Calculate weighted average for loss using the provided function
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Calculate weighted average for accuracy
        accuracies = [
            evaluate_res.metrics["accuracy"] * evaluate_res.num_examples
            for _, evaluate_res in results
        ]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        accuracy_aggregated = (
            sum(accuracies) / sum(examples) if sum(examples) != 0 else 0
        )

        ## Calculate weighted average for F1 score
        f1_scores = [
            evaluate_res.metrics["f1"] * evaluate_res.num_examples
            for _, evaluate_res in results
        ]
        f1_aggregated = (
            sum(f1_scores) / sum(examples) if sum(examples) != 0 else 0
        )

        # Update the Prometheus gauges with the latest aggregated accuracy and loss values
        self.accuracy_gauge.set(accuracy_aggregated)
        self.loss_gauge.set(loss_aggregated)

        ## Compute communication and total time
        communication_time_upstream = self.upstream_end_time - self.upstream_start_time
        total_time = self.upstream_end_time - self.computation_start_time
        self.latency_gauge.labels(round=str(server_round)).set(total_time)  # Include round as a label
       
        ## Log the metrics
        with open(self.log_file, "a") as f:
            f.write(f"{server_round}, {loss_aggregated}, {accuracy_aggregated}, {f1_aggregated}, {self.computation_time}, {communication_time_upstream}, {total_time}\n")

        logger.info(f"Round {server_round} - Loss: {loss_aggregated}, Accuracy: {accuracy_aggregated}, F1: {f1_aggregated}")
        logger.info(f"Computation Time: {self.computation_time}, Communication Time Upstream: {communication_time_upstream}, Total Time: {total_time}")

        metrics_aggregated = {"loss": loss_aggregated, "accuracy": accuracy_aggregated, "f1": f1_aggregated}

         # Early stopping logic
        if loss_aggregated < self.best_loss:
            self.best_loss = loss_aggregated
            self.no_improvement_rounds = 0
        else:
            self.no_improvement_rounds += 1
            if self.no_improvement_rounds >= self.patience:
                logger.info(f"Early stopping triggered. No improvement in loss for {self.patience} rounds.")
                self.early_stop = True

        return loss_aggregated, metrics_aggregated
    
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        if self.early_stop:
            return []
        else:
            clients = client_manager.sample(
                num_clients=self.min_available_clients  # Modified this line
            )
            return [(client, fl.common.FitIns(parameters, {})) for client in clients]
            #return super().configure_fit(server_round, parameters, client_manager)
        
    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        if self.early_stop:
            return []
        else:
            clients = client_manager.sample(
                num_clients=self.min_available_clients  # Modified this line
            )
            return [(client, fl.common.EvaluateIns(parameters, {})) for client in clients]
            #return super().configure_evaluate(server_round, parameters, client_manager)

    def stop_condition(self) -> bool:
        return self.early_stop