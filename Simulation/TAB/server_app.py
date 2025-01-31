"""xgboost_quickstart: A Flower / XGBoost app."""

from typing import Dict

from flwr.common import Context, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from Simulation.TAB.strategy import FedXgbBagging_Save
from flwr.server.strategy import FedXgbBagging


def evaluate_metrics_aggregation(eval_metrics, server_round):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    acc_aggregated = (
        sum([metrics["Accuracy"] * num for num, metrics in eval_metrics]) / total_num
    )
    f1_aggregated = (
        sum([metrics["F1_score"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated, "Accuracy": acc_aggregated, "F1_score": f1_aggregated}
    
    loss_file = f"losses_tab.txt"
    with open(loss_file, "a") as f:
            f.write(f"Rodada {server_round}, F1_score: {f1_aggregated}, AUC: {auc_aggregated}, Acuracia: {acc_aggregated}\n")
    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num_rodadas"]
    fraction_fit = context.run_config["fraction_fit_alvo"]
    fraction_evaluate = context.run_config["fraction_fit_alvo"]

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    # Define strategy
    strategy = FedXgbBagging_Save(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)
