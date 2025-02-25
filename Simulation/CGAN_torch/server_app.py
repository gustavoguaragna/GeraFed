"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from Simulation.CGAN_torch.task import CGAN, get_weights, set_weights
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import torch
import os  # Importar para verificar a existência de arquivos
import random
import numpy as np
from flwr.common import FitIns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Para garantir determinismo total em operações com CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FedAvg_Save(FedAvg):
    def __init__(self, dataset, **kwargs):
        self.agg = kwargs.pop("agg")
        super().__init__(**kwargs)
        self.dataset = dataset

    def configure_fit(
        self, server_round: int, parameters, client_manager):
        """Configure the next round of training."""
        config = {"server_round": server_round}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        # Agrega os resultados da rodada
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None and self.agg == "full":
            # Salva o modelo após a agregação
            self.save_model(aggregated_parameters, server_round)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        # Agrega os resultados da avaliação
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Salva a perda após a avaliação
        self.save_loss(aggregated_loss, server_round)

        return aggregated_loss, aggregated_metrics

    def save_model(self, parameters, server_round):
        # Converte os parâmetros para ndarrays
        ndarrays = parameters_to_ndarrays(parameters)
        # Cria uma instância do modelo
        model = CGAN(dataset=self.dataset)
        # Define os pesos do modelo
        set_weights(model, ndarrays)
        # Salva o modelo no disco com o nome específico do dataset
        model_path = f"model_round_{server_round}_{self.dataset}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Modelo salvo em {model_path}")

    def save_loss(self, loss, server_round):
        # Salva a perda em um arquivo de texto específico do dataset
        loss_file = f"losses_{self.dataset}.txt"
        with open(loss_file, "a") as f:
            f.write(f"Rodada {server_round}, Perda: {loss}\n")
        print(f"Perda da rodada {server_round} salva em {loss_file}")


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""

    # Lê a configuração
    num_rounds = context.run_config["num_rodadas"]
    dataset = context.run_config["dataset"]
    agg = context.run_config["agg"]

    # Define o caminho do checkpoint inicial (opcional)
    initial_model_path = f"Imagens Testes/FULL_FEDAVG/epochs10/model_round_10_mnist.pt"  # Ajuste conforme necessário

    if os.path.exists(initial_model_path):
        # Carrega o modelo existente
        model = CGAN(dataset=dataset)
        model.load_state_dict(torch.load(initial_model_path))
        ndarrays = get_weights(model)
        print(f"Modelo carregado a partir de {initial_model_path}")
    else:
        # Inicializa o modelo a partir do início
        ndarrays = get_weights(CGAN(dataset=dataset))
        print(f"Inicializando modelo do zero para dataset {dataset}")

    parameters = ndarrays_to_parameters(ndarrays)

    # Define a estratégia usando a estratégia personalizada
    strategy = FedAvg_Save(initial_parameters=parameters, dataset=dataset, agg=agg)
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Cria a ServerApp
app = ServerApp(server_fn=server_fn)
