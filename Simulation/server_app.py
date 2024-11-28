# server_app.py

"""fedvaeexample: A Flower / PyTorch app for Federated Variational Autoencoder."""

from fedvaeexample.task import Net, get_weights, set_weights
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import torch
import os  # Importar para verificar a existência de arquivos

class FedAvg_Save(FedAvg):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def aggregate_fit(self, server_round, results, failures):
        # Agrega os resultados da rodada
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
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
        model = Net(dataset=self.dataset)
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
    num_rounds = context.run_config["num-server-rounds"]
    dataset = context.run_config["dataset"]  # Novo parâmetro

    # Define o caminho do checkpoint inicial (opcional)
    initial_model_path = f"model_round_0_{dataset}.pt"  # Ajuste conforme necessário

    if os.path.exists(initial_model_path):
        # Carrega o modelo existente
        model = Net(dataset=dataset)
        model.load_state_dict(torch.load(initial_model_path))
        ndarrays = get_weights(model)
        print(f"Modelo carregado a partir de {initial_model_path}")
    else:
        # Inicializa o modelo a partir do início
        ndarrays = get_weights(Net(dataset=dataset))
        print(f"Inicializando modelo do zero para dataset {dataset}")

    parameters = ndarrays_to_parameters(ndarrays)

    # Define a estratégia usando a estratégia personalizada
    strategy = FedAvg_Save(initial_parameters=parameters, dataset=dataset)
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Cria a ServerApp
app = ServerApp(server_fn=server_fn)
