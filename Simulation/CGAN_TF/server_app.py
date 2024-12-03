# server_app.py

"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from Simulation.task import define_generator, define_discriminator, get_weights
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import tensorflow as tf
import os  # Importar para verificar a existência de arquivos
import logging

class FedAvg_Save(FedAvg):
    def __init__(self, dataset, num_clientes, classes, tam_batch, tam_img, tam_ruido, **kwargs):
        super().__init__(min_available_clients=num_clientes, **kwargs)
        self.dataset = dataset
        self.classes = classes
        self.tam_batch = tam_batch
        self.tam_img = tam_img
        self.tam_ruido = tam_ruido

    def aggregate_fit(self, server_round, results, failures):
        # Agrega os resultados da rodada
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Salva o modelo após a agregação
            if (server_round % 5) == 0:
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
        weights = parameters_to_ndarrays(parameters)

        # Reconstruct the models
        generator = define_generator(noise_dim=self.tam_ruido,
                                     classes=self.classes, 
                                     batch_size=self.tam_batch,
                                     img_size=self.tam_img)
        discriminator = define_discriminator(img_size=self.tam_img,
                                             classes=self.classes,
                                             batch_size=self.tam_batch)

        # Set the weights
        generator_num_weights = len(generator.get_weights())
        generator_weights = weights[:generator_num_weights]
        discriminator_weights = weights[generator_num_weights:]

        generator.set_weights(generator_weights)
        discriminator.set_weights(discriminator_weights)

        generator.save(f'generator_model_round{server_round}.h5')       # Saves architecture and weights
        discriminator.save(f'discriminator_model_round{server_round}.h5')
        print(f"Modelo salvo")

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
    num_clientes = context.run_config["num_clientes"]
    classes = context.run_config["classes"]
    tam_img = context.run_config["tam_img"]
    tam_batch = context.run_config["tam_batch"]
    tam_ruido = context.run_config["tam_ruido"]
    #channels = 1


    # Torne os logs do TensorFlow menos detalhados
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    #os.environ[TF_FORCE_GPU_ALLOW_GROWTH]="1"


    # Configure logging to display client-side information
    logging.basicConfig(level=logging.DEBUG)

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
    strategy = FedAvg_Save(
                            initial_parameters=parameters,
                            dataset=dataset,
                            num_clientes=num_clientes,
                            tam_img=tam_img,
                            tam_batch=tam_batch,
                            tam_ruido=tam_ruido,
                            classes=classes
                           )
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Cria a ServerApp
app = ServerApp(server_fn=server_fn)
