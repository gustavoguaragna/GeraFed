from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context


# Estratégia de agregação: Média dos pesos
strategy = FedAvg()

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""

    # Lê a configuração
    num_rounds = context.run_config["num_rodadas"]

    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


# Configurar e iniciar o servidor
app = ServerApp(server_fn=server_fn)
