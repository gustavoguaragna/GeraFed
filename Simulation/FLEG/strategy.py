"""FLEG: um framework para balancear dados heterogêneos em aprendizado federado, com precupações com a privacidade."""

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

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg

from Simulation.FLEG.task import (
    ClassifierHead1, ClassifierHead2, ClassifierHead3, ClassifierHead4,
    ClassifierHead1_Cifar, ClassifierHead2_Cifar, ClassifierHead3_Cifar, ClassifierHead4_Cifar,
    Net,
    Net_Cifar,
    EmbeddingGAN1, EmbeddingGAN2, EmbeddingGAN3, EmbeddingGAN4,
    EmbeddingGAN1_Cifar, EmbeddingGAN2_Cifar, EmbeddingGAN3_Cifar, EmbeddingGAN4_Cifar,
    FeatureExtractor1, FeatureExtractor2, FeatureExtractor3, FeatureExtractor4,
    FeatureExtractor1_Cifar, FeatureExtractor2_Cifar, FeatureExtractor3_Cifar, FeatureExtractor4_Cifar,
    GeneratedAssetDataset,
    set_weights,
    train_G,
    get_weights,
)
import torch
import pickle
import time
import json
import numpy as np

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
    Setting `min_available_clients` lower than `min_fit_clients` or
    `min_evaluate_clients` can cause the server to fail when there are too few clients
    connected to the server. `min_available_clients` must be set to a value larger
    than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
    """
class FLEG(Strategy):

    """FLEG Strategy.

    Implementation based on Aprendizado Federado com Geração de Embeddings para Controle da Heterogeneidade Estatística

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
        fraction_fit_alvo: float = 1.0,
        fraction_fit_gen: float = 1.0,
        fraction_evaluate_alvo: float = 1.0,
        fraction_evaluate_gen: float = 1.0,
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
        initial_parameters_alvo: Optional[Parameters] = None,
        initial_parameters_gen: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        dataset: str = "mnist",
        img_size: int = 28,
        latent_dim: int = 128,
        gan_arq: str = "f2u_gan",
        gen_epochs: int = 2,
        teste: bool = False,
        lr_gen: float = 0.0002,
        folder: str = ".",
        num_chunks: int = 1,
        patience: int = 10,
        gen,
        optimG_state_dict = None,
        continue_epoch:int = 0,
        valloader
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit_alvo = fraction_fit_alvo
        self.fraction_fit_gen = fraction_fit_gen
        self.fraction_evaluate_alvo = fraction_evaluate_alvo
        self.fraction_evaluate_gen = fraction_evaluate_gen
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters_alvo = initial_parameters_alvo
        self.parameters_alvo = initial_parameters_alvo
        self.parameters_gen = initial_parameters_gen
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.dataset = dataset
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.gan_arq = gan_arq
        self.gen_epochs = gen_epochs
        self.teste = teste
        self.lr_gen = lr_gen
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.folder = folder
        self.gen = gen.to(self.device)
        self.optimG_state_dict = optimG_state_dict if optimG_state_dict else None
        self.num_chunks = num_chunks
        self.freq_save = self.num_chunks // 10 if self.num_chunks >= 10 else 1
        self.continue_epoch = continue_epoch
        self.lvl = 0,
        self.best_accuracy = 0,
        self.patience = patience
        self.valloader = valloader
        self.size_classifier = {
            'cifar10': {1: 0.25, 2: 0.248, 3: 0.238, 4: 0.045, 5: 0.004},
            'mnist': {1: 0.18, 2: 0.179, 3: 0.169, 4: 0.045, 5: 0.004}
        }
        self.size_disc = {
            'cifar10': {1: 18.12, 2: 3.79, 3: 0.79, 4: 0.23},
            'mnist': {1: 5.69, 2: 1.08, 3: 0.8, 4: 0.23}
        }
        self.embds = GeneratedAssetDataset(generator=gen, num_samples=0)
        self.newlvl= True
        self.training_gan = False
        self.metrics_dict = {
                        "g_loss_chunk": [],
                        "d_loss_chunk": [],
                        "net_loss_epoch": [],
                        "local_acc_epoch": [],
                        "val_acc_epoch": [],
                        "time_chunk_gan": [],
                        "time_epoch": [],
                        "time_epoch_gan": [],
                        "net_time": [],
                        "global_net_eval_time": [],
                        "disc_time": [],
                        "gen_time": [],
                        "img_syn_time": [],
                        "test_time": [],
                        "local_test_time": [],
                        "level_time": [],
                        "MB_transmission": [],
                        "traffic_cost_classifier": [],
                        "traffic_cost_embeddings": [],
                        "traffic_cost_discriminator": [],
                        "accuracy_transition": [],
                    },
        self.fe= {
            "mnist": {
                1: FeatureExtractor1, 2: FeatureExtractor2,
                3: FeatureExtractor3, 4: FeatureExtractor4,
            },
            "cifar10": {
                1: FeatureExtractor1_Cifar, 2: FeatureExtractor2_Cifar,
                3: FeatureExtractor3_Cifar, 4: FeatureExtractor4_Cifar,
            },
        },
        self.ch = {
            "mnist": {
                1: ClassifierHead1, 2: ClassifierHead2,
                3: ClassifierHead3, 4: ClassifierHead4,
            },
            "cifar10": {
                1: ClassifierHead1_Cifar, 2: ClassifierHead2_Cifar,
                3: ClassifierHead3_Cifar, 4: ClassifierHead4_Cifar,
            },
        }
        self.gan = {
            "mnist": {
                1: EmbeddingGAN1, 2: EmbeddingGAN2, 3: EmbeddingGAN3, 4: EmbeddingGAN4,
            },
            "cifar10": {
                1: EmbeddingGAN1_Cifar, 2: EmbeddingGAN2_Cifar,
                3: EmbeddingGAN3_Cifar, 4: EmbeddingGAN4_Cifar,
            },
        }


            
    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"GeraFed(accept_failures={self.accept_failures})"
        return rep


    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit_alvo)
        return max(num_clients, self.min_fit_clients), self.min_available_clients


    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate_alvo)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters_alvo
        self.initial_parameters_alvo = None  # Don't keep initial parameters in memory
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
        if self.newlvl:
            self.init_lvl_time = time.time()
            self.epoch = 0
            self.round = 0
        
        self.init_round_time = time.time()

        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_instructions = []
        config = {"level": self.lvl}
        
        if self.training_gan:
            config["model"] = "gan"
            config["round"] = self.round
            fit_ins = FitIns(parameters=self.parameters_gen, config=config)
            self.round += 1
            if self.round % self.num_chunks == 0:
                self.epoch += 1
        else:
            config["model"] = "classifier"
            if self.newlvl:
                config["embds"] = pickle.dumps(self.embds)
                self.newlvl = False
            if self.epochs_no_improve == 0:
                config["best_model"] = True
            fit_ins = FitIns(parameters=self.parameters_alvo, config=config)

        for c in clients:
            fit_instructions.append((c, fit_ins))

        # Return client/config pairs
        return fit_instructions


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate_alvo == 0.0:
            return []

        if self.training_gan:
            round_time = time.time() - self.init_round_time
            self.metrics_dict["time_chunk"].append(round_time)
            return None

        # Parameters and config
        config = {"round": server_round, "level": self.lvl}
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

        if self.training_gan:
             # Define os pesos do modelo
            disc_ndarrays = {fit_res.metrics["cid"]: parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results}
            gan_class = self.gen.__class__
            self.discs = [gan_class().to(self.device) for _ in range(len(disc_ndarrays))]
            for i, disc in enumerate(self.discs):
                set_weights(disc, disc_ndarrays[i])

            gen_time_start = time.time()
            g_loss, self.optimG_state_dict = train_G(
            net=self.gen,
            discs=self.discs,
            epochs=self.gen_epochs,
            lr=self.lr_gen,
            device=self.device,
            latent_dim=self.latent_dim,
            batch_size=1,
            optim_state_dict=self.optimG_state_dict
            )
            gen_time = time.time() - gen_time_start

            self.metrics_dict["g_loss_chunk"].append(g_loss)
            self.metrics_dict["gen_time_chunk"].append(gen_time)
            
            avg_d_loss = weighted_loss_avg(
                [
                    (fit_res.num_examples, fit_res.metrics["avg_d_loss"])
                    for _, fit_res in results
                ]
            )

            self.metrics_dict["d_loss_chunks"].append(avg_d_loss)

            disc_times = [fit_res.metrics["tempo_treino_disc"] for _, fit_res in results]

            max_disc_time = max(disc_times)
           
            self.metrics_dict["disc_time"].append(max_disc_time)

            if self.epoch == 25:
                checkpoint = {
                    'global_net_state_dict': self.global_net.state_dict(),
                    'generated_dataset' : self.embds,
                }
                checkpoint_file = f"{self.folder}/checkpoint_level{self.lvl}.pth"
                torch.save(checkpoint, checkpoint_file)
                print(f"Global net saved to {checkpoint_file}")


                self.metrics_dict["local_acc_epoch"].append(accuracy_aggregated)
                epoch_time = time.time() - self.init_epoch_time
                self.metrics_dict["time_epoch_gan"].append(epoch_time)
                round_time = time.time() - self.init_round_time
                self.metrics_dict["time_chunk"].append(round_time)

                metrics_filename = f"{self.folder}/metrics.json"

                try:
                    with open(metrics_filename, 'w', encoding='utf-8') as f:
                        json.dump(existing_metrics, f, ensure_ascii=False, indent=4) # indent makes it readable
                    print(f"Losses dict successfully saved to {metrics_filename}")
                except Exception as e:
                    print(f"Error saving losses dict to JSON: {e}")


            # Aggregate custom metrics if aggregation fn was provided
            


            # for key, val in list(self.metrics_dict.items()):
            #     if key.endswith("time_chunk") and key != "time_chunk":
            #         epoch_key = key.replace("chunk", "epoch")
            #         sum_value = sum(val)
            #         self.metrics_dict[epoch_key].append(sum_value)

            #     if key.endswith("epoch") or "time" in key:
            #         continue

            #     epoch_key = key.replace('_chunk', '_epoch')

            #     mean_value = sum(val) / len(val)

            #     self.metrics_dict[epoch_key].append(mean_value)
                

            # try:
            #     with open(metrics_filename, 'r', encoding='utf-8') as f:
            #         # Load the dictionary from the JSON file
            #         existing_metrics = json.load(f)
            # except (FileNotFoundError, json.JSONDecodeError):
            #     # If the file is not found or is not valid JSON, we start fresh
            #     existing_metrics = {}
            #     print("Metrics file not found or invalid. A new one will be created.")
            #     print(metrics_filename)

            # for key, new_values in self.metrics_dict.items():
            #     # Get the list that already exists for this key, or an empty list if the key is new
            #     existing_list = existing_metrics.get(key, [])
                
            #     # Use .extend() to add the new items to the existing list
            #     existing_list.extend(new_values)
                
            #     # Update the dictionary with the newly extended list
            #     existing_metrics[key] = existing_list
        
        else:
            if self.inplace:
                # Does in-place weighted average of results
                aggregated_ndarrays= aggregate_inplace(results)
                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
                self.parameters_alvo = parameters_aggregated

            else:
                # Convert results
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results
                ]
                aggregated_ndarrays = aggregate(weights_results)

                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

                self.parameters_alvo = parameters_aggregated

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

            net_times = [fit_res.metrics["tempo_treino_alvo"] for _, fit_res in results]

            max_net_time = max(net_times)
           
            self.metrics_dict["net_time"].append(max_net_time)

            if parameters_aggregated is not None:
                # Salva o modelo após a agregação
                # Cria uma instância do modelo
                if self.dataset == "mnist":
                    self.global_net = Net()
                    self.best_model = Net()
                    image = "image"
                elif self.dataset == "cifar10":
                    self.global_net = Net_Cifar()
                    self.best_model = Net_Cifar()
                    image = "img"
                else:
                    raise ValueError(f"Dataset {self.dataset} nao identificado. Deveria ser 'mnist' ou 'cifar10'")
                # Define os pesos do modelo
                set_weights(self.global_net, aggregated_ndarrays)

                criterion = torch.nn.CrossEntropyLoss()

                # Avaliacao global
                correct, loss = 0, 0.0
                net_global_eval_start_time = time.time()
                with torch.no_grad():
                    for batch in self.valloader:
                        images = batch[image].to(self.device)
                        labels = batch["label"].to(self.device)
                        outputs = self.global_net(images)
                        loss += criterion(outputs, labels).item()
                        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                accuracy = correct / len(self.valloader.dataset)

                self.metrics_dict["global_net_eval_time"].append(time.time() - net_global_eval_start_time)
                self.metrics_dict["val_acc"].append(accuracy)

                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.epochs_no_improve = 0
                    self.best_model.load_state_dict(self.global_net.state_dict())
                else:
                    self.epochs_no_improve += 1
                    print(f"Sem melhorias por {self.epochs_no_improve} épocas. Melhor acurácia: {best_accuracy:.4f}")
                
                if self.epochs_no_improve > self.patience:
                    self.epochs_no_improve = 0
                    self.training_gan = True
                    self.global_net.load_state_dict(self.best_model.state_dict())

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

        local_test_times = [evaluate_res.metrics["local_test_time"]/len(results) for _, evaluate_res in results]
        self.metrics_dict["local_test_time"].append(sum(local_test_times))
        # accuracies = [
        #     evaluate_res.metrics["accuracy"] * evaluate_res.num_examples
        #     for _, evaluate_res in results
        # ]
        # examples = [evaluate_res.num_examples for _, evaluate_res in results]
        # accuracy_aggregated = (
        #     sum(accuracies) / sum(examples) if sum(examples) != 0 else 0
        # )
        # accuracy_aggregated = weighted_loss_avg(
        #     [
        #         (evaluate_res.num_examples, evaluate_res.metrics["local accuracy"])
        #         for _, evaluate_res in results
        #     ]
        # )

        return loss_aggregated, metrics_aggregated

