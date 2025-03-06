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

from collections import Counter

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
import random

from Simulation.task import Net, CGAN, set_weights
import torch

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
        latent_dim: int = 100,
        client_counter: Counter,
        agg: str = "full",
        model: str = "both",
        fid: bool = False
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
        self.client_counter = client_counter
        self.agg = agg
        self.model = model
        self.fid = fid

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

        if self.model == "both":
            print("MODEL BOTH")
            sorted_clients = sorted(clients, key=lambda c: self.client_counter[c])
            metade = len(clients) // 2
            conjunto_gen = sorted_clients[:metade]
            conjunto_alvo = sorted_clients[metade:]
            # Calcula FID
            print(self.fid)
            if self.fid:
                from Simulation.task import InceptionV3, GeneratedDataset, ImagePathDataset, select_samples_per_class
                import numpy as np
                import os
                from scipy import linalg
                import time
                from tqdm import tqdm
                import torchvision.transforms as transforms
                import torchvision.datasets as datasets
                start_time = time.time()
                dims = 2048
                device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
                model = InceptionV3([block_idx]).to(device)
                model.eval()
                ndarrays = parameters_to_ndarrays(self.parameters_gen)
                # Cria uma instância do modelo
                cgan = CGAN(dataset=self.dataset,
                            img_size=self.img_size,
                            latent_dim=self.latent_dim)
                # Define os pesos do modelo
                set_weights(cgan, ndarrays)
                # Create the dataset and dataloader
                generated_dataset = GeneratedDataset(generator=cgan, num_samples=2050, latent_dim=self.latent_dim, num_classes=10, device="cpu")
                gen_dataset = generated_dataset.images
                for c in gen_dataset.keys():
                    gen_dataset[c] = (gen_dataset[c] + 1) / 2 #intervalo entre 0 e 1
                    gen_dataset[c] = gen_dataset[c].repeat(1, 3, 1, 1) # 3 canais como na rede inception

                try:
                    num_cpus = len(os.sched_getaffinity(0))
                except AttributeError:
                    # os.sched_getaffinity is not available under Windows, use
                    # os.cpu_count instead (which may not return the *available* number
                    # of CPUs).
                    num_cpus = os.cpu_count()
                num_workers = min(8, num_cpus)

                dataloaders = [torch.utils.data.DataLoader(gen_dataset[c], batch_size=50, num_workers=num_workers, shuffle=False) for c in range(10)]

                mus_gen = []
                sigmas_gen = []

                for c in range(10):
                    pred_arr = np.empty((len(gen_dataset[c]), dims))
                    start_idx = 0
                    for batch in tqdm(dataloaders[c]):
                            batch = batch.to(device)

                            with torch.no_grad():
                                pred = model(batch)[0]

                            # If model output is not scalar, apply global spatial average pooling.
                            # This happens if you choose a dimensionality not equal 2048.
                            if pred.size(2) != 1 or pred.size(3) != 1:
                                pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

                            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

                            pred_arr[start_idx : start_idx + pred.shape[0]] = pred

                            start_idx += pred.shape[0]

                    mus_gen.append(np.mean(pred_arr, axis=0))
                    sigmas_gen.append(np.cov(pred_arr, rowvar=False))

                # Carrega MNIST test para calculo dist real
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                testset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
                img_reais = select_samples_per_class(testset, 800)
                dataloaders = [torch.utils.data.DataLoader(img_reais[c], batch_size=50, num_workers=num_workers, shuffle=False) for c in range(10)]

                mus_real = []
                sigmas_real = []
                for c in range(10):
                    model = InceptionV3([block_idx]).to(device)
                    model.eval()
                    pred_arr = np.empty((len(img_reais[0]), dims))
                    start_idx = 0
                    for batch in tqdm(dataloaders[c]):
                            batch = batch.to(device)

                            with torch.no_grad():
                                pred = model(batch)[0]

                            # If model output is not scalar, apply global spatial average pooling.
                            # This happens if you choose a dimensionality not equal 2048.
                            if pred.size(2) != 1 or pred.size(3) != 1:
                                pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

                            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

                            pred_arr[start_idx : start_idx + pred.shape[0]] = pred

                            start_idx += pred.shape[0]  # Corrected the operator to +=
                    mus_real.append(np.mean(pred_arr, axis=0))
                    sigmas_real.append(np.cov(pred_arr, rowvar=False))

                mus_gen = [np.atleast_1d(mu_gen) for mu_gen in mus_gen]
                mus_real = [np.atleast_1d(mu_real) for mu_real in mus_real]

                sigmas_gen = [np.atleast_2d(sigma_gen) for sigma_gen in sigmas_gen]
                sigmas_real = [np.atleast_2d(sigma_real) for sigma_real in sigmas_real]

                for mu_gen, mu_real, sigma_gen, sigma_real in zip(mus_gen, mus_real, sigmas_gen, sigmas_real):
                    assert (
                        mu_gen.shape == mu_real.shape
                    ), "Training and test mean vectors have different lengths"
                    assert (
                        sigma_gen.shape == sigma_real.shape
                    ), "Training and test covariances have different dimensions"

                diffs = [mu_gen - mu_real for mu_gen, mu_real in zip(mus_gen, mus_real)]

                # Product might be almost singular
                covmeans = [linalg.sqrtm(sigmas_gen.dot(sigmas_real), disp=False)[0] for sigmas_gen, sigmas_real in zip(sigmas_gen, sigmas_real)]
                for covmean, sigma_gen, sigma_real in zip(covmeans, sigmas_gen, sigmas_real):
                    if not np.isfinite(covmean).all():
                        msg = (
                            "fid calculation produces singular product; "
                            "adding %s to diagonal of cov estimates"
                        ) % 1e-6
                        print(msg)
                        offset = np.eye(sigma_gen.shape[0]) * 1e-6
                        covmean = linalg.sqrtm((sigma_gen + offset).dot(sigma_real + offset))

                # Numerical error might give slight imaginary component
                for i, covmean in enumerate(covmeans):
                    if np.iscomplexobj(covmean):
                        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                            m = np.max(np.abs(covmean.imag))
                            raise ValueError("Imaginary component {}".format(m))
                        covmeans[i] = covmean.real

                tr_covmeans = [np.trace(covmean) for covmean in covmeans]

                fids = [diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_real) - 2 * tr_covmean for diff, sigma_gen, sigma_real, tr_covmean in zip(diffs, sigmas_gen, sigmas_real, tr_covmeans)]
                end_time = time.time()
                open("FID.txt", "a").write(f"Rodada {server_round}, FIDS: {fids}, Tempo: {end_time - start_time}\n")
                

        elif self.model == "alvo":
            print("MODEL ALVO")
            conjunto_alvo = clients
            conjunto_gen = []
        elif self.model == "gen":
            print("MODEL GEN")
            conjunto_alvo = []
            conjunto_gen = clients
        else:
            print(f"Modelo {self.model} não reconhecido. O treinamento será feito para ambos os modelos.")
            sorted_clients = sorted(clients, key=lambda c: self.client_counter[c])
            metade = len(clients) // 2
            conjunto_gen = sorted_clients[:metade]
            conjunto_alvo = sorted_clients[metade:]

        self.client_counter.update(conjunto_gen)

        fit_instructions = []
        config_alvo = {"modelo": "alvo"}
        config_gen = {"modelo": "gen", "round": server_round, "fid": fids}

        for c in conjunto_alvo:
            fit_ins_alvo = FitIns(parameters=self.parameters_alvo, config=config_alvo)
            fit_instructions.append((c, fit_ins_alvo))
        
        for c in conjunto_gen:
            fit_ins_gen = FitIns(parameters=self.parameters_gen, config=config_gen)
            fit_instructions.append((c, fit_ins_gen))

        # Return client/config pairs
        return fit_instructions




    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate_alvo == 0.0:
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
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        results_alvo = [res for res in results if res[1].metrics["modelo"] == "alvo"]
        results_gen = [res for res in results if res[1].metrics["modelo"] == "gen"]

        if self.inplace:
            # Does in-place weighted average of results
            if results_alvo:
                aggregated_ndarrays_alvo = aggregate_inplace(results_alvo)
                parameters_aggregated_alvo = ndarrays_to_parameters(aggregated_ndarrays_alvo)
                self.parameters_alvo = parameters_aggregated_alvo
            if results_gen:
                aggregated_ndarrays_gen = aggregate_inplace(results_gen)
                parameters_aggregated_gen = ndarrays_to_parameters(aggregated_ndarrays_gen)
                self.parameters_gen = parameters_aggregated_gen
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

            parameters_aggregated_alvo = ndarrays_to_parameters(aggregated_ndarrays_alvo)
            parameters_aggregated_gen = ndarrays_to_parameters(aggregated_ndarrays_gen)

            self.parameters_alvo = parameters_aggregated_alvo
            self.parameters_gen = parameters_aggregated_gen

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results_alvo]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        if self.model == "alvo" and parameters_aggregated_alvo is not None:
            # Salva o modelo após a agregação
            ndarrays = parameters_to_ndarrays(parameters_aggregated_alvo)
            # Cria uma instância do modelo
            model = Net()
            # Define os pesos do modelo
            set_weights(model, ndarrays)
            # Salva o modelo no disco com o nome específico do dataset
            model_path = f"modelo_alvo_round_{server_round}_mnist.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Modelo salvo em {model_path}")

        elif self.model == "gen" and parameters_aggregated_gen is not None:
            if self.agg == "full":
                ndarrays = parameters_to_ndarrays(parameters_aggregated_gen)
                # Cria uma instância do modelo
                model = CGAN(dataset=self.dataset,
                            img_size=self.img_size,
                            latent_dim=self.latent_dim)
                # Define os pesos do modelo
                set_weights(model, ndarrays)
                # Salva o modelo no disco com o nome específico do dataset
                model_path = f"modelo_gen_round_{server_round}_mnist.pt"
                torch.save(model.state_dict(), model_path)
                print(f"Modelo salvo em {model_path}")
            return parameters_aggregated_gen, metrics_aggregated

        return parameters_aggregated_alvo, metrics_aggregated



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
        if self.model != "gen":
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
        else:
            metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss_aggregated, metrics_aggregated

