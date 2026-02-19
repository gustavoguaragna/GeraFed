"""
Estratégia customizada FLEG para Flower Simulation.

Responsabilidades do servidor:
  1. Fase Classificador  — FedAvg/FedProx sobre Net (level=0) ou ClassifierHead_N
                           (level 1-4), com early stopping por paciência.
  2. Fase GAN            — Recebe discriminadores treinados dos clientes e treina
                           o gerador usando a lógica Dmax do FLEG.
  3. Transições         — Gerencia as mudanças de nível e de fase, inicializa
                           novos modelos e grava o dataset sintético em
                           shared_state.state para os clientes.

Mapeamento de rounds Flower ↔ FLEG:
  • 1 round = 1 época (fase classificador)
  • 1 round = 1 época GAN   (fase gan — clientes treinam discriminador em todos
                              os chunks; servidor treina gerador por
                              num_chunks × gen_ite passos)
"""

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import shared_state
from task import (
    ClassifierHead1, ClassifierHead2, ClassifierHead3, ClassifierHead4,
    ClassifierHead1_Cifar, ClassifierHead2_Cifar,
    ClassifierHead3_Cifar, ClassifierHead4_Cifar,
    EmbeddingGAN1, EmbeddingGAN2, EmbeddingGAN3, EmbeddingGAN4,
    EmbeddingGAN1_Cifar, EmbeddingGAN2_Cifar,
    EmbeddingGAN3_Cifar, EmbeddingGAN4_Cifar,
    FeatureExtractor1, FeatureExtractor2, FeatureExtractor3, FeatureExtractor4,
    FeatureExtractor1_Cifar, FeatureExtractor2_Cifar,
    FeatureExtractor3_Cifar, FeatureExtractor4_Cifar,
    GeneratedAssetDataset,
    Net, Net_Cifar,
)

# ---------------------------------------------------------------------------
# Tabelas de lookup: dataset × nível → classes de modelo
# ---------------------------------------------------------------------------
_FE_CLS = {
    "mnist": {
        1: FeatureExtractor1, 2: FeatureExtractor2,
        3: FeatureExtractor3, 4: FeatureExtractor4,
    },
    "cifar10": {
        1: FeatureExtractor1_Cifar, 2: FeatureExtractor2_Cifar,
        3: FeatureExtractor3_Cifar, 4: FeatureExtractor4_Cifar,
    },
}
_CH_CLS = {
    "mnist": {
        1: ClassifierHead1, 2: ClassifierHead2,
        3: ClassifierHead3, 4: ClassifierHead4,
    },
    "cifar10": {
        1: ClassifierHead1_Cifar, 2: ClassifierHead2_Cifar,
        3: ClassifierHead3_Cifar, 4: ClassifierHead4_Cifar,
    },
}
_GAN_CLS = {
    "mnist": {
        1: EmbeddingGAN1, 2: EmbeddingGAN2, 3: EmbeddingGAN3, 4: EmbeddingGAN4,
    },
    "cifar10": {
        1: EmbeddingGAN1_Cifar, 2: EmbeddingGAN2_Cifar,
        3: EmbeddingGAN3_Cifar, 4: EmbeddingGAN4_Cifar,
    },
}

# Dimensão do embedding gerado, por dataset × nível
_EMBED_DIM = {
    "mnist":   {1: 864,  2: 256, 3: 120, 4: 84},
    "cifar10": {1: 1176, 2: 400, 3: 120, 4: 84},
}

# Tamanhos de modelo aproximados (MB) — usados apenas para logging
_SIZE_CLASSIFIER = {
    "cifar10": {1: 0.25,  2: 0.248, 3: 0.238, 4: 0.045, 5: 0.004},
    "mnist":   {1: 0.18,  2: 0.179, 3: 0.169, 4: 0.045, 5: 0.004},
}
_SIZE_DISC = {
    "cifar10": {1: 18.12, 2: 3.79, 3: 0.79, 4: 0.23},
    "mnist":   {1: 5.69,  2: 1.08, 3: 0.80, 4: 0.23},
}


# ---------------------------------------------------------------------------
# Helpers de agregação
# ---------------------------------------------------------------------------
def _weighted_average(
    results: List[Tuple[List[np.ndarray], int]]
) -> List[np.ndarray]:
    """Média ponderada de ndarrays pelo número de exemplos."""
    total = sum(n for _, n in results)
    aggregated = [
        np.sum([params[i] * (n / total) for params, n in results], axis=0)
        for i in range(len(results[0][0]))
    ]
    return aggregated


# ---------------------------------------------------------------------------
# Estratégia principal
# ---------------------------------------------------------------------------
class FLEGStrategy(FedAvg):
    """
    Estratégia FLEG customizada.

    Parâmetros
    ----------
    cfg : objeto com atributos de configuração (lido do Context no server_app)
    testloader : DataLoader do conjunto de teste global (criado no server_app)
    """

    def __init__(self, cfg, testloader: DataLoader):
        self.cfg = cfg
        self.testloader = testloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Hyperparameters ─────────────────────────────────────────────────
        self.dataset: str   = cfg.dataset
        self.num_clients: int = cfg.num_clients
        self.seed: int        = cfg.seed
        self.patience: int    = cfg.patience
        self.gan_epochs: int  = cfg.gan_epochs     # épocas de treinamento GAN
        self.num_chunks: int  = cfg.num_chunks
        self.gen_ite: int     = cfg.gen_ite        # iterações do gerador por chunk
        self.levels: int      = cfg.levels         # max nível (default 4)
        self.latent_dim: int  = cfg.tam_ruido
        self.num_syn: str     = getattr(cfg, "num_syn", "dynamic")
        self.lr_gen: float    = cfg.learn_rate_gen
        self.lr_disc: float   = cfg.learn_rate_disc
        self.lr_alvo: float   = cfg.learn_rate_alvo
        self.batch_size: int  = cfg.tam_batch
        self.strategy_name: str = cfg.strategy     # "fedavg" | "fedprox"
        self.mu: float        = getattr(cfg, "mu", 0.5)

        # ── Estado de treinamento ────────────────────────────────────────────
        self.level: int   = 0
        self.phase: str   = "classifier"
        self.epoch: int   = 0           # época corrente na fase classificador
        self.gan_epoch: int = 0         # época corrente na fase GAN
        self.best_accuracy: float = 0.0
        self.epochs_no_improve: int = 0
        self.finished: bool = False

        # ── Modelos do servidor ──────────────────────────────────────────────
        if self.dataset == "mnist":
            self.global_net = Net(self.seed).to(self.device)
            self.best_net    = Net(self.seed).to(self.device)
            self.image_col   = "image"
        else:
            self.global_net = Net_Cifar(self.seed).to(self.device)
            self.best_net    = Net_Cifar(self.seed).to(self.device)
            self.image_col   = "img"

        self.class_head        = None   # ClassifierHead_N (level ≥ 1)
        self.feature_extractor = None   # FeatureExtractor_N (level ≥ 1)
        self.gen               = None   # EmbeddingGAN_N (fase GAN)
        self.disc_models: List = []     # lista de discriminadores (um por cliente)

        # ── Atualiza shared_state ────────────────────────────────────────────
        shared_state.state["image_col"] = self.image_col
        shared_state.state["phase"]     = "classifier"
        shared_state.state["level"]     = 0

        # ── Parâmetros iniciais para o Flower ────────────────────────────────
        init_ndarrays = [v.cpu().numpy() for v in self.global_net.state_dict().values()]
        initial_parameters = ndarrays_to_parameters(init_ndarrays)

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=self.num_clients,
            min_available_clients=self.num_clients,
            initial_parameters=initial_parameters,
        )

    # =========================================================================
    # Helpers internos
    # =========================================================================

    def _classifier_ndarrays(self) -> List[np.ndarray]:
        """Parâmetros do classificador atual como lista de ndarrays."""
        if self.level == 0:
            return [v.cpu().numpy() for v in self.global_net.state_dict().values()]
        return [v.cpu().numpy() for v in self.class_head.state_dict().values()]

    def _gen_ndarrays(self) -> List[np.ndarray]:
        """Parâmetros do gerador como lista de ndarrays."""
        return [v.cpu().numpy() for v in self.gen.state_dict().values()]

    def _load_disc_params(self, ndarrays: List[np.ndarray], model_idx: int):
        """Carrega ndarrays no discriminador `model_idx`."""
        disc = self.disc_models[model_idx]
        params_dict = zip(disc.state_dict().keys(), ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        disc.load_state_dict(state_dict, strict=True)

    def _evaluate_global(self) -> float:
        """Avalia o modelo global no testloader. Retorna acurácia."""
        criterion = torch.nn.CrossEntropyLoss()
        correct = 0
        self.global_net.eval() if self.level == 0 else None
        if self.level > 0:
            self.class_head.eval()
            self.feature_extractor.eval()

        with torch.no_grad():
            for batch in self.testloader:
                images = batch[self.image_col].to(self.device)
                labels = batch["label"].to(self.device)
                if self.level == 0:
                    outputs = self.global_net(images)
                else:
                    outputs = self.class_head(self.feature_extractor(images))
                correct += (outputs.argmax(1) == labels).sum().item()

        return correct / len(self.testloader.dataset)

    def _init_gan_phase(self):
        """
        Inicializa os modelos para a fase GAN do nível `self.level`.
        Atualiza shared_state com feature_extractor para os clientes.
        """
        gan_level = self.level + 1          # GAN_1 após classificador nível 0, etc.
        if gan_level > self.levels:
            print(f"[SERVER] Nível máximo ({self.levels}) atingido. Encerrando.")
            self.finished = True
            return

        FE_cls  = _FE_CLS[self.dataset][gan_level]
        CH_cls  = _CH_CLS[self.dataset][gan_level]
        GAN_cls = _GAN_CLS[self.dataset][gan_level]

        # Cria e inicializa feature extractor a partir do modelo global
        self.feature_extractor = FE_cls(self.seed).to(self.device)
        fe_dict = self.feature_extractor.state_dict()
        pretrained = {k: v for k, v in self.global_net.state_dict().items() if k in fe_dict}
        self.feature_extractor.load_state_dict(pretrained)
        self.feature_extractor.eval()

        # Cria e inicializa classifier head a partir do modelo global
        self.class_head = CH_cls(self.seed).to(self.device)
        ch_dict = self.class_head.state_dict()
        pretrained_ch = {k: v for k, v in self.global_net.state_dict().items() if k in ch_dict}
        self.class_head.load_state_dict(pretrained_ch)

        # Cria gerador e discriminadores
        self.gen = GAN_cls(condition=True, seed=self.seed).to(self.device)
        self.disc_models = [GAN_cls(condition=True, seed=self.seed) for _ in range(self.num_clients)]

        # Atualiza shared_state para os clientes
        shared_state.state["feature_extractor_cls"]   = FE_cls
        shared_state.state["feature_extractor_state"] = OrderedDict(
            {k: v.cpu().clone() for k, v in self.feature_extractor.state_dict().items()}
        )
        shared_state.state["phase"] = "gan"
        shared_state.state["level"] = gan_level

        self.phase     = "gan"
        self.gan_epoch = 0
        print(f"\n[SERVER] Iniciando Fase GAN — nível {gan_level}")

    def _init_next_classifier_phase(self):
        """Transição: GAN concluído → fase classificador do próximo nível."""
        next_level = self.level + 1
        self.level          = next_level
        self.phase          = "classifier"
        self.epoch          = 0
        self.best_accuracy  = 0.0
        self.epochs_no_improve = 0

        shared_state.state["phase"] = "classifier"
        shared_state.state["level"] = next_level
        print(f"\n[SERVER] Iniciando Fase Classificador — nível {next_level}")

    def _train_generator(self):
        """
        Treina o gerador por `num_chunks × gen_ite` passos usando os
        discriminadores recebidos dos clientes (lógica Dmax do FLEG).
        """
        self.gen.to(self.device)
        self.gen.train()
        for disc in self.disc_models:
            disc.to(self.device)
            disc.eval()

        optim_G = torch.optim.Adam(
            self.gen.generator.parameters(),
            lr=self.lr_gen, betas=(0.5, 0.999),
        )

        total_steps = self.num_chunks * self.gen_ite
        total_g_loss = 0.0

        for _ in range(total_steps):
            optim_G.zero_grad()
            z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            fake_labels = torch.randint(0, 10, (self.batch_size,), device=self.device)

            fake_embed = self.gen(z, fake_labels)

            # Dmax: seleciona o discriminador com maior score médio
            with torch.no_grad():
                scores = [
                    torch.mean(disc(fake_embed.detach(), fake_labels)).item()
                    for disc in self.disc_models
                ]
            Dmax = self.disc_models[scores.index(max(scores))]

            real_ident = torch.full((self.batch_size, 1), 1.0, device=self.device)
            y_fake_g   = Dmax(fake_embed, fake_labels)
            g_loss     = self.gen.loss(y_fake_g, real_ident)

            g_loss.backward()
            optim_G.step()
            total_g_loss += g_loss.item()

        avg_g_loss = total_g_loss / total_steps
        print(f"[SERVER] Gerador treinado | passos={total_steps} | G_loss={avg_g_loss:.4f}")
        return avg_g_loss

    def _generate_synthetic_dataset(self):
        """Gera dataset sintético com o gerador treinado e salva no shared_state."""
        gan_level = self.level + 1
        embed_dim = _EMBED_DIM[self.dataset][gan_level]
        D = shared_state.state["total_data_size"]

        if self.num_syn == "dynamic":
            num_samples = int(math.ceil(D / self.num_clients) / 4 * gan_level)
        else:
            num_samples = 48000 if self.dataset == "mnist" else 40000

        num_samples = max(num_samples, 10)  # garante mínimo
        print(f"[SERVER] Gerando {num_samples} amostras sintéticas (nível {gan_level})")

        gen_dataset = GeneratedAssetDataset(
            generator=self.gen,
            num_samples=num_samples,
            latent_dim=self.latent_dim,
            num_classes=10,
            asset_shape=(embed_dim,),
            asset_col_name=self.image_col,
            device=self.device,
        )
        shared_state.state["generated_dataset"] = gen_dataset
        print(f"[SERVER] Dataset sintético criado: {len(gen_dataset)} amostras")

    # =========================================================================
    # Interface Flower
    # =========================================================================

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ):
        if self.finished:
            return []

        config: Dict[str, Scalar] = {
            "level":       self.level,
            "phase":       self.phase,
            "dataset":     self.dataset,
            "seed":        self.seed,
            "batch_size":  self.batch_size,
            "strategy":    self.strategy_name,
            "mu":          self.mu,
            "latent_dim":  self.latent_dim,
            "num_chunks":  self.num_chunks,
            "gen_ite":     self.gen_ite,
            "lr_gen":      self.lr_gen,
            "lr_disc":     self.lr_disc,
            "lr_alvo":     self.lr_alvo,
            "image_col":   self.image_col,
            "num_syn":     self.num_syn,
            "server_round": server_round,
        }

        if self.phase == "classifier":
            send_params = ndarrays_to_parameters(self._classifier_ndarrays())
            config["epoch"] = self.epoch
        else:
            # Fase GAN: envia parâmetros do gerador
            send_params = ndarrays_to_parameters(self._gen_ndarrays())
            config["gan_epoch"] = self.gan_epoch

        sample = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )
        return [(client, FitIns(send_params, config)) for client in sample]

    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results or self.finished:
            return None, {}

        if self.phase == "classifier":
            return self._agg_classifier(results)
        else:
            return self._agg_gan(results)

    # ------------------------------------------------------------------

    def _agg_classifier(self, results):
        """Agrega fase do classificador: FedAvg + avaliação + paciência."""
        self.epoch += 1

        # Atualiza total_data_size apenas uma vez (primeira época do level 0)
        if self.level == 0 and self.epoch == 1:
            total = sum(
                (fit_res.metrics or {}).get("num_train_samples", 0)
                for _, fit_res in results
            )
            shared_state.state["total_data_size"] = int(total)
            print(f"[SERVER] Total de amostras de treinamento (D): {total}")

        # Weighted FedAvg
        ndarrays_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated = _weighted_average(ndarrays_results)

        # Carrega pesos agregados no modelo correto
        if self.level == 0:
            params_dict = zip(self.global_net.state_dict().keys(), aggregated)
            state_dict  = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
            self.global_net.load_state_dict(state_dict)
        else:
            params_dict = zip(self.class_head.state_dict().keys(), aggregated)
            state_dict  = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
            self.class_head.load_state_dict(state_dict)

        # Avaliação global
        accuracy = self._evaluate_global()
        print(f"[SERVER] Round {self.epoch} | Nível {self.level} | Acurácia: {accuracy:.4f}")

        # Atualiza best model e paciência
        if accuracy > self.best_accuracy:
            self.best_accuracy    = accuracy
            self.epochs_no_improve = 0
            if self.level == 0:
                self.best_net.load_state_dict(self.global_net.state_dict())
            else:
                self.best_net.load_state_dict(self.class_head.state_dict(), strict=False)
        else:
            self.epochs_no_improve += 1
            print(f"[SERVER] Sem melhora por {self.epochs_no_improve} épocas "
                  f"(melhor: {self.best_accuracy:.4f})")

        # Verifica condição de parada (paciência esgotada)
        if self.epochs_no_improve > self.patience:
            print(f"[SERVER] Paciência esgotada no nível {self.level}. "
                  f"Melhor acurácia: {self.best_accuracy:.4f}")
            # Restaura best model no global_net
            self.global_net.load_state_dict(self.best_net.state_dict(), strict=False)
            # Reseta paciência para próximo nível
            self.epochs_no_improve = 0

            if self.level >= self.levels:
                print("[SERVER] Todos os níveis concluídos. Treinamento finalizado!")
                self.finished = True
            else:
                self._init_gan_phase()  # → fase GAN do próximo nível

        aggregated_params = ndarrays_to_parameters(aggregated)
        return aggregated_params, {"accuracy": accuracy, "level": self.level}

    # ------------------------------------------------------------------

    def _agg_gan(self, results):
        """Agrega fase GAN: carrega discriminadores, treina gerador."""
        self.gan_epoch += 1

        # Ordena resultados por partition_id para consistência
        sorted_results = sorted(
            results,
            key=lambda x: (x[1].metrics or {}).get("partition_id", 0),
        )

        # Carrega parâmetros do discriminador de cada cliente
        for i, (_, fit_res) in enumerate(sorted_results):
            disc_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            if i < len(self.disc_models):
                self._load_disc_params(disc_ndarrays, i)

        # Treina gerador com os discriminadores atualizados
        g_loss = self._train_generator()

        # Verifica se a fase GAN terminou
        if self.gan_epoch >= self.gan_epochs:
            print(f"[SERVER] Fase GAN concluída (nível {self.level + 1})")
            self._generate_synthetic_dataset()
            self._init_next_classifier_phase()

        # Retorna parâmetros do gerador atualizados
        gen_params = ndarrays_to_parameters(self._gen_ndarrays())
        return gen_params, {"g_loss": g_loss, "gan_epoch": self.gan_epoch}

    # ------------------------------------------------------------------

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Desativa avaliação distribuída — avaliação feita direto no servidor."""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        """Avaliação centralizada (opcional, chamada pelo Flower)."""
        return None
