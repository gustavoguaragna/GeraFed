"""
Client App — FLEG Flower Simulation

Cada instância de cliente recebe sua partição de dados via partition_id
(lido do Context do Flower) e executa:

  • Fase Classificador (phase="classifier"):
      - Carrega o modelo global (Net ou ClassifierHead_N) nos pesos recebidos.
      - Se level > 0: constrói o dataloader com augmentação sintética
        (embeddings reais + gerados) usando o feature_extractor do shared_state.
      - Treina por 1 época local e retorna os pesos atualizados.

  • Fase GAN (phase="gan"):
      - Carrega o gerador recebido do servidor no shared_state.
      - Carrega o feature_extractor fixo do shared_state.
      - Treina o discriminador local em todos os chunks dos dados de treino.
      - Retorna os pesos do discriminador treinado.

Comunicação com o servidor:
  - parameters (entrada): pesos do classificador global | gerador
  - parameters (saída):   pesos do classificador local  | discriminador
  - metrics:              partition_id, num_train_samples, label counts
"""

import math
import random
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, NDArrays, Scalar, ndarrays_to_parameters

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

import numpy as np
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
    Net, Net_Cifar,
    ClassPartitioner,
    EmbeddingPairDataset,
    augment_client_with_generated,
    get_label_counts,
    unpack_batch,
)

# ---------------------------------------------------------------------------
# Tabelas de lookup
# ---------------------------------------------------------------------------
_FE_CLS = {
    "mnist":   {1: FeatureExtractor1, 2: FeatureExtractor2, 3: FeatureExtractor3, 4: FeatureExtractor4},
    "cifar10": {1: FeatureExtractor1_Cifar, 2: FeatureExtractor2_Cifar, 3: FeatureExtractor3_Cifar, 4: FeatureExtractor4_Cifar},
}
_CH_CLS = {
    "mnist":   {1: ClassifierHead1, 2: ClassifierHead2, 3: ClassifierHead3, 4: ClassifierHead4},
    "cifar10": {1: ClassifierHead1_Cifar, 2: ClassifierHead2_Cifar, 3: ClassifierHead3_Cifar, 4: ClassifierHead4_Cifar},
}
_GAN_CLS = {
    "mnist":   {1: EmbeddingGAN1, 2: EmbeddingGAN2, 3: EmbeddingGAN3, 4: EmbeddingGAN4},
    "cifar10": {1: EmbeddingGAN1_Cifar, 2: EmbeddingGAN2_Cifar, 3: EmbeddingGAN3_Cifar, 4: EmbeddingGAN4_Cifar},
}

# FederatedDataset é carregado uma vez globalmente para evitar re-downloads
_fds_cache: Dict[str, FederatedDataset] = {}


def _get_fds(dataset: str, num_partitions: int, partitioner: str, seed: int) -> FederatedDataset:
    """Retorna (ou cria) o FederatedDataset em cache."""
    key = f"{dataset}_{num_partitions}_{partitioner}_{seed}"
    if key not in _fds_cache:
        if partitioner.lower().startswith("dir"):
            alpha = float(partitioner.replace("Dir", "").replace("dir", "")) / 10.0
            part = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=alpha,
                min_partition_size=0,
                self_balancing=False,
            )
        else:
            part = ClassPartitioner(num_partitions=num_partitions, seed=seed, label_column="label")
        _fds_cache[key] = FederatedDataset(dataset=dataset, partitioners={"train": part})
    return _fds_cache[key]


# ---------------------------------------------------------------------------
# Classe principal do cliente
# ---------------------------------------------------------------------------
class FLEGClient(NumPyClient):
    """
    Cliente FLEG para Flower Simulation.

    O cliente é stateless entre rounds — todo o estado persistente (modelos,
    dataloaders, contagens) é reconstruído a cada chamada de `fit` usando o
    `shared_state` e as configurações do round.
    """

    def __init__(self, partition_id: int, num_partitions: int, cfg_dataset: str,
                 cfg_partitioner: str, cfg_seed: int, cfg_local_test_frac: float = 0.2):
        self.partition_id      = partition_id
        self.num_partitions    = num_partitions
        self.dataset           = cfg_dataset
        self.partitioner_name  = cfg_partitioner
        self.seed              = cfg_seed
        self.local_test_frac   = cfg_local_test_frac
        self.device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Carrega e prepara a partição local (uma vez na inicialização)
        self._prepare_data()

    # ------------------------------------------------------------------
    # Preparação dos dados
    # ------------------------------------------------------------------

    def _prepare_data(self):
        """Carrega a partição local e separa train/test."""
        fds = _get_fds(self.dataset, self.num_partitions, self.partitioner_name, self.seed)

        if self.dataset == "mnist":
            self.image_col = "image"
            transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        else:
            self.image_col = "img"
            transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        def apply_transforms(batch):
            batch[self.image_col] = [transforms(img) for img in batch[self.image_col]]
            return batch

        raw_partition = fds.load_partition(self.partition_id, split="train")
        raw_partition = raw_partition.with_transform(apply_transforms)

        n         = len(raw_partition)
        test_size = int(n * self.local_test_frac)
        train_size = n - test_size

        self.train_dataset, self.test_dataset = random_split(
            raw_partition,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Contagem de rótulos para a estratégia de augmentação
        self.label_counts = get_label_counts(self.train_dataset)

        # Registra contagem no shared_state
        shared_state.state["client_counts"][self.partition_id] = self.label_counts

        # Pré-computa chunks para a fase GAN
        self._build_chunks_cache = {}   # num_chunks → list[Subset]

    def _get_chunks(self, num_chunks: int) -> List[Subset]:
        """Particiona o dataset de treino em `num_chunks` chunks embaralhados."""
        if num_chunks in self._build_chunks_cache:
            return self._build_chunks_cache[num_chunks]

        n = len(self.train_dataset)
        indices = list(range(n))
        random.seed(self.seed)
        random.shuffle(indices)

        chunk_size = math.ceil(n / num_chunks)
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end   = min((i + 1) * chunk_size, n)
            chunks.append(Subset(self.train_dataset, indices[start:end]))

        self._build_chunks_cache[num_chunks] = chunks
        return chunks

    # ------------------------------------------------------------------
    # Fase Classificador
    # ------------------------------------------------------------------

    def _fit_classifier(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        """Treina o classificador por 1 época e retorna os pesos atualizados."""
        level      = int(config["level"])
        batch_size = int(config["batch_size"])
        lr         = float(config["lr_alvo"])
        strategy   = str(config["strategy"])
        mu         = float(config.get("mu", 0.0))
        image_col  = str(config["image_col"])
        criterion  = torch.nn.CrossEntropyLoss()

        # ── Instancia e carrega o modelo correto ────────────────────────────
        if level == 0:
            model = Net(self.seed).to(self.device)
        else:
            CH_cls = _CH_CLS[self.dataset][level]
            model  = CH_cls(self.seed).to(self.device)

        # Carrega parâmetros globais recebidos
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)

        # Guarda referência ao modelo global para FedProx
        global_weights = [p.clone().detach() for p in model.parameters()] if strategy == "fedprox" else None

        # ── Dataloader ───────────────────────────────────────────────────────
        if level == 0:
            # Treino sobre dados brutos
            dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        else:
            # Treino sobre embeddings reais + sintéticos
            dataloader = self._build_augmented_loader(level, batch_size, image_col)

        # ── Treinamento ──────────────────────────────────────────────────────
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        running_loss = 0.0
        n_batches    = 0

        for batch in dataloader:
            images = batch[image_col].to(self.device)
            labels = batch["label"].to(self.device)

            if images.size(0) == 1:
                continue  # pula batch de tamanho 1 (instabilidade BatchNorm)

            optimizer.zero_grad()
            outputs = model(images)

            if strategy == "fedprox" and global_weights is not None:
                prox = sum(
                    (lp - gp).norm(2)
                    for lp, gp in zip(model.parameters(), global_weights)
                )
                loss = criterion(outputs, labels) + (mu / 2) * prox
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches    += 1

        avg_loss = running_loss / max(n_batches, 1)

        # ── Coleta parâmetros treinados ──────────────────────────────────────
        trained_ndarrays = [v.cpu().numpy() for v in model.state_dict().values()]

        metrics = {
            "partition_id":      self.partition_id,
            "num_train_samples": len(self.train_dataset),
            "train_loss":        avg_loss,
        }
        return trained_ndarrays, len(self.train_dataset), metrics

    def _build_augmented_loader(self, level: int, batch_size: int, image_col: str) -> DataLoader:
        """
        Constrói um DataLoader com embeddings reais + sintéticos.
        Usa o feature_extractor do shared_state para extrair embeddings dos dados locais.
        """
        fe_state = shared_state.state.get("feature_extractor_state")
        fe_cls   = shared_state.state.get("feature_extractor_cls")
        gen_ds   = shared_state.state.get("generated_dataset")

        if fe_state is None or fe_cls is None or gen_ds is None:
            # Fallback: sem augmentação (não deveria acontecer em level > 0)
            return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        # Carrega feature extractor
        fe = fe_cls(self.seed).to(self.device)
        fe.load_state_dict(fe_state)
        fe.eval()

        # Extrai embeddings dos dados locais
        raw_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=False)
        all_emb, all_lbl = [], []

        with torch.no_grad():
            for batch in raw_loader:
                images, labels = unpack_batch(batch, image_key=image_col, label_key="label")
                emb = fe(images.to(self.device))
                emb = emb.view(emb.size(0), -1)
                all_emb.append(emb.cpu())
                all_lbl.append(labels)

        final_emb = torch.cat(all_emb, dim=0)
        final_lbl = torch.cat(all_lbl, dim=0)

        embedding_ds = EmbeddingPairDataset(
            final_emb, final_lbl,
            asset_col_name=gen_ds.asset_col_name,
            label_col_name=gen_ds.label_col_name,
        )

        # Augmentação com threshold
        n = len(embedding_ds)
        fill_to    = max(int(n / 10), 1)
        threshold  = fill_to

        combined_ds, stats = augment_client_with_generated(
            client_train=embedding_ds,
            gen_dataset=gen_ds,
            counts=self.label_counts,
            strategy="threshold",
            fill_to=fill_to,
            threshold=threshold,
            rng_seed=self.seed,
        )
        print(f"[CLIENT {self.partition_id}] Augmentação: +{stats['gen_selected_count']} "
              f"amostras para as classes {stats['desired_labels']}")

        return DataLoader(combined_ds, batch_size=batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # Fase GAN
    # ------------------------------------------------------------------

    def _fit_gan(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        """
        Treina o discriminador local em todos os chunks e retorna seus pesos.
        O gerador recebido é usado apenas para gerar amostras fake.
        """
        level      = int(config["level"])  # nível da GAN = classificador level + 1
        gan_level  = level + 1 if level < 4 else level  # nível GAN real
        # Na verdade, o shared_state["level"] já é gan_level durante a fase GAN
        gan_level  = shared_state.state["level"]

        num_chunks = int(config["num_chunks"])
        batch_size = int(config["batch_size"])
        lr_disc    = float(config["lr_disc"])
        latent_dim = int(config["latent_dim"])
        image_col  = str(config["image_col"])

        GAN_cls = _GAN_CLS[self.dataset][gan_level]
        FE_cls  = _FE_CLS[self.dataset][gan_level]

        # ── Carrega gerador recebido do servidor ─────────────────────────────
        gen = GAN_cls(condition=True, seed=self.seed).to(self.device)
        params_dict = zip(gen.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        gen.load_state_dict(state_dict, strict=True)
        gen.eval()   # gerador é fixo durante o treinamento do discriminador

        # ── Carrega feature extractor fixo do shared_state ───────────────────
        fe_state = shared_state.state.get("feature_extractor_state")
        if fe_state is None:
            raise RuntimeError("[CLIENT] feature_extractor_state não encontrado no shared_state!")
        fe = FE_cls(self.seed).to(self.device)
        fe.load_state_dict(fe_state)
        fe.eval()

        # ── Instancia discriminador local ────────────────────────────────────
        disc = GAN_cls(condition=True, seed=self.seed).to(self.device)
        optim_D = torch.optim.Adam(disc.discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

        # ── Obtém chunks ─────────────────────────────────────────────────────
        chunks = self._get_chunks(num_chunks)

        # ── Treina discriminador em todos os chunks ──────────────────────────
        disc.train()
        total_d_loss = 0.0
        total_batches = 0

        for chunk_idx, chunk_ds in enumerate(chunks):
            if len(chunk_ds) == 0:
                continue
            chunk_loader = DataLoader(chunk_ds, batch_size=batch_size, shuffle=True)

            for batch in chunk_loader:
                images, labels = unpack_batch(batch, image_key=image_col, label_key="label")
                images = images.to(self.device)
                labels = labels.to(self.device)
                bs     = images.size(0)

                if bs == 1:
                    continue

                # Extrai embeddings reais
                with torch.no_grad():
                    real_embed = fe(images).view(bs, -1)

                real_ident = torch.full((bs, 1), 1.0, device=self.device)
                fake_ident = torch.full((bs, 1), 0.0, device=self.device)

                # Gera amostras falsas
                z           = torch.randn(bs, latent_dim, device=self.device)
                fake_labels = torch.randint(0, 10, (bs,), device=self.device)
                with torch.no_grad():
                    fake_embed = gen(z, fake_labels)

                # Passo de discriminador
                optim_D.zero_grad()

                y_real      = disc(real_embed, labels)
                d_real_loss = disc.loss(y_real, real_ident)

                y_fake      = disc(fake_embed.detach(), fake_labels)
                d_fake_loss = disc.loss(y_fake, fake_ident)

                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optim_D.step()

                total_d_loss  += d_loss.item()
                total_batches += 1

        avg_d_loss = total_d_loss / max(total_batches, 1)
        print(f"[CLIENT {self.partition_id}] D_loss={avg_d_loss:.4f} "
              f"| chunks={len(chunks)} | nível GAN={gan_level}")

        disc_ndarrays = [v.cpu().numpy() for v in disc.state_dict().values()]
        metrics = {
            "partition_id": self.partition_id,
            "d_loss":       avg_d_loss,
        }
        return disc_ndarrays, len(self.train_dataset), metrics

    # ------------------------------------------------------------------
    # Interface NumPyClient
    # ------------------------------------------------------------------

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        phase = str(config.get("phase", "classifier"))

        if phase == "classifier":
            return self._fit_classifier(parameters, config)
        elif phase == "gan":
            return self._fit_gan(parameters, config)
        else:
            raise ValueError(f"Fase desconhecida: '{phase}'")

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Avaliação distribuída não utilizada — feita centralmente no servidor."""
        return 0.0, 0, {}


# ---------------------------------------------------------------------------
# ClientApp (ponto de entrada Flower)
# ---------------------------------------------------------------------------

def client_fn(context: Context) -> NumPyClient:
    """Cria o cliente FLEG a partir do Context do Flower."""
    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    run_config = context.run_config

    dataset         = str(run_config.get("dataset", "mnist"))
    partitioner     = str(run_config.get("partitioner", "Class"))
    seed            = int(run_config.get("seed", 42))
    local_test_frac = float(run_config.get("local_test_frac", 0.2))

    return FLEGClient(
        partition_id=partition_id,
        num_partitions=num_partitions,
        cfg_dataset=dataset,
        cfg_partitioner=partitioner,
        cfg_seed=seed,
        cfg_local_test_frac=local_test_frac,
    )


app = ClientApp(client_fn=client_fn)
