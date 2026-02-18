"""
Server App — FLEG Flower Simulation

Responsabilidades:
  1. Lê as configurações do run_config (pyproject.toml).
  2. Cria o testloader global (split "test" do FederatedDataset).
  3. Instancia a FLEGStrategy com as configurações e o testloader.
  4. Retorna um ServerApp com a estratégia e o número de rounds calculado.

Cálculo de num_rounds:
  O FLEG usa early stopping por paciência, portanto o número exato de rounds
  não é conhecido a priori. Usamos um limite superior generoso:

      num_rounds = (patience + 10) × (levels + 1)   ← classifier rounds
                 + gan_epochs × levels                ← GAN rounds

  A estratégia sinaliza o encerramento retornando parâmetros None ou via a
  flag `finished`, fazendo o Flower parar antecipadamente.
"""

import types
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

from task import ClassPartitioner
from strategy import FLEGStrategy

# ---------------------------------------------------------------------------
# Utilitário de configuração
# ---------------------------------------------------------------------------

def _parse_config(run_config: dict) -> types.SimpleNamespace:
    """
    Converte o dict run_config (lido do pyproject.toml) em um namespace
    com os atributos usados pela estratégia e pelos clientes.

    Mapeia as chaves em português/abreviadas do pyproject.toml para os
    nomes internos do código (compatíveis com FLEG_SBRC.py).
    """
    cfg = types.SimpleNamespace()

    # ── Dataset e particionamento ────────────────────────────────────────────
    cfg.dataset       = str(run_config.get("dataset",     "mnist"))
    cfg.partitioner   = str(run_config.get("partitioner", "Class"))
    cfg.num_clients   = int(run_config.get("num_clients", 4))
    cfg.seed          = int(run_config.get("seed",        42))

    # ── Treinamento do classificador ─────────────────────────────────────────
    cfg.learn_rate_alvo = float(run_config.get("learn_rate_alvo", 0.01))
    cfg.tam_batch       = int(run_config.get("tam_batch",         32))
    cfg.patience        = int(run_config.get("patience",          10))
    cfg.strategy        = str(run_config.get("strategy",          "fedavg"))
    cfg.mu              = float(run_config.get("mu",              0.5))
    cfg.local_test_frac = float(run_config.get("local_test_frac", 0.2))

    # ── GAN ──────────────────────────────────────────────────────────────────
    cfg.learn_rate_gen  = float(run_config.get("learn_rate_gen",  0.0002))
    cfg.learn_rate_disc = float(run_config.get("learn_rate_disc", 0.0002))
    cfg.tam_ruido       = int(run_config.get("tam_ruido",         128))   # latent_dim
    cfg.num_chunks      = int(run_config.get("num_chunks",        100))
    cfg.gan_epochs      = int(run_config.get("gan_epochs",        25))    # épocas GAN
    cfg.gen_ite         = int(run_config.get("gen_ite",           20))    # passos do gerador por chunk
    cfg.levels          = int(run_config.get("levels",            4))
    cfg.num_syn         = str(run_config.get("num_syn",           "dynamic"))

    # ── Modo de teste rápido ─────────────────────────────────────────────────
    if run_config.get("teste", False):
        print("[SERVER] Modo de teste ativo: reduzindo parâmetros...")
        cfg.num_chunks = 2
        cfg.gan_epochs = 2
        cfg.patience   = 2

    return cfg


def _build_testloader(cfg) -> DataLoader:
    """Carrega o split de teste global do FederatedDataset."""
    if cfg.dataset == "mnist":
        transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        image_col  = "image"
    else:
        transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image_col  = "img"

    if cfg.partitioner.lower().startswith("dir"):
        alpha = float(cfg.partitioner.replace("Dir", "").replace("dir", "")) / 10.0
        part  = DirichletPartitioner(
            num_partitions=cfg.num_clients,
            partition_by="label",
            alpha=alpha,
            min_partition_size=0,
            self_balancing=False,
        )
    else:
        part = ClassPartitioner(
            num_partitions=cfg.num_clients,
            seed=cfg.seed,
            label_column="label",
        )

    fds         = FederatedDataset(dataset=cfg.dataset, partitioners={"train": part})
    test_split  = fds.load_split("test")

    def apply_transforms(batch):
        batch[image_col] = [transforms(img) for img in batch[image_col]]
        return batch

    test_split = test_split.with_transform(apply_transforms)
    return DataLoader(test_split, batch_size=64, shuffle=False)


# ---------------------------------------------------------------------------
# ServerApp (ponto de entrada Flower)
# ---------------------------------------------------------------------------

def server_fn(context: Context) -> ServerAppComponents:
    """Cria os componentes do servidor FLEG."""
    run_config = context.run_config
    cfg        = _parse_config(run_config)

    print(f"""
╔══════════════════════════════════════════════════════╗
║           FLEG — Flower Simulation                   ║
╠══════════════════════════════════════════════════════╣
  Dataset       : {cfg.dataset}
  Particionador : {cfg.partitioner}
  Clientes      : {cfg.num_clients}
  Estratégia    : {cfg.strategy}
  Níveis        : {cfg.levels}
  Paciência     : {cfg.patience}
  Épocas GAN    : {cfg.gan_epochs}
  Chunks        : {cfg.num_chunks}
  Gen. iter./chunk: {cfg.gen_ite}
  Latent dim    : {cfg.tam_ruido}
  Num. sintéticas: {cfg.num_syn}
  Seed          : {cfg.seed}
╚══════════════════════════════════════════════════════╝
""")

    # ── Testloader global ────────────────────────────────────────────────────
    testloader = _build_testloader(cfg)

    # ── Estratégia FLEG ──────────────────────────────────────────────────────
    strategy = FLEGStrategy(cfg=cfg, testloader=testloader)

    # ── Número de rounds ────────────────────────────────────────────────────
    # Limite superior: (patience + margem) épocas por nível de classificador
    #                + gan_epochs por nível de GAN
    max_classifier_rounds = (cfg.patience + 15) * (cfg.levels + 1)
    max_gan_rounds        = cfg.gan_epochs * cfg.levels
    num_rounds            = max_classifier_rounds + max_gan_rounds

    print(f"[SERVER] Número máximo de rounds configurado: {num_rounds}")

    server_config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=server_config)


app = ServerApp(server_fn=server_fn)
