"""
Estado compartilhado entre cliente e servidor na simulação Flower do FLEG.

Como a simulação flwr roda no mesmo processo Python (usando threads/processos
leves), podemos compartilhar objetos complexos (tensores, datasets, modelos)
via variáveis de módulo — evitando serialização de objetos PyTorch pelo Flower.

Fluxo de uso:
  - A estratégia (servidor) escreve neste módulo ao final de cada fase.
  - O cliente lê ao início de cada round para obter o estado corrente.
"""

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Estado principal
# ---------------------------------------------------------------------------
state: Dict[str, Any] = {
    # ── Controle de fase ────────────────────────────────────────────────────
    "phase": "classifier",   # "classifier" | "gan"
    "level": 0,              # 0..4 — nível atual de treinamento

    # ── Dataset ─────────────────────────────────────────────────────────────
    "image_col": "image",    # "image" (MNIST) | "img" (CIFAR-10)

    # ── Augmentação com dados sintéticos ────────────────────────────────────
    # Preenchido pelo servidor ao final de cada fase GAN.
    # Lido pelos clientes no início da fase classificador (level > 0).
    "generated_dataset": None,   # GeneratedAssetDataset | None

    # ── Feature extractor fixo durante a fase GAN ───────────────────────────
    # Classe e state_dict do extrator de features do nível corrente.
    # Clientes carregam este extrator para criar embeddings dos dados locais.
    "feature_extractor_cls": None,    # tipo nn.Module
    "feature_extractor_state": None,  # OrderedDict (state_dict)

    # ── Contagem de rótulos por cliente ─────────────────────────────────────
    # Preenchido pelos clientes na primeira rodada (metrics) e lido pela
    # estratégia para a lógica de augmentação threshold.
    # chave: partition_id (int), valor: dict {label -> count}
    "client_counts": {},

    # ── Tamanho total do dataset de treino (D) ──────────────────────────────
    "total_data_size": 0,
}
