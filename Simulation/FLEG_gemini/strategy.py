"""FLEG: um framework para balancear dados heterogêneos em aprendizado federado, com precupações com a privacidade."""

import torch
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import (
    Parameters,
    FitRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
import numpy as np
from collections import OrderedDict

# Importa os modelos definidos no task.py
from Simulation.FLEG_gemini.task import (
    Net, Net_Cifar,
    ClassifierHead1, ClassifierHead2, ClassifierHead3, ClassifierHead4,
    ClassifierHead1_Cifar, ClassifierHead2_Cifar, ClassifierHead3_Cifar, ClassifierHead4_Cifar,
    FeatureExtractor1, FeatureExtractor2, FeatureExtractor3, FeatureExtractor4,
    FeatureExtractor1_Cifar, FeatureExtractor2_Cifar, FeatureExtractor3_Cifar, FeatureExtractor4_Cifar,
    EmbeddingGAN1, EmbeddingGAN2, EmbeddingGAN3, EmbeddingGAN4,
    EmbeddingGAN1_Cifar, EmbeddingGAN2_Cifar, EmbeddingGAN3_Cifar, EmbeddingGAN4_Cifar,
)

class FlegStrategy(FedAvg):
    def __init__(
        self,
        num_partitions: int,
        dataset_name: str,
        device: torch.device,
        total_levels: int = 4,
        gan_epochs_per_level: int = 25,
        patience: int = 5,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_partitions = num_partitions
        self.dataset_name = dataset_name
        self.device = device
        self.total_levels = total_levels
        self.gan_epochs_per_level = gan_epochs_per_level
        self.patience = patience

        # Estado da Simulação
        self.current_level = 0
        self.phase = "classification"  # 'classification' ou 'gan'
        self.best_accuracy = 0.0
        self.epochs_no_improve = 0
        self.gan_epoch_counter = 0

        # Inicialização dos Modelos Globais
        self.seed = 42
        self.global_net = self._init_net()
        self.feature_extractor = None
        self.class_head = None
        self.generator = None
        self.discriminator = None # Usado apenas como template/placeholder no servidor
        
        # Mapeamento de Classes por Nível
        self._setup_model_classes()

    def _init_net(self):
        if self.dataset_name == "mnist":
            return Net(self.seed).to(self.device)
        return Net_Cifar(self.seed).to(self.device)

    def _setup_model_classes(self):
        """Define quais classes usar dependendo do dataset e nível."""
        suffix = "_Cifar" if self.dataset_name == "cifar10" else ""
        self.model_map = {
            1: (globals()[f"FeatureExtractor1{suffix}"], globals()[f"ClassifierHead1{suffix}"], globals()[f"EmbeddingGAN1{suffix}"]),
            2: (globals()[f"FeatureExtractor2{suffix}"], globals()[f"ClassifierHead2{suffix}"], globals()[f"EmbeddingGAN2{suffix}"]),
            3: (globals()[f"FeatureExtractor3{suffix}"], globals()[f"ClassifierHead3{suffix}"], globals()[f"EmbeddingGAN3{suffix}"]),
            4: (globals()[f"FeatureExtractor4{suffix}"], globals()[f"ClassifierHead4{suffix}"], globals()[f"EmbeddingGAN4{suffix}"]),
        }

    def _update_architecture_for_level(self, level):
        """Atualiza feature_extractor, class_head e gan para o novo nível."""
        if level == 0:
            return

        FeatClass, HeadClass, GanClass = self.model_map[level]
        
        # Instancia novos modelos
        self.feature_extractor = FeatClass(self.seed).to(self.device)
        self.class_head = HeadClass(self.seed).to(self.device)
        self.generator = GanClass(condition=True, seed=self.seed).to(self.device).generator
        # Discriminador é local, mas precisamos inicializar pesos para enviar
        self.discriminator = GanClass(condition=True, seed=self.seed).to(self.device).discriminator

        # Carrega pesos do nível anterior (Progressive Growing)
        # Pega pesos da global_net e carrega nas partes correspondentes
        global_state = self.global_net.state_dict()
        
        # Carrega Feature Extractor
        feat_dict = self.feature_extractor.state_dict()
        pretrained_feat = {k: v for k, v in global_state.items() if k in feat_dict}
        self.feature_extractor.load_state_dict(pretrained_feat)
        
        # Carrega Classifier Head
        head_dict = self.class_head.state_dict()
        pretrained_head = {k: v for k, v in global_state.items() if k in head_dict}
        self.class_head.load_state_dict(pretrained_head)

    def initialize_parameters(self, client_manager):
        """Retorna os parâmetros iniciais (Nível 0: Net completa)."""
        return ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.global_net.state_dict().items()]
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitRes]]:
        
        # Configuração enviada para os clientes
        config = {
            "level": self.current_level,
            "phase": self.phase,
            "dataset_name": self.dataset_name,
        }

        # Serialização de modelos auxiliares (Feature Extractor e Generator)
        # Clientes precisam disso para:
        # 1. Classification (Level > 0): Feature Extractor para extrair features, Generator para Data Augmentation.
        # 2. GAN: Feature Extractor para input real, Generator para input fake.
        
        aux_params = {}
        if self.current_level > 0:
            # Serializa Feature Extractor
            fe_ndarrays = [val.cpu().numpy() for _, val in self.feature_extractor.state_dict().items()]
            config["feature_extractor_params"] = ndarrays_to_parameters(fe_ndarrays)
            
            # Serializa Generator (se já treinado ou para treino da GAN)
            gen_ndarrays = [val.cpu().numpy() for _, val in self.generator.state_dict().items()]
            config["generator_params"] = ndarrays_to_parameters(gen_ndarrays)

        # Decide quais parâmetros principais enviar para treino
        if self.phase == "classification":
            if self.current_level == 0:
                fit_params = parameters # Global Net
            else:
                # Envia apenas o ClassHead
                head_ndarrays = [val.cpu().numpy() for _, val in self.class_head.state_dict().items()]
                fit_params = ndarrays_to_parameters(head_ndarrays)
        else:
            # Fase GAN: Envia o Discriminador para os clientes treinarem
            disc_ndarrays = [val.cpu().numpy() for _, val in self.discriminator.state_dict().items()]
            fit_params = ndarrays_to_parameters(disc_ndarrays)

        # Amostragem padrão (todos ou fração)
        clients = client_manager.sample(num_clients=self.num_partitions, min_num_clients=1)
        
        return [(client, fit_params, config) for client in clients if config.update({"feature_extractor_params": config.get("feature_extractor_params"), "generator_params": config.get("generator_params")}) or True] # Hack to pass large config inside standard structure is tricky, ideally use fit_ins with custom config, but standard flow accepts dict. Note: Passing huge params in config dict is bad practice in production (bloats JSON), but functional for simulation.
        
        # Better approach for passing model weights in config:
        # In Flower simulation, passing ndarrays in config works but is heavy. 
        # Ideally, we pack everything into 'parameters' vector, but models are separate.
        # For this implementation, we will utilize the FitIns parameters for the MODEL TO BE TRAINED,
        # and pack fixed models (Extractor, Generator) into the config dictionary as serialized bytes or lists.
        # Note: Above I put ndarrays_to_parameters object in config, client needs to unpack.

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        # Agregação padrão (FedAvg) dos parâmetros retornados
        aggregated_ndarrays = super().aggregate_fit(server_round, results, failures)[0]
        
        if aggregated_ndarrays:
            params_np = parameters_to_ndarrays(aggregated_ndarrays)

            if self.phase == "classification":
                # Atualiza o modelo correspondente
                if self.current_level == 0:
                    params_dict = zip(self.global_net.state_dict().keys(), params_np)
                    state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
                    self.global_net.load_state_dict(state_dict, strict=True)
                else:
                    params_dict = zip(self.class_head.state_dict().keys(), params_np)
                    state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
                    self.class_head.load_state_dict(state_dict, strict=True)
                    
                    # Sincroniza o global_net com o novo head
                    self.global_net.load_state_dict(self.class_head.state_dict(), strict=False)

            elif self.phase == "gan":
                # Atualiza o Discriminador Global (Média dos Discriminadores locais)
                params_dict = zip(self.discriminator.state_dict().keys(), params_np)
                state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
                self.discriminator.load_state_dict(state_dict, strict=True)

                # --- SERVER-SIDE GENERATOR UPDATE ---
                # No FLEG original, o servidor treina o Generator para enganar os Discriminadores.
                # Aqui, usamos o Discriminador Agregado como proxy.
                self._train_server_generator()
                
                self.gan_epoch_counter += 1
                print(f"GAN Epoch {self.gan_epoch_counter}/{self.gan_epochs_per_level}")

        return aggregated_ndarrays, {"level": self.current_level, "phase": self.phase}

    def _train_server_generator(self):
        """Treina o Generator no servidor usando o Discriminador agregado."""
        # Configuração simplificada para simulação
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.generator.train()
        self.discriminator.eval() # O discriminador é fixo neste passo (foi treinado pelos clientes)
        
        # Simula algumas iterações de treino do gerador (ex: 20 iterações como no original)
        batch_size = 32
        latent_dim = 128
        adv_loss = torch.nn.BCEWithLogitsLoss()
        
        for _ in range(20):
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, latent_dim, device=self.device)
            gen_labels = torch.randint(0, 10, (batch_size,), device=self.device)
            
            # Gera imagens falsas (embeddings)
            # Nota: O EmbeddingGAN original lida com concatenação interna de labels
            # Precisamos garantir que a chamada seja compatível com a definição em task.py
            # Em task.py, EmbedinGAN.forward(x, labels)
            
            gen_imgs = self.generator(z, gen_labels) # task.py forward generator handle z+labels concat
            
            # Passa pelo discriminador
            # Importante: O discriminador espera input concatenado se condition=True
            validity = self.discriminator(gen_imgs, gen_labels)
            
            # Loss: Generator quer que validity seja 1 (Real)
            real_labels = torch.full((batch_size, 1), 1.0, device=self.device)
            g_loss = adv_loss(validity, real_labels)
            
            g_loss.backward()
            optimizer_G.step()

    def evaluate(self, server_round: int, parameters: Parameters):
        """Avalia o modelo global e gerencia a transição de fases/níveis."""
        # Se estivermos treinando GAN, não avaliamos acurácia de classificação, apenas checamos épocas
        if self.phase == "gan":
            if self.gan_epoch_counter >= self.gan_epochs_per_level:
                # Fim do treino da GAN para este nível
                print(f">>> Fim da Fase GAN do Nível {self.current_level}. Avançando nível.")
                self.current_level += 1
                self.phase = "classification"
                self.gan_epoch_counter = 0
                self.best_accuracy = 0.0
                self.epochs_no_improve = 0
                
                if self.current_level > self.total_levels:
                    return None, {"status": "finished"}

                self._update_architecture_for_level(self.current_level)
            return None, {}

        # Fase de Classificação: Avaliar Acurácia
        loss, accuracy = self._evaluate_global_model()
        print(f"Round {server_round} (Level {self.current_level}): Acc: {accuracy:.4f}, Loss: {loss:.4f}")

        # Lógica de Paciência
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        
        # Transição para GAN
        if self.epochs_no_improve >= self.patience:
            print(f">>> Convergência do Classificador Nível {self.current_level}. Iniciando Fase GAN.")
            self.phase = "gan"
            self.gan_epoch_counter = 0
            
            # Se for nível 0 e for avançar, precisamos preparar a arquitetura do nível 1 para a GAN
            # O FLEG treina a GAN do Nível X DEPOIS de treinar o classificador do Nível X?
            # Olhando o original:
            # 1. Treina Classifier Level L.
            # 2. Alterna para GAN Level L+1 (Prepara GAN1, Head1, Extractor1).
            # 3. Treina GAN1.
            # 4. Gera dados.
            # 5. Loop reinicia para Level L+1 (Classification).
            
            # Ajuste na lógica:
            if self.current_level < self.total_levels:
                # Prepara a arquitetura do próximo nível para treinar a GAN correspondente
                # (A GAN treinada agora será usada para augmentation no treino de classificação do próximo nível)
                # Na verdade, a GAN do nível 1 gera dados para ajudar o treino do classificador nível 1?
                # Original: "Alternando para o nível {level} de treinamento da GAN" -> if level+1 == 1...
                # Isso significa que ao terminar Classificador Level 0, treinamos GAN Level 1.
                
                # Atualiza arquiteturas para o contexto da GAN (que é do próximo nível)
                next_lvl = self.current_level + 1
                self._update_architecture_for_level(next_lvl) 
                # Nota: self.current_level ainda é 0 (no print), mas os modelos agora são do nível 1
                # A variável self.current_level será incrementada APÓS o fim da fase GAN.

        return loss, {"accuracy": accuracy, "level": self.current_level, "phase": self.phase}

    def _evaluate_global_model(self):
        """Avaliação centralizada usando dados de teste (MNIST/CIFAR)."""
        # Carrega dados de teste
        from torchvision.datasets import MNIST, CIFAR10
        from torchvision.transforms import Compose, ToTensor, Normalize
        from torch.utils.data import DataLoader

        if self.dataset_name == "mnist":
            transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
            test_set = MNIST("./data", train=False, download=True, transform=transform)
        else:
            transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            test_set = CIFAR10("./data", train=False, download=True, transform=transform)

        testloader = DataLoader(test_set, batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        
        self.global_net.eval()
        if self.current_level > 0:
            self.feature_extractor.eval()
            self.class_head.eval()

        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                if self.current_level == 0:
                    outputs = self.global_net(images)
                else:
                    emb = self.feature_extractor(images)
                    outputs = self.class_head(emb)
                
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return loss / len(testloader), correct / total