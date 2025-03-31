import os
import glob
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_curve, auc, roc_auc_score
import seaborn as sns
from PIL import Image
import pandas as pd
import segmentation_models_pytorch as smp
import time

# Configuração do logging
logging.basicConfig(
    filename='training_output_unet_smp_balanced_augmented_custom.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Definir seed para reprodutibilidade
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if torch.cuda.is_available() else None
np.random.seed(SEED)
random.seed(SEED)

# Configurações gerais
DATA_DIR = "/home/Ayrton/Neural_network_cancer"
BATCH_SIZE = 32
INPUT_SIZE = 224  # Tamanho para a maioria dos modelos pré-treinados
NUM_CLASSES = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BACKBONE = "mobilenet_v2"  # Pode ser alterado para resnet18, efficientnet-b0, etc.
UNDERSAMPLE_RATIO = 0.4  # Manter 40% das amostras da classe majoritária
EARLY_STOPPING_PATIENCE = 15
CHECKPOINT_DIR = os.path.join(DATA_DIR, f'checkpoints_unet_{BACKBONE}_balanced_augmented_custom')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Caminho para os pesos pré-treinados personalizados
PRETRAINED_WEIGHTS_PATH = "/home/Ayrton/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth"

# Verificar se CUDA está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Utilizando dispositivo: {device}")

# Configuração para mixed precision training
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Classe de dataset personalizada com suporte para undersampling
class CancerDataset(Dataset):
    def __init__(self, data_dir, subset='Training', transform=None, undersample_majority=False, undersample_ratio=0.5):
        """
        Args:
            data_dir (string): Diretório com todas as imagens
            subset (string): 'Training', 'Validation', ou 'Test'
            transform (callable, optional): Transformação opcional para aplicar à amostra
            undersample_majority (bool): Se deve realizar undersampling da classe majoritária
            undersample_ratio (float): Proporção de amostras da classe majoritária a manter (0.0-1.0)
        """
        self.transform = transform
        self.classes = ['AC', 'LSCC']  # Actinic cheilitis e Labial squamous cell carcinoma
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Coletar amostras por classe
        class_samples = {cls: [] for cls in self.classes}
        class_labels = {cls: [] for cls in self.classes}
        
        for cls in self.classes:
            class_path = os.path.join(data_dir, subset, cls)
            if not os.path.exists(class_path):
                logging.warning(f"Caminho não existe: {class_path}")
                continue
                
            for img_ext in ['*.jpg', '*.JPG']:
                paths = glob.glob(os.path.join(class_path, img_ext))
                class_samples[cls].extend(paths)
                class_labels[cls].extend([self.class_to_idx[cls]] * len(paths))
        
        # Contar amostras em cada classe
        class_counts = {cls: len(samples) for cls, samples in class_samples.items()}
        logging.info(f"Antes do undersampling - Contagem de classes: {class_counts}")
        
        # Encontrar classes majoritárias e minoritárias
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        # Aplicar undersampling se solicitado e apenas para o conjunto de treinamento
        if undersample_majority and subset == 'Training':
            # Calcular quantas amostras manter da classe majoritária
            keep_count = int(class_counts[majority_class] * undersample_ratio)
            
            # Usar random.sample para selecionar aleatoriamente
            indices = random.sample(range(class_counts[majority_class]), keep_count)
            class_samples[majority_class] = [class_samples[majority_class][i] for i in indices]
            class_labels[majority_class] = [class_labels[majority_class][i] for i in indices]
            
            # Atualizar contagens
            class_counts[majority_class] = len(class_samples[majority_class])
            logging.info(f"Após undersampling - Contagem de classes: {class_counts}")
        
        # Combinar todas as amostras
        self.image_paths = []
        self.labels = []
        
        for cls in self.classes:
            self.image_paths.extend(class_samples[cls])
            self.labels.extend(class_labels[cls])
        
        logging.info(f"Carregadas {len(self.image_paths)} imagens para o conjunto {subset}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logging.error(f"Erro ao carregar imagem {img_path}: {e}")
            # Retornar uma imagem placeholder se a imagem original não puder ser carregada
            placeholder = torch.zeros((3, INPUT_SIZE, INPUT_SIZE))
            return placeholder, label

# Definir transformações com aumento de dados para treinamento
def get_transforms(input_size, is_training=False):
    if is_training:
        # Transformações geométricas que preservam características da lesão
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),  # Rotação moderada
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0),  # Alterações sutis de cor
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Apenas redimensionar e normalizar para validação/teste
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Modelo adaptado para classificação baseado em U-Net do SMP com melhorias
class UNetClassifier(nn.Module):
    def __init__(self, encoder_name=BACKBONE, in_channels=3, num_classes=NUM_CLASSES, dropout_rate=0.3, pretrained_weights_path=None):
        super(UNetClassifier, self).__init__()
        
        # Verificar se os pesos personalizados existem
        use_pretrained_weights = False
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            logging.info(f"Usando pesos pré-treinados personalizados de: {pretrained_weights_path}")
            use_pretrained_weights = True
        else:
            logging.warning(f"Arquivo de pesos não encontrado: {pretrained_weights_path}")
            logging.info("Usando pesos padrão do ImageNet")
        
        # Configurar o SMP para usar pesos personalizados ou padrão
        if use_pretrained_weights:
            # Primeiro, carregamos o modelo sem pesos pré-treinados
            self.unet = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,  # Não usar os pesos pré-treinados padrão
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
            
            # Agora, vamos carregar manualmente os pesos pré-treinados para o encoder
            try:
                # Carregar o estado do modelo
                state_dict = torch.load(pretrained_weights_path, map_location=device)
                
                # Tentar aplicar diretamente ao encoder do backbone
                self.unet.encoder.model.load_state_dict(state_dict, strict=False)
                logging.info("Pesos pré-treinados carregados com sucesso para o encoder")
            except Exception as e:
                logging.error(f"Erro ao carregar pesos pré-treinados: {e}")
                # Se falhar, tentar recriar o modelo com os pesos padrão
                self.unet = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",  # Fallback para os pesos padrão
                    in_channels=in_channels,
                    classes=1,
                    activation=None,
                )
                logging.info("Usando pesos padrão do ImageNet como fallback após erro")
        else:
            # Criar o modelo de segmentação U-Net com o backbone escolhido e pesos padrão
            self.unet = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",  # Usar pesos pré-treinados do ImageNet
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
        
        # GAP (Global Average Pooling) para obter um vetor de features
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Feature extractor de profundidade variável dependendo do backbone
        if 'efficientnet' in encoder_name:
            enc_channels = self.unet.encoder.out_channels
            feature_dim = enc_channels[-1]
        elif 'mobilenet' in encoder_name:
            feature_dim = 1280  # MobileNetV2
        elif 'resnet' in encoder_name:
            if '18' in encoder_name or '34' in encoder_name:
                feature_dim = 512
            else:
                feature_dim = 2048
        else:
            feature_dim = 512  # valor padrão para outros backbones
        
        # Camadas de classificação com melhor regularização
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),  # Adicionado normalização
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),  # Um pouco menor para a camada interna
            nn.Linear(256, num_classes)
        )
        
        # Congelar camadas iniciais do encoder para fine-tuning mais eficiente
        self._freeze_encoder_layers()

    def _freeze_encoder_layers(self):
        """Congela camadas iniciais do encoder para evitar overfitting"""
        # Estratégia de congelamento progressivo
        if hasattr(self.unet.encoder, 'stages'):
            # Para EfficientNet e outros
            num_stages = len(self.unet.encoder.stages)
            for i, stage in enumerate(self.unet.encoder.stages):
                if i < num_stages - 2:  # Congelar todos menos os últimos 2 estágios
                    for param in stage.parameters():
                        param.requires_grad = False
        else:
            # Abordagem genérica para outros encoders
            count = 0
            total_layers = sum(1 for _ in self.unet.encoder.parameters())
            freeze_ratio = 0.7  # Congelar 70% das camadas iniciais
            
            for param in self.unet.encoder.parameters():
                count += 1
                if count < total_layers * freeze_ratio:
                    param.requires_grad = False

    def forward(self, x):
        # Extrair features usando a U-Net encoder
        features = self.unet.encoder(x)
        
        # Pegar o feature map mais profundo
        x = features[-1]
        
        # Aplicar pooling global
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classificação
        x = self.classifier(x)
        return x

# Função de treinamento com mixed precision e early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_auc = 0.0
    best_model_wts = None
    early_stop_counter = 0
    
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Fase de treinamento
        model.train()
        running_loss = 0.0
        total = 0
        all_preds = []
        all_labels = []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Usar mixed precision se disponível
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                # Scale loss e realizar backward pass
                scaler.scale(loss).backward()
                
                # Aplicar gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Atualizar optimizer e scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Fallback para precisão padrão
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                # Aplicar gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            # Estatísticas
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / total
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        train_losses.append(epoch_loss)
        train_f1s.append(epoch_f1)
        
        logging.info(f'Época {epoch+1}/{num_epochs}')
        logging.info(f'Treinamento Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')
        
        # Fase de validação
        model.eval()
        running_loss = 0.0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Coletar probabilidades para curva ROC
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        
        epoch_loss = running_loss / total
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        val_losses.append(epoch_loss)
        val_f1s.append(epoch_f1)
        
        # Calcular AUC e outras métricas
        if len(np.unique(all_labels)) > 1:
            epoch_auc = roc_auc_score(all_labels, all_probs)
            
            # Calcular sensibilidade e especificidade
            cm = confusion_matrix(all_labels, all_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                logging.info(f'Validação Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f} AUC: {epoch_auc:.4f}')
                logging.info(f'Sensibilidade: {sensitivity:.4f} Especificidade: {specificity:.4f}')
            else:
                logging.info(f'Validação Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f} AUC: {epoch_auc:.4f}')
            
            # Salvar o melhor modelo e verificar early stopping
            if epoch_auc > best_val_auc:
                best_val_auc = epoch_auc
                best_model_wts = model.state_dict().copy()
                early_stop_counter = 0  # Resetar contador
                
                logging.info(f'Nova melhor AUC de validação: {best_val_auc:.4f}')
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'best_unet_{BACKBONE}_balanced_augmented_custom.pth'))
            else:
                early_stop_counter += 1
                logging.info(f'Sem melhoria há {early_stop_counter} épocas')
                
                # Verificar early stopping
                if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    logging.info(f'Early stopping acionado após época {epoch+1}')
                    break
        else:
            logging.info(f'Validação Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')
        
        # Atualizar taxa de aprendizado
        scheduler.step()
        
        # Plotar métricas a cada 10 épocas ou no final
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1 or early_stop_counter >= EARLY_STOPPING_PATIENCE:
            plot_training_history(train_losses, val_losses, train_f1s, val_f1s, BACKBONE, CHECKPOINT_DIR)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Treinamento concluído em {elapsed_time/60:.2f} minutos")
    
    # Carregar os melhores pesos do modelo
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    
    return model, train_losses, val_losses, train_f1s, val_f1s

# Função de avaliação aprimorada
def evaluate_model(model, data_loader, criterion=None):
    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilidade da classe positiva

    # Calcular métricas
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Métricas adicionais para classificação binária
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Valor preditivo negativo
    else:
        sensitivity = specificity = precision_val = npv = float('nan')
    
    # AUC-ROC
    if len(np.unique(all_labels)) > 1:
        auc_roc = roc_auc_score(all_labels, all_probs)
    else:
        auc_roc = float('nan')
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    auc_pr = auc(recall, precision)
    
    # Log dos resultados
    logging.info("Matriz de Confusão:")
    logging.info(f"{cm}")
    logging.info(f"Acurácia: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Sensibilidade (Recall): {sensitivity:.4f}")
    logging.info(f"Especificidade: {specificity:.4f}")
    logging.info(f"Precisão: {precision_val:.4f}")
    logging.info(f"Valor Preditivo Negativo: {npv:.4f}")
    logging.info(f"AUC-ROC: {auc_roc:.4f}")
    logging.info(f"AUC-PR: {auc_pr:.4f}")
    
    # Criar DataFrame com resultados individuais para análise
    results_df = pd.DataFrame({
        'True_Label': all_labels,
        'Predicted_Label': all_preds,
        'Probability': all_probs
    })
    
    results_df.to_csv(os.path.join(CHECKPOINT_DIR, f'prediction_results_{BACKBONE}_balanced_augmented_custom.csv'), index=False)
    logging.info(f"Resultados detalhados da predição salvos em 'prediction_results_{BACKBONE}_balanced_augmented_custom.csv'")
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision_val,
        'npv': npv,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr
    }
    
    if criterion:
        avg_loss = total_loss / len(data_loader.dataset)
        logging.info(f"Perda média: {avg_loss:.4f}")
        metrics['loss'] = avg_loss
    
    return cm, metrics, all_labels, all_probs

# Funções de visualização aprimoradas
def plot_training_history(train_loss, val_loss, train_f1, val_f1, backbone, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_loss, label='Perda de Treinamento')
    ax1.plot(val_loss, label='Perda de Validação')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Perda')
    ax1.set_title('Perdas de Treinamento e Validação')
    ax1.legend()
    
    ax2.plot(train_f1, label='F1 de Treinamento')
    ax2.plot(val_f1, label='F1 de Validação')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Scores de Treinamento e Validação')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_history_unet_{backbone}_balanced_augmented_custom.png'))
    logging.info(f'Gráfico de histórico de treinamento salvo como training_history_unet_{backbone}_balanced_augmented_custom.png')
    plt.close(fig)

def plot_confusion_matrix(cm, class_names, backbone, save_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Rótulos Preditos')
    plt.ylabel('Rótulos Verdadeiros')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_unet_{backbone}_balanced_augmented_custom.png'))
    logging.info(f'Gráfico de matriz de confusão salvo como confusion_matrix_unet_{backbone}_balanced_augmented_custom.png')
    plt.close()

def plot_roc_curve(labels, probs, backbone, save_dir):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Característica Operacional do Receptor (ROC)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'roc_curve_unet_{backbone}_balanced_augmented_custom.png'))
    logging.info(f'Gráfico da curva ROC salvo como roc_curve_unet_{backbone}_balanced_augmented_custom.png')
    plt.close()
    
    return roc_auc

def plot_pr_curve(labels, probs, backbone, save_dir):
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'Curva PR (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title('Curva Precisão-Recall')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pr_curve_unet_{backbone}_balanced_augmented_custom.png'))
    logging.info(f'Gráfico da curva Precisão-Recall salvo como pr_curve_unet_{backbone}_balanced_augmented_custom.png')
    plt.close()
    
    return pr_auc

def main():
    logging.info(f"Iniciando treinamento com balanceamento e aumento de dados do modelo U-Net com backbone {BACKBONE} e pesos pré-treinados customizados para classificação de câncer")
    
    # Criar datasets com undersampling e data augmentation
    train_dataset = CancerDataset(
        DATA_DIR, 
        subset='Training', 
        transform=get_transforms(INPUT_SIZE, is_training=True),
        undersample_majority=True,
        undersample_ratio=UNDERSAMPLE_RATIO
    )
    
    val_dataset = CancerDataset(
        DATA_DIR, 
        subset='Validation', 
        transform=get_transforms(INPUT_SIZE, is_training=False)
    )
    
    test_dataset = CancerDataset(
        DATA_DIR, 
        subset='Test', 
        transform=get_transforms(INPUT_SIZE, is_training=False)
    )
    
    # Criar data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Calcular pesos de classe para loss function balanceada
    class_counts = np.bincount([label for _, label in train_dataset])
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    logging.info(f"Pesos das classes para loss function: {class_weights.cpu().numpy()}")
    
    # Inicializar modelo com regularização aprimorada e pesos pré-treinados personalizados
    model = UNetClassifier(
        encoder_name=BACKBONE, 
        dropout_rate=0.3,
        pretrained_weights_path=PRETRAINED_WEIGHTS_PATH
    )
    model = model.to(device)
    logging.info(f"Inicializado modelo U-Net com backbone {BACKBONE} usando pesos pré-treinados personalizados de {PRETRAINED_WEIGHTS_PATH}")
    
    # Definir função de perda com pesos de classe
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Definir parâmetros de treinamento para cada camada (aprendizado diferenciado)
    encoder_params = [p for n, p in model.named_parameters() if 'unet.encoder' in n and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters() if 'unet.decoder' in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]
    
    # Definir otimizador com weight decay e taxas de aprendizado diferentes
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': LEARNING_RATE * 0.1},      # Taxa menor para o encoder pré-treinado
        {'params': decoder_params, 'lr': LEARNING_RATE * 0.5},      # Taxa média para o decoder
        {'params': classifier_params, 'lr': LEARNING_RATE}          # Taxa normal para o classificador
    ], weight_decay=WEIGHT_DECAY)
    
    # Scheduler para ajuste da taxa de aprendizado
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.01)
    
    # Treinar modelo
    logging.info("Iniciando treinamento do modelo com balanceamento, aumento de dados e pesos pré-treinados personalizados")
    model, train_losses, val_losses, train_f1s, val_f1s = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS
    )
    logging.info("Treinamento do modelo concluído")
    
    # Plotar histórico de treinamento
    plot_training_history(train_losses, val_losses, train_f1s, val_f1s, BACKBONE, CHECKPOINT_DIR)
    
    # Avaliar no conjunto de teste
    logging.info("Avaliando modelo no conjunto de teste")
    cm, metrics, test_labels, test_probs = evaluate_model(model, test_loader, criterion)
    
    # Plotar matriz de confusão
    plot_confusion_matrix(cm, train_dataset.classes, BACKBONE, CHECKPOINT_DIR)
    
    # Plotar curvas ROC e PR
    if len(np.unique(test_labels)) > 1:
        roc_auc = plot_roc_curve(test_labels, test_probs, BACKBONE, CHECKPOINT_DIR)
        pr_auc = plot_pr_curve(test_labels, test_probs, BACKBONE, CHECKPOINT_DIR)
    
    # Salvar métricas do teste em formato de texto
    with open(os.path.join(CHECKPOINT_DIR, 'test_metrics.txt'), 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
        
        # Adicionar matriz de confusão
        f.write("\nMatriz de Confusão:\n")
        for row in cm:
            f.write(f"{row}\n")
    
    # Salvar modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'classes': train_dataset.classes,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
        'test_metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'backbone': BACKBONE,
        'pretrained_weights_path': PRETRAINED_WEIGHTS_PATH
    }, os.path.join(CHECKPOINT_DIR, f'final_unet_{BACKBONE}_balanced_augmented_custom_model.pth'))
    
    logging.info(f"Modelo salvo como final_unet_{BACKBONE}_balanced_augmented_custom_model.pth")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Ocorreu um erro: {e}")