import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
import glob
import logging
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_curve, auc, roc_auc_score, roc_curve

# Configuração do logging
logging.basicConfig(
    filename='training_output_vgg16_balanced_augmented.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Verificar se CUDA está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Utilizando dispositivo: {device}")

# Definir seed para reprodutibilidade
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if torch.cuda.is_available() else None
np.random.seed(SEED)
random.seed(SEED)

# Configurações gerais
DATA_DIR = "/home/Ayrton/Neural_network_cancer"
BATCH_SIZE = 32
INPUT_SIZE = 224  # VGG usa 224x224
NUM_CLASSES = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
UNDERSAMPLE_RATIO = 0.4  # Manter 40% das amostras da classe majoritária
EARLY_STOPPING_PATIENCE = 15
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints_vgg16_balanced_augmented')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Caminho para os pesos pré-treinados personalizados
PRETRAINED_WEIGHTS_PATH = "/home/Ayrton/.cache/torch/hub/checkpoints/vgg16-397923af.pth"

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

# Definição da arquitetura VGG16 com pesos pré-treinados personalizados
class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5, weights_path=None):
        super(VGG16Classifier, self).__init__()
        
        # Inicializar o modelo VGG16 sem pesos pré-treinados primeiro
        self.backbone = models.vgg16(weights=None)
        
        # Carregar os pesos pré-treinados do arquivo especificado
        if weights_path and os.path.exists(weights_path):
            logging.info(f"Carregando pesos pré-treinados de: {weights_path}")
            try:
                # Carregar estado do modelo
                state_dict = torch.load(weights_path, map_location=device)
                self.backbone.load_state_dict(state_dict)
                logging.info("Pesos pré-treinados carregados com sucesso")
            except Exception as e:
                logging.error(f"Erro ao carregar pesos pré-treinados: {e}")
                # Fallback para pesos do ImageNet caso ocorra algum erro
                logging.info("Usando pesos padrão do ImageNet como fallback")
                self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            logging.warning(f"Arquivo de pesos não encontrado: {weights_path}")
            logging.info("Usando pesos padrão do ImageNet como fallback")
            self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Estratégia de congelamento progressivo
        # Primeiro congela todas as camadas
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Depois descongela as últimas camadas convolucionais
        for i, param in enumerate(self.backbone.features.parameters()):
            # VGG16 tem 13 camadas convolucionais, descongelamos as últimas 3
            if i >= 20:  # Aproximadamente a partir do bloco 4
                param.requires_grad = True
        
        # Modificar o classificador final com melhor regularização
        num_features = self.backbone.classifier[6].in_features
        
        # Substituir o classificador inteiro para melhor controle de regularização
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes)
        )
        
        # Certificar-se de que as camadas do classificador são treináveis
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

# Função para carregar dados com undersampling e data augmentation
def load_data(data_dir, batch_size=32, input_size=224):
    train_dir = os.path.join(data_dir, 'Training')
    val_dir = os.path.join(data_dir, 'Validation')
    test_dir = os.path.join(data_dir, 'Test')
    
    # Criar datasets com transforms apropriados
    train_dataset = CancerDataset(
        data_dir, 
        subset='Training', 
        transform=get_transforms(input_size, is_training=True),
        undersample_majority=True,
        undersample_ratio=UNDERSAMPLE_RATIO
    )
    
    val_dataset = CancerDataset(
        data_dir, 
        subset='Validation', 
        transform=get_transforms(input_size, is_training=False)
    )
    
    test_dataset = CancerDataset(
        data_dir, 
        subset='Test', 
        transform=get_transforms(input_size, is_training=False)
    )
    
    # Criar data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Calcular pesos para loss function balanceada
    class_counts = np.bincount([label for _, label in train_dataset])
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    logging.info(f"Pesos das classes para loss function: {class_weights.cpu().numpy()}")
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return dataloaders, train_dataset.classes, class_weights

# Função para avaliar o modelo com métricas detalhadas
def evaluate_model(model, dataloader, criterion):
    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calcular probabilidades e predições
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            # Coletar para métricas
            total_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            # Pegando probabilidades da classe positiva (assumindo classificação binária)
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calcular métricas
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Métricas adicionais para classificação binária
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Valor preditivo negativo
    else:
        sensitivity = specificity = precision_val = npv = float('nan')
    
    # Só calcular AUC se houver ambas as classes
    if len(np.unique(all_labels)) > 1:
        auc_value = roc_auc_score(all_labels, all_probs)
        # PR AUC
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)
        logging.info(f"AUC-ROC: {auc_value:.4f}, AUC-PR: {pr_auc:.4f}")
    else:
        auc_value = float('nan')
        pr_auc = float('nan')
        logging.info("AUCs não puderam ser calculados (apenas uma classe presente)")
    
    # Logging das métricas
    logging.info(f"Acurácia: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Sensibilidade: {sensitivity:.4f}")
    logging.info(f"Especificidade: {specificity:.4f}")
    logging.info(f"Precisão: {precision_val:.4f}")
    logging.info(f"Valor Preditivo Negativo: {npv:.4f}")
    logging.info("Matriz de Confusão:")
    logging.info(f"{cm}")

    # Confusão matriz como dataframe para facilitar visualização
    cm_df = pd.DataFrame(cm)
    
    metrics = {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': accuracy,
        'f1': f1,
        'auc_roc': auc_value,
        'auc_pr': pr_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision_val,
        'npv': npv,
        'cm': cm,
        'cm_df': cm_df,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_probs': all_probs
    }
    
    return metrics

# Função para plotar a matriz de confusão
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'confusion_matrix_vgg16_balanced_augmented.png'))
    plt.close()

# Função para plotar a curva ROC
def plot_roc_curve(all_labels, all_probs):
    plt.figure(figsize=(10, 8))
    
    # Calcular ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'roc_curve_vgg16_balanced_augmented.png'))
    plt.close()
    
    return roc_auc

# Função para plotar a curva de precisão-recall
def plot_pr_curve(all_labels, all_probs):
    plt.figure(figsize=(10, 8))
    
    # Calcular PR
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'pr_curve_vgg16_balanced_augmented.png'))
    plt.close()
    
    return pr_auc

# Função para plotar histórico de treinamento
def plot_training_history(history):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Treinamento')
    plt.plot(history['val_loss'], label='Validação')
    plt.title('Perda durante o treinamento')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Treinamento')
    plt.plot(history['val_acc'], label='Validação')
    plt.title('Acurácia durante o treinamento')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Treinamento')
    plt.plot(history['val_f1'], label='Validação')
    plt.title('F1 Score durante o treinamento')
    plt.xlabel('Época')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_history_vgg16_balanced_augmented.png'))
    plt.close()

# Função de treinamento melhorada com mixed precision e early stopping
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, class_names=None):
    model = model.to(device)
    
    best_model_wts = model.state_dict().copy()
    best_val_auc = 0.0
    early_stop_counter = 0
    
    # Histórico para monitoramento
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        logging.info(f"Época {epoch + 1}/{num_epochs}")
        logging.info("-" * 20)

        # Fase de treinamento
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_train_preds = []
        all_train_labels = []

        # Iterar sobre os dados de treinamento
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero aos gradientes
            optimizer.zero_grad()

            # Forward pass com mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                # Backward + optimize com scaler
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Fallback para precisão padrão
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

            # Estatísticas
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        # Calcular métricas da época de treinamento
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        epoch_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')

        # Salvar histórico de treinamento
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        history['train_f1'].append(epoch_f1)

        logging.info(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")

        # Fase de validação
        val_metrics = evaluate_model(model, dataloaders['val'], criterion)
        
        # Salvar histórico de validação
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Verificar se é o melhor modelo baseado em AUC
        current_val_auc = val_metrics['auc_roc']
        
        if not np.isnan(current_val_auc) and current_val_auc > best_val_auc:
            logging.info(f"Nova melhor AUC de validação: {current_val_auc:.4f}")
            best_val_auc = current_val_auc
            best_model_wts = model.state_dict().copy()
            
            # Salvar o melhor modelo
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1']
            }, os.path.join(CHECKPOINT_DIR, 'best_vgg16_balanced_augmented.pth'))
            
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logging.info(f"Sem melhoria na AUC por {early_stop_counter} épocas")
            
            # Early stopping
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping acionado na época {epoch + 1}")
                break

        # Ajuste do learning rate
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
            
        # Mostrar learning rate atual
        for param_group in optimizer.param_groups:
            logging.info(f"LR: {param_group['lr']:.6f}")
        
        # Plotar histórico de treinamento periodicamente
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs or early_stop_counter >= EARLY_STOPPING_PATIENCE:
            plot_training_history(history)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Treinamento concluído em {elapsed_time/60:.2f} minutos")
    logging.info(f"Melhor AUC na validação: {best_val_auc:.4f}")
    
    # Carregar os melhores pesos
    model.load_state_dict(best_model_wts)
    
    # Plotar histórico de treinamento final
    plot_training_history(history)
    
    return model, history

def main():
    try:
        logging.info("Iniciando treinamento balanceado e aumentado do modelo VGG16 com pesos específicos")
        
        # Carregar os dados com undersampling e data augmentation
        dataloaders, class_names, class_weights = load_data(DATA_DIR, BATCH_SIZE, INPUT_SIZE)
        
        # Instanciar o modelo com os pesos pré-treinados personalizados
        model = VGG16Classifier(
            num_classes=NUM_CLASSES, 
            dropout_rate=0.5,
            weights_path=PRETRAINED_WEIGHTS_PATH
        )
        logging.info(f"Modelo VGG16 com pesos pré-treinados do arquivo {PRETRAINED_WEIGHTS_PATH} carregado")
        
        # Função de perda com pesos de classe
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logging.info(f"Usando CrossEntropyLoss ponderada com pesos: {class_weights.cpu().numpy()}")
        
        # Separar parâmetros para diferentes taxas de aprendizado
        feature_params = [param for name, param in model.named_parameters() 
                        if "features" in name and param.requires_grad]
        classifier_params = [param for name, param in model.named_parameters() 
                           if "classifier" in name and param.requires_grad]
        
        # Otimizador com diferentes learning rates para camadas diferentes
        optimizer = optim.AdamW([
            {'params': feature_params, 'lr': LEARNING_RATE * 0.1},
            {'params': classifier_params, 'lr': LEARNING_RATE}
        ], weight_decay=WEIGHT_DECAY)
        
        # Verificar quantos parâmetros serão treinados
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Parâmetros treináveis: {trainable_params:,} de {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # Scheduler para ajuste da taxa de aprendizado - CosineAnnealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=NUM_EPOCHS, 
            eta_min=LEARNING_RATE * 0.01
        )
        
        # Treinar o modelo
        logging.info(f"Iniciando treinamento do modelo VGG16 com balanceamento e aumento por {NUM_EPOCHS} épocas")
        model, history = train_model(
            model, 
            dataloaders, 
            criterion, 
            optimizer, 
            scheduler, 
            num_epochs=NUM_EPOCHS, 
            class_names=class_names
        )
        
        # Avaliar no conjunto de teste
        logging.info("Avaliando modelo no conjunto de teste")
        test_metrics = evaluate_model(model, dataloaders['test'], criterion)
        
        # Plotar resultados de teste
        plot_confusion_matrix(test_metrics['cm'], class_names)
    except Exception as e:
        logging.exception(f"Ocorreu um erro: {e}")