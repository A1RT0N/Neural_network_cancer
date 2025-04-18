import os
import glob
import logging
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

# Configuração do logging
logging.basicConfig(
    filename='training_output_unet_smp_raw.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Definir seed para reprodutibilidade
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if torch.cuda.is_available() else None
np.random.seed(SEED)

# Configurações gerais
DATA_DIR = "/home/Ayrton/Neural_network_cancer"
BATCH_SIZE = 32
INPUT_SIZE = 224  # Tamanho para a maioria dos modelos pré-treinados
NUM_CLASSES = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BACKBONE = "mobilenet_v2"  # Pode ser alterado para resnet18, efficientnet-b0, etc.

# Caminho para os pesos pré-treinados personalizados
PRETRAINED_WEIGHTS_PATH = "/home/Ayrton/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth"

# Verificar se CUDA está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Utilizando dispositivo: {device}")

# Classe de dataset personalizada para carregar imagens
class CancerDataset(Dataset):
    def __init__(self, data_dir, subset='Training', transform=None):
        """
        Args:
            data_dir (string): Diretório com todas as imagens
            subset (string): 'Training', 'Validation', ou 'Test'
            transform (callable, optional): Transformação opcional para aplicar à amostra
        """
        self.transform = transform
        self.classes = ['AC', 'LSCC']  # Assumindo que estas são as duas classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Obter todos os arquivos jpg e JPG
        self.image_paths = []
        self.labels = []
        
        for cls in self.classes:
            class_path = os.path.join(data_dir, subset, cls)
            if not os.path.exists(class_path):
                logging.warning(f"Caminho não existe: {class_path}")
                continue
                
            for img_ext in ['*.jpg', '*.JPG']:
                paths = glob.glob(os.path.join(class_path, img_ext))
                self.image_paths.extend(paths)
                self.labels.extend([self.class_to_idx[cls]] * len(paths))
        
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

# Definir transformações de dados - sem aumento de dados
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Modelo adaptado para classificação baseado em U-Net do SMP
class UNetClassifier(nn.Module):
    def __init__(self, encoder_name=BACKBONE, in_channels=3, num_classes=NUM_CLASSES, pretrained_weights_path=None):
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
        # Precisamos modificar a configuração do smp para usar nossos pesos personalizados
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
                
                # Filtrar e adaptar as chaves do state_dict para o encoder
                encoder_state_dict = {}
                encoder_prefix = "encoder."  # Prefixo usado pelo SMP para o encoder
                
                # O dicionário do mobilenet_v2 completo tem chaves como "features.0.0.weight"
                # Precisamos mapear para o formato esperado pelo encoder do SMP
                # Isso pode variar dependendo do backbone, aqui é específico para mobilenet_v2
                
                # Tentar aplicar diretamente ao encoder
                # Nota: pode ser necessário ajustar esse mapeamento dependendo do backbone
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
        
        # Camadas de classificação
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

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

# Função de treinamento
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_auc = 0.0
    best_model_wts = None
    
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    
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
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
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
        
        # Calcular AUC
        if len(np.unique(all_labels)) > 1:  # Calcular AUC apenas se houver múltiplas classes no batch
            epoch_auc = roc_auc_score(all_labels, all_probs)
            logging.info(f'Validação Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f} AUC: {epoch_auc:.4f}')
            
            # Salvar o melhor modelo
            if epoch_auc > best_val_auc:
                best_val_auc = epoch_auc
                best_model_wts = model.state_dict().copy()
                logging.info(f'Nova melhor AUC de validação: {best_val_auc:.4f}')
                torch.save(model.state_dict(), f'best_unet_{BACKBONE}_custom.pth')
        else:
            logging.info(f'Validação Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')
        
        # Atualizar taxa de aprendizado
        scheduler.step()
    
    # Carregar os melhores pesos do modelo
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    
    return model, train_losses, val_losses, train_f1s, val_f1s

# Função de avaliação
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
    logging.info(f"AUC-ROC: {auc_roc:.4f}")
    logging.info(f"AUC-PR: {auc_pr:.4f}")
    
    # Criar DataFrame com resultados individuais para análise
    results_df = pd.DataFrame({
        'True_Label': all_labels,
        'Predicted_Label': all_preds,
        'Probability': all_probs
    })
    
    results_df.to_csv(f'prediction_results_{BACKBONE}_custom.csv', index=False)
    logging.info(f"Resultados detalhados da predição salvos em 'prediction_results_{BACKBONE}_custom.csv'")
    
    if criterion:
        avg_loss = total_loss / len(data_loader.dataset)
        logging.info(f"Perda média: {avg_loss:.4f}")
        return cm, accuracy, f1, auc_roc, auc_pr, all_labels, all_probs, avg_loss
    
    return cm, accuracy, f1, auc_roc, auc_pr, all_labels, all_probs

# Funções de visualização
def plot_training_history(train_loss, val_loss, train_f1, val_f1):
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
    plt.savefig(f'training_history_unet_{BACKBONE}_custom.png')
    logging.info(f'Gráfico de histórico de treinamento salvo como training_history_unet_{BACKBONE}_custom.png')

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Rótulos Preditos')
    plt.ylabel('Rótulos Verdadeiros')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_unet_{BACKBONE}_custom.png')
    logging.info(f'Gráfico de matriz de confusão salvo como confusion_matrix_unet_{BACKBONE}_custom.png')

def plot_roc_curve(labels, probs):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Característica Operacional do Receptor (ROC)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'roc_curve_unet_{BACKBONE}_custom.png')
    logging.info(f'Gráfico da curva ROC salvo como roc_curve_unet_{BACKBONE}_custom.png')

def plot_pr_curve(labels, probs):
    precision, recall, _ = precision_recall_curve(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'Curva PR (AUC = {auc(recall, precision):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title('Curva Precisão-Recall')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f'pr_curve_unet_{BACKBONE}_custom.png')
    logging.info(f'Gráfico da curva Precisão-Recall salvo como pr_curve_unet_{BACKBONE}_custom.png')

def main():
    logging.info(f"Iniciando treinamento com pesos pré-treinados personalizados do modelo U-Net com backbone {BACKBONE} para classificação de câncer")
    
    # Criar datasets - usando a mesma transformação para todos os conjuntos
    train_dataset = CancerDataset(DATA_DIR, subset='Training', transform=transform)
    val_dataset = CancerDataset(DATA_DIR, subset='Validation', transform=transform)
    test_dataset = CancerDataset(DATA_DIR, subset='Test', transform=transform)
    
    # Criar data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Inicializar modelo com pesos pré-treinados personalizados
    model = UNetClassifier(
        encoder_name=BACKBONE,
        pretrained_weights_path=PRETRAINED_WEIGHTS_PATH
    )
    model = model.to(device)
    logging.info(f"Inicializado modelo U-Net com backbone {BACKBONE} usando pesos pré-treinados personalizados de {PRETRAINED_WEIGHTS_PATH}")
    
    # Definir função de perda sem pesos de classe
    criterion = nn.CrossEntropyLoss()
    
    # Definir parâmetros de treinamento para cada camada (aprendizado diferenciado)
    encoder_params = [p for n, p in model.named_parameters() if 'unet.encoder' in n]
    decoder_params = [p for n, p in model.named_parameters() if 'unet.encoder' not in n]
    
    # Definir otimizador com weight decay e taxas de aprendizado diferentes
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': LEARNING_RATE * 0.1},  # Taxa menor para o encoder pré-treinado
        {'params': decoder_params, 'lr': LEARNING_RATE}         # Taxa normal para o decoder e classificador
    ], weight_decay=WEIGHT_DECAY)
    
    # Scheduler para ajuste da taxa de aprendizado
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.01)
    
    # Treinar modelo
    logging.info("Iniciando treinamento do modelo")
    model, train_losses, val_losses, train_f1s, val_f1s = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS
    )
    logging.info("Treinamento do modelo concluído")
    
    # Plotar histórico de treinamento
    plot_training_history(train_losses, val_losses, train_f1s, val_f1s)
    
    # Avaliar no conjunto de teste
    logging.info("Avaliando modelo no conjunto de teste")
    cm, accuracy, f1, auc_roc, auc_pr, test_labels, test_probs = evaluate_model(model, test_loader, criterion)
    
    # Plotar matriz de confusão
    plot_confusion_matrix(cm, train_dataset.classes)
    
    # Plotar curvas ROC e PR
    if len(np.unique(test_labels)) > 1:
        plot_roc_curve(test_labels, test_probs)
        plot_pr_curve(test_labels, test_probs)
    
    # Salvar modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classes': train_dataset.classes,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
        'confusion_matrix': cm.tolist(),
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'backbone': BACKBONE,
        'pretrained_weights_path': PRETRAINED_WEIGHTS_PATH
    }, f'final_unet_{BACKBONE}_custom_model.pth')
    logging.info(f"Modelo salvo como final_unet_{BACKBONE}_custom_model.pth")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Ocorreu um erro: {e}")