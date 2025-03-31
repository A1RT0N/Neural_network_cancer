import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc

# Configuração do logging
logging.basicConfig(
    filename='training_output_vgg16_raw.log',
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

# Definição da arquitetura VGG16 com pesos pré-treinados
class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(VGG16Classifier, self).__init__()
        # Carrega o VGG16 com pesos pré-treinados no ImageNet
        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Congela os parâmetros da rede base para evitar sobreajuste
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Modificar o classificador final
        num_features = self.backbone.classifier[6].in_features
        
        # Substituir a última camada por uma nova com o número correto de classes
        self.backbone.classifier[6] = nn.Linear(num_features, num_classes)
        
        # Adicionar dropout para reduzir sobreajuste
        self.backbone.classifier[5] = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.backbone(x)

# Função para carregar os dados sem aumento e sem balanceamento
def load_data(data_dir, batch_size=32, input_size=224):
    # Transformação básica sem aumento de dados
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'Training')
    val_dir = os.path.join(data_dir, 'Test')

    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=transform),
        'val': datasets.ImageFolder(root=val_dir, transform=transform)
    }
    
    # Sem balanceamento de classes
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'], 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        ),
        'val': DataLoader(
            image_datasets['val'], 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
    }

    return dataloaders, image_datasets

# Função para avaliar o modelo com métricas detalhadas
def evaluate_model(model, dataloader, criterion, class_names):
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
    
    # Só calcular AUC se houver ambas as classes
    if len(np.unique(all_labels)) > 1:
        auc_value = roc_auc_score(all_labels, all_probs)
        logging.info(f"AUC: {auc_value:.4f}")
    else:
        auc_value = float('nan')
        logging.info("AUC não pôde ser calculado (apenas uma classe presente)")

    # Calcular precisão global
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    
    # Logging das métricas
    logging.info(f"Acurácia: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info("Matriz de Confusão:")
    logging.info(f"{cm}")

    # Confusão matriz como dataframe para facilitar visualização
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Criar DataFrame de resultados para análise posterior
    results_df = pd.DataFrame({
        'True_Label': [class_names[i] for i in all_labels],
        'Predicted_Label': [class_names[i] for i in all_preds],
        'Confidence': all_probs
    })
    
    # Salvar resultados detalhados para análise
    results_df.to_csv('vgg16_raw_prediction_results.csv', index=False)
    
    return {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc_value,
        'cm': cm,
        'cm_df': cm_df,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_probs': all_probs
    }

# Função para plotar a matriz de confusão
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig('confusion_matrix_vgg16_raw.png')
    plt.close()

# Função para plotar a curva ROC
def plot_roc_curve(all_labels, all_probs, class_names):
    plt.figure(figsize=(10, 8))
    
    # Calcular ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_vgg16_raw.png')
    plt.close()

# Função para plotar a curva de precisão-recall
def plot_pr_curve(all_labels, all_probs):
    plt.figure(figsize=(10, 8))
    
    # Calcular PR
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="lower left")
    plt.savefig('pr_curve_vgg16_raw.png')
    plt.close()

# Função para plotar histórico de treinamento
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Treinamento')
    plt.plot(history['val_loss'], label='Validação')
    plt.title('Perda durante o treinamento')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Treinamento')
    plt.plot(history['val_acc'], label='Validação')
    plt.title('Acurácia durante o treinamento')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_vgg16_raw.png')
    plt.close()

# Função de treinamento melhorada
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, class_names=None):
    model = model.to(device)
    
    best_model_wts = model.state_dict().copy()
    best_acc = 0.0
    
    # Histórico para monitoramento
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        logging.info(f"Época {epoch + 1}/{num_epochs}")
        logging.info("-" * 20)

        # Cada época tem uma fase de treinamento e validação
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Modo de treinamento
            else:
                model.eval()   # Modo de avaliação

            running_loss = 0.0
            running_corrects = 0

            # Iterar sobre os dados
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero aos gradientes
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize apenas no treinamento
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Estatísticas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calcular métricas da época
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Salvar histórico
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Salvar o melhor modelo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
                torch.save(model.state_dict(), 'best_vgg16_raw.pth')
                logging.info("Modelo salvo com nova melhor acurácia!")
        
        # Ajuste do learning rate
        if scheduler:
            scheduler.step()
            
        logging.info(f"LR atual: {optimizer.param_groups[0]['lr']:.6f}")
        
    # Carregar os melhores pesos
    model.load_state_dict(best_model_wts)
    
    # Plotar histórico de treinamento
    plot_training_history(history)
    
    logging.info(f"Melhor acurácia na validação: {best_acc:.4f}")
    
    # Avaliar o modelo final
    results = evaluate_model(model, dataloaders['val'], criterion, class_names)
    
    # Plotar resultados
    plot_confusion_matrix(results['cm_df'], class_names)
    
    if len(np.unique(results['all_labels'])) > 1:
        plot_roc_curve(results['all_labels'], results['all_probs'], class_names)
        plot_pr_curve(results['all_labels'], results['all_probs'])
    
    return model, history, results

def main():
    # Configurações gerais
    DATA_DIR = "/home/Ayrton/Neural_network_cancer"
    BATCH_SIZE = 32
    INPUT_SIZE = 224  # VGG usa 224x224
    NUM_CLASSES = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Carregar os dados sem aumento e sem balanceamento
    dataloaders, image_datasets = load_data(DATA_DIR, BATCH_SIZE, INPUT_SIZE)
    class_names = image_datasets['train'].classes
    
    logging.info(f"Classes: {class_names}")
    
    # Instanciar o modelo com pesos pré-treinados
    model = VGG16Classifier(num_classes=NUM_CLASSES, freeze_backbone=True)
    logging.info("Modelo VGG16 com pesos pré-treinados carregado")
    
    # Função de perda sem pesos de classe
    criterion = nn.CrossEntropyLoss()
    
    # Otimizador com diferente learning rate para camadas diferentes
    # Congelamos as camadas convolucionais e treinamos apenas o classificador
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:  # Apenas parâmetros não congelados
            params_to_update.append(param)
            
    # Verificar quantos parâmetros serão treinados
    logging.info(f"Número de parâmetros treináveis: {len(params_to_update)}")
    
    optimizer = optim.AdamW(params_to_update, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler para ajuste da taxa de aprendizado
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5, 
        verbose=True
    )
    
    # Treinar o modelo
    logging.info(f"Iniciando treinamento cru do modelo VGG16 por {NUM_EPOCHS} épocas")
    model, history, results = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=NUM_EPOCHS, 
        class_names=class_names
    )
    
    # Salvar modelo final
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': class_names,
        'results': {
            'accuracy': results['accuracy'],
            'f1': results['f1'],
            'auc': results['auc'],
            'confusion_matrix': results['cm'].tolist()
        }
    }, 'final_vgg16_raw_model.pth')
    
    logging.info("Treinamento cru concluído e modelo final salvo!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Ocorreu um erro: {e}")