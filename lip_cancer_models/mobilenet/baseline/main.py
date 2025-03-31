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
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_recall_curve, auc

# Configuração do logging
logging.basicConfig(
    filename='training_output_mobilenet_v2_raw.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configurações gerais
DATA_DIR = "/home/Ayrton/Neural_network_cancer"
BATCH_SIZE = 32
INPUT_SIZE = 224  # MobileNetV2 usa 224x224
NUM_CLASSES = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints_mobilenet_v2_raw')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Path to the pre-downloaded weights
PRETRAINED_WEIGHTS = "/home/Ayrton/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth"

# Verificar disponibilidade de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Definição da arquitetura MobileNetV2 com ajustes para classificação e carregamento de pesos pré-treinados
class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None):
        super(MobileNetV2Classifier, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=False)
        
        # Carregar pesos pré-treinados se especificado
        if pretrained_path and os.path.exists(pretrained_path):
            logging.info(f"Loading pre-trained weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=device)
            self.backbone.load_state_dict(state_dict)
            logging.info("Pre-trained weights loaded successfully")
        else:
            logging.warning(f"Pre-trained weights file not found at {pretrained_path}. Using random initialization.")
        
        # Congelar camadas iniciais para evitar overfitting
        for param in list(self.backbone.parameters())[:-30]:  # Só treina as últimas camadas
            param.requires_grad = False
            
        # Modificar classificador final
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Função para carregar os dados sem aumento de dados
def load_data(data_dir, batch_size=32, input_size=224):
    # Transformação básica sem aumento de dados
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'Training')
    val_dir = os.path.join(data_dir, 'Validation')
    test_dir = os.path.join(data_dir, 'Test')

    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=transform),
        'val': datasets.ImageFolder(root=val_dir, transform=transform),
        'test': datasets.ImageFolder(root=test_dir, transform=transform)
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, 
                     shuffle=(x == 'train'), num_workers=4, pin_memory=True)
        for x in ['train', 'val', 'test']
    }
    
    # Registrar informações sobre os datasets
    class_to_idx = image_datasets['train'].class_to_idx
    class_names = list(class_to_idx.keys())
    
    for phase in ['train', 'val', 'test']:
        logging.info(f"{phase} dataset size: {len(image_datasets[phase])}")
    
    logging.info(f"Classes: {class_names}, Class mapping: {class_to_idx}")
    
    return dataloaders, class_names

# Função de avaliação do modelo com métricas e visualizações detalhadas
def evaluate_model(model, dataloader, criterion, class_names, phase='val'):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            # Extrair probabilidades da classe positiva (assumindo classe 1 como positiva)
            if NUM_CLASSES == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                # Se for multiclasse, pegamos a maior probabilidade
                all_probs.extend(probs.max(dim=1).values.cpu().numpy())

    # Calcular métricas
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    # Matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)
    
    # F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # AUC-ROC (apenas para classificação binária)
    if NUM_CLASSES == 2:
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(all_labels, all_probs)
            pr_auc = auc(recall, precision)
        except:
            logging.warning("Couldn't calculate ROC-AUC, possibly only one class present in batch")
            roc_auc = float('nan')
            pr_auc = float('nan')
    else:
        roc_auc = float('nan')
        pr_auc = float('nan')
    
    # Logging de resultados
    logging.info(f"{phase} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, F1: {f1:.4f}")
    if NUM_CLASSES == 2:
        logging.info(f"{phase} - ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    logging.info(f"{phase} - Confusion Matrix:\n{cm}")
    
    # Criar visualizações só para o conjunto de teste
    if phase == 'test':
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(CHECKPOINT_DIR, 'confusion_matrix.png'))
        
        # Plot ROC curve para classificação binária
        if NUM_CLASSES == 2 and not np.isnan(roc_auc):
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, marker='.', label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')  # linha diagonal
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(CHECKPOINT_DIR, 'roc_curve.png'))
            
            # Plot Precision-Recall curve
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, marker='.', label=f'PR curve (AUC = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            plt.savefig(os.path.join(CHECKPOINT_DIR, 'pr_curve.png'))
    
    results = {
        'loss': epoch_loss,
        'accuracy': epoch_acc.item(),
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }
    
    return results

# Função de treinamento aprimorada
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25):
    model = model.to(device)
    
    # Tracking de métricas para visualização
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []
    
    best_val_f1 = 0.0
    best_model_wts = None
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info("-" * 20)

        # Fase de treino
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        train_f1s.append(epoch_f1)
        
        logging.info(f"Train - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}")
        
        # Fase de validação
        val_results = evaluate_model(model, dataloaders['val'], criterion, class_names, 'val')
        
        val_losses.append(val_results['loss'])
        val_accs.append(val_results['accuracy'])
        val_f1s.append(val_results['f1_score'])
        
        # Atualizar learning rate se scheduler fornecido
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_results['loss'])
            else:
                scheduler.step()
        
        # Salvar melhor modelo baseado no F1 score
        if val_results['f1_score'] > best_val_f1:
            best_val_f1 = val_results['f1_score']
            best_model_wts = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_results['f1_score'],
                'val_loss': val_results['loss'],
                'val_acc': val_results['accuracy'],
            }, os.path.join(CHECKPOINT_DIR, 'best_mobilenet_v2_raw_model.pth'))
            logging.info(f"Saved new best model with F1 score: {best_val_f1:.4f}")
        
        # Plotar métricas a cada 10 épocas
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            plot_training_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s)
    
    logging.info(f"Best validation F1 score: {best_val_f1:.4f}")
    
    # Carregar os melhores pesos no modelo final
    model.load_state_dict(best_model_wts)
    return model

# Função para plotar métricas de treinamento
def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    
    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label='Train F1 Score')
    plt.plot(val_f1s, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score over Epochs')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_metrics.png'))
    logging.info("Saved training metrics plot")

# Função principal
def main():
    try:
        logging.info("Starting raw cancer classification with MobileNetV2")
        
        # Carregar os dados
        dataloaders, class_names = load_data(DATA_DIR, BATCH_SIZE, INPUT_SIZE)
        
        # Instanciar o modelo com pesos pré-treinados
        model = MobileNetV2Classifier(num_classes=NUM_CLASSES, pretrained_path=PRETRAINED_WEIGHTS)
        
        # Definir função de perda sem pesos de classe
        criterion = nn.CrossEntropyLoss()
        
        # Otimizador com diferenciação de learning rate para camadas diferentes
        optimizer = optim.AdamW([
            {'params': [param for name, param in model.named_parameters() if 'classifier' not in name], 'lr': LEARNING_RATE * 0.1},
            {'params': model.backbone.classifier.parameters(), 'lr': LEARNING_RATE}
        ], weight_decay=WEIGHT_DECAY)
        
        # Scheduler para ajustar learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Treinar o modelo
        model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)
        
        # Avaliar no conjunto de teste
        logging.info("Evaluating model on test set...")
        test_results = evaluate_model(model, dataloaders['test'], criterion, class_names, 'test')
        
        # Salvar resultados finais
        with open(os.path.join(CHECKPOINT_DIR, 'test_results.txt'), 'w') as f:
            f.write(f"Test Loss: {test_results['loss']:.4f}\n")
            f.write(f"Test Accuracy: {test_results['accuracy']:.4f}\n")
            f.write(f"Test F1 Score: {test_results['f1_score']:.4f}\n")
            f.write(f"Test ROC-AUC: {test_results['roc_auc']:.4f}\n")
            f.write(f"Test PR-AUC: {test_results['pr_auc']:.4f}\n")
        
        logging.info("Training and evaluation completed successfully")
        
    except Exception as e:
        logging.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()