import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd

# Import metrics from scikit-learn
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

logging.basicConfig(
    filename='training_output_efficientnet_finetuning.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Definição da arquitetura EfficientNet com fine-tuning
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetClassifier, self).__init__()
        # Carregar o modelo EfficientNet com pesos pré-treinados
        self.backbone = models.efficientnet_b0(pretrained=True)  # Com pré-treino
        # Ajustar a última camada para o número de classes desejado
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Função para carregar os dados
def load_data(data_dir, batch_size=32, input_size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dir = os.path.join(data_dir, 'Training')
    val_dir = os.path.join(data_dir, 'Test')

    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(root=val_dir, transform=data_transforms['val'])
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    return dataloaders

# Função para congelar algumas camadas do modelo
def freeze_layers(model):
    for param in model.backbone.features.parameters():
        param.requires_grad = False

# Função de avaliação do modelo
def evaluate_model(model, dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.max(dim=1).values.cpu().numpy())

    all_labels = pd.Series(all_labels, name='True Labels')
    all_preds = pd.Series(all_preds, name='Predicted Labels')
    all_probs = pd.Series(all_probs, name='Confidence Scores')

    # Calcular métricas
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    logging.info("Confusion Matrix:")
    logging.info(f"{cm}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"AUC: {auc:.4f}")

    return total_loss / len(dataloader.dataset)

# Função de treinamento
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_acc = 0.0
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_efficientnet_finetuning.pth')
                logging.info("Modelo salvo com nova melhor acurácia!")

    logging.info(f"Melhor acurácia na validação: {best_acc:.4f}")
    return model

# Configurações gerais
DATA_DIR = "/home/Ayrton/Neural_network_cancer"
BATCH_SIZE = 32
INPUT_SIZE = 224  # Tamanho de entrada para EfficientNet
NUM_CLASSES = 2
NUM_EPOCHS = 50

# Carregar os dados
dataloaders = load_data(DATA_DIR, BATCH_SIZE, INPUT_SIZE)

# Instanciar o modelo
model = EfficientNetClassifier(num_classes=NUM_CLASSES)

# Congelar camadas do backbone para fine-tuning
freeze_layers(model)

# Definir função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Treinar o modelo
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS)
