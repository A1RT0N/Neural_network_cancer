import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import f1_score
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dinov2.models import dinov2_vitb14  # Certifique-se de que a biblioteca DinoV2 está instalada

logging.basicConfig(
    filename='training_dinov2.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Classe para usar DINOv2
class DinoV2Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DinoV2Classifier, self).__init__()
        # Carregar o modelo DINOv2
        self.backbone = dinov2_vitb14(pretrained=True)  # Usar pesos pré-treinados
        self.backbone.head = nn.Identity()  # Remover a camada final

        # Adicionar uma nova cabeça para classificação
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def evaluate_model(model, dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    all_labels = []
    all_preds = []
    all_probs = []
    all_sample_names = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.max(dim=1).values.cpu().numpy())
            all_sample_names.extend(paths)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    df = pd.DataFrame({
        'sample_name': all_sample_names,
        'y_test': all_labels,
        'pred_score': all_probs,
        'pred_class': all_preds
    })
    df.to_csv('confidence_scores_dinov2.csv', index=False)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    f1 = f1_score(all_labels, all_preds)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    auc = roc_auc_score(all_labels, all_probs)

    logging.info("Confusion Matrix:")
    logging.info(f"{cm}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"False Positive Rate (FPR): {fpr:.4f}")
    logging.info(f"False Negative Rate (FNR): {fnr:.4f}")
    logging.info(f"AUC: {auc:.4f}")

    plt.figure(figsize=(10, 6))
    plt.hist(df.loc[df['y_test'] == 1, 'pred_score'], bins=50, alpha=0.6, label='Correct Positive')
    plt.hist(df.loc[df['y_test'] == 0, 'pred_score'], bins=50, alpha=0.6, label='Correct Negative')
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Confidence Score Distribution")
    plt.savefig('confidence_score_distribution_dinov2.png')

    return total_loss / len(dataloader.dataset)

def load_data(data_dir, batch_size=32, input_size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    train_dir = os.path.join(data_dir, 'Training')
    val_dir = os.path.join(data_dir, 'Test')

    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(root=val_dir, transform=data_transforms['val'])
    }

    dataloaders = {
        x: DataLoader(
            [
                (sample[0], sample[1], image_datasets[x].samples[i][0]) 
                for i, sample in enumerate(image_datasets[x])
            ],
            batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in ['train', 'val']
    }

    return dataloaders, {x: len(image_datasets[x]) for x in ['train', 'val']}

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

            for inputs, labels, _ in dataloaders[phase]:
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

            if phase == 'val':
                evaluate_model(model, dataloaders['val'], criterion)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'best_model_dinov2.pth')
                    logging.info("Modelo salvo com nova melhor acurácia!")

    logging.info(f"Melhor acurácia na validação: {best_acc:.4f}")
    return model

# Configurações
DATA_DIR = "/home/Ayrton/Neural_network_cancer"
BATCH_SIZE = 32
INPUT_SIZE = 224
NUM_CLASSES = 2
NUM_EPOCHS = 50

# Carregar os dados
dataloaders, dataset_sizes = load_data(DATA_DIR, BATCH_SIZE, INPUT_SIZE)

# Instanciar o modelo com DINOv2
model = DinoV2Classifier(num_classes=NUM_CLASSES)

# Definir função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Treinar o modelo
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS)
