import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import logging
import os

# Configurar logging para salvar logs em arquivo
logging.basicConfig(
    filename='training_output.log',  # Nome do arquivo de log
    level=logging.INFO,              # Nível de log
    format='%(asctime)s - %(levelname)s - %(message)s'
)




class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNeXtClassifier, self).__init__()
        # Carregar ConvNeXt pré-treinado
        self.backbone = models.convnext_base(pretrained=True)
        # Ajustar a última camada para o número de classes
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
    
    
def load_data(data_dir, batch_size=32, input_size=224):
    # Transformações para treinamento e validação
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

    # Ajustar pastas conforme estrutura
    train_dir = os.path.join(data_dir, 'Training')
    val_dir = os.path.join(data_dir, 'Test')

    # Carregar datasets
    image_datasets = {
        'train': datasets.ImageFolder(
            root=train_dir,
            transform=data_transforms['train']
        ),
        'val': datasets.ImageFolder(
            root=val_dir,
            transform=data_transforms['val']
        )
    }

    # DataLoader para treino e validação
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    return dataloaders, {x: len(image_datasets[x]) for x in ['train', 'val']}

import logging
import os

# Configurar logging para salvar logs em arquivo
logging.basicConfig(
    filename='training_output.log',  # Nome do arquivo de log
    level=logging.INFO,              # Nível de log
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

            # Salvar melhores pesos
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')
                logging.info("Modelo salvo com nova melhor acurácia!")

    logging.info(f"Melhor acurácia na validação: {best_acc:.4f}")
    return model




DATA_DIR = "/home/Ayrton/Neural_network_cancer"  # Atualize com o caminho correto no cluster
BATCH_SIZE = 16  
INPUT_SIZE = 128  
NUM_CLASSES = 2  
NUM_EPOCHS = 20 

# Preparação dos dados
dataloaders, dataset_sizes = load_data(DATA_DIR, BATCH_SIZE, INPUT_SIZE)

# Inicialização do modelo
model = ConvNeXtClassifier(num_classes=NUM_CLASSES)


# Definir a função de perda e otimizador

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Treinamento
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS)

