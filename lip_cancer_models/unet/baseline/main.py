# https://arxiv.org/abs/2310.02486

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os
import logging
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

# Configuração do logging
logging.basicConfig(
    filename='training_output_unet_mobilenetv2.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Definição do módulo Squeeze-and-Excitation (SE)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.adaptive_avg_pool2d(x, (1, 1))
        y = y.view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Definição do módulo Atrous Spatial Pyramid Pooling (ASPP)
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        for rate in rates:
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate))
        self.bottleneck = nn.Conv2d(len(rates) * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        x = torch.cat(features, dim=1)
        x = self.bottleneck(x)
        return x

# Definição do bloco residual
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

# Definição da arquitetura U-Net com MobileNetV2 como backbone
class UNetWithMobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(UNetWithMobileNetV2, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.encoder = mobilenet.features

        # Camadas de expansão (decoder)
        self.decoder4 = self._decoder_block(1280, 512)
        self.decoder3 = self._decoder_block(512, 256)
        self.decoder2 = self._decoder_block(256, 128)
        self.decoder1 = self._decoder_block(128, 64)

        # Camada final
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # Módulos adicionais
        self.se_block = SEBlock(1280)
        self.aspp = ASPP(1280, 256, rates=[1, 6, 12, 18])
        self.res_block = ResidualBlock(1280, 1280)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Codificador
        enc1 = self.encoder[0][0](x)
        enc2 = self.encoder[0][1](enc1)
        enc3 = self.encoder[0][2](enc2)
        enc4 = self.encoder[0][3](enc3)
        enc5 = self.encoder[0][4](enc4)
        enc6 = self.encoder[0][5](enc5)
        enc7 = self.encoder[0][6](enc6)
        enc8 = self.encoder[0][7](enc7)
        enc9 = self.encoder[0][8](enc8)
        enc10 = self.encoder[0][9](enc9)
        enc11 = self.encoder[0][10](enc10)
        enc12 = self.encoder[0][11](enc11)
        enc13 = self.encoder[0][12](enc12)
        enc14 = self.encoder[0][13](enc13)
        enc15 = self.encoder[0][14](enc14)
        enc16 = self.encoder[0][15](enc15)
        enc17 = self.encoder[0][16](enc16)
        enc18 = self.encoder[0][17](enc17)

        # Aplicar módulos adicionais
        x = self.se_block(enc18)
        x = self.aspp(x)
        x = self.res_block(x)

        # Decodificador
        dec4 = self.decoder4(x)
        dec3 = self.decoder3(dec4)
        dec2 = self.decoder2(dec3)
        dec1 = self.decoder1(dec2)

        # Saída final
        out = self.final_conv(dec1)
        return out

# Função para carregar os dados
def load_data(data_dir, batch_size=32, input_size=224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
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
                torch.save(model.state_dict(), 'best_unet_mobilenetv2.pth')
                logging.info("Modelo salvo com nova melhor acurácia!")

    logging.info(f"Melhor acurácia na validação: {best_acc:.4f}")
    return model

# Configurações gerais
DATA_DIR = "/home/Ayrton/Neural_network_cancer"
BATCH_SIZE = 32
INPUT_SIZE = 224  # U-Net e MobileNetV2 usa 224x224
NUM_CLASSES = 2
NUM_EPOCHS = 50

# Carregar os dados
dataloaders = load_data(DATA_DIR, BATCH_SIZE, INPUT_SIZE)

# Instanciar o modelo
model = UNetWithMobileNetV2(num_classes=NUM_CLASSES)

# Definir função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Treinar o modelo
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS)
