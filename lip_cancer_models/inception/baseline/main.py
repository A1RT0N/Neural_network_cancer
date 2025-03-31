import os
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_curve, auc, roc_auc_score
import seaborn as sns
from PIL import Image

# Setup logging
logging.basicConfig(
    filename='training_output_inception_resnet_v2_raw.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
DATA_DIR = "/home/Ayrton/Neural_network_cancer"
BATCH_SIZE = 16
INPUT_SIZE = 299  # InceptionResNetV2 uses 299x299 input
NUM_CLASSES = 2
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints_inception_resnet_v2_raw')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Path to the pre-downloaded weights
PRETRAINED_WEIGHTS = "/home/Ayrton/.cache/torch/hub/checkpoints/inception_resnet_v2-940b1cd6.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Data transformations - only basic resizing and normalization, no augmentation
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset that handles XML files alongside images
class CancerDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    # Only process image files
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        path = os.path.join(class_dir, filename)
                        self.samples.append((path, self.class_to_idx[class_name]))
        
        logging.info(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logging.error(f"Error loading image {path}: {str(e)}")
            # Return a placeholder if an image fails to load
            return torch.zeros((3, INPUT_SIZE, INPUT_SIZE)), label

# Load datasets
def load_data():
    train_dir = os.path.join(DATA_DIR, 'Training')
    val_dir = os.path.join(DATA_DIR, 'Validation')
    test_dir = os.path.join(DATA_DIR, 'Test')
    
    # Use the same transform for all datasets - no augmentation
    train_dataset = CancerDataset(train_dir, transform=transform)
    val_dataset = CancerDataset(val_dir, transform=transform)
    test_dataset = CancerDataset(test_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    logging.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

# Define InceptionResNetV2 model (not available in standard torchvision)
class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionResNetV2, self).__init__()
        # We'll load the pretrained weights from the file
        # This is just a placeholder class structure
        self.features = nn.Sequential(
            # Placeholder for the feature extraction layers
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1536, num_classes)  # InceptionResNetV2 has 1536 features
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Model definition - Using timm library for InceptionResNetV2
def create_model(num_classes):
    # Since standard torchvision doesn't have InceptionResNetV2, we'll import timm
    try:
        import timm
        logging.info("Using timm library for InceptionResNetV2")
        # Initialize model with pretrained=False to avoid download
        model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=num_classes)
        
        # Load pre-trained weights manually from local file
        if os.path.exists(PRETRAINED_WEIGHTS):
            logging.info(f"Loading pre-trained weights from: {PRETRAINED_WEIGHTS}")
            state_dict = torch.load(PRETRAINED_WEIGHTS, map_location=device)
            
            # Filter out classifier weights if needed
            if 'last_linear.weight' in state_dict and 'fc.weight' in model.state_dict():
                state_dict['fc.weight'] = state_dict.pop('last_linear.weight')
                state_dict['fc.bias'] = state_dict.pop('last_linear.bias')
            
            # Handle potential naming differences between model and state_dict
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            logging.info(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained weights")
        else:
            logging.warning(f"Pre-trained weights file not found at {PRETRAINED_WEIGHTS}. Using random initialization.")
        
        # Freeze early layers
        for name, param in model.named_parameters():
            if 'conv2d_7b' not in name and 'fc' not in name:  # Only fine-tune last conv block and fc
                param.requires_grad = False
        
        # Replace the final classifier layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    except ImportError:
        logging.error("timm library not found. Using a placeholder InceptionResNetV2 model structure.")
        logging.error("Please install timm using: pip install timm")
        model = InceptionResNetV2(num_classes=num_classes)
        
        # Load pre-trained weights would require adjusting to match the placeholder structure
        # This is challenging without the proper model architecture
        # It's strongly recommended to install timm instead
        
    model = model.to(device)
    logging.info(f"Created InceptionResNetV2 model with {num_classes} output classes")
    return model

# Training function
def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if i % 10 == 0:
            logging.info(f'Epoch: {epoch}, Batch: {i}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100. * correct / total
    logging.info(f'Epoch {epoch} training completed. Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    return epoch_loss, epoch_acc

# Validation function
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = 100. * correct / total
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    logging.info(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {f1:.4f}')
    return val_loss, val_acc, f1, all_preds, all_targets

# Function to plot metrics
def plot_metrics(train_losses, val_losses, train_accs, val_accs, f1_scores):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over epochs')
    
    plt.subplot(1, 3, 3)
    plt.plot(f1_scores, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score over epochs')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_metrics.png'))
    logging.info("Saved training metrics plot")

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'confusion_matrix.png'))
    logging.info("Saved confusion matrix plot")

# Function to plot precision-recall curve
def plot_pr_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, marker='.', label=f'AUPRC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'pr_curve.png'))
    logging.info(f"Saved PR curve plot. AUPRC: {pr_auc:.4f}")
    return pr_auc

# Function to plot ROC curve
def plot_roc_curve(y_true, y_scores):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, marker='.', label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'roc_curve.png'))
    logging.info(f"Saved ROC curve plot. ROC AUC: {roc_auc:.4f}")
    return roc_auc

# Main training loop
def train_model():
    train_loader, val_loader, test_loader, class_idx = load_data()
    class_names = {v: k for k, v in class_idx.items()}
    logging.info(f"Class mapping: {class_names}")
    
    model = create_model(NUM_CLASSES)
    
    # Use standard cross entropy loss without class weights
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay - two groups with different learning rates
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad], 'lr': LEARNING_RATE * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'fc' in n], 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                   patience=5, verbose=True)
    
    # Initialize tracking variables
    best_val_f1 = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    f1_scores = []
    
    start_time = time.time()
    
    # Main training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        logging.info(f"Starting epoch {epoch}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        f1_scores.append(val_f1)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save model if it's the best so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            logging.info(f"Saved new best model with F1 score: {val_f1:.4f}")
        
        # Plot and save metrics every 10 epochs
        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            plot_metrics(train_losses, val_losses, train_accs, val_accs, f1_scores)
    
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time/60:.2f} minutes")
    
    # Final evaluation
    evaluate_model(model, test_loader, criterion, class_names)
    
    return model

# Function to evaluate model on test set
def evaluate_model(model, test_loader, criterion, class_names):
    logging.info("Starting final evaluation on test set...")
    
    # Regular validation metrics
    test_loss, test_acc, test_f1, all_preds, all_targets = validate(model, test_loader, criterion)
    
    # Confusion matrix
    plot_confusion_matrix(all_targets, all_preds, list(class_names.values()))
    
    # Get class probabilities for ROC and PR curves
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
            all_targets.extend(targets.numpy())
    
    # PR curve and AUC
    pr_auc = plot_pr_curve(all_targets, all_probs)
    
    # ROC curve and AUC
    roc_auc = plot_roc_curve(all_targets, all_probs)
    
    # Log all metrics
    logging.info(f"""
    Final Test Results:
    -------------------
    Loss:      {test_loss:.4f}
    Accuracy:  {test_acc:.2f}%
    F1 Score:  {test_f1:.4f}
    AUPRC:     {pr_auc:.4f}
    ROC AUC:   {roc_auc:.4f}
    """)
    
    # Save results to a separate file
    with open(os.path.join(CHECKPOINT_DIR, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
        f.write(f"AUPRC: {pr_auc:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")

if __name__ == "__main__":
    logging.info("Starting raw cancer classification with InceptionResNetV2")
    try:
        trained_model = train_model()
        logging.info("Program completed successfully")
    except Exception as e:
        logging.exception(f"An error occurred: {e}")