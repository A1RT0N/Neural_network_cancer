import os
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_curve, auc, roc_auc_score
import seaborn as sns
from PIL import Image

# Set up logging
logging.basicConfig(
    filename='training_output_efficientnet_b7_raw.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if torch.cuda.is_available() else None
np.random.seed(SEED)

# Configuration
DATA_DIR = "/home/Ayrton/Neural_network_cancer"
BATCH_SIZE = 16
INPUT_SIZE = 600
NUM_CLASSES = 2
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Custom dataset class for loading images
class CancerDataset(Dataset):
    def __init__(self, data_dir, subset='Training', transform=None):
        """
        Args:
            data_dir (string): Directory with all the images
            subset (string): 'Training', 'Validation', or 'Test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.transform = transform
        self.classes = ['AC', 'LSCC']  # Assuming these are the two classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get all jpg and JPG files
        self.image_paths = []
        self.labels = []
        
        for cls in self.classes:
            class_path = os.path.join(data_dir, subset, cls)
            if not os.path.exists(class_path):
                logging.warning(f"Path does not exist: {class_path}")
                continue
                
            for img_ext in ['*.jpg', '*.JPG']:
                paths = glob.glob(os.path.join(class_path, img_ext))
                self.image_paths.extend(paths)
                self.labels.extend([self.class_to_idx[cls]] * len(paths))
        
        logging.info(f"Loaded {len(self.image_paths)} images for {subset} set")
    
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
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder image if the original image cannot be loaded
            placeholder = torch.zeros((3, INPUT_SIZE, INPUT_SIZE))
            return placeholder, label

# Define data transformations - only basic resizing and normalization, no augmentation
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets - using the same transform for all sets
train_dataset = CancerDataset(DATA_DIR, subset='Training', transform=transform)
val_dataset = CancerDataset(DATA_DIR, subset='Validation', transform=transform)
test_dataset = CancerDataset(DATA_DIR, subset='Test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Verification of pretrained weights
def check_pretrained_weights():
    weights_path = "/home/Ayrton/.cache/torch/hub/checkpoints/efficientnet_b7_lukemelas-c5b4e57e.pth"
    if not os.path.exists(weights_path):
        logging.error(f"Pretrained weights file not found at: {weights_path}")
        raise FileNotFoundError(f"Pretrained weights file not found at: {weights_path}")
    else:
        logging.info(f"Pretrained weights file found at: {weights_path}")
    return weights_path

# EfficientNet B7 model with local pretrained weights
def get_model():
    # Initialize the model with random weights first
    model = models.efficientnet_b7(weights=None)
    
    # Load the pretrained weights from local file
    pretrained_weights_path = "/home/Ayrton/.cache/torch/hub/checkpoints/efficientnet_b7_lukemelas-c5b4e57e.pth"
    state_dict = torch.load(pretrained_weights_path)
    model.load_state_dict(state_dict)
    
    # Freeze initial layers to prevent overfitting
    # B7 has more layers, so we freeze more initial layers
    for param in list(model.parameters())[:-40]:  # Adjusted for B7
        param.requires_grad = False
        
    # Replace the classifier head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    return model.to(device)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_auc = 0.0
    best_model_wts = None
    
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    
    for epoch in range(num_epochs):
        # Training phase
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
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        logging.info(f'Train Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')
        
        # Validation phase
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
                
                # Collect probabilities for ROC curve
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
        
        # Calculate AUC
        if len(np.unique(all_labels)) > 1:  # Only calculate AUC if there are multiple classes in the batch
            epoch_auc = roc_auc_score(all_labels, all_probs)
            logging.info(f'Val Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f} AUC: {epoch_auc:.4f}')
            
            # Save the best model
            if epoch_auc > best_val_auc:
                best_val_auc = epoch_auc
                best_model_wts = model.state_dict().copy()
                logging.info(f'New best validation AUC: {best_val_auc:.4f}')
                torch.save(model.state_dict(), 'best_efficientnet_b7_raw_model.pth')
        else:
            logging.info(f'Val Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')
        
        # Update learning rate
        scheduler.step()
    
    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    
    return model, train_losses, val_losses, train_f1s, val_f1s

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Get probability scores for ROC and PR curves
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    
    # Calculate metrics
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
    
    logging.info(f'Confusion Matrix:\n{cm}')
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')
    logging.info(f'AUC-ROC: {auc_roc:.4f}')
    logging.info(f'AUC-PR: {auc_pr:.4f}')
    
    return cm, accuracy, f1, auc_roc, auc_pr, all_labels, all_probs

# Visualization functions
def plot_training_history(train_loss, val_loss, train_f1, val_f1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_loss, label='Training Loss')
    ax1.plot(val_loss, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    ax2.plot(train_f1, label='Training F1')
    ax2.plot(val_f1, label='Validation F1')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Training and Validation F1 Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_b7_raw.png')
    logging.info('Training history plot saved as training_history_b7_raw.png')

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_b7_raw.png')
    logging.info('Confusion matrix plot saved as confusion_matrix_b7_raw.png')

def plot_roc_curve(labels, probs):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve_b7_raw.png')
    logging.info('ROC curve plot saved as roc_curve_b7_raw.png')

def plot_pr_curve(labels, probs):
    precision, recall, _ = precision_recall_curve(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {auc(recall, precision):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('pr_curve_b7_raw.png')
    logging.info('Precision-Recall curve plot saved as pr_curve_b7_raw.png')

def main():
    logging.info("Starting raw cancer classification model training with EfficientNet B7")
    
    # Verify if pretrained weights are available
    check_pretrained_weights()
    
    # Initialize model
    model = get_model()
    logging.info(f"Initialized EfficientNet B7 model with lukemelas pretrained weights")
    
    # Define loss function without class weights
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer with weight decay
    optimizer = optim.AdamW(
        [
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' not in n], 'lr': LEARNING_RATE * 0.1},
            {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
        ],
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.01)
    
    # Train model
    logging.info("Starting model training")
    model, train_losses, val_losses, train_f1s, val_f1s = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS
    )
    logging.info("Model training completed")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_f1s, val_f1s)
    
    # Evaluate on test set
    logging.info("Evaluating model on test set")
    cm, accuracy, f1, auc_roc, auc_pr, test_labels, test_probs = evaluate_model(model, test_loader)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, train_dataset.classes)
    
    # Plot ROC and PR curves
    if len(np.unique(test_labels)) > 1:
        plot_roc_curve(test_labels, test_probs)
        plot_pr_curve(test_labels, test_probs)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classes': train_dataset.classes
    }, 'final_efficientnet_b7_raw_model.pth')
    logging.info("Model saved as final_efficientnet_b7_raw_model.pth")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"An error occurred: {e}")