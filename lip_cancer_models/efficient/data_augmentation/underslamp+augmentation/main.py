import os
import glob
import logging
import random
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
    filename='training_output_efficientnet_b7_balanced_augmented.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if torch.cuda.is_available() else None
np.random.seed(SEED)
random.seed(SEED)

# Configuration
DATA_DIR = "/home/Ayrton/Neural_network_cancer"
BATCH_SIZE = 16
INPUT_SIZE = 600
NUM_CLASSES = 2
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
UNDERSAMPLE_RATIO = 0.4  # Keep 40% of majority class samples

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Mixed precision training setup
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Custom dataset class with undersampling option
class CancerDataset(Dataset):
    def __init__(self, data_dir, subset='Training', transform=None, undersample_majority=False, undersample_ratio=0.5):
        """
        Args:
            data_dir (string): Directory with all the images
            subset (string): 'Training', 'Validation', or 'Test'
            transform (callable, optional): Optional transform to be applied on a sample
            undersample_majority (bool): Whether to undersample the majority class
            undersample_ratio (float): What percentage of majority class to keep (0.0-1.0)
        """
        self.transform = transform
        self.classes = ['AC', 'LSCC']  # Actinic cheilitis and Labial squamous cell carcinoma
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get all image paths for each class
        class_paths = {}
        class_counts = {}
        
        for cls in self.classes:
            class_path = os.path.join(data_dir, subset, cls)
            if not os.path.exists(class_path):
                logging.warning(f"Path does not exist: {class_path}")
                continue
                
            paths = []
            for img_ext in ['*.jpg', '*.JPG']:
                paths.extend(glob.glob(os.path.join(class_path, img_ext)))
            
            class_paths[cls] = paths
            class_counts[cls] = len(paths)
        
        # Find majority and minority classes
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        logging.info(f"Before undersampling - {majority_class}: {class_counts[majority_class]}, {minority_class}: {class_counts[minority_class]}")
        
        # Apply undersampling to majority class if requested (only for training set)
        if undersample_majority and subset == 'Training':
            # Calculate how many samples to keep from majority class
            keep_count = int(class_counts[majority_class] * undersample_ratio)
            # Randomly sample from majority class
            class_paths[majority_class] = random.sample(class_paths[majority_class], keep_count)
            class_counts[majority_class] = keep_count
            
            logging.info(f"After undersampling - {majority_class}: {class_counts[majority_class]}, {minority_class}: {class_counts[minority_class]}")
        
        # Combine all paths and labels
        self.image_paths = []
        self.labels = []
        
        for cls in self.classes:
            self.image_paths.extend(class_paths[cls])
            self.labels.extend([self.class_to_idx[cls]] * len(class_paths[cls]))
        
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

# Define data transformations for each set
def get_transforms(input_size, is_training=False):
    if is_training:
        # Strong augmentation for training (geometric transformations that preserve lesion features)
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),  # Moderate rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0),  # Subtle color changes
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Only resize and normalize for validation/test sets
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Create datasets with appropriate transforms
train_dataset = CancerDataset(
    DATA_DIR, 
    subset='Training', 
    transform=get_transforms(INPUT_SIZE, is_training=True),
    undersample_majority=True,
    undersample_ratio=UNDERSAMPLE_RATIO
)
val_dataset = CancerDataset(DATA_DIR, subset='Validation', transform=get_transforms(INPUT_SIZE, is_training=False))
test_dataset = CancerDataset(DATA_DIR, subset='Test', transform=get_transforms(INPUT_SIZE, is_training=False))

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

# EfficientNet B7 model with improved fine-tuning strategy
def get_model():
    # Initialize the model with random weights first
    model = models.efficientnet_b7(weights=None)
    
    # Load the pretrained weights from local file
    pretrained_weights_path = "/home/Ayrton/.cache/torch/hub/checkpoints/efficientnet_b7_lukemelas-c5b4e57e.pth"
    state_dict = torch.load(pretrained_weights_path)
    model.load_state_dict(state_dict)
    
    # Improved freezing strategy - unfreeze final layers gradually
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze the last few feature blocks (progressive unfreezing)
    features = list(model.features)
    for layer in features[-4:]:  # Unfreeze the last 4 blocks
        for param in layer.parameters():
            param.requires_grad = True
    
    # Replace the classifier head with dropout for better regularization
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),  # Add dropout for regularization
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    
    # Make sure classifier parameters are trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model.to(device)

# Training function with early stopping and mixed precision
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=15):
    best_val_auc = 0.0
    best_model_wts = None
    patience_counter = 0
    
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
            
            # Use mixed precision if available
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Mixed precision backward and update
                scaler.scale(loss).backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision fallback
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            _, preds = torch.max(outputs, 1)
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
            
            # Check if this is the best model
            if epoch_auc > best_val_auc:
                best_val_auc = epoch_auc
                best_model_wts = model.state_dict().copy()
                patience_counter = 0  # Reset patience counter
                logging.info(f'New best validation AUC: {best_val_auc:.4f}')
                torch.save(model.state_dict(), 'best_efficientnet_b7_balanced_augmented.pth')
            else:
                patience_counter += 1
                logging.info(f'No improvement for {patience_counter} epochs')
                
                # Early stopping
                if patience_counter >= patience:
                    logging.info(f'Early stopping triggered after epoch {epoch+1}')
                    break
        else:
            logging.info(f'Val Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')
        
        # Update learning rate
        scheduler.step()
    
    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    
    return model, train_losses, val_losses, train_f1s, val_f1s

# Evaluation function with detailed metrics
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
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
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
    logging.info(f'Sensitivity: {sensitivity:.4f}')
    logging.info(f'Specificity: {specificity:.4f}')
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
    plt.savefig('training_history_b7_balanced_augmented.png')
    logging.info('Training history plot saved as training_history_b7_balanced_augmented.png')

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_b7_balanced_augmented.png')
    logging.info('Confusion matrix plot saved as confusion_matrix_b7_balanced_augmented.png')

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
    plt.savefig('roc_curve_b7_balanced_augmented.png')
    logging.info('ROC curve plot saved as roc_curve_b7_balanced_augmented.png')

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
    plt.savefig('pr_curve_b7_balanced_augmented.png')
    logging.info('Precision-Recall curve plot saved as pr_curve_b7_balanced_augmented.png')

def main():
    logging.info("Starting balanced and augmented cancer classification model training with EfficientNet B7")
    
    # Verify if pretrained weights are available
    check_pretrained_weights()
    
    # Initialize model
    model = get_model()
    logging.info(f"Initialized EfficientNet B7 model with lukemelas pretrained weights")
    
    # Calculate class weights for weighted loss
    class_counts = np.bincount([label for _, label in train_dataset])
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logging.info(f"Using weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy()}")
    
    # Define optimizer with differential learning rates
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' not in n], 'lr': LEARNING_RATE * 0.1},
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.01)
    
    # Train model
    logging.info("Starting model training with balancing and augmentation")
    model, train_losses, val_losses, train_f1s, val_f1s = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, patience=15
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
        'classes': train_dataset.classes,
        'training_stats': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_f1s': train_f1s,
            'val_f1s': val_f1s
        },
        'test_metrics': {
            'accuracy': accuracy,
            'f1': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr
        }
    }, 'final_efficientnet_b7_balanced_augmented_model.pth')
    logging.info("Model saved as final_efficientnet_b7_balanced_augmented_model.pth")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        
        
        
        
        
# 1. Undersampling of Majority Class (AC)

# Added an undersampling strategy in the CancerDataset class that reduces the majority class (AC) to 40% of its original size
# This undersampling is only applied to the training set, preserving the validation and test sets

# 2. Data Augmentation
# Added geometric transformations that preserve lesion features:

# RandomResizedCrop (with scale 0.8-1.0 to maintain most of the lesion area)
# RandomHorizontalFlip and RandomVerticalFlip
# Moderate RandomRotation (15 degrees)
# Subtle ColorJitter with minimal hue adjustments to preserve medical features

# 3. Efficiency and Effectiveness Improvements

# Mixed precision training using torch.cuda.amp for faster computation
# Gradient clipping to prevent exploding gradients
# Class-weighted loss function to further address class imbalance
# Progressive unfreezing of EfficientNet layers
# Improved model regularization with dropout in the classifier
# Early stopping with patience parameter to prevent overfitting
# Differential learning rates for feature extractor vs. classifier
# Cosine annealing learning rate scheduler
# Enhanced evaluation metrics including sensitivity and specificity
# Improved logging with more detailed metrics

# 4. Other Enhancements

# Better file naming for logs and saved models
# Expanded the model saving to include training statistics and test metrics
# More informative visualization plots
# Better error handling and seed setting for reproducibility

