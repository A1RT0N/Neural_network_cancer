import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
import logging
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_recall_curve, auc

# Set up logging
logging.basicConfig(
    filename='training_output_mobilenet_v2_balanced_augmented.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if torch.cuda.is_available() else None
np.random.seed(SEED)
random.seed(SEED)

# General configurations
DATA_DIR = "/home/Ayrton/Neural_network_cancer"
BATCH_SIZE = 32
INPUT_SIZE = 224  # MobileNetV2 uses 224x224
NUM_CLASSES = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
UNDERSAMPLE_RATIO = 0.4  # Keep 40% of majority class samples
EARLY_STOPPING_PATIENCE = 15
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints_mobilenet_v2_balanced_augmented')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Path to the pre-downloaded weights
PRETRAINED_WEIGHTS = "/home/Ayrton/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth"

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Mixed precision training setup
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Custom dataset class with undersampling capability
class CancerDataset(Dataset):
    def __init__(self, root_dir, transform=None, undersample_majority=False, undersample_ratio=0.5):
        """
        Args:
            root_dir (string): Directory with class subdirectories
            transform (callable, optional): Transform to be applied on images
            undersample_majority (bool): Whether to undersample the majority class
            undersample_ratio (float): Proportion of majority class samples to keep (0.0-1.0)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect samples by class
        class_samples = {cls: [] for cls in self.classes}
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    img_path = os.path.join(class_dir, filename)
                    class_samples[class_name].append(img_path)
        
        # Count samples in each class
        class_counts = {cls: len(samples) for cls, samples in class_samples.items()}
        logging.info(f"Before undersampling - Class counts: {class_counts}")
        
        # Find majority and minority classes
        if len(class_counts) > 0:
            majority_class = max(class_counts, key=class_counts.get)
            minority_class = min(class_counts, key=class_counts.get)
            
            # Apply undersampling if requested
            if undersample_majority and "Training" in root_dir:
                # Calculate how many samples to keep
                keep_count = int(class_counts[majority_class] * undersample_ratio)
                # Randomly sample from majority class
                class_samples[majority_class] = random.sample(class_samples[majority_class], keep_count)
                # Update counts
                class_counts[majority_class] = len(class_samples[majority_class])
                logging.info(f"After undersampling - Class counts: {class_counts}")
        
        # Prepare final samples list
        self.samples = []
        for class_name, paths in class_samples.items():
            for path in paths:
                self.samples.append((path, self.class_to_idx[class_name]))
        
        logging.info(f"Loaded {len(self.samples)} total samples from {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            # Return a placeholder image if loading fails
            return torch.zeros((3, INPUT_SIZE, INPUT_SIZE)), label

# Define transforms with augmentation for training
def get_transforms(input_size, is_training=False):
    if is_training:
        # Strong data augmentation for training (geometric transforms that preserve lesion features)
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),  # Moderate rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0),  # Subtle color changes
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Basic transforms for validation/test
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# Enhanced MobileNetV2 classifier with improved regularization
class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None, dropout_rate=0.3):
        super(MobileNetV2Classifier, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=False)
        
        # Load pre-trained weights if available
        if pretrained_path and os.path.exists(pretrained_path):
            logging.info(f"Loading pre-trained weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=device)
            self.backbone.load_state_dict(state_dict)
            logging.info("Pre-trained weights loaded successfully")
        else:
            logging.warning(f"Pre-trained weights file not found at {pretrained_path}. Using random initialization.")
        
        # Progressive freezing strategy - freeze early layers
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Then selectively unfreeze later layers
        # MobileNetV2 features has 18 blocks (0-17)
        for i, block in enumerate(self.backbone.features):
            if i >= 14:  # Only unfreeze last 4 blocks (adjust as needed)
                for param in block.parameters():
                    param.requires_grad = True
        
        # Enhanced classifier with dropout for regularization
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        # Make sure classifier parameters are trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    def forward(self, x):
        return self.backbone(x)

# Function to load datasets with undersampling
def load_data(data_dir, batch_size=32, input_size=224):
    train_dir = os.path.join(data_dir, 'Training')
    val_dir = os.path.join(data_dir, 'Validation')
    test_dir = os.path.join(data_dir, 'Test')

    # Create datasets with appropriate transforms
    train_dataset = CancerDataset(
        train_dir,
        transform=get_transforms(input_size, is_training=True),
        undersample_majority=True,
        undersample_ratio=UNDERSAMPLE_RATIO
    )
    
    val_dataset = CancerDataset(
        val_dir,
        transform=get_transforms(input_size, is_training=False)
    )
    
    test_dataset = CancerDataset(
        test_dir,
        transform=get_transforms(input_size, is_training=False)
    )

    # Create data loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }
    
    # Get class names and compute class weights for loss function
    class_to_idx = train_dataset.class_to_idx
    class_names = list(class_to_idx.keys())
    
    # Compute class weights for weighted loss function
    class_counts = np.bincount([label for _, label in train_dataset])
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    
    logging.info(f"Classes: {class_names}, Class mapping: {class_to_idx}")
    logging.info(f"Class weights for loss function: {class_weights}")
    
    return dataloaders, class_names, class_weights

# Enhanced model evaluation function
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
            
            # Extract class probabilities
            if NUM_CLASSES == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                all_probs.extend(probs.max(dim=1).values.cpu().numpy())

    # Calculate metrics
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Calculate additional metrics for binary classification
    if NUM_CLASSES == 2 and len(np.unique(all_labels)) > 1:
        try:
            # ROC-AUC
            roc_auc = roc_auc_score(all_labels, all_probs)
            
            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(all_labels, all_probs)
            pr_auc = auc(recall, precision)
            
            # Sensitivity (Recall) and Specificity
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            logging.info(f"{phase} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, F1: {f1:.4f}")
            logging.info(f"{phase} - ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
            logging.info(f"{phase} - Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Precision: {precision_val:.4f}")
        except Exception as e:
            logging.warning(f"Error calculating ROC-AUC: {e}")
            roc_auc = float('nan')
            pr_auc = float('nan')
            sensitivity = float('nan')
            specificity = float('nan')
            precision_val = float('nan')
    else:
        logging.info(f"{phase} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, F1: {f1:.4f}")
        roc_auc = float('nan')
        pr_auc = float('nan')
        sensitivity = float('nan') 
        specificity = float('nan')
        precision_val = float('nan')
    
    logging.info(f"{phase} - Confusion Matrix:\n{cm}")
    
    # Create visualizations for test set
    if phase == 'test':
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(CHECKPOINT_DIR, 'confusion_matrix.png'))
        
        # Plot ROC curve for binary classification
        if NUM_CLASSES == 2 and not np.isnan(roc_auc):
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, marker='.', label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
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
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision_val,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }
    
    return results

# Enhanced training function with mixed precision and early stopping
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25):
    model = model.to(device)
    
    # Tracking metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []
    
    best_val_f1 = 0.0
    best_model_wts = None
    early_stop_counter = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info("-" * 20)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Use mixed precision if available
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                # Scale loss and perform backward pass
                scaler.scale(loss).backward()
                
                # Apply gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step optimizer and update scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision fallback
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
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
        
        # Validation phase
        val_results = evaluate_model(model, dataloaders['val'], criterion, class_names, 'val')
        
        val_losses.append(val_results['loss'])
        val_accs.append(val_results['accuracy'])
        val_f1s.append(val_results['f1_score'])
        
        # Update learning rate
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_results['loss'])
            else:
                scheduler.step()
        
        # Check for best model and save
        if val_results['f1_score'] > best_val_f1:
            best_val_f1 = val_results['f1_score']
            best_model_wts = model.state_dict().copy()
            early_stop_counter = 0  # Reset counter
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_f1': val_results['f1_score'],
                'val_loss': val_results['loss'],
                'val_acc': val_results['accuracy'],
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'train_f1s': train_f1s,
                'val_f1s': val_f1s
            }, os.path.join(CHECKPOINT_DIR, 'best_mobilenet_v2_balanced_augmented_model.pth'))
            
            logging.info(f"Saved new best model with F1 score: {best_val_f1:.4f}")
        else:
            # Increment early stopping counter
            early_stop_counter += 1
            logging.info(f"No improvement for {early_stop_counter} epochs")
            
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Plot metrics every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1 or early_stop_counter >= EARLY_STOPPING_PATIENCE:
            plot_training_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {elapsed_time/60:.2f} minutes")
    logging.info(f"Best validation F1 score: {best_val_f1:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Function to plot training metrics
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

# Main function
def main():
    try:
        logging.info("Starting balanced and augmented cancer classification with MobileNetV2")
        
        # Load data with undersampling and augmentation
        dataloaders, class_names, class_weights = load_data(DATA_DIR, BATCH_SIZE, INPUT_SIZE)
        
        # Create model with improved regularization
        model = MobileNetV2Classifier(
            num_classes=NUM_CLASSES, 
            pretrained_path=PRETRAINED_WEIGHTS,
            dropout_rate=0.3
        )
        
        # Use weighted cross entropy loss
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logging.info(f"Using weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy()}")
        
        # Optimizer with differential learning rates
        optimizer = optim.AdamW([
            {'params': [param for name, param in model.named_parameters() 
                       if 'classifier' not in name and param.requires_grad], 
             'lr': LEARNING_RATE * 0.1},
            {'params': model.backbone.classifier.parameters(), 
             'lr': LEARNING_RATE}
        ], weight_decay=WEIGHT_DECAY)
        
        # Cosine annealing scheduler for smoother decay
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=NUM_EPOCHS, 
            eta_min=LEARNING_RATE * 0.01
        )
        
        # Train the model with all improvements
        model = train_model(
            model, 
            dataloaders, 
            criterion, 
            optimizer, 
            scheduler, 
            num_epochs=NUM_EPOCHS
        )
        
        # Evaluate on test set
        logging.info("Evaluating model on test set...")
        test_results = evaluate_model(model, dataloaders['test'], criterion, class_names, 'test')
        
        # Save final test results
        with open(os.path.join(CHECKPOINT_DIR, 'test_results.txt'), 'w') as f:
            f.write(f"Test Loss: {test_results['loss']:.4f}\n")
            f.write(f"Test Accuracy: {test_results['accuracy']:.4f}\n")
            f.write(f"Test F1 Score: {test_results['f1_score']:.4f}\n")
            f.write(f"Test ROC-AUC: {test_results['roc_auc']:.4f}\n")
            f.write(f"Test PR-AUC: {test_results['pr_auc']:.4f}\n")
            f.write(f"Test Sensitivity: {test_results['sensitivity']:.4f}\n")
            f.write(f"Test Specificity: {test_results['specificity']:.4f}\n")
            f.write(f"Test Precision: {test_results['precision']:.4f}\n")
            
            # Add confusion matrix
            f.write("\nConfusion Matrix:\n")
            cm = test_results['confusion_matrix']
            for row in cm:
                f.write(f"{row}\n")
        
        logging.info("Training and evaluation completed successfully")
        
    except Exception as e:
        logging.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()