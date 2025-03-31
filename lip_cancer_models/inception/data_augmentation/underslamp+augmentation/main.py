import os
import logging
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_curve, auc, roc_auc_score
import seaborn as sns
from PIL import Image

# Setup logging
logging.basicConfig(
    filename='training_output_inception_resnet_v2_balanced_augmented.log',
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
INPUT_SIZE = 299  # InceptionResNetV2 uses 299x299 input
NUM_CLASSES = 2
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
UNDERSAMPLE_RATIO = 0.4  # Keep 40% of majority class samples
EARLY_STOPPING_PATIENCE = 15
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints_inception_resnet_v2_balanced_augmented')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Path to the pre-downloaded weights
PRETRAINED_WEIGHTS = "/home/Ayrton/.cache/torch/hub/checkpoints/inception_resnet_v2-940b1cd6.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Mixed precision training setup
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Custom dataset that handles undersampling and class balancing
class CancerDataset(torch.utils.data.Dataset):
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
        self.classes = sorted(os.listdir(root_dir))  # Sort to ensure consistent class indexing
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Collect samples by class
        class_samples = {cls: [] for cls in self.classes}
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    # Only process image files
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        path = os.path.join(class_dir, filename)
                        class_samples[class_name].append((path, self.class_to_idx[class_name]))
        
        # Get class counts for logging
        class_counts = {cls: len(samples) for cls, samples in class_samples.items()}
        logging.info(f"Before undersampling - Class counts: {class_counts}")
        
        # Identify majority class
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        # Apply undersampling to majority class if requested
        if undersample_majority and "Training" in root_dir:
            # Calculate how many samples to keep from majority class
            keep_count = int(class_counts[majority_class] * undersample_ratio)
            # Randomly select samples to keep
            class_samples[majority_class] = random.sample(class_samples[majority_class], keep_count)
            
            # Update class counts after undersampling
            class_counts[majority_class] = len(class_samples[majority_class])
            logging.info(f"After undersampling - Class counts: {class_counts}")
        
        # Combine all samples
        self.samples = []
        for samples in class_samples.values():
            self.samples.extend(samples)
        
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

# Define data transformations with augmentation for training
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

# Load datasets with appropriate transforms and undersampling
def load_data():
    train_dir = os.path.join(DATA_DIR, 'Training')
    val_dir = os.path.join(DATA_DIR, 'Validation')
    test_dir = os.path.join(DATA_DIR, 'Test')
    
    # Use different transforms for training vs. validation/testing
    train_dataset = CancerDataset(
        train_dir, 
        transform=get_transforms(INPUT_SIZE, is_training=True),
        undersample_majority=True,
        undersample_ratio=UNDERSAMPLE_RATIO
    )
    val_dataset = CancerDataset(val_dir, transform=get_transforms(INPUT_SIZE, is_training=False))
    test_dataset = CancerDataset(test_dir, transform=get_transforms(INPUT_SIZE, is_training=False))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    logging.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Compute class weights for the loss function
    class_counts = np.bincount([label for _, label in train_dataset])
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    logging.info(f"Class weights for loss function: {class_weights}")
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx, class_weights

# Define InceptionResNetV2 model (placeholder for when timm isn't available)
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

# Model definition - Using timm library for InceptionResNetV2 with improved fine-tuning
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
        
        # First freeze all parameters for better controlled fine-tuning
        for param in model.parameters():
            param.requires_grad = False
        
        # Progressively unfreeze the last few blocks
        # This is a more controlled approach than the original code
        for name, param in model.named_parameters():
            # Only fine-tune later layers for transfer learning
            if any(layer_name in name for layer_name in ['conv2d_7a', 'conv2d_7b', 'block8', 'fc']):
                param.requires_grad = True
        
        # Replace the final classifier layer with dropout for regularization
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),  # Add dropout for regularization
            nn.Linear(in_features, num_classes)
        )
        
        # Make sure all classifier parameters are trainable
        for param in model.fc.parameters():
            param.requires_grad = True
        
    except ImportError:
        logging.error("timm library not found. Using a placeholder InceptionResNetV2 model structure.")
        logging.error("Please install timm using: pip install timm")
        model = InceptionResNetV2(num_classes=num_classes)
        
    model = model.to(device)
    
    # Count and log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Created InceptionResNetV2 model with {num_classes} output classes")
    logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model

# Training function with mixed precision support
def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Use mixed precision if available
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
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
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Statistics
        _, predicted = outputs.max(1)
        running_loss += loss.item() * inputs.size(0)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        if i % 10 == 0:
            logging.info(f'Epoch: {epoch}, Batch: {i}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
    
    logging.info(f'Epoch {epoch} training completed. Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, F1: {epoch_f1:.4f}')
    return epoch_loss, epoch_acc, epoch_f1

# Validation function with probability scores for ROC/PR curves
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Get class predictions
            _, predicted = outputs.max(1)
            
            # Get probability scores for ROC/PR curves
            probs = torch.softmax(outputs, dim=1)
            pos_probs = probs[:, 1].cpu().numpy()  # Probability of positive class
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(pos_probs)
    
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # Calculate sensitivity, specificity, and AUC if possible
    if len(np.unique(all_targets)) > 1:
        cm = confusion_matrix(all_targets, all_preds)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc_score = roc_auc_score(all_targets, all_probs)
        
        logging.info(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {f1:.4f}, '
                     f'Sens: {sensitivity:.4f}, Spec: {specificity:.4f}, AUC: {auc_score:.4f}')
    else:
        logging.info(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {f1:.4f}')
    
    return val_loss, val_acc, f1, all_preds, all_targets, all_probs

# Function to plot metrics
def plot_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s):
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
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Validation F1')
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
    
    # Calculate sensitivity, specificity and other metrics from confusion matrix
    if cm.shape == (2, 2):  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value (precision)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        
        logging.info(f"Sensitivity (Recall): {sensitivity:.4f}")
        logging.info(f"Specificity: {specificity:.4f}")
        logging.info(f"Positive Predictive Value (Precision): {ppv:.4f}")
        logging.info(f"Negative Predictive Value: {npv:.4f}")
    
    return cm

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

# Main training loop with early stopping
def train_model():
    train_loader, val_loader, test_loader, class_idx, class_weights = load_data()
    class_names = {v: k for k, v in class_idx.items()}
    logging.info(f"Class mapping: {class_names}")
    
    model = create_model(NUM_CLASSES)
    
    # Use weighted cross entropy loss
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logging.info(f"Using weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy()}")
    
    # Optimizer with weight decay - two groups with different learning rates
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad], 'lr': LEARNING_RATE * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'fc' in n], 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler - CosineAnnealingLR for smoother decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.01)
    
    # Initialize tracking variables
    best_val_f1 = 0.0
    early_stop_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    
    start_time = time.time()
    
    # Main training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        logging.info(f"Starting epoch {epoch}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_targets, val_probs = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        # Update learning rate
        scheduler.step()
        
        # Save model if it's the best so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0  # Reset counter
            
            # Save the model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'class_names': class_names,
                'class_weights': class_weights.cpu()
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            
            logging.info(f"Saved new best model with F1 score: {val_f1:.4f}")
        else:
            # Increment early stopping counter
            early_stop_counter += 1
            logging.info(f"No improvement for {early_stop_counter} epochs")
            
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Plot and save metrics every 10 epochs or on the last epoch
        if epoch % 10 == 0 or epoch == NUM_EPOCHS or early_stop_counter >= EARLY_STOPPING_PATIENCE:
            plot_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s)
    
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time/60:.2f} minutes")
    
    # Load the best model for final evaluation
    best_checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    logging.info(f"Loaded best model from epoch {best_checkpoint['epoch']} with validation F1 score: {best_checkpoint['val_f1']:.4f}")
    
    # Final evaluation
    evaluate_model(model, test_loader, criterion, class_names)
    
    return model

# Function to evaluate model on test set
def evaluate_model(model, test_loader, criterion, class_names):
    logging.info("Starting final evaluation on test set...")
    
    # Regular validation metrics
    test_loss, test_acc, test_f1, all_preds, all_targets, all_probs = validate(model, test_loader, criterion)
    
    # Confusion matrix
    cm = plot_confusion_matrix(all_targets, all_preds, list(class_names.values()))
    
    # PR curve and AUC
    pr_auc = plot_pr_curve(all_targets, all_probs)
    
    # ROC curve and AUC
    roc_auc = plot_roc_curve(all_targets, all_probs)
    
    # Calculate detailed metrics from confusion matrix
    if cm.shape == (2, 2):  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # F1 score can also be calculated from precision and recall
        f1_from_pr = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Log all metrics
        logging.info(f"""
        Final Test Results:
        -------------------
        Loss:         {test_loss:.4f}
        Accuracy:     {test_acc:.2f}%
        F1 Score:     {test_f1:.4f}
        Sensitivity:  {sensitivity:.4f}
        Specificity:  {specificity:.4f}
        Precision:    {precision:.4f}
        AUPRC:        {pr_auc:.4f}
        ROC AUC:      {roc_auc:.4f}
        """)
        
        # Save results to a separate file
        with open(os.path.join(CHECKPOINT_DIR, 'test_results.txt'), 'w') as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
            f.write(f"F1 Score: {test_f1:.4f}\n")
            f.write(f"Sensitivity (Recall): {sensitivity:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"AUPRC: {pr_auc:.4f}\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"TN: {tn}, FP: {fp}\n")
            f.write(f"FN: {fn}, TP: {tp}\n")
    else:
        # Multi-class metrics
        logging.info(f"""
        Final Test Results:
        -------------------
        Loss:      {test_loss:.4f}
        Accuracy:  {test_acc:.2f}%
        F1 Score:  {test_f1:.4f}
        AUPRC:     {pr_auc:.4f}
        ROC AUC:   {roc_auc:.4f}
        """)
        
        with open(os.path.join(CHECKPOINT_DIR, 'test_results.txt'), 'w') as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
            f.write(f"F1 Score: {test_f1:.4f}\n")
            f.write(f"AUPRC: {pr_auc:.4f}\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n")

if __name__ == "__main__":
    logging.info("Starting balanced and augmented cancer classification with InceptionResNetV2")
    try:
        trained_model = train_model()
        logging.info("Program completed successfully")
    except Exception as e:
        logging.exception(f"An error occurred: {e}")