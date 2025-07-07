"""
Rigorous training system for 3 methods comparison
"""

# Import necessary modules
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import IRIS components
from iris_implementation_fixed import ORACLE, CASTLE, NEXUS, compute_similarity, evaluate_retrieval

# Import Adaptive Hybrid components
from adaptive_hybrid_retrieval_complete import AdaptiveHybridModel, QAFF, ContrastiveLoss

class RigorousDataset(Dataset):
    """High-quality dataset for rigorous training"""
    
    def __init__(self, num_samples=2000, num_classes=50, split='train', seed=42):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.split = split
        self.image_size = 224
        
        # Set seed for reproducibility
        np.random.seed(seed + hash(split) % 1000)
        
        # Generate consistent class patterns
        self.class_patterns = {}
        for class_id in range(num_classes):
            # Each class has unique color and pattern characteristics
            base_color = np.array([
                (class_id * 73) % 256,
                (class_id * 137) % 256, 
                (class_id * 211) % 256
            ], dtype=np.uint8)
            
            pattern_type = class_id % 4  # 4 different pattern types
            self.class_patterns[class_id] = {
                'base_color': base_color,
                'pattern_type': pattern_type,
                'noise_level': 20 + (class_id % 30)
            }
        
        # Generate sample assignments
        self.samples = []
        samples_per_class = num_samples // num_classes
        for class_id in range(num_classes):
            for i in range(samples_per_class):
                self.samples.append(class_id)
        
        # Add remaining samples
        remaining = num_samples - len(self.samples)
        for i in range(remaining):
            self.samples.append(i % num_classes)
        
        # Shuffle samples
        np.random.shuffle(self.samples)
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        class_id = self.samples[idx]
        pattern = self.class_patterns[class_id]
        
        # Create deterministic but varied image for this index and class
        np.random.seed(idx + class_id * 1000)
        
        # Base image with class color
        image = np.full((self.image_size, self.image_size, 3), 
                       pattern['base_color'], dtype=np.uint8)
        
        # Add pattern based on pattern type
        if pattern['pattern_type'] == 0:  # Circles
            center_x, center_y = self.image_size // 2, self.image_size // 2
            radius = 30 + (class_id % 40)
            y, x = np.ogrid[:self.image_size, :self.image_size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            image[mask] = (image[mask] * 0.7 + 50).astype(np.uint8)
            
        elif pattern['pattern_type'] == 1:  # Stripes
            stripe_width = 10 + (class_id % 20)
            for i in range(0, self.image_size, stripe_width * 2):
                image[i:i+stripe_width, :] = (image[i:i+stripe_width, :] * 0.8 + 30).astype(np.uint8)
                
        elif pattern['pattern_type'] == 2:  # Checkerboard
            block_size = 15 + (class_id % 25)
            for i in range(0, self.image_size, block_size):
                for j in range(0, self.image_size, block_size):
                    if (i // block_size + j // block_size) % 2 == 0:
                        image[i:i+block_size, j:j+block_size] = (
                            image[i:i+block_size, j:j+block_size] * 0.6 + 40
                        ).astype(np.uint8)
                        
        else:  # Gradient
            for i in range(self.image_size):
                factor = i / self.image_size
                image[i, :] = (image[i, :] * (0.5 + factor * 0.5) + factor * 30).astype(np.uint8)
        
        # Add controlled noise
        noise = np.random.randint(-pattern['noise_level'], pattern['noise_level'], 
                                 (self.image_size, self.image_size, 3))
        image = np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, class_id

class IRISModelWrapper(nn.Module):
    """Wrapper for IRIS model with proper interface"""
    
    def __init__(self, output_dim=512, num_classes=50):
        super().__init__()
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Initialize IRIS components with correct parameters
        self.oracle = ORACLE()
        self.castle = CASTLE()
        self.nexus = NEXUS()
        
        # Add projection layer to match output_dim
        self.feature_projector = nn.Linear(512, output_dim)  # IRIS outputs 512 by default
        
        # Classification head
        self.classifier = nn.Linear(output_dim, num_classes)
        
    def forward(self, x, return_features=False):
        # IRIS forward pass
        oracle_features = self.oracle(x)
        castle_features = self.castle(oracle_features)
        nexus_features = self.nexus(castle_features)
        
        # Project to desired output dimension
        projected_features = self.feature_projector(nexus_features)
        
        if return_features:
            return projected_features
        
        # Classification
        logits = self.classifier(projected_features)
        return logits

class SpCaModelWrapper(nn.Module):
    """Improved SpCa model wrapper"""
    
    def __init__(self, output_dim=512, num_classes=50):
        super().__init__()
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Backbone
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=True)
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(backbone_dim, backbone_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_dim, backbone_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim // 4, backbone_dim),
            nn.Sigmoid()
        )
        
        # Feature projector
        self.feature_projector = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        # Classifier
        self.classifier = nn.Linear(output_dim, num_classes)
        
    def forward(self, x, return_features=False):
        # Extract backbone features
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        # Apply attention mechanisms
        spatial_att = self.spatial_attention(features)
        channel_att = self.channel_attention(features).unsqueeze(-1).unsqueeze(-1)
        
        # Apply attention
        attended_features = features * spatial_att * channel_att
        
        # Global average pooling
        pooled_features = torch.mean(attended_features, dim=[2, 3])
        
        # Project features
        projected_features = self.feature_projector(pooled_features)
        
        if return_features:
            return projected_features
        
        # Classification
        logits = self.classifier(projected_features)
        return logits

class AdaptiveHybridWrapper(nn.Module):
    """Wrapper for Adaptive Hybrid model"""
    
    def __init__(self, output_dim=512, num_classes=50, **kwargs):
        super().__init__()
        # Create the model with correct parameters
        self.model = AdaptiveHybridModel(
            backbone='resnet18',
            pretrained=True,
            output_dim=output_dim
        )
        
        # Add classification head
        self.classifier = nn.Linear(output_dim, num_classes)
        
    def forward(self, x, return_features=False):
        # Extract features
        sc_gem, regional_gem, scale_gem = self.model(x)
        
        if return_features:
            return sc_gem  # Return main feature for retrieval
        else:
            # Classification
            logits = self.classifier(sc_gem)
            return logits

class RigorousTrainer:
    """Rigorous training system with comprehensive monitoring"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.training_history = {}
        
    def create_model(self, method_name, **kwargs):
        """Create model based on method name"""
        if method_name == 'iris':
            return IRISModelWrapper(**kwargs)
        elif method_name == 'spca':
            return SpCaModelWrapper(**kwargs)
        elif method_name == 'adaptive':
            return AdaptiveHybridWrapper(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    def train_model(self, model, train_loader, val_loader, method_name, 
                   num_epochs=20, lr=0.001, save_dir='rigorous_models'):
        """Train model with rigorous monitoring"""
        
        print(f"\n{'='*60}")
        print(f"TRAINING {method_name.upper()}")
        print(f"{'='*60}")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        method_dir = os.path.join(save_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        
        # Setup training
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_times': []
        }
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        
        print(f"Training configuration:")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Learning rate: {lr}")
        print(f"  - Device: {self.device}")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch_idx, (images, labels) in enumerate(train_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_acc = 100.0 * train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for images, labels in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            epoch_time = time.time() - epoch_start_time
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            history['train_loss'].append(train_loss_avg)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss_avg)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            history['epoch_times'].append(epoch_time)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc,
                    'history': history
                }, os.path.join(method_dir, 'best_model.pth'))
                print(f"  New best model saved! Val Loss: {val_loss_avg:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc,
                    'history': history
                }, os.path.join(method_dir, 'best_acc_model.pth'))
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc,
                    'history': history
                }, os.path.join(method_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            scheduler.step()
            print()
        
        # Save final model
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss_avg,
            'val_acc': val_acc,
            'history': history
        }, os.path.join(method_dir, 'final_model.pth'))
        
        # Save training history
        with open(os.path.join(method_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # Create training curves
        self.plot_training_curves(history, method_name, method_dir)
        
        print(f"Training completed for {method_name.upper()}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Models saved in: {method_dir}")
        
        return history
    
    def plot_training_curves(self, history, method_name, save_dir):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{method_name.upper()} Training Curves', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3.plot(epochs, history['lr'], 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        
        # Epoch times
        ax4.plot(epochs, history['epoch_times'], 'm-', linewidth=2)
        ax4.set_title('Training Time per Epoch')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main training function"""
    print("="*80)
    print("RIGOROUS 3-METHOD TRAINING SYSTEM")
    print("="*80)
    
    # Configuration
    config = {
        'num_epochs': 20,
        'batch_size': 16,
        'learning_rate': 0.001,
        'output_dim': 512,
        'num_classes': 50,
        'train_samples': 2000,
        'val_samples': 500,
        'test_samples': 500,
        'num_workers': 0
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = RigorousDataset(
        num_samples=config['train_samples'],
        num_classes=config['num_classes'],
        split='train',
        seed=42
    )
    
    val_dataset = RigorousDataset(
        num_samples=config['val_samples'],
        num_classes=config['num_classes'],
        split='val',
        seed=123
    )
    
    test_dataset = RigorousDataset(
        num_samples=config['test_samples'],
        num_classes=config['num_classes'],
        split='test',
        seed=456
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize trainer
    trainer = RigorousTrainer(device=device)
    
    # Train each method
    methods = ['iris', 'spca', 'adaptive']
    all_histories = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"INITIALIZING {method.upper()} MODEL")
        print(f"{'='*80}")
        
        try:
            # Create model
            model = trainer.create_model(
                method,
                output_dim=config['output_dim'],
                num_classes=config['num_classes']
            )
            
            print(f"Model created successfully for {method}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Train model
            history = trainer.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                method_name=method,
                num_epochs=config['num_epochs'],
                lr=config['learning_rate']
            )
            
            all_histories[method] = history
            
        except Exception as e:
            print(f"Error training {method}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall training summary
    summary = {
        'config': config,
        'device': str(device),
        'training_date': datetime.now().isoformat(),
        'methods_trained': list(all_histories.keys()),
        'histories': all_histories
    }
    
    with open('rigorous_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for method, history in all_histories.items():
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        best_val_loss = min(history['val_loss'])
        best_val_acc = max(history['val_acc'])
        
        print(f"\n{method.upper()}:")
        print(f"  Final Train Loss: {final_train_loss:.4f}")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Final Train Acc: {final_train_acc:.2f}%")
        print(f"  Final Val Acc: {final_val_acc:.2f}%")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.2f}%")
    
    print(f"\nAll models and training data saved in 'rigorous_models/' directory")
    print(f"Training summary saved in 'rigorous_training_summary.json'")

if __name__ == '__main__':
    main()

