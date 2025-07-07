"""
IRIS training test script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import time
import json
from datetime import datetime

from iris_implementation_corrected import IRISWrapper, IRISLoss

class SimpleDataset(Dataset):
    """Simple synthetic dataset for testing"""
    def __init__(self, num_samples=1000, num_classes=50, image_size=224, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Generate synthetic data
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def test_iris_training():
    """Test IRIS training functionality"""
    print("="*60)
    print("IRIS TRAINING TEST")
    print("="*60)
    
    # Configuration
    config = {
        'num_epochs': 3,
        'batch_size': 8,
        'learning_rate': 1e-3,
        'output_dim': 512,
        'num_classes': 50,
        'train_samples': 200,
        'val_samples': 50,
        'num_workers': 0
    }
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SimpleDataset(
        num_samples=config['train_samples'],
        num_classes=config['num_classes'],
        seed=42
    )
    
    val_dataset = SimpleDataset(
        num_samples=config['val_samples'],
        num_classes=config['num_classes'],
        seed=123
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Create model
    print("\nCreating IRIS model...")
    model = IRISWrapper(
        backbone='resnet18',
        output_dim=config['output_dim'],
        num_classes=config['num_classes']
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = IRISLoss(
        classification_weight=1.0,
        retrieval_weight=0.5
    )
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images)
            features = model(images, return_features=True)
            
            # Calculate loss
            loss = criterion(logits, features, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100.0 * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.1f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                logits = model(images)
                features = model(images, return_features=True)
                
                # Calculate loss
                loss = criterion(logits, features, labels)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_acc = 100.0 * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.1f}%'
                })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{config['num_epochs']}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'config': config
            }, 'iris_best_model.pth')
            print(f"  New best model saved! Val Loss: {epoch_val_loss:.4f}")
        
        print()
    
    # Save final model
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': epoch_val_loss,
        'val_acc': epoch_val_acc,
        'config': config
    }, 'iris_final_model.pth')
    
    # Save training history
    training_summary = {
        'config': config,
        'device': str(device),
        'training_date': datetime.now().isoformat(),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_val_loss': best_val_loss,
        'final_val_loss': epoch_val_loss,
        'final_val_acc': epoch_val_acc,
        'history': history
    }
    
    with open('iris_training_history.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print("="*60)
    print("IRIS TRAINING TEST COMPLETED")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {epoch_val_acc:.2f}%")
    print(f"Models saved: iris_best_model.pth, iris_final_model.pth")
    print(f"History saved: iris_training_history.json")
    
    return training_summary

if __name__ == "__main__":
    test_iris_training()

