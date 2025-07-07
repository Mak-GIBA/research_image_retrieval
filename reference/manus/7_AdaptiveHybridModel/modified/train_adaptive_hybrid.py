"""
Training script for Adaptive Hybrid Feature Learning for Efficient Image Retrieval
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

from adaptive_hybrid_retrieval_complete import AdaptiveHybridModel, QAFF, ContrastiveLoss

class ImageRetrievalDataset(Dataset):
    """Dataset for image retrieval training"""
    def __init__(self, root_dir, transform=None, num_classes=100, samples_per_class=50):
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        
        # Create mock dataset
        self._create_mock_dataset()
        
        # Load data
        self.images = []
        self.labels = []
        self.paths = []
        
        for cls in range(num_classes):
            for sample in range(samples_per_class):
                img_path = os.path.join(root_dir, f'class_{cls}', f'img_{sample}.jpg')
                if os.path.exists(img_path):
                    self.images.append(img_path)
                    self.labels.append(cls)
                    self.paths.append(img_path)
    
    def _create_mock_dataset(self):
        """Create mock training dataset"""
        os.makedirs(self.root_dir, exist_ok=True)
        
        for cls in range(self.num_classes):
            class_dir = os.path.join(self.root_dir, f'class_{cls}')
            os.makedirs(class_dir, exist_ok=True)
            
            for sample in range(self.samples_per_class):
                img_path = os.path.join(class_dir, f'img_{sample}.jpg')
                if not os.path.exists(img_path):
                    # Create diverse mock images
                    color = (
                        (cls * 37 + sample * 13) % 255,
                        (cls * 73 + sample * 29) % 255,
                        (cls * 101 + sample * 41) % 255
                    )
                    Image.new('RGB', (224, 224), color=color).save(img_path)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_loaders(train_dir, val_dir, batch_size=32, num_workers=4):
    """Create training and validation data loaders"""
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageRetrievalDataset(train_dir, transform=train_transform, num_classes=50, samples_per_class=40)
    val_dataset = ImageRetrievalDataset(val_dir, transform=val_transform, num_classes=20, samples_per_class=20)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def train_epoch(model, qaff, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    qaff.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Extract features
        sc_gem, regional_gem, scale_gem = model(images)
        
        # Apply QAFF - use SC-GeM as query guidance
        batch_size = images.size(0)
        query_features = sc_gem  # Use SC-GeM as representative query feature
        gallery_features_list = [sc_gem, regional_gem, scale_gem]
        
        # Apply QAFF to get fused features
        fused_features = qaff(query_features, gallery_features_list)
        
        # Compute loss
        loss = criterion(fused_features, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, qaff, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    qaff.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            # Extract features
            sc_gem, regional_gem, scale_gem = model(images)
            
            # Apply QAFF
            query_features = sc_gem
            gallery_features_list = [sc_gem, regional_gem, scale_gem]
            fused_features = qaff(query_features, gallery_features_list)
            
            # Compute loss
            loss = criterion(fused_features, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description='Train Adaptive Hybrid Retrieval model')
    parser.add_argument('--train_dir', type=str, default='./data/train', help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='./data/val', help='Validation data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone network')
    parser.add_argument('--dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint frequency')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        args.train_dir, args.val_dir, args.batch_size, args.num_workers
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create models
    print("Creating models...")
    model = AdaptiveHybridModel(backbone=args.backbone, pretrained=True, output_dim=args.dim)
    qaff = QAFF(feature_dim=args.dim, num_feature_types=3)
    
    # Move to device
    model = model.to(device)
    qaff = qaff.to(device)
    
    # Create optimizer and loss function
    optimizer = optim.AdamW(
        list(model.parameters()) + list(qaff.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    criterion = ContrastiveLoss(temperature=0.07)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, qaff, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, qaff, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'qaff_state_dict': qaff.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'qaff_state_dict': qaff.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'qaff_state_dict': qaff.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'args': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.output_dir}")

if __name__ == '__main__':
    main()

