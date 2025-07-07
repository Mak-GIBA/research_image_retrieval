"""
ULTRON: Unifying Local Transformer and Convolution for Large-scale Image Retrieval
Training System Implementation

論文記載の学習条件に基づく厳密な実装
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import math
import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

from simple_ultron_test import SimpleULTRON
from madacos_loss import MadaCosLoss, ULTRONTrainingLoss

class GLDv2Dataset(Dataset):
    """
    Google Landmarks v2-clean Dataset (Simulated)
    論文記載の設定に基づく実装
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        num_classes: int = 81313,
        samples_per_class: int = 20
    ):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        
        # シミュレートされたデータセット
        if split == "train":
            self.num_samples = num_classes * samples_per_class  # 約1.6M画像
        else:
            self.num_samples = min(10000, num_classes)  # 検証用
        
        # ランダムシードを固定してデータの一貫性を保つ
        np.random.seed(42)
        self.labels = np.random.randint(0, num_classes, self.num_samples)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # シミュレートされた画像データ (実際の実装では画像ファイルを読み込み)
        image = torch.randn(3, 224, 224)
        label = self.labels[idx]
        
        if self.transform:
            # 実際の実装ではPIL Imageに変換してからtransformを適用
            pass
        
        return image, label

class ULTRONTrainer:
    """
    ULTRON Training System
    論文記載の学習条件に基づく実装
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 81313,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        
        # 損失関数 (論文設定: ρ=0.04)
        self.criterion = ULTRONTrainingLoss(
            num_classes=num_classes,
            embed_dim=512,
            rho=0.04
        ).to(device)
        
        # 最適化器設定 (論文記載)
        self.setup_optimizers()
        
        # 学習履歴
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": []
        }
    
    def setup_optimizers(self):
        """
        論文記載の最適化設定:
        - AdamW: 最初10エポック, lr=1e-3
        - SGD: 残り30エポック, lr=1e-2, momentum=0.9, weight_decay=1e-4
        """
        # AdamW optimizer (ウォームアップ用)
        self.adamw_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        # SGD optimizer (メイン学習用)
        self.sgd_optimizer = optim.SGD(
            self.model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # コサインスケジューラー
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.sgd_optimizer,
            T_max=30,  # 30エポック
            eta_min=1e-6
        )
        
        self.current_optimizer = self.adamw_optimizer
        self.warmup_epochs = 5  # 論文では5エポック
    
    def get_data_transforms(self):
        """
        論文記載のデータ拡張:
        - Random cropping
        - Color jittering
        - Resize to 512x512
        """
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def create_dataloaders(self, data_path: str, batch_size: int = 128):
        """
        データローダーの作成
        論文設定: batch_size=128, 4 GPUs
        """
        train_transform, val_transform = self.get_data_transforms()
        
        # データセット作成 (シミュレート版)
        train_dataset = GLDv2Dataset(
            data_path, split="train", transform=train_transform,
            num_classes=min(1000, self.num_classes),  # テスト用に縮小
            samples_per_class=10
        )
        
        val_dataset = GLDv2Dataset(
            data_path, split="val", transform=val_transform,
            num_classes=min(1000, self.num_classes),
            samples_per_class=5
        )
        
        # データローダー
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # 実際の実装では4-8
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """1エポックの学習"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # オプティマイザーの切り替え
        if epoch >= self.warmup_epochs and self.current_optimizer == self.adamw_optimizer:
            print(f"エポック {epoch}: SGDオプティマイザーに切り替え")
            self.current_optimizer = self.sgd_optimizer
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.current_optimizer.zero_grad()
            
            # 順伝播
            features = self.model.extract_features(data)
            
            # 損失計算
            loss_dict = self.criterion(features, None, target)
            loss = loss_dict["total_loss"]
            
            # 逆伝播
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.current_optimizer.step()
            
            # 統計更新
            total_loss += loss.item()
            
            # 精度計算 (簡易版)
            with torch.no_grad():
                pred = self.model(data)
                predicted = pred.argmax(dim=1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # プログレスバー更新
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })
        
        # スケジューラー更新
        if self.current_optimizer == self.sgd_optimizer:
            self.scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                # 順伝播
                features = self.model.extract_features(data)
                output = self.model(data)
                
                # 損失計算
                loss_dict = self.criterion(features, None, target)
                loss = loss_dict["total_loss"]
                
                total_loss += loss.item()
                
                # 精度計算
                predicted = output.argmax(dim=1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def train(
        self,
        data_path: str,
        num_epochs: int = 40,
        batch_size: int = 32,  # テスト用に縮小
        save_dir: str = "ultron_checkpoints"
    ):
        """
        メイン学習ループ
        論文設定: 40エポック, batch_size=128, 4 GPUs
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # データローダー作成
        train_loader, val_loader = self.create_dataloaders(data_path, batch_size)
        
        print(f"学習開始: {num_epochs}エポック")
        print(f"学習データ: {len(train_loader.dataset)}サンプル")
        print(f"検証データ: {len(val_loader.dataset)}サンプル")
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\n=== エポック {epoch+1}/{num_epochs} ===")
            
            # 学習
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 検証
            val_metrics = self.validate(val_loader)
            
            # 履歴更新
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["lr"].append(self.current_optimizer.param_groups[0]["lr"])            
            # 結果表示
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Learning Rate: {self.current_optimizer.param_groups[0]['lr']:.6f}")
            
            # ベストモデル保存
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.current_optimizer.state_dict(),
                    "val_acc": val_metrics["accuracy"],
                    "history": self.history
                }, os.path.join(save_dir, "best_model.pth"))
                print(f"ベストモデル更新: Val Acc {val_metrics['accuracy']:.2f}%")
        
        # 最終モデル保存
        torch.save({
            "epoch": num_epochs - 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.current_optimizer.state_dict(),
            "history": self.history
        }, os.path.join(save_dir, "final_model.pth"))
        
        # 履歴保存
        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n学習完了! ベスト検証精度: {best_val_acc:.2f}%")
        
        return self.history

def test_ultron_training():
    """ULTRON学習システムのテスト"""
    print("=== ULTRON Training System テスト ===")
    
    # モデル作成
    model = SimpleULTRON(num_classes=1000)
    
    # トレーナー作成
    trainer = ULTRONTrainer(model, num_classes=1000)
    
    # 短時間テスト (2エポック)
    history = trainer.train(
        data_path="./data",  # ダミーパス
        num_epochs=2,
        batch_size=16,
        save_dir="ultron_test_checkpoints"
    )
    
    print("\n=== 学習結果 ===")
    print(f"最終Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"最終Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"最終Train Acc: {history['train_acc'][-1]:.2f}%")
    print(f"最終Val Acc: {history['val_acc'][-1]:.2f}%")
    
    print("✓ ULTRON Training System テスト完了")

if __name__ == "__main__":
    test_ultron_training()

