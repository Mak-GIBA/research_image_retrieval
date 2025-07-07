"""
ULTRON 簡易テストスクリプト
次元不整合問題を解決するための簡易版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 簡易版CDConv
class SimpleCDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SimpleCDConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.bn(self.conv(x))

# 簡易版SCALABlock
class SimpleSCALABlock(nn.Module):
    def __init__(self, dim):
        super(SimpleSCALABlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        
    def forward(self, x):
        # Attention
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        
        # MLP
        identity = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + identity
        
        return x

# 簡易版ULTRON
class SimpleULTRON(nn.Module):
    def __init__(self, num_classes=1000):
        super(SimpleULTRON, self).__init__()
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, 64, 4, 4)
        
        # Stage 1: CDConv (64 -> 128)
        self.stage1 = nn.Sequential(
            SimpleCDConv(64, 128, stride=2),
            nn.ReLU(inplace=True),
            SimpleCDConv(128, 128),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: CDConv (128 -> 320)
        self.stage2 = nn.Sequential(
            SimpleCDConv(128, 320, stride=2),
            nn.ReLU(inplace=True),
            SimpleCDConv(320, 320),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: SCALA (320 -> 512)
        self.stage3 = nn.Sequential(
            SimpleCDConv(320, 512, stride=2),
            SimpleSCALABlock(512),
            SimpleSCALABlock(512)
        )
        
        # Stage 4: SCALA (512)
        self.stage4 = nn.Sequential(
            SimpleSCALABlock(512),
            SimpleSCALABlock(512)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.head = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, 64, 56, 56]
        
        # Stages
        x = self.stage1(x)  # [B, 128, 28, 28]
        x = self.stage2(x)  # [B, 320, 14, 14]
        x = self.stage3(x)  # [B, 512, 7, 7]
        x = self.stage4(x)  # [B, 512, 7, 7]
        
        # Global pooling
        x = self.global_pool(x)  # [B, 512, 1, 1]
        x = x.flatten(1)  # [B, 512]
        
        # Classification
        x = self.head(x)  # [B, num_classes]
        
        return x
    
    def extract_features(self, x):
        """特徴抽出"""
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return x

def test_simple_ultron():
    print("=== Simple ULTRON テスト ===")
    
    # モデル作成
    model = SimpleULTRON(num_classes=1000)
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"パラメータ数: {total_params:,}")
    
    # テスト
    x = torch.randn(2, 3, 224, 224)
    print(f"入力形状: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        features = model.extract_features(x)
        print(f"出力形状: {output.shape}")
        print(f"特徴形状: {features.shape}")
    
    # 勾配テスト
    x_grad = torch.randn(2, 3, 224, 224, requires_grad=True)
    output_grad = model(x_grad)
    loss = output_grad.sum()
    loss.backward()
    
    print(f"勾配計算成功: {x_grad.grad is not None}")
    print(f"勾配ノルム: {x_grad.grad.norm().item():.4f}")
    
    print("✓ Simple ULTRON テスト完了")

if __name__ == "__main__":
    test_simple_ultron()

