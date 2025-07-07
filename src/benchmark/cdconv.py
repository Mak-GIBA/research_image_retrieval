"""
ULTRON: Unifying Local Transformer and Convolution for Large-scale Image Retrieval
CDConv (Channel-wise Dilated Convolution) Implementation

論文の式(1), (2)に基づく厳密な実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class CDConv(nn.Module):
    """
    Channel-wise Dilated Convolution
    
    論文の式(1), (2)に基づく実装:
    - 式(1): ac = sigmoid(∑wk · (1/HW ∑∑ Xc+k-Pk,j))
    - 式(2): CDConvc,i,j(X) = ∑∑ wc,m,n · Xc+m·dc,j+n·dc
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        tau1: float = 0.75,  # 上位75%閾値
        tau2: float = 0.50,  # 上位50%閾値
        delta1: int = 3,     # 中膨張値
        delta2: int = 6,     # 大膨張値
        groups: int = 1
    ):
        super(CDConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tau1 = tau1
        self.tau2 = tau2
        self.delta1 = delta1
        self.delta2 = delta2
        self.groups = groups
        
        # チャネル注意のための1D畳み込み重み
        self.channel_attention_weights = nn.Parameter(
            torch.randn(kernel_size, in_channels) / math.sqrt(in_channels)
        )
        
        # 各膨張率に対応する畳み込み層
        self.conv_dilation_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation=1, groups=groups, bias=False
        )
        self.conv_dilation_delta1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding * delta1,
            dilation=delta1, groups=groups, bias=False
        )
        self.conv_dilation_delta2 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding * delta2,
            dilation=delta2, groups=groups, bias=False
        )
        
        # バッチ正規化
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def compute_channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        チャネル注意の計算 (式1)
        ac = sigmoid(∑wk · (1/HW ∑∑ Xc+k-Pk,j))
        """
        B, C, H, W = x.shape
        K = self.kernel_size
        
        # グローバル平均プーリング: 1/HW ∑∑ X
        global_avg = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # [B, C]
        
        # チャネル注意スコアの計算
        attention_scores = torch.zeros(B, C, device=x.device)
        
        for k in range(K):
            # wk · (global average of channel c+k-P)
            for c in range(C):
                # チャネルインデックスの調整 (循環的)
                channel_idx = (c + k) % C
                weight = self.channel_attention_weights[k, c]
                attention_scores[:, c] += weight * global_avg[:, channel_idx]
        
        # Sigmoid活性化
        attention_scores = torch.sigmoid(attention_scores)  # [B, C]
        
        return attention_scores
    
    def determine_dilation_rates(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        膨張率の決定
        dc = {1 if τ1 < ac, δ1 if τ2 < ac ≤ τ1, δ2 if ac ≤ τ2}
        """
        B, C = attention_scores.shape
        dilation_rates = torch.ones(B, C, dtype=torch.long, device=attention_scores.device)
        
        # 閾値に基づく膨張率の決定
        mask_delta1 = (attention_scores <= self.tau1) & (attention_scores > self.tau2)
        mask_delta2 = attention_scores <= self.tau2
        
        dilation_rates[mask_delta1] = self.delta1
        dilation_rates[mask_delta2] = self.delta2
        
        return dilation_rates
    
    def apply_cdconv(self, x: torch.Tensor, dilation_rates: torch.Tensor) -> torch.Tensor:
        """
        チャネルワイズ膨張畳み込みの適用 (式2)
        CDConvc,i,j(X) = ∑∑ wc,m,n · Xc+m·dc,j+n·dc
        """
        B, C, H, W = x.shape
        
        # 各膨張率に対する出力を計算
        output_1 = self.conv_dilation_1(x)
        output_delta1 = self.conv_dilation_delta1(x)
        output_delta2 = self.conv_dilation_delta2(x)
        
        # 膨張率の分布に基づいて重み付き平均を計算
        mask_1 = (dilation_rates == 1).float().mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        mask_delta1 = (dilation_rates == self.delta1).float().mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        mask_delta2 = (dilation_rates == self.delta2).float().mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        
        # 正規化
        total_mask = mask_1 + mask_delta1 + mask_delta2 + 1e-8
        mask_1 = mask_1 / total_mask
        mask_delta1 = mask_delta1 / total_mask
        mask_delta2 = mask_delta2 / total_mask
        
        # 重み付き結合
        final_output = mask_1 * output_1 + mask_delta1 * output_delta1 + mask_delta2 * output_delta2
        
        return final_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        CDConvの順伝播
        """
        # 1. チャネル注意の計算
        attention_scores = self.compute_channel_attention(x)
        
        # 2. 膨張率の決定
        dilation_rates = self.determine_dilation_rates(attention_scores)
        
        # 3. チャネルワイズ膨張畳み込みの適用
        output = self.apply_cdconv(x, dilation_rates)
        
        # 4. バッチ正規化
        output = self.bn(output)
        
        return output

class CDConvBlock(nn.Module):
    """
    CDConv Block with residual connection
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        tau1: float = 0.75,
        tau2: float = 0.50,
        delta1: int = 3,
        delta2: int = 6
    ):
        super(CDConvBlock, self).__init__()
        
        self.cdconv1 = CDConv(
            in_channels, out_channels, kernel_size, stride,
            padding=kernel_size//2, tau1=tau1, tau2=tau2, delta1=delta1, delta2=delta2
        )
        self.relu1 = nn.ReLU(inplace=True)
        
        self.cdconv2 = CDConv(
            out_channels, out_channels, kernel_size, 1,
            padding=kernel_size//2, tau1=tau1, tau2=tau2, delta1=delta1, delta2=delta2
        )
        
        # ダウンサンプリング層の自動作成
        if downsample is None and (in_channels != out_channels or stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = downsample
            
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.cdconv1(x)
        out = self.relu1(out)
        
        out = self.cdconv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu2(out)
        
        return out

def test_cdconv():
    """CDConvのテスト"""
    print("=== CDConv テスト ===")
    
    # テスト設定
    batch_size = 2
    in_channels = 64
    out_channels = 128
    height, width = 56, 56
    
    # モデル作成
    cdconv = CDConv(in_channels, out_channels, kernel_size=3, tau1=0.75, tau2=0.50)
    cdconv_block = CDConvBlock(in_channels, out_channels)
    
    # 入力データ
    x = torch.randn(batch_size, in_channels, height, width)
    
    print(f"入力形状: {x.shape}")
    
    # CDConv単体テスト
    with torch.no_grad():
        # チャネル注意の計算
        attention_scores = cdconv.compute_channel_attention(x)
        print(f"チャネル注意スコア形状: {attention_scores.shape}")
        print(f"注意スコア範囲: [{attention_scores.min():.4f}, {attention_scores.max():.4f}]")
        
        # 膨張率の決定
        dilation_rates = cdconv.determine_dilation_rates(attention_scores)
        print(f"膨張率形状: {dilation_rates.shape}")
        print(f"膨張率の分布: {torch.bincount(dilation_rates.flatten())}")
        
        # 順伝播
        output = cdconv(x)
        print(f"CDConv出力形状: {output.shape}")
        
        # CDConvBlockテスト
        block_output = cdconv_block(x)
        print(f"CDConvBlock出力形状: {block_output.shape}")
    
    # 勾配テスト
    x_grad = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    output_grad = cdconv(x_grad)
    loss = output_grad.sum()
    loss.backward()
    
    print(f"勾配計算成功: {x_grad.grad is not None}")
    print(f"勾配ノルム: {x_grad.grad.norm().item():.4f}")
    
    print("✓ CDConv テスト完了")

if __name__ == "__main__":
    test_cdconv()

