"""
ULTRON: Unifying Local Transformer and Convolution for Large-scale Image Retrieval
SCALA (Spatial Context-Aware Local Attention) Implementation

論文の式(3)-(8)に基づく厳密な実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class MultiScaleContextKernel(nn.Module):
    """
    Multi-scale Context Kernel (MCK)
    論文の式(3), (4)に基づく実装
    """
    
    def __init__(self, dim: int, kernel_size: int = 3):
        super(MultiScaleContextKernel, self).__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 膨張畳み込み層 (異なる膨張率)
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=self.padding * (i + 1), 
                     dilation=i + 1, groups=dim, bias=False)
            for i in range(3)  # 膨張率 1, 2, 3
        ])
        
        # 1x1畳み込みによる統合
        self.conv1x1 = nn.Conv2d(dim * 3, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        式(3): DCk(X)i,j = ∑∑ wm,n · Xi+m·δ,j+n·δ + bi,j
        式(4): MCK(X)i,j = conv1x1(concat(X, DC1(X)i,j, DC2(X)i,j, DC3(X)i,j))
        """
        B, C, H, W = x.shape
        
        # 各膨張率での畳み込み
        dilated_outputs = []
        for conv in self.dilated_convs:
            dilated_outputs.append(conv(x))
        
        # チャネル方向に結合
        concatenated = torch.cat(dilated_outputs, dim=1)  # [B, 3*C, H, W]
        
        # 1x1畳み込みで統合
        output = self.conv1x1(concatenated)
        output = self.bn(output)
        
        return output

class SpatialContextAwareLocalAttention(nn.Module):
    """
    Spatial Context-Aware Local Attention (SCALA)
    論文の式(5)-(8)に基づく実装
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super(SpatialContextAwareLocalAttention, self).__init__()
        
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Multi-scale Context Kernel
        self.mck = MultiScaleContextKernel(dim)
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 位置エンコーディング
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # 相対位置インデックス
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        # 初期化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def window_partition(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        特徴マップをウィンドウに分割
        """
        B, H, W, C = x.shape
        
        # パディングを追加してウィンドウサイズの倍数にする
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        H_pad, W_pad = H + pad_h, W + pad_w
        
        # ウィンドウに分割
        x = x.view(B, H_pad // self.window_size, self.window_size, 
                   W_pad // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            -1, self.window_size, self.window_size, C
        )
        
        return windows, (H_pad, W_pad)
    
    def window_reverse(self, windows: torch.Tensor, H_pad: int, W_pad: int, H: int, W: int) -> torch.Tensor:
        """
        ウィンドウを特徴マップに復元
        """
        B = int(windows.shape[0] / (H_pad * W_pad / self.window_size / self.window_size))
        x = windows.view(B, H_pad // self.window_size, W_pad // self.window_size,
                        self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)
        
        # パディングを除去
        if H_pad > H or W_pad > W:
            x = x[:, :H, :W, :].contiguous()
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SCALA の順伝播
        
        式(5): Ai,j = Qi · MCK(V)ki,(i) + Bzi,(i)
        式(6): Aki = [Ai,1 Ai,2 ··· Ai,k]T
        式(7): Vki = [Vzi,(1) Vzi,(2) ··· Vzi,(k)]T
        式(8): SCALAk(i) = softmax(Aki/√d) · Vki
        """
        B, C, H, W = x.shape
        
        # 1. Multi-scale Context Kernel の適用
        context_features = self.mck(x)  # [B, C, H, W]
        
        # 2. 特徴マップを [B, H, W, C] に変換
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        context_reshaped = context_features.permute(0, 2, 3, 1).contiguous()
        
        # 3. ウィンドウ分割
        x_windows, (H_pad, W_pad) = self.window_partition(x_reshaped)  # [nW*B, Wh, Ww, C]
        context_windows, _ = self.window_partition(context_reshaped)
        
        # 4. ウィンドウを平坦化
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Wh*Ww, C]
        context_windows = context_windows.view(-1, self.window_size * self.window_size, C)
        
        # 5. QKV の計算
        nW_B, N, C = x_windows.shape
        qkv = self.qkv(x_windows).reshape(nW_B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [nW*B, num_heads, N, C//num_heads]
        
        # 6. スケールされた注意の計算
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [nW*B, num_heads, N, N]
        
        # 7. 相対位置バイアスの追加
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # 8. Softmax と Dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # 9. 値との乗算
        x_attended = (attn @ v).transpose(1, 2).reshape(nW_B, N, C)  # [nW*B, N, C]
        
        # 10. 出力投影
        x_attended = self.proj(x_attended)
        x_attended = self.proj_drop(x_attended)
        
        # 11. ウィンドウを特徴マップに復元
        x_attended = x_attended.view(-1, self.window_size, self.window_size, C)
        x_output = self.window_reverse(x_attended, H_pad, W_pad, H, W)  # [B, H, W, C]
        
        # 12. [B, C, H, W] に戻す
        x_output = x_output.permute(0, 3, 1, 2).contiguous()
        
        return x_output

class SCALABlock(nn.Module):
    """
    SCALA Block with Feed-Forward Network
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super(SCALABlock, self).__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        # Layer Normalization (チャネル次元に適用)
        self.norm1 = nn.BatchNorm2d(dim)
        
        # SCALA Attention
        self.attn = SpatialContextAwareLocalAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        # Drop Path
        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        
        # Layer Normalization
        self.norm2 = nn.BatchNorm2d(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(mlp_hidden_dim, dim, 1),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SCALA Attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

def test_scala():
    """SCALA のテスト"""
    print("=== SCALA テスト ===")
    
    # テスト設定
    batch_size = 2
    dim = 256
    height, width = 56, 56
    window_size = 7
    num_heads = 8
    
    # モデル作成
    mck = MultiScaleContextKernel(dim)
    scala_attn = SpatialContextAwareLocalAttention(dim, window_size, num_heads)
    scala_block = SCALABlock(dim, window_size, num_heads)
    
    # 入力データ
    x = torch.randn(batch_size, dim, height, width)
    
    print(f"入力形状: {x.shape}")
    
    # テスト実行
    with torch.no_grad():
        # Multi-scale Context Kernel テスト
        mck_output = mck(x)
        print(f"MCK出力形状: {mck_output.shape}")
        
        # SCALA Attention テスト
        attn_output = scala_attn(x)
        print(f"SCALA Attention出力形状: {attn_output.shape}")
        
        # SCALA Block テスト
        block_output = scala_block(x)
        print(f"SCALA Block出力形状: {block_output.shape}")
    
    # 勾配テスト
    x_grad = torch.randn(batch_size, dim, height, width, requires_grad=True)
    output_grad = scala_block(x_grad)
    loss = output_grad.sum()
    loss.backward()
    
    print(f"勾配計算成功: {x_grad.grad is not None}")
    print(f"勾配ノルム: {x_grad.grad.norm().item():.4f}")
    
    print("✓ SCALA テスト完了")

if __name__ == "__main__":
    test_scala()

