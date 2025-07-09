"""
Modified multi_model_train.py to support Table 1 implementations
既存のmulti_model_train.pyを改造してTable 1の手法を統合

主な変更点:
1. Table 1モデルの条件付きインポート追加
2. モデル選択ロジックにTable 1手法を追加
3. 既存コードの構造は最大限保持
"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch import cuda, optim
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms

# SummaryWriterは使用されていないため削除
# from torch.utils.tensorboard import SummaryWriter

from config import get_args
from dataset import ImageFromList, GLDv2_imdb
from networks import Token, SpCa
from utils import MetricLogger, create_optimizer, init_distributed_mode, is_main_process
from utils import compute_map_and_print, extract_vectors
from utils.helpfunc import get_checkpoint_root, freeze_weights, unfreeze_weights

# ULTRON関連のインポートを追加
try:
    from modls.ultron_modules.ultron import ULTRON
    from madacos_loss import MadaCosLoss, ULTRONTrainingLoss
    ULTRON_AVAILABLE = True
except ImportError:
    print("Warning: ULTRON modules not found. ULTRON training will be disabled")
    ULTRON_AVAILABLE = False

# CVNet関連のインポートを追加
try:
    from modls.cvnet_modules.CVNet_Rerank_model import CVNet_Rerank
    from modls.cvnet_modules.CVlearner import CVLearner
    CVNET_AVAILABLE = True
except ImportError:
    print("Warning: CVNet modules not found. CVNet training will be disabled")
    CVNET_AVAILABLE = False

# DOLGNet関連のインポートを追加
try:
    from models.dolg_net import DOLGNet, create_dolg_model
    DOLGNET_AVAILABLE = True
except ImportError:
    print("Warning: DOLGNet modules not found. DOLGNet training will be disabled")
    DOLGNET_AVAILABLE = False

# Table 1実装のインポートを追加
try:
    from models import (
        get_model as get_table1_model,
        get_optimizer as get_table1_optimizer,
        list_available_models,
        get_model_from_table1_name,
        TABLE1_TO_MODEL_MAPPING
    )
    TABLE1_AVAILABLE = True
    print(f"Table 1 models available: {list_available_models()}")
except ImportError:
    print("Warning: Table 1 models not found. Table 1 training will be disabled")
    TABLE1_AVAILABLE = False


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(preds, max(ks), dim=1, largest=True, sorted=True)
    # (batch_size, ) -> (batch_size, 1) -> (batch_size, max_k)
    rep_max_k_labels = labels.reshape(-1, 1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks]
    return [(1.0 - x / preds.size(0)) * 100.0 for x in topks_correct]


class WarmupCos_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr=0, iter_per_epoch=1):
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        return self.lr_schedule[self.iter]

    def state_dict(self):
        state_dict = {}
        state_dict['base_lr'] = self.base_lr
        state_dict['lr_schedule'] = self.lr_schedule
        state_dict['iter'] = self.iter
        return state_dict


class CVNetWrapper(nn.Module):
    """
    CVNetモデルをSPCA学習フレームワークに適合させるためのラッパー
    """
    def __init__(self, num_classes, resnet_depth=50, reduction_dim=512):
        super(CVNetWrapper, self).__init__()
        self.backbone = CVNet_Rerank(resnet_depth, reduction_dim)
        self.classifier = nn.Linear(reduction_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        """
        SPCA学習フレームワークと互換性のある順伝播

        Args:
            x: 入力画像 [B, C, H, W]
            targets: ターゲットラベル [B] (学習時のみ)

        Returns:
            loss: 損失値 (学習時のみ)
            logits: 分類ロジット [B, num_classes]
        """
        if self.training and targets is not None:
            # 学習時: 特徴抽出 + 分類
            features = self.backbone.extract_global_descriptor(x)
            logits = self.classifier(features)
            loss = self.criterion(logits, targets)
            return loss, logits
        else:
            # 推論時: 分類のみ
            features = self.backbone.extract_global_descriptor(x)
            logits = self.classifier(features)
            return None, logits


def create_cvnet_optimizer(args, model):
    """
    CVNet用の最適化器作成
    """
    # CVNet用のSGD最適化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    return optimizer


class DOLGNetWrapper(nn.Module):
    """
    DOLGNetモデルをSPCA学習フレームワークに適合させるためのラッパー
    """
    def __init__(self, num_classes, backbone='resnet50', local_dim=1024, global_dim=2048,
                 pretrained=True,
                 local_dim=local_dim,
                 global_dim=global_dim,
                 output_dim=output_dim,
                 num_classes=num_classes
                 ):
        super(DOLGNetWrapper, self).__init__()
        self.backbone = create_dolg_model(
            backbone=backbone,
            pretrained=pretrained,
            local_dim=local_dim,
            global_dim=global_dim,
            output_dim=output_dim,
            num_classes=num_classes
        )

    def forward(self, x, targets=None):
        """
        SPCA学習フレームワークと互換性のある順伝播

        Args:
            x: 入力画像 [B, C, H, W]
            targets: ターゲットラベル [B] (学習時のみ)

        Returns:
            loss: 損失値 (学習時のみ)
            logits: 分類ロジット [B, num_classes]
        """
        return self.backbone(x, targets)

    def extract_descriptor(self, x):
        """特徴抽出の抽出"""
        return self.backbone.extract_descriptor(x)


def create_dolgnet_optimizer(args, model):
    """
    DOLGNet用の最適化器作成
    """
    # DOLGNet用のAdamW最適化器 (論文推奨)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    return optimizer


class ULTRONWrapper(nn.Module):
    """
    ULTRONモデルをSPCA学習フレームワークに適合させるためのラッパー
    """
    def __init__(self, num_classes, embed_dim=512, rho=0.04):
        super(ULTRONWrapper, self).__init__()
        self.backbone = ULTRON(num_classes=num_classes)
        self.criterion = ULTRONTrainingLoss(
            num_classes=num_classes,
            embed_dim=embed_dim,
            rho=rho
        )

    def forward(self, x, targets=None):
        """
        SPCA学習フレームワークと互換性のある順伝播

        Args:
            x: 入力画像 [B, C, H, W]
            targets: ターゲットラベル [B] (学習時のみ)

        Returns:
            loss: 損失値 (学習時のみ)
            logits: 分類ロジット [B, num_classes]
        """
        if self.training and targets is not None:
            # 学習時: 特徴抽出 + 損失計算
            features = self.backbone.extract_features(x)
            logits = self.backbone(x)

            # ULTRON用損失計算
            loss_dict = self.criterion(features, None, targets)
            loss = loss_dict["total_loss"]

            return loss, logits
        else:
            # 推論時: 分類のみ
            logits = self.backbone(x)
            return None, logits


def create_ultron_optimizer(args, model):
    """
    ULTRON用の最適化器作成
    論文記載の設定: AdamW(5epoch) → SGD(35epoch)
    """
    # 最初はAdamW
    if hasattr(args, 'ultron_phase') and args.ultron_phase == 'sgd':
        # SGDフェーズ
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
    else:
        # AdamWフェーズ
        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )

    return optimizer


def main(args):
    print('distributed parallel mode only')
    init_distributed_mode(args)

    for key in vars(args):
        print(key + " : " + str(vars(args)[key]))

    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # set the path for model save:
    args.directory = get_checkpoint_root()
    os.makedirs(args.directory, exist_ok=True)

    # モデル名に基づいてパス生成を調整
    if args.model.startswith('ultron'):
        path = '{}-bb{}-bl{}-f{}-{}-nep{}-speedup-githubv1'.format(
            args.model, args.backbone, args.base_lr, args.final_lr,
            args.batch_size, args.num_epochs, getattr(args, 'rho', 0.04)
        )
    elif args.model.startswith('dolgnet'):
        path = '{}-bb{}-bl{}-f{}-{}-gd{}-od{}-speedup-githubv1'.format(
            args.model, args.backbone, args.base_lr, args.final_lr,
            args.batch_size, args.num_epochs, getattr(args, 'local_dim', 1024),
            getattr(args, 'global_dim', 2048), getattr(args, 'output_dim', 512)
        )
    else:
        path = '{}-bb{}-bl{}-f{}-{}-m{}-b{}-nep{}-speedup-githubv1'.format(
            args.model, args.backbone, args.base_lr, args.final_lr,
            args.tau, args.margin, args.batch_size, args.num_epochs
        )

    if args.seed != 11:
        path += '-seed{}'.format(args.seed)
    if args.split is not None:
        path += '-split{}'.format(args.split)
    if args.model.startswith('spca'):
        path += '-k{}'.format(args.codebook_size)
        path += '-mul{}'.format(args.multi)
        path += '-c{}'.format(args.combine)

    directory = os.path.join(args.directory, path)
    os.makedirs(directory, exist_ok=True)

    # distributed parallel setting:
    if args.distributed:
        ngpus_per_node = cuda.device_count()
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(f'>> batch_size per node: {args.batch_size}')

    # データセット作成
    train_dataset, val_dataset, class_num = GLDv2_imdb(args.imsize, args.seed,
                                                       args.classifier_num = class_num)

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=True)
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler,
                                  num_workers=args.num_workers, pin_memory=True)
        val_sampler = DistributedSampler(val_dataset)
        val_batch_sampler = BatchSampler(val_sampler, args.batch_size, drop_last=False)
        val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_batch_sampler,
                                num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    # モデル作成分岐を条件分岐で拡張
    if args.model.startswith('token'):
        model = Token(1024, args.classifier_num, args.pretrained, args.backbone)
    elif args.model.startswith('spca'):
        meta = {}
        meta['outputdim'] = args.outputdim
        meta['k'] = args.codebook_size
        meta['local_dim'] = args.local_dim
        meta['combine'] = args.combine
        meta['multi'] = args.multi
        meta['pretrained'] = args.pretrained
        model = SpCa(args.outputdim, args.classifier_num, meta, args.tau, args.margin)
    elif args.model.startswith('ultron'):
        # ULTRON モデルの作成
        if not ULTRON_AVAILABLE:
            raise ValueError('ULTRON modules are not available. Please ensure ULTRON is properly installed.')

        embed_dim = getattr(args, 'embed_dim', 512)
        rho = getattr(args, 'rho', 0.04)

        model = ULTRONWrapper(
            num_classes=args.classifier_num,
            embed_dim=embed_dim,
            rho=rho
        ).to(device)

        print(f'>> ULTRON model created with {args.classifier_num} classes, embed_dim={embed_dim}, rho={rho}')
    elif args.model.startswith('cvnet'):
        # CVNet モデルの作成
        if not CVNET_AVAILABLE:
            raise ValueError('CVNet modules are not available. Please ensure CVNet is properly installed.')

        # CVNet用のパラメータ設定
        resnet_depth = getattr(args, 'resnet_depth', 50)
        reduction_dim = getattr(args, 'reduction_dim', 512)

        model = CVNetWrapper(
            num_classes=args.classifier_num,
            resnet_depth=resnet_depth,
            reduction_dim=reduction_dim
        ).to(device)

        print(f'>> CVNet model created with {args.classifier_num} classes, resnet_depth={resnet_depth}')
    elif args.model.startswith('dolgnet'):
        # DOLGNet モデルの作成
        if not DOLGNET_AVAILABLE:
            raise ValueError('DOLGNet modules are not available. Please ensure DOLGNet is properly installed.')

        # DOLGNet用のパラメータ設定
        local_dim = getattr(args, 'local_dim', 1024)
        global_dim = getattr(args, 'global_dim', 2048)
        output_dim = getattr(args, 'output_dim', 512)
        backbone = getattr(args, 'backbone', 'resnet50')

        model = DOLGNetWrapper(
            num_classes=args.classifier_num,
            backbone=backbone,
            local_dim=local_dim,
            global_dim=global_dim,
            output_dim=output_dim
        ).to(device)

        print(f'>> DOLGNet model created with {args.classifier_num} classes, backbone={backbone}')
    elif TABLE1_AVAILABLE and (args.model in list_available_models() or args.model in TABLE1_TO_MODEL_MAPPING):
        # Table 1 モデルの作成
        try:
            if args.model in TABLE1_TO_MODEL_MAPPING:
                # Table 1名前で指定された場合
                actual_model_name = TABLE1_TO_MODEL_MAPPING[args.model]
                model = get_table1_model(actual_model_name, num_classes=args.classifier_num)
                print(f'>> Table 1 model created: {args.model} -> {actual_model_name}')
            else:
                # 直接モデル名で指定された場合
                model = get_table1_model(args.model, num_classes=args.classifier_num)
                print(f'>> Table 1 model created: {args.model}')

            model = model.to(device)
        except Exception as e:
            raise ValueError(f'Failed to create Table 1 model {args.model}: {e}')
    else:
        raise ValueError(f'Unknown model: {args.model}. Available models: {list_available_models() if TABLE1_AVAILABLE else "Table 1 models not available"}')

    # 分散学習設定
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # オプティマイザ作成
    if args.model.startswith('ultron'):
        optimizer = create_ultron_optimizer(args, model)
    elif args.model.startswith('cvnet'):
        optimizer = create_cvnet_optimizer(args, model)
    elif args.model.startswith('dolgnet'):
        optimizer = create_dolgnet_optimizer(args, model)
    elif TABLE1_AVAILABLE and (args.model in list_available_models() or args.model in TABLE1_TO_MODEL_MAPPING):
        # Table 1 モデル用のオプティマイザ
        try:
            if args.model in TABLE1_TO_MODEL_MAPPING:
                actual_model_name = TABLE1_TO_MODEL_MAPPING[args.model]
                optimizer = get_table1_optimizer(actual_model_name, args, model)
            else:
                optimizer = get_table1_optimizer(args.model, args, model)
        except Exception as e:
            print(f'Warning: Failed to create Table 1 optimizer for {args.model}: {e}')
            print('Falling back to default optimizer')
            optimizer = create_optimizer(args, model)
    else:
        optimizer = create_optimizer(args, model)

    # 学習率スケジューラ
    if args.model.startswith('ultron') or args.model.startswith('cvnet') or args.model.startswith('dolgnet') or \
       (TABLE1_AVAILABLE and (args.model in list_available_models() or args.model in TABLE1_TO_MODEL_MAPPING)):
        # 新しいモデル用のスケジューラ
        scheduler = WarmupCos_Scheduler(
            optimizer,
            warmup_epochs=5,
            warmup_lr=1e-6,
            num_epochs=args.num_epochs,
            base_lr=args.base_lr,
            final_lr=args.final_lr,
            iter_per_epoch=len(train_loader)
        )
    else:
        # 既存のスケジューラ
        scheduler = WarmupCos_Scheduler(
            optimizer,
            warmup_epochs=5,
            warmup_lr=1e-6,
            num_epochs=args.num_epochs,
            base_lr=args.base_lr,
            final_lr=args.final_lr,
            iter_per_epoch=len(train_loader)
        )

    # 学習ループ
    print("Start training")
    start_time = time.time()

    for epoch in range(args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # 学習フェーズ
        model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for i, (images, targets) in enumerate(metric_logger.log_every(train_loader, 10, header)):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 順伝播
            loss, logits = model(images, targets)

            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # スケジューラ更新
            scheduler.step()

            # ログ更新
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 検証フェーズ
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_samples = 0

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    loss, logits = model(images, targets)

                    if loss is not None:
                        val_loss += loss.item()

                    # 精度計算
                    _, predicted = torch.max(logits, 1)
                    val_acc += (predicted == targets).sum().item()
                    val_samples += targets.size(0)

            val_loss /= len(val_loader)
            val_acc /= val_samples

            print(f'Epoch [{epoch}/{args.num_epochs}] - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # モデル保存
        if is_main_process() and (epoch % 20 == 0 or epoch == args.num_epochs - 1):
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            torch.save(checkpoint, os.path.join(directory, f'checkpoint_epoch_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args()
    main(args)

