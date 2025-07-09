import math
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch import cuda, optim
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
import torchvision.transforms as transforms
# SummaryWriterは使用されていないため削除
# from torch.utils.tensorboard import SummaryWriter

from config import get_args
from dataset import ImageFromList, GLDv2_lmdb
from networks import Token, SpCa
from utils import MetricLogger, create_optimizer, init_distributed_mode, is_main_process, get_rank, optimizer_to
from utils import compute_map_and_print, extract_vectors
from utils.helpfunc import get_checkpoint_root, freeze_weights, unfreeze_weights, load_checkpoint

# ULTRON関連のインポートを追加
try:
    from ultron import ULTRON
    from madacos_loss import MadaCosLoss, ULTRONTrainingLoss
    ULTRON_AVAILABLE = True
except ImportError:
    print("Warning: ULTRON modules not found. ULTRON training will be disabled.")
    ULTRON_AVAILABLE = False

# CVNet関連のインポートを追加
try:
    from CVNet_Rerank_model import CVNet_Rerank
    from CVlearner import CVLearner
    CVNET_AVAILABLE = True
except ImportError:
    print("Warning: CVNet modules not found. CVNet training will be disabled.")
    CVNET_AVAILABLE = False

# DOLGNet関連のインポートを追加
try:
    from dolg_net import DOLGNet, create_dolg_model
    DOLGNET_AVAILABLE = True
except ImportError:
    print("Warning: DOLGNet modules not found. DOLGNet training will be disabled.")
    DOLGNET_AVAILABLE = False

def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(preds, max(ks), dim=1, largest=True, sorted=True)
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.reshape(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks]
    return [(1.0 - x / preds.size(0)) * 100.0 for x in topks_correct]


class WarmupCos_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(math.pi * np.arange(decay_iter) / decay_iter))
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

    def load_state_dict(self, state_dict):
        self.base_lr = state_dict['base_lr']
        self.lr_schedule = state_dict['lr_schedule']
        self.iter = state_dict['iter']


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
            loss: 損失値 (学習時)
            logits: 分類ロジット [B, num_classes]
        """
        if self.training and targets is not None:
            # 学習時: 特徴抽出 + 分類
            features = self.backbone.extract_global_descriptor(x)
            logits = self.classifier(features)
            
            # 分類損失計算
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
    def __init__(self, num_classes, backbone='resnet50', local_dim=1024, global_dim=2048, output_dim=512):
        super(DOLGNetWrapper, self).__init__()
        self.backbone = create_dolg_model(
            backbone=backbone,
            pretrained=True,
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
            loss: 損失値 (学習時)
            logits: 分類ロジット [B, num_classes]
        """
        return self.backbone(x, targets)
    
    def extract_descriptor(self, x):
        """特徴記述子の抽出"""
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
            loss: 損失値 (学習時)
            logits: 分類ロジット [B, num_classes]
        """
        if self.training and targets is not None:
            # 学習時: 特徴抽出 + 損失計算
            features = self.backbone.extract_features(x)
            logits = self.backbone(x)
            
            # ULTRON損失計算
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
        # AdamWフェーズ (デフォルト)
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
        print(key + ":" + str(vars(args)[key]))
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # set the path for model save:
    args.directory = get_checkpoint_root()
    os.makedirs(args.directory, exist_ok=True)
    
    # モデル名に基づいてパス生成を調整
    if args.model.startswith('ultron'):
        path = '{}-bb{}-bl{}-fl{}-b{}-nep{}-rho{}-speedup-githubv'.format(
            args.model, args.backbone, args.base_lr, args.final_lr, 
            args.batch_size, args.num_epochs, getattr(args, 'rho', 0.04)
        )
    elif args.model.startswith('cvnet'):
        path = '{}-bb{}-bl{}-fl{}-b{}-nep{}-rd{}-speedup-githubv'.format(
            args.model, args.backbone, args.base_lr, args.final_lr, 
            args.batch_size, args.num_epochs, getattr(args, 'reduction_dim', 512)
        )
    elif args.model.startswith('dolgnet'):
        path = '{}-bb{}-bl{}-fl{}-b{}-nep{}-ld{}-gd{}-od{}-speedup-githubv'.format(
            args.model, args.backbone, args.base_lr, args.final_lr, 
            args.batch_size, args.num_epochs, getattr(args, 'local_dim', 1024),
            getattr(args, 'global_dim', 2048), getattr(args, 'output_dim', 512)
        )
    else:
        path = '{}-bb{}-bl{}-fl{}-s{}-m{}-b{}-nep{}-speedup-githubv'.format(
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

    # distributed paralell setting:
    if args.distributed:
        ngpus_per_node = cuda.device_count()
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print('>> batch size per node:{}'.format(args.batch_size))
        print('>> num workers per node:{}'.format(args.num_workers))

    train_dataset, val_dataset, class_num = GLDv2_lmdb(args.imsize, args.seed, args.split)
    args.classifier_num = class_num

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=False)
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=True)   
        val_sampler = DistributedSampler(val_dataset)
        val_batch_sampler = BatchSampler(val_sampler, args.batch_size, drop_last=False)
        val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_batch_sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=False)

    # モデル作成部分を条件分岐で拡張
    if args.model.startswith('token'):
        model = Token(1024, args.classifier_num, args.pretrained, args.backbone).to(device)
    elif args.model.startswith('spca'):
        meta = {}
        meta['outputdim'] = args.outputdim
        meta['K'] = args.codebook_size
        meta['local_dim'] = args.local_dim
        meta['combine'] = args.combine
        meta['multi'] = args.multi
        meta['pretrained'] = args.pretrained
        model = SpCa(args.outputdim, args.classifier_num, meta, args.tau, args.margin, args.backbone).to(device)
    elif args.model.startswith('ultron'):
        # ULTRON モデルの作成
        if not ULTRON_AVAILABLE:
            raise ValueError('ULTRON modules are not available. Please ensure ULTRON implementation files are present.')
        
        # ULTRON用のパラメータ設定
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
            raise ValueError('CVNet modules are not available. Please ensure CVNet implementation files are present.')
        
        # CVNet用のパラメータ設定
        resnet_depth = getattr(args, 'resnet_depth', 50)
        reduction_dim = getattr(args, 'reduction_dim', 512)
        
        model = CVNetWrapper(
            num_classes=args.classifier_num,
            resnet_depth=resnet_depth,
            reduction_dim=reduction_dim
        ).to(device)
        
        print(f'>> CVNet model created with {args.classifier_num} classes, resnet_depth={resnet_depth}, reduction_dim={reduction_dim}')
    elif args.model.startswith('dolgnet'):
        # DOLGNet モデルの作成
        if not DOLGNET_AVAILABLE:
            raise ValueError('DOLGNet modules are not available. Please ensure DOLGNet implementation files are present.')
        
        # DOLGNet用のパラメータ設定
        local_dim = getattr(args, 'local_dim', 1024)
        global_dim = getattr(args, 'global_dim', 2048)
        output_dim = getattr(args, 'output_dim', 512)
        
        model = DOLGNetWrapper(
            num_classes=args.classifier_num,
            backbone=args.backbone,
            local_dim=local_dim,
            global_dim=global_dim,
            output_dim=output_dim
        ).to(device)
        
        print(f'>> DOLGNet model created with {args.classifier_num} classes, backbone={args.backbone}, local_dim={local_dim}, global_dim={global_dim}, output_dim={output_dim}')
    else:
        raise ValueError('Unsupported or unknown model: {}!'.format(args.model))

    # 最適化器の作成を条件分岐
    if args.model.startswith('ultron'):
        # ULTRON用の最適化器
        optimizer = create_ultron_optimizer(args, model)
        print(f'>> ULTRON optimizer created: {type(optimizer).__name__}')
    elif args.model.startswith('cvnet'):
        # CVNet用の最適化器
        optimizer = create_cvnet_optimizer(args, model)
        print(f'>> CVNet optimizer created: {type(optimizer).__name__}')
    elif args.model.startswith('dolgnet'):
        # DOLGNet用の最適化器
        optimizer = create_dolgnet_optimizer(args, model)
        print(f'>> DOLGNet optimizer created: {type(optimizer).__name__}')
    else:
        # 既存のSPCA/Token用最適化器
        param_dicts = create_optimizer(args.weight_decay, model)
        optimizer = optim.SGD(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True, dampening=0.0)
    
    model_without_ddp = model

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('>> number of params:{:.2f}M'.format(n_parameters / (1024 * 1024)))
    
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optim'])
            optimizer_to(optimizer, device)
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))

    # 学習率スケジューラーの設定を条件分岐
    if args.model.startswith('ultron'):
        # ULTRON用のスケジューラー (論文設定に基づく)
        warmup_epochs = getattr(args, 'ultron_warmup_epochs', 5)
        if hasattr(args, 'ultron_phase') and args.ultron_phase == 'sgd':
            # SGDフェーズ用のコサインスケジューラー
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.num_epochs - warmup_epochs, eta_min=1e-6
            )
        else:
            # AdamWフェーズ用の固定学習率
            lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif args.model.startswith('cvnet'):
        # CVNet用のスケジューラー (標準的なWarmupCosスケジューラー)
        lr_scheduler = WarmupCos_Scheduler(optimizer=optimizer,
                                           warmup_epochs=args.warmup_epochs,
                                           warmup_lr=args.warmup_lr * args.update_every,
                                           num_epochs=args.num_epochs,
                                           base_lr=args.base_lr * args.update_every,
                                           final_lr=args.final_lr * args.update_every,
                                           iter_per_epoch=int(len(train_loader) / args.update_every))

        lr_scheduler.iter = max(int(len(train_loader) / args.update_every) * start_epoch - 1, 0)
    elif args.model.startswith('dolgnet'):
        # DOLGNet用のスケジューラー (AdamW用のWarmupCosスケジューラー)
        lr_scheduler = WarmupCos_Scheduler(optimizer=optimizer,
                                           warmup_epochs=args.warmup_epochs,
                                           warmup_lr=args.warmup_lr * args.update_every,
                                           num_epochs=args.num_epochs,
                                           base_lr=args.base_lr * args.update_every,
                                           final_lr=args.final_lr * args.update_every,
                                           iter_per_epoch=int(len(train_loader) / args.update_every))

        lr_scheduler.iter = max(int(len(train_loader) / args.update_every) * start_epoch - 1, 0)
    else:
        # 既存のWarmupCosスケジューラー
        lr_scheduler = WarmupCos_Scheduler(optimizer=optimizer,
                                           warmup_epochs=args.warmup_epochs,
                                           warmup_lr=args.warmup_lr * args.update_every,
                                           num_epochs=args.num_epochs,
                                           base_lr=args.base_lr * args.update_every,
                                           final_lr=args.final_lr * args.update_every,
                                           iter_per_epoch=int(len(train_loader) / args.update_every))

        lr_scheduler.iter = max(int(len(train_loader) / args.update_every) * start_epoch - 1, 0)

    # Start training
    metric_logger = MetricLogger(delimiter=" ")
    val_metric_logger = MetricLogger(delimiter=" ")
    print_freq = 200
    model_path = None
    
    # ログ用の辞書を条件分岐で設定
    if args.model.startswith('ultron'):
        Loss_logger = {'ULTRON loss': []}
        val_Loss_logger = {'ULTRON loss': []}
    elif args.model.startswith('cvnet'):
        Loss_logger = {'CVNet loss': []}
        val_Loss_logger = {'CVNet loss': []}
    elif args.model.startswith('dolgnet'):
        Loss_logger = {'DOLG loss': []}
        val_Loss_logger = {'DOLG loss': []}
    else:
        Loss_logger = {'ArcFace loss': []}
        val_Loss_logger = {'ArcFace loss': []}
    
    Error_Logger = {'Top1 error': [], 'Top5 error': []}
    LR_Logger = {'Learning Rate': []}
    val_Error_Logger = {'Top1 error': [], 'Top5 error': []}
    min_val = 100.0

    # ULTRON用のオプティマイザー切り替え管理
    ultron_optimizer_switched = False

    for epoch in range(start_epoch, args.num_epochs): 
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch + 1 + get_rank())
        
        # ULTRON用のオプティマイザー切り替え (エポック5でAdamW→SGD)
        if args.model.startswith('ultron') and epoch == 5 and not ultron_optimizer_switched:
            print(">> Switching ULTRON optimizer from AdamW to SGD")
            args.ultron_phase = 'sgd'
            optimizer = create_ultron_optimizer(args, model_without_ddp)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.num_epochs - 5, eta_min=1e-6
            )
            ultron_optimizer_switched = True
            
            # 分散学習の場合は再設定
            if args.distributed:
                optimizer_to(optimizer, device)
        
        header = '>> Train Epoch: [{}]'.format(epoch)
        
        optimizer.zero_grad()
        for idx, (images, targets) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            model.train()
            targets = targets.to(device, non_blocking=True)
            loss, logits = model(images.to(device, non_blocking=True), targets)         
            loss.backward()

            # ログ名を条件分岐で設定
            if args.model.startswith('ultron'):
                loss_key = 'ULTRON loss'
            elif args.model.startswith('cvnet'):
                loss_key = 'CVNet loss'
            elif args.model.startswith('dolgnet'):
                loss_key = 'DOLG loss'
            else:
                loss_key = 'ArcFace loss'
            
            metric_logger.meters[loss_key].update(loss.item())
            
            with torch.no_grad():
                desc_top1_err, desc_top5_err = topk_errors(logits, targets, [1, 5])
                metric_logger.meters['Top1 error'].update(desc_top1_err.item())
                metric_logger.meters['Top5 error'].update(desc_top5_err.item())

            # 更新頻度の条件分岐
            update_condition = True
            if args.model.startswith('ultron'):
                # ULTRONは毎ステップ更新
                update_condition = True
            elif args.model.startswith('cvnet'):
                # CVNetは既存の更新頻度
                update_condition = (idx + 1) % args.update_every == 0 and lr_scheduler.iter < len(lr_scheduler.lr_schedule) - 1
            elif args.model.startswith('dolgnet'):
                # DOLGNetは既存の更新頻度
                update_condition = (idx + 1) % args.update_every == 0 and lr_scheduler.iter < len(lr_scheduler.lr_schedule) - 1
            else:
                # SPCA/Tokenは既存の更新頻度
                update_condition = (idx + 1) % args.update_every == 0 and lr_scheduler.iter < len(lr_scheduler.lr_schedule) - 1

            if update_condition:
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                
                if args.model.startswith('ultron'):
                    # ULTRON用のスケジューラー更新
                    if hasattr(lr_scheduler, 'step'):
                        lr_scheduler.step()
                    lr = optimizer.param_groups[0]['lr']
                elif args.model.startswith('cvnet'):
                    # CVNet用のスケジューラー更新
                    lr = lr_scheduler.step()
                elif args.model.startswith('dolgnet'):
                    # DOLGNet用のスケジューラー更新
                    lr = lr_scheduler.step()
                else:
                    # 既存のスケジューラー更新
                    lr = lr_scheduler.step()
                
                optimizer.step()
                optimizer.zero_grad()

            if (idx + 1) % 10 == 0:
                if is_main_process():
                    Loss_logger[loss_key].append(metric_logger.meters[loss_key].avg)
                    Error_Logger['Top1 error'].append(metric_logger.meters['Top1 error'].avg)
                    Error_Logger['Top5 error'].append(metric_logger.meters['Top5 error'].avg)
                    LR_Logger['Learning Rate'].append(lr if isinstance(lr, (int, float)) else lr.squeeze())
                    
                    # プロット作成
                    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
                    fig.tight_layout()
                    axes = axes.flatten()
                    for (key, value) in Loss_logger.items():
                        axes[0].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                    axes[0].legend(loc='upper right', shadow=True, fontsize='medium')
                    axes[0].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                    axes[0].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                    axes[0].set_xlabel('iter')
                    axes[0].set_ylabel("loss")
                    axes[0].minorticks_on()
                    for (key, value) in LR_Logger.items():
                        axes[1].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                    axes[1].legend(loc='upper right', shadow=True, fontsize='medium')
                    axes[1].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                    axes[1].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                    axes[1].set_xlabel('iter')
                    axes[1].set_ylabel("learning rate")
                    axes[1].minorticks_on()
                    for (key, value) in Error_Logger.items():
                        axes[2].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                    axes[2].legend(loc='upper right', shadow=True, fontsize='medium')
                    axes[2].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                    axes[2].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                    axes[2].set_xlabel('iter')
                    axes[2].set_ylabel("Error rate (%)")
                    axes[2].minorticks_on()
                    plt.savefig(os.path.join(directory, 'training_{}_{}_logger.png'.format(args.model, args.comment)))
                    plt.close()

        if (epoch + 1) % args.val_epoch == 0:
            with torch.no_grad():
                # Enable eval mode
                model.eval()
                for idx, (inputs, labels) in enumerate(val_metric_logger.log_every(val_loader, print_freq, '>> Val Epoch: [{}]'.format(epoch))):
                    # Transfer the data to the current GPU device
                    inputs, labels = inputs.to(device), labels.to(device, non_blocking=True)
                    # Compute the predictions
                    loss, logits = model(inputs, labels)

                    val_metric_logger.meters[loss_key].update(loss.item())
                    # Compute the errors
                    desc_top1_err, desc_top5_err = topk_errors(logits, labels, [1, 5])
                    val_metric_logger.meters['Top1 error'].update(desc_top1_err.item())
                    val_metric_logger.meters['Top5 error'].update(desc_top5_err.item())

                    if (idx + 1) % 10 == 0:
                        if is_main_process():
                            val_Loss_logger[loss_key].append(val_metric_logger.meters[loss_key].avg)
                            val_Error_Logger['Top1 error'].append(val_metric_logger.meters['Top1 error'].avg)
                            val_Error_Logger['Top5 error'].append(val_metric_logger.meters['Top5 error'].avg)
                            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
                            fig.tight_layout()
                            axes = axes.flatten()
                            for (key, value) in val_Loss_logger.items():
                                axes[0].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                            axes[0].legend(loc='upper right', shadow=True, fontsize='medium')
                            axes[0].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                            axes[0].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                            axes[0].set_xlabel('iter')
                            axes[0].set_ylabel("loss")
                            axes[0].minorticks_on()
                            for (key, value) in val_Error_Logger.items():
                                axes[1].plot(value, 'o-', label=key, linewidth=1, markersize=2)
                            axes[1].legend(loc='upper right', shadow=True, fontsize='medium')
                            axes[1].grid(b=True, which='major', color='gray', linestyle='-', alpha=0.1)
                            axes[1].grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.1)
                            axes[1].set_xlabel('iter')
                            axes[1].set_ylabel("Error rate (%)")
                            axes[1].minorticks_on()
                            plt.savefig(os.path.join(directory, 'val_logger_{}.png'.format(args.comment)))
                            plt.close()

        if is_main_process():
            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0:
                model_path = os.path.join(directory, 'epoch{}.pth'.format(epoch + 1))
                torch.save({'epoch': epoch + 1, 'state_dict': model_without_ddp.state_dict(), 'optim': optimizer.state_dict()}, model_path)
                model_path_pre = os.path.join(directory, 'epoch{}.pth'.format(epoch + 1 - args.save_freq))
                if epoch % 5 != 0:
                    try:
                        os.remove(model_path_pre)
                    except:
                        pass

    print('>> Training finished.')


if __name__ == '__main__':
    args = get_args()
    
    # ULTRON用のデフォルトパラメータを追加
    if not hasattr(args, 'embed_dim'):
        args.embed_dim = 512
    if not hasattr(args, 'rho'):
        args.rho = 0.04
    if not hasattr(args, 'ultron_warmup_epochs'):
        args.ultron_warmup_epochs = 5
    
    # CVNet用のデフォルトパラメータを追加
    if not hasattr(args, 'resnet_depth'):
        args.resnet_depth = 50
    if not hasattr(args, 'reduction_dim'):
        args.reduction_dim = 512
    
    # DOLGNet用のデフォルトパラメータを追加
    if not hasattr(args, 'local_dim'):
        args.local_dim = 1024
    if not hasattr(args, 'global_dim'):
        args.global_dim = 2048
    if not hasattr(args, 'output_dim'):
        args.output_dim = 512
    
    main(args)

