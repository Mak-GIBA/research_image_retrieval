import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch import cuda, optim
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# IRIS関連のインポート
from iris_implementation import IRIS, IRISLoss, IRISRetrieval
from iris_implementation import compute_similarity, evaluate_retrieval

# 分散学習とユーティリティ関数のインポート
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# データセット関連のインポート（SpCaのデータセットを使用）
# 注: 実際の実装では、必要に応じてデータセット関連のコードを修正する必要があります
try:
    from dataset import ImageFromList, GLDv2_lmdb
except ImportError:
    print("Warning: SpCa dataset modules not found. Using placeholder implementations.")
    # プレースホルダーのデータセットクラス
    class GLDv2_lmdb:
        def __init__(self, imsize, seed, split):
            self.imsize = imsize
            self.seed = seed
            self.split = split
            print(f"Placeholder GLDv2_lmdb with imsize={imsize}, seed={seed}, split={split}")
            # 実際のデータセットの代わりにダミーデータを返す
            import torch.utils.data as data
            class DummyDataset(data.Dataset):
                def __init__(self, size=1000, dim=3, imsize=224):
                    self.size = size
                    self.dim = dim
                    self.imsize = imsize
                
                def __getitem__(self, index):
                    img = torch.randn(self.dim, self.imsize, self.imsize)
                    label = torch.tensor(index % 100)  # 100クラスを想定
                    return img, label
                
                def __len__(self):
                    return self.size
            
            self.train_dataset = DummyDataset(size=1000)
            self.val_dataset = DummyDataset(size=200)
            self.class_num = 100
        
        def __call__(self):
            return self.train_dataset, self.val_dataset, self.class_num

# 引数解析のためのインポート
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='IRIS Training Script')
    
    # モデル関連
    parser.add_argument('--model', type=str, default='iris', help='Model name')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='Backbone network')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained backbone')
    parser.add_argument('--dim', type=int, default=512, help='Feature dimension')
    
    # データセット関連
    parser.add_argument('--imsize', type=int, default=224, help='Image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split', type=int, default=None, help='Dataset split')
    
    # 学習関連
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--base_lr', type=float, default=0.01, help='Base learning rate')
    parser.add_argument('--final_lr', type=float, default=0.0001, help='Final learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=0.0001, help='Warmup learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--update_every', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--clip_max_norm', type=float, default=0.0, help='Clip max norm (0 for no clipping)')
    
    # IRIS特有のパラメータ
    parser.add_argument('--use_spatial_features', type=bool, default=True, help='Use spatial features')
    parser.add_argument('--oracle_num_objects', type=int, default=8, help='Number of objects for ORACLE module')
    parser.add_argument('--castle_num_heads', type=int, default=8, help='Number of heads for CASTLE module')
    parser.add_argument('--nexus_sparsity', type=float, default=0.5, help='Sparsity threshold for NEXUS module')
    parser.add_argument('--margin', type=float, default=0.2, help='Margin for triplet loss')
    
    # 損失関連
    parser.add_argument('--classification_weight', type=float, default=1.0, help='Classification loss weight')
    parser.add_argument('--triplet_weight', type=float, default=1.0, help='Triplet loss weight')
    parser.add_argument('--structure_weight', type=float, default=0.5, help='Structure consistency loss weight')
    
    # 分散学習関連
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', help='Distributed backend')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='Rank of the process')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank')
    
    # その他
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--directory', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--val_epoch', type=int, default=5, help='Validation frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')
    parser.add_argument('--comment', type=str, default='', help='Comment for tensorboard')
    
    args = parser.parse_args()
    return args

# 学習率スケジューラ
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
        return self.lr_schedule[self.iter-1]
    
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

# メトリックロガー
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
            
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)
        
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)
    
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if header is not None:
            print(header)
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0:
                print(f"{i}/{len(iterable)}: {self}")

class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        return f"{self.avg:.4f}"

# 分散学習の初期化
def init_distributed_mode(args):
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            print('Not using distributed mode')
            args.distributed = False
            return
            
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank
        )
        torch.distributed.barrier()
    else:
        args.gpu = 0

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

# オプティマイザの作成
def create_optimizer(weight_decay, model):
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr_scale": 0.1,
        },
    ]
    return param_dicts

# デバイス間のオプティマイザ状態の移動
def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

# チェックポイントの保存ディレクトリを取得
def get_checkpoint_root():
    return os.path.join(os.getcwd(), 'checkpoints')

# モデルの重みを凍結
def freeze_weights(model, pattern):
    for name, param in model.named_parameters():
        if pattern in name:
            param.requires_grad = False

# モデルの重みを解凍
def unfreeze_weights(model, pattern):
    for name, param in model.named_parameters():
        if pattern in name:
            param.requires_grad = True

# チェックポイントの読み込み
def load_checkpoint(path, model, optimizer=None):
    if not os.path.isfile(path):
        print(f"No checkpoint found at '{path}'")
        return 0
    
    print(f"Loading checkpoint from '{path}'")
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optim'])
    
    print(f"Loaded checkpoint from '{path}' (epoch {checkpoint['epoch']})")
    return checkpoint['epoch']

# トップkエラーの計算
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

# メイン関数
def main(args):
    print('IRIS Training Script')
    
    # 分散学習の初期化
    if args.distributed:
        init_distributed_mode(args)
    
    # 引数の表示
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    
    # デバイスの設定
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    # チェックポイント保存ディレクトリの設定
    args.directory = get_checkpoint_root()
    os.makedirs(args.directory, exist_ok=True)
    
    # モデル名とパラメータに基づいてパスを生成
    path = f'{args.model}-bb{args.backbone}-dim{args.dim}-bs{args.batch_size}-nep{args.num_epochs}'
    if args.seed != 42:
        path += f'-seed{args.seed}'
    if args.split is not None:
        path += f'-split{args.split}'
    path += f'-oracle{args.oracle_num_objects}-castle{args.castle_num_heads}-nexus{args.nexus_sparsity}'
    path += f'-{args.comment}'
    
    directory = os.path.join(args.directory, path)
    os.makedirs(directory, exist_ok=True)
    
    # 分散学習の設定
    if args.distributed:
        ngpus_per_node = cuda.device_count()
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(f'>> batch size per node: {args.batch_size}')
        print(f'>> num workers per node: {args.num_workers}')
    
    # データセットの読み込み
    dataset_fn = GLDv2_lmdb(args.imsize, args.seed, args.split)
    train_dataset, val_dataset, class_num = dataset_fn()
    args.classifier_num = class_num
    
    # データローダーの設定
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=False)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_sampler = DistributedSampler(val_dataset)
        val_batch_sampler = BatchSampler(val_sampler, args.batch_size, drop_last=False)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=None,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=None,
            drop_last=False
        )
    
    # モデルの初期化
    model = IRIS(
        backbone=args.backbone,
        pretrained=args.pretrained,
        dim=args.dim,
        num_classes=args.classifier_num,
        oracle_num_objects=args.oracle_num_objects,
        castle_num_heads=args.castle_num_heads,
        nexus_sparsity=args.nexus_sparsity,
        use_spatial_features=args.use_spatial_features
    ).to(device)
    
    # オプティマイザの設定
    param_dicts = create_optimizer(args.weight_decay, model)
    optimizer = optim.SGD(
        param_dicts,
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=True,
        dampening=0.0
    )
    
    # 分散学習の設定
    model_without_ddp = model
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # モデルのパラメータ数を表示
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print(f'>> number of params: {n_parameters / (1024 * 1024):.2f}M')
    
    # チェックポイントからの復元
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f">> Loading checkpoint: '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optim'])
            optimizer_to(optimizer, device)
            print(f">>>> loaded checkpoint: '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f">> No checkpoint found at '{args.resume}'")
    
    # 学習率スケジューラの設定
    lr_scheduler = WarmupCos_Scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr * args.update_every,
        num_epochs=args.num_epochs,
        base_lr=args.base_lr * args.update_every,
        final_lr=args.final_lr * args.update_every,
        iter_per_epoch=int(len(train_loader) / args.update_every)
    )
    lr_scheduler.iter = max(int(len(train_loader) / args.update_every) * start_epoch - 1, 0)
    
    # 損失関数の設定
    criterion = IRISLoss(
        margin=args.margin,
        classification_weight=args.classification_weight,
        triplet_weight=args.triplet_weight,
        structure_weight=args.structure_weight
    )
    
    # メトリックロガーの設定
    metric_logger = MetricLogger(delimiter=" ")
    val_metric_logger = MetricLogger(delimiter=" ")
    print_freq = 20
    
    # ロギング用の設定
    Loss_logger = {'Classification loss': [], 'Triplet loss': [], 'Structure loss': [], 'Total loss': []}
    Error_Logger = {'Top1 error': [], 'Top5 error': []}
    LR_Logger = {'Learning Rate': []}
    val_Loss_logger = {'Classification loss': [], 'Triplet loss': [], 'Structure loss': [], 'Total loss': []}
    val_Error_Logger = {'Top1 error': [], 'Top5 error': []}
    
    # TensorBoardの設定
    if is_main_process():
        writer = SummaryWriter(log_dir=os.path.join(directory, 'tensorboard'))
    
    # 最小検証エラーの初期化
    min_val = 100.0
    
    # 学習ループ
    for epoch in range(start_epoch, args.num_epochs):
        # 分散サンプラーのエポック設定
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch + 1 + get_rank())
        
        header = f'>> Train Epoch: [{epoch}]'
        
        # オプティマイザのリセット
        optimizer.zero_grad()
        
        # 学習ループ
        for idx, (images, targets) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            # モデルをトレーニングモードに設定
            model.train()
            
            # データをデバイスに移動
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # 順伝播
            outputs = model(images)
            
            # 損失の計算
            losses = criterion(outputs, targets)
            loss = losses['total']
            
            # 逆伝播
            loss.backward()
            
            # 勾配のクリッピング
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            
            # 勾配の更新
            if (idx + 1) % args.update_every == 0 or (idx + 1) == len(train_loader):
                # 学習率の更新
                lr = lr_scheduler.step()
                
                # オプティマイザのステップ
                optimizer.step()
                optimizer.zero_grad()
            
            # メトリックの更新
            metric_logger.update(loss=loss.item())
            metric_logger.update(cls_loss=losses['classification'].item())
            metric_logger.update(triplet_loss=losses['triplet'].item())
            metric_logger.update(structure_loss=losses['structure'].item())
            
            # エラー率の計算
            with torch.no_grad():
                desc_top1_err, desc_top5_err = topk_errors(outputs['logits'], targets, [1, 5])
                metric_logger.meters['Top1 error'].update(desc_top1_err.item())
                metric_logger.meters['Top5 error'].update(desc_top5_err.item())
            
            # ロギング（10イテレーションごと）
            if (idx + 1) % 10 == 0 and is_main_process():
                # ロガーの更新
                Loss_logger['Classification loss'].append(metric_logger.meters['cls_loss'].avg)
                Loss_logger['Triplet loss'].append(metric_logger.meters['triplet_loss'].avg)
                Loss_logger['Structure loss'].append(metric_logger.meters['structure_loss'].avg)
                Loss_logger['Total loss'].append(metric_logger.meters['loss'].avg)
                Error_Logger['Top1 error'].append(metric_logger.meters['Top1 error'].avg)
                Error_Logger['Top5 error'].append(metric_logger.meters['Top5 error'].avg)
                LR_Logger['Learning Rate'].append(lr)
                
                # TensorBoardへの書き込み
                writer.add_scalar('Train/Loss/Classification', metric_logger.meters['cls_loss'].avg, epoch * len(train_loader) + idx)
                writer.add_scalar('Train/Loss/Triplet', metric_logger.meters['triplet_loss'].avg, epoch * len(train_loader) + idx)
                writer.add_scalar('Train/Loss/Structure', metric_logger.meters['structure_loss'].avg, epoch * len(train_loader) + idx)
                writer.add_scalar('Train/Loss/Total', metric_logger.meters['loss'].avg, epoch * len(train_loader) + idx)
                writer.add_scalar('Train/Error/Top1', metric_logger.meters['Top1 error'].avg, epoch * len(train_loader) + idx)
                writer.add_scalar('Train/Error/Top5', metric_logger.meters['Top5 error'].avg, epoch * len(train_loader) + idx)
                writer.add_scalar('Train/LR', lr, epoch * len(train_loader) + idx)
        
        # 検証（val_epochごと）
        if (epoch + 1) % args.val_epoch == 0:
            # 検証ループ
            header = f'>> Val Epoch: [{epoch}]'
            
            with torch.no_grad():
                # モデルを評価モードに設定
                model.eval()
                
                for idx, (inputs, labels) in enumerate(val_metric_logger.log_every(val_loader, print_freq, header)):
                    # データをデバイスに移動
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # 順伝播
                    outputs = model(inputs)
                    
                    # 損失の計算
                    losses = criterion(outputs, labels)
                    
                    # メトリックの更新
                    val_metric_logger.update(loss=losses['total'].item())
                    val_metric_logger.update(cls_loss=losses['classification'].item())
                    val_metric_logger.update(triplet_loss=losses['triplet'].item())
                    val_metric_logger.update(structure_loss=losses['structure'].item())
                    
                    # エラー率の計算
                    desc_top1_err, desc_top5_err = topk_errors(outputs['logits'], labels, [1, 5])
                    val_metric_logger.meters['Top1 error'].update(desc_top1_err.item())
                    val_metric_logger.meters['Top5 error'].update(desc_top5_err.item())
                
                # ロギング
                if is_main_process():
                    # ロガーの更新
                    val_Loss_logger['Classification loss'].append(val_metric_logger.meters['cls_loss'].avg)
                    val_Loss_logger['Triplet loss'].append(val_metric_logger.meters['triplet_loss'].avg)
                    val_Loss_logger['Structure loss'].append(val_metric_logger.meters['structure_loss'].avg)
                    val_Loss_logger['Total loss'].append(val_metric_logger.meters['loss'].avg)
                    val_Error_Logger['Top1 error'].append(val_metric_logger.meters['Top1 error'].avg)
                    val_Error_Logger['Top5 error'].append(val_metric_logger.meters['Top5 error'].avg)
                    
                    # TensorBoardへの書き込み
                    writer.add_scalar('Val/Loss/Classification', val_metric_logger.meters['cls_loss'].avg, epoch)
                    writer.add_scalar('Val/Loss/Triplet', val_metric_logger.meters['triplet_loss'].avg, epoch)
                    writer.add_scalar('Val/Loss/Structure', val_metric_logger.meters['structure_loss'].avg, epoch)
                    writer.add_scalar('Val/Loss/Total', val_metric_logger.meters['loss'].avg, epoch)
                    writer.add_scalar('Val/Error/Top1', val_metric_logger.meters['Top1 error'].avg, epoch)
                    writer.add_scalar('Val/Error/Top5', val_metric_logger.meters['Top5 error'].avg, epoch)
                    
                    # 最良モデルの保存
                    if val_metric_logger.meters['Top1 error'].avg < min_val:
                        min_val = val_metric_logger.meters['Top1 error'].avg
                        model_path = os.path.join(directory, 'best_checkpoint.pth')
                        torch.save({
                            'epoch': epoch + 1,
                            'state_dict': model_without_ddp.state_dict(),
                            'optim': optimizer.state_dict(),
                        }, model_path)
        
        # チェックポイントの保存（save_freqごと）
        if is_main_process() and (epoch + 1) % args.save_freq == 0:
            model_path = os.path.join(directory, f'epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model_without_ddp.state_dict(),
                'optim': optimizer.state_dict(),
            }, model_path)
            
            # 古いチェックポイントの削除（5エポックごとに保存）
            if epoch % 5 != 0:
                model_path_pre = os.path.join(directory, f'epoch{epoch}.pth')
                try:
                    os.remove(model_path_pre)
                    print(f'The previous saved model <<model_epoch{epoch}.pth.tar>> is deleted from disk to save space')
                except:
                    print(f'The previous saved model <<model_epoch{epoch}.pth.tar>> does not exist')
    
    # 最終モデルの保存
    if is_main_process():
        model_path = os.path.join(directory, 'final_model.pth')
        torch.save({
            'epoch': args.num_epochs,
            'state_dict': model_without_ddp.state_dict(),
            'optim': optimizer.state_dict(),
        }, model_path)
        
        print(f"Training completed. Final model saved to {model_path}")
        
        # TensorBoardの終了
        writer.close()
    
    return model_path

if __name__ == '__main__':
    args = get_args()
    main(args)
