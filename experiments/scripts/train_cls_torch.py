# ============================================================
# train_cls_torch.py — Phase 3 (定版)
# 職責：PyTorch 分類模型訓練
# 支援：ResNet-50, EfficientNet-B0, MobileNetV3-L, ViT-S/16
# 不呼叫 logger（由 run_experiments.py orchestrator 負責）
# ============================================================
import copy
import os
import sys
import time
import argparse
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

BASE_DIR = Path(__file__).parent.parent.parent

CLASS_NAMES = ['Abrasions', 'Bruises', 'Burns', 'Cut',
               'Ingrown_nails', 'Laceration', 'Stab_wound']
NUM_CLASSES = 7

TORCH_MODELS = {'resnet50', 'efficientnet_b0', 'mobilenet_v3_large', 'vit_s_16'}


# ── 模型建構 ──────────────────────────────────────────────────
def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    if model_name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)

    elif model_name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)

    elif model_name == 'mobilenet_v3_large':
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)

    elif model_name == 'vit_s_16':
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        # ViT-B 換成 Small 的方式：使用 B 規格替代（ViT-S 在 torchvision 中無官方預訓練）
        m.heads.head = nn.Linear(m.heads.head.in_features, NUM_CLASSES)

    else:
        raise ValueError(f"Unknown model: {model_name}. Options: {TORCH_MODELS}")

    return m


# ── 資料增強 ──────────────────────────────────────────────────
def build_transforms(img_size: int, aug: dict, is_train: bool):
    if is_train:
        t = [transforms.Resize((img_size + 32, img_size + 32)),
             transforms.RandomCrop(img_size)]
        if aug.get('fliplr', 0) > 0:
            t.append(transforms.RandomHorizontalFlip(p=aug['fliplr']))
        if aug.get('flipud', 0) > 0:
            t.append(transforms.RandomVerticalFlip(p=aug['flipud']))
        if aug.get('degrees', 0) > 0:
            t.append(transforms.RandomRotation(degrees=aug['degrees']))
        if aug.get('scale', 0) > 0:
            s = 1.0 - aug['scale']
            t.append(transforms.RandomResizedCrop(img_size, scale=(max(s, 0.3), 1.0)))
        t += [
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        if aug.get('erasing', 0) > 0:
            t.append(transforms.RandomErasing(p=aug['erasing']))
    else:
        t = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    return transforms.Compose(t)


# ── 主訓練函式 ─────────────────────────────────────────────────
def train_torch_cls(flat_cfg: dict) -> dict:
    """
    執行 PyTorch 分類模型訓練。

    Args:
        flat_cfg: _cfg_to_flat() 產生的扁平 dict

    Returns:
        dict 含 weights_path, best_epoch, train_samples, val_samples
    """
    model_name   = flat_cfg['model']
    run_name     = flat_cfg['run_name']
    seed         = flat_cfg.get('seed', 42)
    epochs       = flat_cfg.get('epochs', 150)
    batch        = flat_cfg.get('batch_size', 32)
    imgsz        = flat_cfg.get('img_size', 224)
    lr0          = flat_cfg.get('lr0', 0.001)
    weight_decay = flat_cfg.get('weight_decay', 0.0005)
    optimizer_name = flat_cfg.get('optimizer', 'adam')
    dataset      = flat_cfg.get('dataset', '')
    aug          = flat_cfg.get('augmentation', {})

    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset_path = Path(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"   [{run_name}] Model: {model_name}  Device: {device}")

    # DataLoaders
    train_tf = build_transforms(imgsz, aug, is_train=True)
    val_tf   = build_transforms(imgsz, {}, is_train=False)
    train_ds = datasets.ImageFolder(str(dataset_path / 'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(dataset_path / 'val'),   transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                              num_workers=0, pin_memory=torch.cuda.is_available())

    train_samples = len(train_ds)
    val_samples   = len(val_ds)

    # 模型
    model = build_model(model_name).to(device)

    # Optimizer
    opt_name = optimizer_name.lower()
    if opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr0,
                              momentum=0.937, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr0, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr0, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None
    patience_ctr = 0
    patience_max = 50

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), labels).backward()
            optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = model(imgs.to(device)).argmax(1)
                correct += (preds == labels.to(device)).sum().item()
                total   += labels.size(0)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}/{epochs}  Val: {val_acc:.4f}  Best: {best_val_acc:.4f}")

        if patience_ctr >= patience_max:
            print(f"   EarlyStopping at epoch {epoch}")
            break

    # 儲存最佳權重
    run_dir = BASE_DIR / 'experiments' / 'results' / 'raw' / run_name / 'weights'
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = str(run_dir / 'best.pt')
    torch.save({
        'model_name':  model_name,
        'state_dict':  best_state,
        'classes':     CLASS_NAMES,
        'best_epoch':  best_epoch,
        'val_acc':     best_val_acc,
    }, weights_path)

    print(f"   [{run_name}] Done — Best Val: {best_val_acc*100:.2f}%  Best Epoch: {best_epoch}")

    return {
        'weights_path':  weights_path,
        'best_epoch':    best_epoch,
        'val_top1':      round(best_val_acc * 100, 2),
        'train_samples': train_samples,
        'val_samples':   val_samples,
    }


# ── 允許直接執行（供測試用）──────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from experiment_logger import log_start, log_done, log_failed

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg['experiment']['id']
    flat_cfg = {
        'exp_id':       exp_id,
        'model':        cfg['model']['name'],
        'run_name':     f"{cfg['experiment']['id']}_{cfg['experiment']['name']}",
        'dataset':      str(BASE_DIR / cfg['dataset']['path']),
        'epochs':       cfg['training']['epochs'],
        'batch_size':   cfg['training']['batch'],
        'img_size':     cfg['training']['imgsz'],
        'seed':         cfg['training']['seed'],
        'optimizer':    cfg['optimizer']['name'],
        'lr0':          cfg['optimizer']['lr0'],
        'weight_decay': cfg['optimizer']['weight_decay'],
        'augmentation': cfg.get('augmentation', {}),
    }

    log_start(exp_id, cfg)
    try:
        record = train_torch_cls(flat_cfg)
        log_done(exp_id, cfg, {
            'training_time_min': 0,
            'model_path': record['weights_path'],
            'best_epoch': record['best_epoch'],
            'train_samples': record['train_samples'],
            'val_samples': record['val_samples'],
        })
    except Exception as e:
        log_failed(exp_id, cfg, error=str(e))
        raise
