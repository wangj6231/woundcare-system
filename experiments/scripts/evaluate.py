# ============================================================
# evaluate.py — Phase 4：完整分類評估模組
# 計算所有論文所需指標，並可儲存 Confusion Matrix / ROC
# ============================================================
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, roc_auc_score, roc_curve
)

BASE_DIR = Path(__file__).parent.parent.parent
CLASS_NAMES = ['Abrasions', 'Bruises', 'Burns', 'Cut',
               'Ingrown_nails', 'Laceration', 'Stab_wound']
FIGURES_DIR = Path(__file__).parent.parent / 'results' / 'figures'


def predict_yolo(weights_path: str, data_dir: str) -> tuple:
    """YOLO 分類模型預測，回傳 (y_true, y_pred, y_prob)"""
    from ultralytics import YOLO
    model = YOLO(weights_path)
    y_true, y_pred, y_prob = [], [], []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(data_dir) / cls_name
        if not cls_dir.exists():
            continue
        imgs = list(cls_dir.glob('*.[jp][pn]g'))
        for img_path in imgs:
            res = model(str(img_path), verbose=False)
            probs = res[0].probs.data.cpu().numpy()
            y_true.append(cls_idx)
            y_pred.append(int(probs.argmax()))
            y_prob.append(probs.tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def predict_torch(weights_path: str, data_dir: str) -> tuple:
    """PyTorch 模型預測，回傳 (y_true, y_pred, y_prob)"""
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader

    # weights_only=False 是必要的，因為 checkpoint dict 含非 tensor 物件（model_name, classes 等）
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)  # noqa: FutureWarning
    model_name = checkpoint['model_name']
    NUM_CLASSES = 7

    if model_name == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    elif model_name == 'vit_s_16':
        model = models.vit_b_16()
        model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    else:
        raise ValueError(f'Unknown model in checkpoint: {model_name}')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(str(data_dir), transform=tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs.tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def compute_all_metrics(y_true, y_pred, y_prob=None) -> dict:
    """計算所有分類指標，回傳 dict"""
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        'accuracy':          round(accuracy_score(y_true, y_pred) * 100, 2),
        'macro_precision':   round(report['macro avg']['precision'] * 100, 2),
        'macro_recall':      round(report['macro avg']['recall'] * 100, 2),
        'macro_f1':          round(report['macro avg']['f1-score'] * 100, 2),
        'weighted_f1':       round(report['weighted avg']['f1-score'] * 100, 2),
        'n_samples':         int(len(y_true)),
    }

    # Per-class 指標
    for cls_name in CLASS_NAMES:
        safe_key = cls_name.lower().replace(' ', '_').replace('/', '_')
        cls_data = report.get(cls_name, {})
        metrics[f'{safe_key}_precision'] = round(cls_data.get('precision', 0) * 100, 2)
        metrics[f'{safe_key}_recall']    = round(cls_data.get('recall', 0) * 100, 2)
        metrics[f'{safe_key}_f1']        = round(cls_data.get('f1-score', 0) * 100, 2)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # ROC-AUC（One-vs-Rest，需要 probability）
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            metrics['roc_auc_macro'] = round(roc_auc, 4)
        except Exception:
            metrics['roc_auc_macro'] = ''

    return metrics


def save_confusion_matrix(y_true, y_pred, exp_id: str, split: str = 'val'):
    """儲存歸一化混淆矩陣圖"""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel='True Label',
        xlabel='Predicted Label',
        title=f'Normalized Confusion Matrix — {exp_id} ({split})'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f'{cm[i,j]:.2f}',
                    ha='center', va='center',
                    color='white' if cm[i, j] > 0.5 else 'black', fontsize=8)
    fig.tight_layout()

    out_dir = FIGURES_DIR / 'confusion_matrix'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{exp_id}_{split}_confusion_matrix.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   💾 Confusion matrix saved: {out_path}")
    return str(out_path)


def evaluate(weights_path: str, data_dir: str,
             framework: str = 'ultralytics',
             exp_id: str = 'unknown',
             split: str = 'val',
             save_figures: bool = True) -> dict:
    """
    完整評估入口：
      weights_path: best.pt 路徑
      data_dir: 含子資料夾（各類別）的目錄，例如 dataset/val/
      framework: 'ultralytics' 或 'torch'
      exp_id: 實驗 ID（用於圖檔命名）
      split: 'val' 或 'test'（不影響計算，用於命名）
    """
    print(f"\n🔍 Evaluating [{exp_id}] on '{split}' split...")
    print(f"   Weights:  {weights_path}")
    print(f"   Data dir: {data_dir}")

    if not os.path.exists(weights_path):
        print(f"⚠️  Weights not found: {weights_path}")
        return {}

    if framework == 'ultralytics':
        y_true, y_pred, y_prob = predict_yolo(weights_path, data_dir)
    else:
        y_true, y_pred, y_prob = predict_torch(weights_path, data_dir)

    if len(y_true) == 0:
        print(f"⚠️  No images found in {data_dir}")
        return {}

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    metrics['split'] = split

    print(f"\n   Accuracy:  {metrics['accuracy']}%")
    print(f"   Macro-F1:  {metrics['macro_f1']}%")
    print(f"   Stab Recall: {metrics.get('stab_wound_recall', 'N/A')}%")

    if save_figures:
        save_confusion_matrix(y_true, y_pred, exp_id=exp_id, split=split)

    # 儲存原始預測到 predictions/ 目錄
    preds_dir = Path(__file__).parent.parent / 'results' / 'predictions' / split
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f'{exp_id}_predictions.json'
    with open(preds_path, 'w') as f:
        json.dump({
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist() if y_prob is not None else [],
            'class_names': CLASS_NAMES,
            'metrics': {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
        }, f, indent=2)
    print(f"   💾 Predictions saved: {preds_path}")

    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    required=True)
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--framework',  default='ultralytics', choices=['ultralytics', 'torch'])
    parser.add_argument('--exp_id',     default='manual_eval')
    parser.add_argument('--split',      default='val', choices=['val', 'test'])
    args = parser.parse_args()

    metrics = evaluate(
        weights_path=args.weights,
        data_dir=args.data_dir,
        framework=args.framework,
        exp_id=args.exp_id,
        split=args.split
    )
    print("\nFull metrics:")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"  {k}: {v}")
