"""
Training script for DroughtCNN.

Usage
-----
    python train.py
    python train.py --label_col drought_class_spi6
    python train.py --epochs 150 --lr 5e-4
    python train.py --timeseries_cv

Checkpoints saved to CNN/model/checkpoints/best.pt
Training history saved to CNN/model/checkpoints/history.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

SCRIPT_DIR = Path(__file__).resolve().parent
CNN_DIR = SCRIPT_DIR.parent
NPZ_DIR = CNN_DIR / 'SMAP_regions'
CSV_DIR = CNN_DIR / 'CHIRPS_processing'
CKPT_DIR = SCRIPT_DIR / 'checkpoints'
TEST_RANGE = ('2024-01-01', '2026-03-01')
TS_CV_FOLDS = [
    (('2015-04-01', '2018-12-01'), ('2019-01-01', '2019-12-01')),
    (('2015-04-01', '2019-12-01'), ('2020-01-01', '2020-12-01')),
    (('2015-04-01', '2020-12-01'), ('2021-01-01', '2021-12-01')),
    (('2015-04-01', '2021-12-01'), ('2022-01-01', '2022-12-01')),
    (('2015-04-01', '2022-12-01'), ('2023-01-01', '2023-12-01')),
]

sys.path.insert(0, str(SCRIPT_DIR))
from dataset import make_dataloaders, make_dataloaders_for_ranges
from model import DroughtCNN


def make_class_weights(labels: list[int],
                       n_classes: int,
                       power: float,
                       device: torch.device) -> torch.Tensor | None:
    if power <= 0:
        return None

    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long),
                            minlength=n_classes).float()
    majority = counts.max().clamp(min=1.0)
    weights = (majority / counts.clamp(min=1.0)) ** power
    weights = weights / weights.mean()
    return weights.to(device)


def run_epoch(model: nn.Module,
              loader,
              criterion: nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              train: bool = True):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.set_grad_enabled(train):
        for image, labels, months in loader:
            image = image.to(device)
            labels = labels.to(device)
            months = months.to(device)

            logits = model(image, months)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    macro_f1 = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0)
    return avg_loss, macro_f1


def evaluate_loader(model: nn.Module, loader, device: torch.device):
    preds_all, labels_all = [], []
    with torch.no_grad():
        for image, labels, months in loader:
            preds = model(image.to(device), months.to(device)).argmax(dim=-1).cpu().numpy()
            preds_all.extend(preds.tolist())
            labels_all.extend(labels.numpy().tolist())
    return np.array(labels_all), np.array(preds_all)


def build_model(args: argparse.Namespace, device: torch.device) -> DroughtCNN:
    model = DroughtCNN(
        in_channels=10,
        n_classes=2,
        dropout=args.dropout,
        width=args.width,
        use_month=not args.no_month,
    ).to(device)
    return model


def train_once(args: argparse.Namespace,
               device: torch.device,
               train_loader,
               val_loader,
               test_loader=None,
               checkpoint_path: Path | None = None,
               history_path: Path | None = None,
               run_name: str | None = None):
    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params:,}')
    print(f'Month encoding: {"off" if args.no_month else "on"}')

    class_weights = make_class_weights(
        train_loader.dataset._labels,
        n_classes=2,
        power=args.loss_weight_power,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=args.label_smoothing)
    if class_weights is None:
        print(f'Loss: CrossEntropyLoss(label_smoothing={args.label_smoothing})')
    else:
        print('Loss: CrossEntropyLoss('
              f'label_smoothing={args.label_smoothing}, '
              f'class_weights={class_weights.cpu().numpy().round(3)})')

    optimizer = Adam(model.parameters(),
                     lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=args.lr_patience)

    best_val_f1_ema = -1.0
    val_f1_ema = None
    ema_alpha = 0.3
    patience_ctr = 0
    history: list[dict] = []
    best_payload = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_f1 = run_epoch(model, train_loader, criterion,
                                   optimizer, device, train=True)
        vl_loss, vl_f1 = run_epoch(model, val_loader, criterion,
                                   optimizer, device, train=False)
        scheduler.step(vl_loss)

        if val_f1_ema is None:
            val_f1_ema = vl_f1
        else:
            val_f1_ema = ema_alpha * vl_f1 + (1 - ema_alpha) * val_f1_ema

        row = dict(epoch=epoch,
                   tr_loss=round(tr_loss, 5), tr_f1=round(tr_f1, 4),
                   vl_loss=round(vl_loss, 5), vl_f1=round(vl_f1, 4),
                   vl_f1_ema=round(val_f1_ema, 4))
        history.append(row)
        print(f'Epoch {epoch:3d}  '
              f'tr_loss={tr_loss:.4f} tr_f1={tr_f1:.3f}  '
              f'vl_loss={vl_loss:.4f} vl_f1={vl_f1:.3f} (ema={val_f1_ema:.3f})')

        if val_f1_ema > best_val_f1_ema:
            best_val_f1_ema = val_f1_ema
            patience_ctr = 0
            best_payload = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': vl_loss,
                'val_f1': vl_f1,
                'val_f1_ema': val_f1_ema,
                'channel_mean': torch.from_numpy(train_loader.dataset.channel_mean),
                'channel_std': torch.from_numpy(train_loader.dataset.channel_std),
                'args': vars(args),
            }
            if checkpoint_path is not None:
                torch.save(best_payload, checkpoint_path)
            print(f'  ✓ best model saved (ema={val_f1_ema:.3f}, val_f1={vl_f1:.3f})')
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f'Early stopping at epoch {epoch} '
                      f'(no improvement for {args.patience} epochs)')
                break

    if history_path is not None:
        with open(history_path, 'w') as fh:
            json.dump(history, fh, indent=2)
        print(f'History saved → {history_path}')

    result = {
        'history': history,
        'best_val_f1_ema': best_val_f1_ema,
        'best_epoch': best_payload['epoch'] if best_payload else None,
    }

    if best_payload is not None:
        model.load_state_dict(best_payload['model_state_dict'])

    if test_loader is not None:
        print(f'\n=== Evaluation ({run_name or "best checkpoint"}) ===')
        labels_all, preds_all = evaluate_loader(model, test_loader, device)
        report = classification_report(
            labels_all,
            preds_all,
            target_names=['normal', 'drought'],
            zero_division=0,
            output_dict=True,
        )
        print(classification_report(
            labels_all,
            preds_all,
            target_names=['normal', 'drought'],
            zero_division=0,
        ))
        result['report'] = report

    return result


def train_default_split(args: argparse.Namespace, device: torch.device):
    train_loader, val_loader, test_loader, _, _ = make_dataloaders(
        NPZ_DIR, CSV_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_col=args.label_col,
        sampler_power=args.sampler_power,
    )
    print(f'Samples — train: {len(train_loader.dataset)}  '
          f'val: {len(val_loader.dataset)}  '
          f'test: {len(test_loader.dataset)}')
    return train_once(
        args=args,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        checkpoint_path=CKPT_DIR / 'best.pt',
        history_path=CKPT_DIR / 'history.json',
        run_name='best checkpoint',
    )


def train_timeseries_cv(args: argparse.Namespace, device: torch.device):
    print('\n=== Rolling time-series CV ===')
    print(f'Final holdout test range remains fixed at {TEST_RANGE[0]} → {TEST_RANGE[1]}')
    fold_results = []

    for fold_idx, (train_range, val_range) in enumerate(TS_CV_FOLDS, start=1):
        print('\n' + '=' * 80)
        print(f'Fold {fold_idx}: train {train_range[0]} → {train_range[1]}  '
              f'val {val_range[0]} → {val_range[1]}')

        train_loader, val_loader, _, _, _ = make_dataloaders_for_ranges(
            NPZ_DIR, CSV_DIR,
            train_range=train_range,
            val_range=val_range,
            test_range=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_col=args.label_col,
            sampler_power=args.sampler_power,
        )
        print(f'Samples — train: {len(train_loader.dataset)}  '
              f'val: {len(val_loader.dataset)}')

        fold_ckpt = CKPT_DIR / f'cv_fold{fold_idx}_best.pt'
        fold_hist = CKPT_DIR / f'cv_fold{fold_idx}_history.json'
        result = train_once(
            args=args,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=val_loader,
            checkpoint_path=fold_ckpt,
            history_path=fold_hist,
            run_name=f'fold {fold_idx} validation set',
        )
        report = result['report']
        fold_results.append({
            'fold': fold_idx,
            'train_range': train_range,
            'val_range': val_range,
            'best_epoch': result['best_epoch'],
            'best_val_f1_ema': result['best_val_f1_ema'],
            'macro_f1': report['macro avg']['f1-score'],
            'accuracy': report['accuracy'],
            'drought_precision': report['drought']['precision'],
            'drought_recall': report['drought']['recall'],
            'drought_f1': report['drought']['f1-score'],
        })

    summary_path = CKPT_DIR / 'timeseries_cv_summary.json'
    with open(summary_path, 'w') as fh:
        json.dump(fold_results, fh, indent=2)

    macro_f1 = np.mean([f['macro_f1'] for f in fold_results])
    drought_f1 = np.mean([f['drought_f1'] for f in fold_results])
    drought_precision = np.mean([f['drought_precision'] for f in fold_results])
    drought_recall = np.mean([f['drought_recall'] for f in fold_results])
    accuracy = np.mean([f['accuracy'] for f in fold_results])

    print('\n=== CV Summary ===')
    for row in fold_results:
        print(f"Fold {row['fold']}: macro_f1={row['macro_f1']:.3f}  "
              f"drought_f1={row['drought_f1']:.3f}  accuracy={row['accuracy']:.3f}")
    print(f'Average macro F1: {macro_f1:.3f}')
    print(f'Average drought precision: {drought_precision:.3f}')
    print(f'Average drought recall: {drought_recall:.3f}')
    print(f'Average drought F1: {drought_f1:.3f}')
    print(f'Average accuracy: {accuracy:.3f}')
    print(f'Summary saved → {summary_path}')

    return fold_results


def train(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if args.timeseries_cv:
        return train_timeseries_cv(args, device)
    return train_default_split(args, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DroughtCNN')
    parser.add_argument('--label_col', default='drought_class_spi3',
                        choices=['drought_class_spi3', 'drought_class_spi6'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--lr_patience', type=int, default=8,
                        help='LR scheduler patience (epochs)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--width', type=float, default=0.25,
                        help='Backbone channel width multiplier (0.25 ≈ 700K params, 1.0 = 11M)')
    parser.add_argument('--sampler_power', type=float, default=1.0,
                        help='Exponent for drought oversampling (1.0 = full inverse-frequency, 0.0 = off)')
    parser.add_argument('--loss_weight_power', type=float, default=0.0,
                        help='Exponent for class-weighted CE on raw train counts (0.0 = off)')
    parser.add_argument('--no_month', action='store_true',
                        help='Disable sinusoidal month-of-year encoding')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for CrossEntropyLoss (0 = off, 0.1 recommended)')
    parser.add_argument('--timeseries_cv', action='store_true',
                        help='Run rolling yearly time-series cross-validation on 2015-2023 and keep 2024-2026 untouched')
    train(parser.parse_args())