"""
Training script for DroughtCNN.

Usage
-----
    python train.py
    python train.py --label_col drought_class_spi6
    python train.py --epochs 150 --lr 5e-4

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
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
CNN_DIR    = SCRIPT_DIR.parent
NPZ_DIR    = CNN_DIR / 'SMAP_regions'
CSV_DIR    = CNN_DIR / 'CHIRPS_processing'
CKPT_DIR   = SCRIPT_DIR / 'checkpoints'

sys.path.insert(0, str(SCRIPT_DIR))
from model   import DroughtCNN
from dataset import make_dataloaders


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
            image  = image.to(device)
            labels = labels.to(device)
            months = months.to(device)

            logits = model(image, months)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    macro_f1 = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0)
    return avg_loss, macro_f1


def train(args: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)


    train_loader, val_loader, test_loader, ch_mean, ch_std = make_dataloaders(
        NPZ_DIR, CSV_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_col=args.label_col,
    )
    print(f'Samples — train: {len(train_loader.dataset)}  '
          f'val: {len(val_loader.dataset)}  '
          f'test: {len(test_loader.dataset)}')


    model = DroughtCNN(in_channels=10, n_classes=2,
                       dropout=args.dropout, width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params:,}')


    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print(f'Loss: CrossEntropyLoss(label_smoothing={args.label_smoothing})')

    optimizer = Adam(model.parameters(),
                     lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=args.lr_patience)

    best_val_f1_ema = -1.0
    val_f1_ema      = None
    ema_alpha       = 0.3
    patience_ctr    = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_f1 = run_epoch(model, train_loader, criterion,
                                   optimizer, device, train=True)
        vl_loss, vl_f1 = run_epoch(model, val_loader,   criterion,
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
            patience_ctr    = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':         vl_loss,
                'val_f1':           vl_f1,
                'val_f1_ema':       val_f1_ema,
                'channel_mean':     torch.from_numpy(ch_mean),
                'channel_std':      torch.from_numpy(ch_std),
                'args':             vars(args),
            }, CKPT_DIR / 'best.pt')
            print(f'  ✓ best model saved (ema={val_f1_ema:.3f}, val_f1={vl_f1:.3f})')
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f'Early stopping at epoch {epoch} '
                      f'(no improvement for {args.patience} epochs)')
                break


    with open(CKPT_DIR / 'history.json', 'w') as fh:
        json.dump(history, fh, indent=2)
    print(f'History saved → {CKPT_DIR / "history.json"}')

    print('\n=== Test-set evaluation (best checkpoint) ===')
    ckpt = torch.load(CKPT_DIR / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_preds, test_labels_all = [], []
    with torch.no_grad():
        for image, labels, months in test_loader:
            preds = model(image.to(device), months.to(device)).argmax(dim=-1).cpu().numpy()
            test_preds.extend(preds.tolist())
            test_labels_all.extend(labels.numpy().tolist())

    print(classification_report(
        test_labels_all, test_preds,
        target_names=['normal', 'drought'],
        zero_division=0,
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DroughtCNN')
    parser.add_argument('--label_col',    default='drought_class_spi3',
                        choices=['drought_class_spi3', 'drought_class_spi6'])
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--patience',     type=int,   default=20,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--lr_patience',  type=int,   default=8,
                        help='LR scheduler patience (epochs)')
    parser.add_argument('--num_workers',  type=int,   default=4,
                        help='DataLoader worker processes')
    parser.add_argument('--seed',         type=int,   default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--width',        type=float, default=0.25,
                        help='Backbone channel width multiplier (0.25 ≈ 700K params, 1.0 = 11M)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for CrossEntropyLoss (0 = off, 0.1 recommended)')
    train(parser.parse_args())
