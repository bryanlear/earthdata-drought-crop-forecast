import argparse
import json
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text())


def fold_labels(summary: list[dict]) -> list[str]:
    return [f"F{row['fold']}\n{row['val_range'][0][:4]}" for row in summary]


def metric_values(summary: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in summary]


def mean_metric(summary: list[dict], key: str) -> float:
    values = metric_values(summary, key)
    return float(np.mean(values)) if values else float('nan')


def plot_current_metrics(summary: list[dict], out_path: Path) -> None:
    labels = fold_labels(summary)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    axes[0].plot(x, metric_values(summary, 'macro_f1'), marker='o', linewidth=2, label='Macro F1')
    axes[0].plot(x, metric_values(summary, 'drought_f1'), marker='o', linewidth=2, label='Drought F1')
    axes[0].plot(x, metric_values(summary, 'accuracy'), marker='o', linewidth=2, label='Accuracy')
    axes[0].set_title('Current Time-Series CV: Overall Metrics by Fold')
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, metric_values(summary, 'drought_precision'), marker='o', linewidth=2, label='Drought Precision')
    axes[1].plot(x, metric_values(summary, 'drought_recall'), marker='o', linewidth=2, label='Drought Recall')
    axes[1].plot(x, metric_values(summary, 'drought_f1'), marker='o', linewidth=2, label='Drought F1')
    axes[1].set_title('Current Time-Series CV: Drought-Class Metrics by Fold')
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_compare_previous(current: list[dict], previous: list[dict], out_path: Path) -> None:
    labels = fold_labels(current)
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    compare_specs = [
        ('macro_f1', 'Macro F1 by Fold'),
        ('drought_f1', 'Drought F1 by Fold'),
        ('accuracy', 'Accuracy by Fold'),
    ]
    for ax, (metric, title) in zip(axes.flat[:3], compare_specs):
        ax.bar(x - width / 2, metric_values(previous, metric), width=width, label='Previous', alpha=0.8)
        ax.bar(x + width / 2, metric_values(current, metric), width=width, label='Current', alpha=0.8)
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()

    avg_metrics = ['macro_f1', 'drought_precision', 'drought_recall', 'drought_f1', 'accuracy']
    prev_means = [mean_metric(previous, metric) for metric in avg_metrics]
    curr_means = [mean_metric(current, metric) for metric in avg_metrics]
    ax = axes.flat[3]
    x2 = np.arange(len(avg_metrics))
    ax.bar(x2 - width / 2, prev_means, width=width, label='Previous', alpha=0.8)
    ax.bar(x2 + width / 2, curr_means, width=width, label='Current', alpha=0.8)
    ax.set_title('Average Metrics Across Folds')
    ax.set_xticks(x2, ['Macro F1', 'D Prec', 'D Rec', 'D F1', 'Acc'])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_histories(summary: list[dict], histories: list[list[dict]], out_path: Path) -> None:
    cols = 2
    rows = ceil(len(histories) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.8 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, fold_summary, history in zip(axes, summary, histories):
        epochs = [row['epoch'] for row in history]
        tr_f1 = [row['tr_f1'] for row in history]
        vl_f1 = [row['vl_f1'] for row in history]
        vl_f1_ema = [row['vl_f1_ema'] for row in history]
        ax.plot(epochs, tr_f1, label='Train F1', linewidth=1.8)
        ax.plot(epochs, vl_f1, label='Val F1', linewidth=1.8)
        ax.plot(epochs, vl_f1_ema, label='Val F1 EMA', linestyle='--', linewidth=1.8)
        best_epoch = fold_summary.get('best_epoch')
        if best_epoch is not None:
            ax.axvline(best_epoch, color='black', linestyle=':', linewidth=1)
        ax.set_title(
            f"Fold {fold_summary['fold']} ({fold_summary['val_range'][0][:4]})  "
            f"best EMA={fold_summary['best_val_f1_ema']:.3f}"
        )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1')
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(histories):]:
        ax.axis('off')

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create time-series CV plots')
    parser.add_argument('--checkpoints-dir', default='checkpoints',
                        help='Directory containing current and previous CV artifacts')
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    plots_dir = checkpoints_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    current_summary_path = checkpoints_dir / 'timeseries_cv_summary.json'
    previous_summary_path = checkpoints_dir / 'previous' / 'timeseries_cv_summary.json'

    current_summary = load_json(current_summary_path)

    histories = []
    for row in current_summary:
        history_path = checkpoints_dir / f"cv_fold{row['fold']}_history.json"
        histories.append(load_json(history_path))

    plot_current_metrics(current_summary, plots_dir / 'cv_metrics_current.png')
    plot_histories(current_summary, histories, plots_dir / 'cv_training_histories.png')

    if previous_summary_path.exists():
        previous_summary = load_json(previous_summary_path)
        plot_compare_previous(current_summary, previous_summary,
                              plots_dir / 'cv_compare_previous.png')

    print(f'Plots written to {plots_dir}')


if __name__ == '__main__':
    main()