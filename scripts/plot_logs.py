#!/usr/bin/env python3
"""Plot training/dev metrics across epochs from data/log.txt using matplotlib.

Usage:
    python3 plot_logs.py [path/to/log.txt] [output.png]

If run from `quill/scripts`, default log path is `../data/log.txt` and default
output is `../data/training_plot.png`.
"""
import sys
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(path: Path):
    metrics = {
        'epoch': [],
        'train_loss': [],
        'dev_loss': [],
        'train_map': [],
        'dev_map': [],
        'train_rprec': [],
        'dev_rprec': []
    }

    epoch = None
    float_re = re.compile(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Epoch'):
                # Line like: "Epoch 12" or "Epoch 17"
                parts = line.split()
                try:
                    epoch = int(parts[1])
                except Exception:
                    # fallback: increment
                    epoch = (metrics['epoch'][-1] + 1) if metrics['epoch'] else 0
                metrics['epoch'].append(epoch)
                # ensure placeholders (in case some metrics are missing)
                for k in ['train_loss', 'dev_loss', 'train_map', 'dev_map', 'train_rprec', 'dev_rprec']:
                    # append None for now; we'll replace when we see values
                    metrics[k].append(None)
                continue

            if line.startswith('Train loss:'):
                v = float_re.search(line).group(1)
                metrics['train_loss'][-1] = float(v)
            elif line.startswith('Dev loss:'):
                v = float_re.search(line).group(1)
                metrics['dev_loss'][-1] = float(v)
            elif line.startswith('Train mAP:'):
                v = float_re.search(line).group(1)
                metrics['train_map'][-1] = float(v)
            elif line.startswith('Dev mAP:'):
                v = float_re.search(line).group(1)
                metrics['dev_map'][-1] = float(v)
            elif line.startswith('Train R-Precision:'):
                v = float_re.search(line).group(1)
                metrics['train_rprec'][-1] = float(v)
            elif line.startswith('Dev R-Precision:'):
                v = float_re.search(line).group(1)
                metrics['dev_rprec'][-1] = float(v)

    # Convert None to NaN for plotting
    import math
    for k in ['train_loss', 'dev_loss', 'train_map', 'dev_map', 'train_rprec', 'dev_rprec']:
        metrics[k] = [float('nan') if x is None else x for x in metrics[k]]

    return metrics


def plot(metrics, out: Path):
    epochs = metrics['epoch']
    try:
        plt.style.use('seaborn-darkgrid')
    except Exception:
        plt.style.use('default')
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Losses
    axes[0].plot(epochs, metrics['train_loss'], marker='o', label='Train loss')
    axes[0].plot(epochs, metrics['dev_loss'], marker='o', label='Dev loss')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # mAP
    axes[1].plot(epochs, metrics['train_map'], marker='o', label='Train mAP')
    axes[1].plot(epochs, metrics['dev_map'], marker='o', label='Dev mAP')
    axes[1].set_ylabel('mAP')
    axes[1].legend()

    # R-Precision
    axes[2].plot(epochs, metrics['train_rprec'], marker='o', label='Train R-Precision')
    axes[2].plot(epochs, metrics['dev_rprec'], marker='o', label='Dev R-Precision')
    axes[2].set_ylabel('R-Precision')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f'Saved plot to {out}')


def main(argv):
    log_path = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / 'data' / 'log.txt'
    out_path = Path(argv[2]) if len(argv) > 2 else Path(__file__).resolve().parents[1] / 'data' / 'image' / 'training_plot.png'

    if not log_path.exists():
        print('Log file not found:', log_path)
        return 2

    metrics = parse_log(log_path)
    plot(metrics, out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
