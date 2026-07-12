import sys
import os
import json
import argparse
import pickle
import shutil
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _clear_python_cache(root: str) -> None:
    root_path = Path(root).resolve()
    if not root_path.exists():
        return

    cache_dirs = [p for p in root_path.rglob('__pycache__') if p.is_dir()]
    pyc_files = [p for p in root_path.rglob('*.py[cod]') if p.is_file()]

    for cache_dir in cache_dirs:
        shutil.rmtree(cache_dir)
    for pyc_file in pyc_files:
        pyc_file.unlink(missing_ok=True)

    if cache_dirs or pyc_files:
        print(f'Cleared Python bytecode cache under {root_path}')
    else:
        print(f'No Python bytecode cache found under {root_path}')


_clear_python_cache(ROOT)

# Compatibility shim: some SciPy versions changed `scipy.linalg.logm` signature
# Older code (in quill) expects `logm(..., disp=False)` to return `(log, info)`.
# Newer SciPy returns a single matrix and does not accept `disp` kwarg.
try:
    import scipy.linalg as _scla
    _orig_logm = _scla.logm
    def _logm_shim(A, *args, **kwargs):
        if 'disp' in kwargs:
            kwargs.pop('disp')
            res = _orig_logm(A, *args, **kwargs)
            return res, None
        return _orig_logm(A, *args, **kwargs)
    _scla.logm = _logm_shim
except Exception:
    # If SciPy isn't available or something goes wrong, continue and let import errors surface later
    pass

from quill.nn.training import TrainCfg, Trainer, Logger
from quill.nn.batching import discard_empty, split_by_length, Sampler, Collator
from quill.nn.utils.schedules import make_schedule

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def _apply_cpu_safety(config: TrainCfg, device: str) -> None:
    if device != 'cpu':
        return

    if config.get('batch_size_h', 32) > 4:
        print(f"Reducing batch_size_h from {config['batch_size_h']} to 4 for CPU memory safety.")
        config['batch_size_h'] = 4
    if config.get('max_tokens', 16384) > 1024:
        print(f"Reducing max_tokens from {config['max_tokens']} to 1024 for CPU memory safety.")
        config['max_tokens'] = 1024

    model_cfg = config['model_config']
    if model_cfg.get('depth', 1) > 2:
        print(f"Reducing model depth from {model_cfg['depth']} to 2 for CPU memory safety.")
        model_cfg['depth'] = 2
    if model_cfg.get('dim', 256) > 64:
        print(f"Reducing model dim from {model_cfg['dim']} to 64 for CPU memory safety.")
        model_cfg['dim'] = 64
    if model_cfg.get('num_heads', 8) > 4:
        print(f"Reducing model num_heads from {model_cfg['num_heads']} to 4 for CPU memory safety.")
        model_cfg['num_heads'] = 4
    if model_cfg.get('head_dim', 16) > 8:
        print(f"Reducing model head_dim from {model_cfg['head_dim']} to 8 for CPU memory safety.")
        model_cfg['head_dim'] = 8


def train(
        config: TrainCfg,
        data_path: str,
        store_path: str,
        log_path: str,
        checkpoint_path: str,
        device: str,
        max_batches: int | None = None):
    logger = Logger(sys.stdout, log_path)
    sys.stdout = logger
    print(config['model_config'])

    #_apply_cpu_safety(config, device)
    if max_batches is None and device == 'cpu':
        max_batches = 6

    with open(data_path, 'rb') as f:
        files = pickle.load(f)
        print(f'Read {len(files)} files with {sum(len(file.hole_asts) for file in files)} holes.')
    files = discard_empty(files)
    print(f'Of which {len(files)} have at least 1 hole.')
    train_files = [file for file in files if file.file.name in config['train_files']]
    dev_files = [file for file in files if file.file.name in config['dev_files']]
    train_files, _ = split_by_length(train_files, config['max_tokens'])
    dev_files, _ = split_by_length(dev_files, config['max_tokens'])
    #if device == 'cpu' and len(train_files) > 16:
    #    print(f"Using a subset of {16} training files for CPU runtime safety.")
    #    train_files = train_files[:16]
    #if device == 'cpu' and len(dev_files) > 8:
    #    print(f"Using a subset of {8} dev files for CPU runtime safety.")
    #    dev_files = dev_files[:8]

    if len(train_files) == 0:
        raise RuntimeError('No training files found after filtering by config["train_files"]')

    train_sampler = Sampler(train_files)
    epoch_size = train_sampler.itersize(config['batch_size_s'] * config['backprop_every'], config['batch_size_h'])
    collator = Collator(pad_value=-1, device=device, allow_self_loops=config['allow_self_loops'])

    model = Trainer(config['model_config']).to(device)
    optimizer = AdamW(params=model.parameters(), lr=1, weight_decay=1e-02)
    schedule = make_schedule(
        warmup_steps=config['warmup_epochs'] * epoch_size,
        warmdown_steps=config['warmdown_epochs'] * epoch_size,
        max_lr=config['max_lr'],
        min_lr=config['min_lr'],
        total_steps=config['num_epochs'] * epoch_size
    )
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=schedule, last_epoch=-1)

    start_epoch, best_ap = 0, -1e08
    if os.path.exists(checkpoint_path):
        start_epoch, best_ap = model.load_checkpoint(checkpoint_path, optimizer, scheduler, device)
        print(f'Resuming from checkpoint at epoch {start_epoch}.')
    for epoch in range(start_epoch, config['num_epochs']):
        print(f'Epoch {epoch}')
        print('-' * 64)
        batches = []
        for batch_idx, batch in enumerate(train_sampler.iter(
                batch_size_s=config['batch_size_s'],
                batch_size_h=config['batch_size_h'])):
            batches.append(collator(batch))
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
        train_epoch = model.train_epoch(
            epoch=iter(batches),
            optimizer=optimizer,
            scheduler=scheduler,
            backprop_every=config['backprop_every'])
        if len(train_epoch.loss) > 0:
            print(f'Train loss: {sum(train_epoch.loss)/len(train_epoch.loss)}')
        else:
            print('Train loss: N/A')
        if len(train_epoch.ap) > 0:
            print(f'Train mAP: {sum(train_epoch.ap)/len(train_epoch.ap)}')
        else:
            print('Train mAP: N/A')
        if len(train_epoch.rp) > 0:
            print(f'Train R-Precision: {sum(train_epoch.rp) / len(train_epoch.rp)}')
        else:
            print('Train R-Precision: N/A')

        dev_epoch = None
        if len(dev_files) > 0:
            dev_epoch = model.eval_epoch(map(lambda x: collator([x]), dev_files))
            if len(dev_epoch.loss) > 0:
                print(f'Dev loss: {sum(dev_epoch.loss)/len(dev_epoch.loss)}')
            else:
                print('Dev loss: N/A')
            if len(dev_epoch.ap) > 0:
                print(f'Dev mAP: {sum(dev_epoch.ap) / len(dev_epoch.ap)}')
            else:
                print('Dev mAP: N/A')
            if len(dev_epoch.rp) > 0:
                print(f'Dev R-Precision: {sum(dev_epoch.rp) / len(dev_epoch.rp)}')
            else:
                print('Dev R-Precision: N/A')
        else:
            print('Skipping dev evaluation (no dev files).')

        if dev_epoch is not None and len(dev_epoch.ap) > 0 and sum(dev_epoch.ap) > best_ap:
            print('Saving...')
            model.save(store_path)
            best_ap = sum(dev_epoch.ap)
        model.save_checkpoint(checkpoint_path, optimizer, scheduler, epoch, best_ap)
        print('=' * 64 + '\n')
    logger.flush()


def parse_args():
    parser = argparse.ArgumentParser(description='Run a single training iteration')
    parser.add_argument('--data_path', type=str, help='Path to data file',
                        default='../data/tokenized.p')
    parser.add_argument('--config_path', type=str, help='Path to config file',
                        default='../data/config.json')
    parser.add_argument('--store_path', type=str, help='Where to store the trained model',
                        default='../data/model.pt')
    parser.add_argument('--log_path', type=str, help='Where to log results',
                        default='../data/log.txt')
    parser.add_argument('--checkpoint_path', type=str, help='Where to store/resume the training checkpoint',
                        default='../data/checkpoint.pt')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                        help='Device to run on (cpu or cuda)', default='cpu')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit the number of training batches per epoch; useful for quick smoke tests')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_cfg: TrainCfg = json.load(open(args.config_path, 'r'))
    train(
        config=train_cfg,
        data_path=args.data_path,
        store_path=args.store_path,
        log_path=args.log_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        max_batches=args.max_batches,
    )
