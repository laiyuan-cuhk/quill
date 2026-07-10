import sys
import os
import json
import argparse
import pickle

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


def train(
        config: TrainCfg,
        data_path: str,
        store_path: str,
        log_path: str,
        checkpoint_path: str,
        device: str):
    logger = Logger(sys.stdout, log_path)
    sys.stdout = logger
    print(config['model_config'])

    with open(data_path, 'rb') as f:
        files = pickle.load(f)
        print(f'Read {len(files)} files with {sum(len(file.hole_asts) for file in files)} holes.')
    files = discard_empty(files)
    print(f'Of which {len(files)} have at least 1 hole.')
    train_files = [file for file in files if file.file.name in config['train_files']]
    dev_files = [file for file in files if file.file.name in config['dev_files']]
    train_files, _ = split_by_length(train_files, config['max_tokens'])
    dev_files, _ = split_by_length(dev_files, config['max_tokens'])

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
        train_epoch = model.train_epoch(
            epoch=map(collator, train_sampler.iter(
                batch_size_s=config['batch_size_s'],
                batch_size_h=config['batch_size_h'])),
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
    )
