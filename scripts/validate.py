import json
import os
import sys
import pickle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch

from quill.nn.training import TrainCfg, Trainer
from quill.nn.batching import discard_empty, split_by_length, Collator
from quill.nn.utils.ranking import average_precision, rprecision


def _apply_cpu_safety(config: TrainCfg, device: str) -> None:
    if device != 'cpu':
        return
    if config.get('batch_size_h', 32) > 4:
        config['batch_size_h'] = 4
    if config.get('max_tokens', 16384) > 1024:
        config['max_tokens'] = 1024
    model_cfg = config['model_config']
    if model_cfg.get('depth', 1) > 2:
        model_cfg['depth'] = 2
    if model_cfg.get('dim', 256) > 64:
        model_cfg['dim'] = 64
    if model_cfg.get('num_heads', 8) > 4:
        model_cfg['num_heads'] = 4
    if model_cfg.get('head_dim', 16) > 8:
        model_cfg['head_dim'] = 8


def evaluate(
        config: TrainCfg,
        data_path: str,
        model_paths: list[str],
        device: str,
        dev: bool = True,
        short: bool = True,
        long: bool = False):

    #_apply_cpu_safety(config, device)
    model = Trainer(config['model_config']).to(device)
    with open(data_path, 'rb') as f:
        files = pickle.load(f)
        print(f'Read {len(files)} files with {sum(len(file.hole_asts) for file in files)} holes.')
    files = discard_empty(files)
    files = [f for f in files if f.file.name != 'foundation.morphisms-cospans']
    print(f'Of which {len(files)} have at least 1 hole.')

    files = [file for file in files if file.file.name in (config['dev_files'] if dev else config['train_files'])]
    print(f'Of which {len(files)} are {"dev" if dev else "train"} files.')
    match (short, long):
        case False, False:
            raise ValueError('Well, you must evaluate on something')
        case True, False:
            files, _ = split_by_length(files, config['max_tokens'])
        case False, True:
            _, files = split_by_length(files, config['max_tokens'])
        case True, True:
            pass
    print(f'Evaluating on {len(files)} files with {sum(len(file.hole_asts) for file in files)} holes.')

    AP, RP, R1 = [], [], []
    with torch.no_grad():
        collator = Collator(pad_value=-1, device=device, allow_self_loops=config['allow_self_loops'])
        for model_path in model_paths:
            try:
                model.load(model_path, strict=True, map_location=device)
            except Exception as exc:
                print(f'Skipping {model_path}: {exc}')
                continue
            model.eval()
            print(model_path)

            predictions, truths = model.infer_epoch(map(lambda x: collator([x]), files))
            aps = [average_precision(x, y) for x, y in zip(predictions, truths)]
            rps = [rprecision(x, y) for x, y in zip(predictions, truths)]
            r1s = [x[0] in y for x, y in zip(predictions, truths)]
            ap_stats = stats(aps)
            rp_stats = stats(rps)
            r1_stats = stats(r1s)
            AP.append(ap_stats[2]*100)
            RP.append(rp_stats[2]*100)
            R1.append(r1_stats[2]*100)
    print(stats(AP)[2])
    print(stats(RP)[2])
    print(stats(R1)[2])


def stats(xs: list[float]) -> tuple[float, float, float, float]:
    mu = sum(xs) / len(xs)
    var = sum((x - mu)**2 for x in xs) ** 0.5
    return min(xs), max(xs), mu, var/len(xs)


if __name__ == '__main__':
    train_cfg: TrainCfg = json.load(open(os.path.join(ROOT, 'data', 'config.json'), 'r'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_paths = [os.path.join(ROOT, 'data', 'model.pt')]
    evaluate(
        config=train_cfg,
        data_path=os.path.join(ROOT, 'data', 'tokenized.p'),
        model_paths=model_paths,
        long=True,
        short=False,
        device=device
    )
