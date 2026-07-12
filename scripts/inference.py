import os
import sys
from pathlib import Path

from tqdm import tqdm

import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quill.data.agda.reader import parse_file
from quill.nn.inference import Inferer


train_cfg = json.load(open('../data/config.json', 'r'))
train_files, dev_files = train_cfg['train_files'], train_cfg['dev_files']

inferer = Inferer(train_cfg['model_config'], 'cuda')
inferer.load('../data/model.pt', strict=True, map_location='cpu')
inferer.eval()

file: str = dev_files[13]
selected = inferer.select_premises(
    file=parse_file(f'../data/stdlib/{file}.json', validate=True))
print(selected[:10])

cache_files = []
for f in tqdm(train_files[:50]):
    try:
        parsed = parse_file(f'../data/stdlib/{f}.json', validate=True)
        if len(parsed.scope):
            cache_files.append(parsed)
    except AssertionError:
        pass



inferer.precompute(cache_files)
selected = inferer.select_premises(
    file=parse_file(f'../data/stdlib/{file}.json', validate=True),
    use_cache=True)
print(selected[:10])
