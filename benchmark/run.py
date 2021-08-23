import argparse

import yaml

from models import MODELS
from datasets import DATASETS

parser = argparse.ArgumentParser("Benchmarks for OpenCV Zoo.")
parser.add_argument('--cfg', '-c', type=str,
                    help='Benchmarking on the given config.')
args = parser.parse_args()

def build_from_cfg(cfg, registery):
    obj_name = cfg.pop('name')
    obj = registery.get(obj_name)
    return obj(**cfg)

if __name__ == '__main__':
    assert args.cfg.endswith('yaml')
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    model = build_from_cfg(cfg=cfg['Model'], registery=MODELS)
    dataset = build_from_cfg(cfg=cfg['Dataset'], registery=DATASETS)

    dataset.benchmark(model)