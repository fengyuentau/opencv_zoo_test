import os
import argparse

import yaml
import tqdm
import numpy as np
import cv2 as cv

from models import MODELS
from download import Downloader

parser = argparse.ArgumentParser("Benchmarks for OpenCV Zoo.")
parser.add_argument('--cfg', '-c', type=str,
                    help='Benchmarking on the given config.')
args = parser.parse_args()

class Benchmark:
    def __init__(self, **kwargs):
        self._fileList = kwargs.pop('fileList')
        self._sizes = kwargs.pop('sizes', None)
        self._repeat = kwargs.pop('repeat', 100)
        self._parentPath = kwargs.pop('parentPath', '')

        self._tm = cv.TickMeter()

    def run(self, model):
        avg_infer_time = dict.fromkeys(self._fileList, dict())
        for img_name in self._fileList:
            img = cv.imread(os.path.join(self._parentPath, img_name))

            for size in self._sizes:
                infer_times = []
                pbar = tqdm.tqdm(range(self._repeat))
                for _ in pbar:
                    pbar.set_description('Benchmarking on {} of size {}'.format(img_name, str(size)))

                    self._tm.start()
                    result = model.infer(img, size)
                    self._tm.stop()

                    infer_times.append(self._tm.getTimeMilli())
                    self._tm.reset()

                avg_infer_time[img_name][str(size)] = sum(infer_times) / self._repeat

        print(avg_infer_time)

def build_from_cfg(cfg, registery):
    obj_name = cfg.pop('name')
    obj = registery.get(obj_name)
    return obj(**cfg)

def prepend_pythonpath(cfg, key1, key2):
    pythonpath = os.environ['PYTHONPATH']
    if cfg[key1][key2].startswith('/'):
        return
    cfg[key1][key2] = os.path.join(pythonpath, cfg[key1][key2])

if __name__ == '__main__':
    assert args.cfg.endswith('yaml')
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # prepend PYTHONPATH to each path
    prepend_pythonpath(cfg, key1='Data', key2='parentPath')
    prepend_pythonpath(cfg, key1='Benchmark', key2='parentPath')
    prepend_pythonpath(cfg, key1='Model', key2='modelPath')


    # Download requested data if does not exist
    downloader = Downloader(**cfg['Data'])
    downloader.get()

    # Instantiate benchmarking
    benchmark = Benchmark(**cfg['Benchmark'])

    # Instantiate model
    model = build_from_cfg(cfg=cfg['Model'], registery=MODELS)

    # Run benchmarking
    benchmark.run(model)