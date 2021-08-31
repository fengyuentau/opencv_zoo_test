# OpenCV Zoo Benchmark

Benchmarking different models in the zoo.

## Preparation

1. `python >= 3.6` is installed.
2. Run `pip install -r requirements.txt` to install dependencies.

## Benchmarking

Run the following command to benchmark on a given config:

```shell

mkdir data
PYTHONPATH=.. python benchmark.py --cfg ./config/YuNet.yaml
```