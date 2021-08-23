# OpenCV Zoo Benchmark

Benchmarking different models in the zoo.

## Preparation

1. `python >= 3.6` is installed.
2. Run `python download.py` to download required datasets.
3. Run `pip install -r requirements.txt` to install dependencies.

## Benchmarking

Run the following command to benchmark on a given config:

```shell
cd $opencv_zoo
PYTHONPATH=. python benchmark/run.py --cfg ./benchmark/config/YuNet.yaml
```