# EAST And CRNN for Text Detection

EAST is an efficient and accurate detector for text, and CRNN is a convolutional recurrent neural network for text recognition.

## Demo

Run the following command to try the demo:
```shell
python detect.py --input /path/to/image --model ./text_detection_east.pb --ocr ./text_recognition_crnn.onnx
```

## Licence

Please see [LICENCE](./LICENCE).

## Reference

### EAST

- https://arxiv.org/abs/1704.03155v2
- https://github.com/argman/EAST

### CRNN

- https://arxiv.org/abs/1507.05717
- https://github.com/bgshih/crnn
- https://github.com/meijieru/crnn.pytorch
