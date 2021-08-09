# CRNN for Text Recognition

CRNN is a convolutional recurrent neural network for text recognition.

***NOTE***: To run this demo, you need to have `$zoo/models/text_detection_east`.

## Demo

Run the following command to try the demo:
```shell
# recognize from an image
python recognize_image.py --input /path/to/image --model text_recognition_crnn.onnx
# recognize using a camera
python recognize_camera.py --model ./text_recognition_crnn.onnx
```

## Licence

Please see [LICENCE](./LICENCE).

## Reference

- https://arxiv.org/abs/1507.05717
- https://github.com/bgshih/crnn
- https://github.com/meijieru/crnn.pytorch
