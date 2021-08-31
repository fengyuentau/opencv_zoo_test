# DB Text Detection

Real-time Scene Text Detection with Differentiable Binarization.

## Demo

Run the following command to try the demo:
```shell
# detect from an image
python detect_image.py --input /path/to/image --model ./text_detection_db.onnx
# detect using a camera
python detect_camera.py --model ./text_detection_db.onnx
```

## Model Descripton
`text_detection_db.onnx` detects both English & Chinese instances, and is renamed and obtained from `DB_TD500_resnet18.onnx`(https://docs.opencv.org/master/d4/d43/tutorial_dnn_text_spotting.html).

## Licence

Please see [LICENCE](./LICENCE).

## Reference

- https://arxiv.org/abs/1911.08947
- https://github.com/MhLiao/DB