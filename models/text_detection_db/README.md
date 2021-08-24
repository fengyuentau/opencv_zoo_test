# DB Text Detection

Real-time Scene Text Detection with Differentiable Binarization.

## Demo

Run the following command to try the demo:
```shell
# detect from an image
python detect_image.py --input /path/to/image --model ./DB_IC15_resnet18_en.onnx
# detect using a camera
python detect_camera.py --model ./DB_IC15_resnet18_en.onnx
```

## Model Descripton
There are two pre-trained model,
- DB_IC15_resnet18_en.onnx is for English.
- DB_TD500_resnet18_cn.onnx is for Chinese.

And if you wan to get more pre-trained DB models, please refer https://docs.opencv.org/4.5.1/d4/d43/tutorial_dnn_text_spotting.html.

## Licence

Please see [LICENCE](./LICENCE).

## Reference

- https://arxiv.org/abs/1911.08947
- https://github.com/MhLiao/DB