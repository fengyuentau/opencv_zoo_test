# EAST And CRNN for Text Detection

EAST is an efficient and accurate detector for text.

## Demo

Run the following command to try the demo:
```shell
# detect from an image
python detect_image.py --input /path/to/image --model text_detection_east.pb
# detect using a camera
python detect_camera.py --model ./text_detection_east.pb
```

## Licence

Please see [LICENCE](./LICENCE).

## Reference

- https://arxiv.org/abs/1704.03155v2
- https://github.com/argman/EAST