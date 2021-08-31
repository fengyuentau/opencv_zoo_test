# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import cv2 as cv
import numpy as np

from db import DB

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser(
    description="Real-time Scene Text Detection with Differentiable Binarization (https://arxiv.org/abs/1911.08947)"
    "Use this script to run Pytorch implementation (https://github.com/MhLiao/DB) of ")
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='text_detection_db.onnx', help='Path to the model.')
parser.add_argument('--width', type=int, default=736,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=736,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
parser.add_argument('--binary_threshold', type=float, default=0.3, help='Threshold of the binary map.')
parser.add_argument('--polygon_threshold', type=float, default=0.5, help='Threshold of polygons.')
parser.add_argument('--max_candidates', type=int, default=200, help='Max candidates of polygons.')
parser.add_argument('--unclip_ratio', type=np.float64, default=2.0, help=' The unclip ratio of the detected text region, which determines the output size.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(image, results, color=(0, 255, 0), isClosed=True, thickness=2):
    output = image.copy()

    pts = np.array(results[0])
    output = cv.polylines(output, pts, isClosed, color, thickness)
    return output

if __name__ == '__main__':
    # Instantiate DB
    model = DB(modelPath=args.model,
               inputNames='',
               outputNames='',
               inputSize=[args.width, args.height],
               binaryThreshold=args.binary_threshold,
               polygonThreshold=args.polygon_threshold,
               maxCandidates=args.max_candidates,
               unclipRatio=args.unclip_ratio
    )

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)

        # Inference
        faces = model.infer(image)

        # Draw results on the input image
        result = visualize(image, faces)

        # Save results if save is true
        if args.save:
            print('Resutls saved to result.jpg\n')
            cv.imwrite('result.jpg', result)

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, result)
            cv.waitKey(0)
    else: # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Inference
            tm.start()
            faces = model.infer(frame) # faces is a tuple
            tm.stop()

            # Draw results on the input image
            frame = visualize(frame, faces)

            cv.putText(frame, 'FPS: {}'.format(tm.getFPS()), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            # Visualize results in a new Window
            cv.imshow('DB Demo', frame)

            tm.reset()