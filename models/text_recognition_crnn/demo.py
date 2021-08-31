# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import sys
import argparse

import cv2 as cv
import numpy as np

from crnn import CRNN

sys.path.append('../text_detection_db')
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
parser.add_argument('--model', '-m', type=str, default='text_recognition_crnn.onnx', help='Path to the model.')
parser.add_argument('--width', type=int, default=100,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=32,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(image, boxes, texts, color=(0, 255, 0), isClosed=True, thickness=2):
    output = image.copy()

    pts = np.array(boxes[0])
    output = cv.polylines(output, pts, isClosed, color, thickness)
    for box, text in zip(boxes[0], texts):
        cv.putText(output, text, (box[1].astype(np.int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25))
    return output

if __name__ == '__main__':
    # Instantiate CRNN for text recognition
    recognizer = CRNN(modelPath=args.model,
               inputNames='',
               outputNames='',
               inputSize=[args.width, args.height]
    )
    # Instantiate DB for text detection
    detector = DB(modelPath='../text_detection_db/text_detection_db.onnx',
               inputNames='',
               outputNames='',
               inputSize=[736, 736],
               binaryThreshold=0.3,
               polygonThreshold=0.5,
               maxCandidates=200,
               unclipRatio=2.0
    )

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)

        # Inference
        results = detector.infer(image)
        texts = []
        for box, score in zip(results[0], results[1]):
            result = np.hstack(
                (box.reshape(8), score)
            )
            texts.append(
                recognizer.infer(image, result)
            )

        # Draw results on the input image
        image = visualize(image, results, texts)

        # Save results if save is true
        if args.save:
            print('Resutls saved to result.jpg\n')
            cv.imwrite('result.jpg', image)

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, image)
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

            # Inference of text detector
            tm.start()
            results = detector.infer(frame)
            tm.stop()
            fps_detector = tm.getFPS()
            tm.reset()
            # Inference of text recognizer
            texts = []
            tm.start()
            for box in results:
                texts.append(
                    recognizer.infer(frame, box)
                )
            tm.stop()
            fps_recognizer = tm.getFPS()
            tm.reset()

            # Draw results on the input image
            frame = visualize(frame, results, texts)

            cv.putText(frame, 'Detector DB FPS: {}'.format(fps_detector), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv.putText(frame, 'Recognizer CRNN FPS: {}'.format(fps_recognizer), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            # Visualize results in a new Window
            cv.imshow('CRNN Demo', frame)