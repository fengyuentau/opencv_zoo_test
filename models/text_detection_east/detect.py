# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import cv2 as cv
import numpy as np

from east import EAST

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser(
    description="EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)"
    "Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of ")
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='text_detection_east.pb', help='Path to the model.')
parser.add_argument('--score_threshold', type=float, default=0.3, help='Filter out bboxes of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.5, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(image, results, color=(0, 255, 0), thickness=2):
    for box in results:
        cv.line(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
        cv.line(image, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), color, thickness)
        cv.line(image, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, thickness)
        cv.line(image, (int(box[6]), int(box[7])), (int(box[0]), int(box[1])), color, thickness)
    return image

if __name__ == '__main__':
    # Instantiate DB
    model = EAST(modelPath=args.model,
                 inputNames='',
                 outputNames=["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"],
                 inputSize=[320, 320],
                 confThreshold=args.score_threshold,
                 nmsThreshold=args.nms_threshold)

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)

        # Inference
        results = model.infer(image)

        # Draw results on the input image
        image = visualize(image, results)

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

            # Inference
            tm.start()
            results = model.infer(frame) # faces is a tuple
            tm.stop()

            # Draw results on the input image
            frame = visualize(frame, results)

            cv.putText(frame, 'FPS: {}'.format(tm.getFPS()), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            # Visualize results in a new Window
            cv.imshow('DB Demo', frame)

            tm.reset()