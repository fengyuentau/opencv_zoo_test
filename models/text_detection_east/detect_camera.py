# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.
# This file is taken and modified from https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py.


# Import required modules
import argparse

import cv2 as cv
import numpy as np

from east import EAST

parser = argparse.ArgumentParser(
    description="Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of "
                "EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)")
parser.add_argument('--input',
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', '-m', default='text_detection_east.pb',
                    help='Path to a binary .pb file contains trained detector network.')
parser.add_argument('--width', type=int, default=320,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=320,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
parser.add_argument('--conf', type=float, default=0.5,
                    help='Confidence threshold.')
parser.add_argument('--nms', type=float, default=0.4,
                    help='Non-maximum suppression threshold.')
args = parser.parse_args()

def draw_rbbox(image, rbbox, color=(0, 255, 0)):
    '''Draw the given rotated bounding boxes on the given image

    Parameters:
        image   -   input image to be drawn on
        rbbox   -   rotated bounding boxes in format of [x0, y0, x1, y1, x2, y2, x3, y3, conf]
        color   -   a tuple for rgb values
    '''
    for box in rbbox:
        cv.line(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color)
        cv.line(image, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), color)
        cv.line(image, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color)
        cv.line(image, (int(box[6]), int(box[7])), (int(box[0]), int(box[1])), color)
    return image

def main():
    # Instantiate EAST
    model = EAST(model=args.model,
                 inputNames='',
                 outputNames=["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"],
                 inputSize=[args.height, args.width],
                 confThreshold=args.conf,
                 nmsThreshold=args.nms)

    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)

    # vis
    kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)

    tickmeter = cv.TickMeter()
    while cv.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        tickmeter.start()
        results = model.infer(frame)
        tickmeter.stop()

        # Put efficiency information
        label = 'Inference time: %.2f ms' % (tickmeter.getTimeMilli())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Draw rotated bounding boxes
        frame = draw_rbbox(frame, results)

        # Display the frame
        cv.imshow(kWinName, frame)
        tickmeter.reset()


if __name__ == "__main__":
    main()
