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

sys.path.append('../text_detection_east')
from east import EAST

parser = argparse.ArgumentParser(
    description="The OCR model can be obtained from converting the pretrained CRNN model to .onnx format from the github repository https://github.com/meijieru/crnn.pytorch")
parser.add_argument('--input',
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', '-m', required=True,
                    help='Path to a binary .pb file contains trained detector network.')
parser.add_argument('--width', type=int, default=100,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=32,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
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
    # Instantiate EAST for text detection
    detector = EAST(model='../text_detection_east/text_detection_east.pb',
                    inputNames='',
                    outputNames=["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"],
                    inputSize=[320, 320])

    # Instantiate CRNN for text recognition
    recognizer = CRNN(model=args.model,
                      inputNames='',
                      outputNames='',
                      inputSize=[args.width, args.height])

    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)

    # Set up a window for visualization
    kWinName = "CRNN"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)

    tickmeter = cv.TickMeter()
    while cv.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        # Run text detection
        tickmeter.start()
        rbboxes = detector.infer(frame)
        tickmeter.stop()

        # Run text recognition for each bounding box
        for rbbox in rbboxes:
            tickmeter.start()
            results = recognizer.infer(frame, rbbox)
            tickmeter.stop()

            cv.putText(frame, results, (int(rbbox[2]), int(rbbox[3])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

        # Draw rotated bounding boxes
        frame = draw_rbbox(frame, rbboxes)

        # Put efficiency information
        time_msg = 'Inference time: %.2f ms' % (tickmeter.getTimeMilli())
        cv.putText(frame, time_msg, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Display the frame
        cv.imshow(kWinName, frame)
        tickmeter.reset()


if __name__ == "__main__":
    main()