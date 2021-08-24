# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import cv2 as cv
import numpy as np

from db import DB

parser = argparse.ArgumentParser(
    description="Use this script to run Pytorch implementation (https://github.com/MhLiao/DB) of "
                "Real-time Scene Text Detection with Differentiable Binarization (https://arxiv.org/abs/1911.08947)")
parser.add_argument('--input',
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', '-m', default='./DB_IC15_resnet18_en.onnx',
                    help='Path to a binary .onnx file contains trained detector network.')
parser.add_argument('--width', type=int, default=736,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=736,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
parser.add_argument('--binThresh', type=float, default=0.3,
                    help='Confidence threshold of the binary map.')
parser.add_argument('--polyThresh', type=float, default=0.5,
                    help='Confidence threshold of polygons.')
parser.add_argument('--maxCandidates', type=int, default=200,
                    help='Max candidates of polygons.')
parser.add_argument('--unclipRatio', type=np.float64, default=2.0,
                    help='unclip ratio.')
args = parser.parse_args()


def main():
    # Instantiate DB Text Detection
    model = DB(modelPath=args.model,
                 inputSize=[args.width, args.height],
                 binThresh=args.binThresh,
                 polyThresh=args.polyThresh,
                 maxCandidates=args.maxCandidates,
                 unclipRatio=args.unclipRatio
                 )

    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)

    # Set up a window for visualization
    kWinName = "Real-time Scene Text Detection with Differentiable Binarization"
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
        # Draw rotated bounding boxes
        isClosed = True
        # Blue color in BGR
        color = (0, 255, 0)
        # Line thickness of 2 px
        thickness = 2

        # Image Drawing
        pts = np.array(results[0])
        frame = cv.polylines(frame, pts, isClosed, color, thickness)

        # Put efficiency information
        time_msg = 'Inference time: %.2f ms' % (tickmeter.getTimeMilli())
        cv.putText(frame, time_msg, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Display the frame
        cv.imshow(kWinName, frame)
        tickmeter.reset()


if __name__ == "__main__":
    main()