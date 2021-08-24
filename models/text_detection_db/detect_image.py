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

from db import DB

def str2bool(s):
    if s.lower() in ['on', 'true', 't', 'yes', 'y']:
        return True
    elif s.lower() in ['off', 'false', 'f', 'no', 'n']:
        return False
    else:
        return NotImplementedError


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
parser.add_argument('--vis', type=str2bool, default=True,
                    help='If true, a window will be open up for visualization. If false, results are saved as JPG files.')
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

    # Open an image
    img = cv.imread(args.input)

    # Inference and timer
    tickmeter = cv.TickMeter()
    tickmeter.start()
    results = model.infer(img)
    tickmeter.stop()

    # Draw rotated bounding boxes
    isClosed = True
    # Blue color in BGR
    color = (0, 255, 0)
    # Line thickness of 2 px
    thickness = 2

    # Image Drawing
    pts = np.array(results[0])
    image = cv.polylines(img, pts, isClosed, color, thickness)

    # Print results
    print('{} text detected; Inference time: {:.2f} ms'.format(pts.shape[0], tickmeter.getTimeMilli()))

    if args.vis:
        # Display the img
        kWinName = "Real-time Scene Text Detection with Differentiable Binarization"
        cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
        cv.imshow(kWinName, image)

        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        # Save Results
        cv.imwrite('result_{}'.format(args.input.split('/')[-1]), img)


if __name__ == "__main__":
    main()