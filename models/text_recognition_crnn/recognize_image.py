# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.
# This file is taken and modified from https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py.

import sys
import argparse

import cv2 as cv
import numpy as np

from crnn import CRNN

sys.path.append('../text_detection_east')
from east import EAST

def str2bool(s):
    if s.lower() in ['on', 'true', 't', 'yes', 'y']:
        return True
    elif s.lower() in ['off', 'false', 'f', 'no', 'n']:
        return False
    else:
        return NotImplementedError


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
parser.add_argument('--vis', type=str2bool, default=True,
                    help='If true, a window will be open up for visualization. If false, results are saved as JPG files.')
args = parser.parse_args()

def draw_rbbox(image, rbboxes, texts, box_color=(0, 255, 0), text_color=(255, 0, 0)):
    '''Draw the given rotated bounding boxes on the given image

    Parameters:
        image   -   input image to be drawn on
        rbboxes -   rotated bounding boxes in format of [x0, y0, x1, y1, x2, y2, x3, y3, conf]
        texts   -   texts recognized
        color   -   a tuple for rgb values
    '''
    for box, text in zip(rbboxes, texts):
        # Draw the bounding box
        cv.line(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color)
        cv.line(image, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), box_color)
        cv.line(image, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), box_color)
        cv.line(image, (int(box[6]), int(box[7])), (int(box[0]), int(box[1])), box_color)

        # Put the text
        cv.putText(image, text, (int(box[2]), int(box[3])), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
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

    # Open an image
    img = cv.imread(args.input)

    # Set up a timer
    tickmeter = cv.TickMeter()

    # Inference
    tickmeter.start()
    rbboxes = detector.infer(img)
    tickmeter.stop()

    # Run text recognition for each bounding box
    texts = []
    tickmeter.start()
    for rbbox in rbboxes:
        texts.append(recognizer.infer(img, rbbox))
    tickmeter.stop()

    # Draw rotated bounding boxes and 
    img = draw_rbbox(img, rbboxes, texts)

    # Print results
    print('{} text detected and recognized; Inference time: {:.2f} ms'.format(rbboxes.shape[0], tickmeter.getTimeMilli()))
    for idx, (box, text) in enumerate(zip(rbboxes, texts)):
        print('{}: [{:.0f}, {:.0f}] [{:.0f}, {:.0f}] [{:.0f}, {:.0f}] [{:.0f}, {:.0f}], {}'.format(idx, box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], text))


    if args.vis:
        # Display the img
        kWinName = "CRNN"
        cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
        cv.imshow(kWinName, img)

        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        # Save Results
        cv.imwrite('result_{}'.format(args.input.split('/')[-1]), img)


if __name__ == "__main__":
    main()