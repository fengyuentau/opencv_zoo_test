# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import cv2 as cv
import numpy as np

from siamrpnpp import SiamRPNPP

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser(description='SiamRPN++: Evolution of siamese visual tracking with very deep networks (https://arxiv.org/abs/1812.11703)')
parser.add_argument('--input', '-i', type=str, help='Path to the input video. Omit for using default camera.')
parser.add_argument('--target_net', type=str, default='object_tracking_siamrpnpp-target_net.onnx', help='Path to the target net for SiamRPN++.')
parser.add_argument('--search_net', type=str, default='object_tracking_siamrpnpp-search_net.onnx', help='Path to the search net for SiamRPN++.')
parser.add_argument('--rpn_head', type=str, default='object_tracking_siamrpnpp-rpn_head.onnx', help='Path to the target net for SiamRPN++.')
args = parser.parse_args()

def visualize(image, results, box_color=(0, 255, 0)):
    pass

if __name__ == '__main__':
    # Instantiate SiamRPN++
    model = SiamRPNPP(
        target_net=args.target_net,
        search_net=args.search_net,
        rpn_head=args.rpn_head
    )

    # Open a video file or camera stream
    isFirstFrame = True
    cap = cv.VideoCapture(args.input if args.input else 0)
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        if isFirstFrame:
            try:
                roi = cv.selectROI('Select a ROI for SiamRPN++ to track', frame, False, False)
            except:
                exit()
            model.initialize(frame, roi)
            isFirstFrame = False
        else:
            outputs = model.track(frame)
            bbox = list(map(int, outputs['bbox']))
            x, y, w, h = bbox
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv.imshow('Track using SiamRPN++', frame)