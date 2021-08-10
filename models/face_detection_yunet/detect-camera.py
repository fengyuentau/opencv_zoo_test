# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import cv2 as cv
import numpy as np

from yunet import YuNet

def str2bool(s):
    if s.lower() in ['on', 'true', 't', 'yes', 'y']:
       return True
    elif s.lower() in ['off', 'false', 'f', 'no', 'n']:
       return False
    else:
       return NotImplementedError

parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--input', help='Path to a video file.')
parser.add_argument('--model', type=str, help='Path to .onnx model file.')
parser.add_argument('--width', type=int, default=320, help='Fixed input size.')
parser.add_argument('--height', type=int, default=320, help='Fixed input size.')
parser.add_argument('--conf', type=float, default=0.6, help='Confidence threshold.')
parser.add_argument('--nms', type=float, default=0.4, help='Non-maximum suppression threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k for NMS.')
parser.add_argument('--keep_top_k', type=int, default=750, help='Keep keep_top_k after NMS.')
parser.add_argument('--vis', type=str2bool, default=True,
                    help='If true, a window will be open up for visualization. If false, results are saved as JPG files.')
args = parser.parse_args()

def draw_results(image, results, box_color=(0, 255, 0), text_color=(255, 255, 255)):
    '''Draw the given results (bounding boxes and landmarks) on the given image.

    Parameters:
        image     -    input image to be drawn on.
        resuls    -    num x [x1, y1, w, h, *[landmarks_x, landmarks_y], conf]
        color     -    a tuple for rgb values.
    '''
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # mouth right
        (  0, 255, 255)  # mouth left
    ]
    for det in results:
        # Draw bounding box
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        # Draw confidence
        conf = det[-1]
        cv.putText(image, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        # Draw landmarks
        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(image, landmark, 2, landmark_color[idx], 2)

    return image

def main():
    # Instantiate YuNet
    model = YuNet(model=args.model,
                inputNames='',
                outputNames=['loc', 'conf', 'iou'],
                inputSize=[args.width, args.height],
                confThreshold=args.conf,
                nmsThreshold=args.nms,
                topK=args.top_k,
                keepTopK=args.keep_top_k)

    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)

    # Set up a window for visualization
    kWinName = 'YuNet Demo'
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
        time_msg = 'Inference time: %.2f ms' % (tickmeter.getTimeMilli())
        cv.putText(frame, time_msg, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Draw rotated bounding boxes
        frame = draw_results(frame, results)

        # Display the frame
        cv.imshow(kWinName, frame)
        tickmeter.reset()

if __name__ == '__main__':
    main()