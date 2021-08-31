# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import cv2 as cv
import numpy as np

from yunet import YuNet

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='face_detection_yunet.onnx', help='Path to the model.')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--keep_top_k', type=int, default=750, help='Keep keep_top_k bounding boxes after NMS.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(image, results, box_color=(0, 255, 0), text_color=(255, 255, 255)):
    '''Draw the given results (bounding boxes and landmarks) on the given image.

    Parameters:
        image     -    input image to be drawn on.
        resuls    -    num x [x1, y1, w, h, *[landmarks_x, landmarks_y], conf]
        color     -    a tuple for rgb values.
    '''
    output = image.copy()
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
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        # Draw confidence
        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        # Draw landmarks
        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output

if __name__ == '__main__':

    # Instantiate YuNet
    model = YuNet(modelPath=args.model,
                  inputNames='',
                  outputNames=['loc', 'conf', 'iou'],
                  inputSize=[320, 320],
                  confThreshold=args.score_threshold,
                  nmsThreshold=args.nms_threshold,
                  topK=args.top_k,
                  keepTopK=args.keep_top_k)

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)

        # Inference
        faces = model.infer(image)

        # Print results
        for idx, det in enumerate(faces):
            print('{}: [{:.0f}, {:.0f}] [{:.0f}, {:.0f}], {:.2f}'.format(
                idx, det[0], det[1], det[2], det[3], det[-1])
            )

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
            cv.imshow('YuNet Demo', frame)

            tm.reset()