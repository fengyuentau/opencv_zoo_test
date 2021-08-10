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
parser.add_argument('--input', help='Path to the image.')
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

     # Open an image
     img = cv.imread(args.input)

     # Set up a timer
     tickmeter = cv.TickMeter()

     # Inference
     tickmeter.start()
     dets = model.infer(img)
     tickmeter.stop()

     # Draw detection results on the input image
     img = draw_results(img, dets)

     # Print results
     print('{} faces detected; Inference time: {:.2f} ms'.format(dets.shape[0], tickmeter.getTimeMilli()))
     for idx, det in enumerate(dets):
          print('{}: [{:.0f}, {:.0f}] [{:.0f}, {:.0f}], {:.2f}'.format(idx, det[0], det[1], det[2], det[3], det[-1]))

     # Visualization
     if args.vis:
          # Display the image
          kWinName = "CRNN"
          cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
          cv.imshow(kWinName, img)

          cv.waitKey(0)
          cv.destroyAllWindows()
     else:
          # Save Results
          cv.imwrite('result_{}'.format(args.input.split('/')[-1]), img)

if __name__ == '__main__':
     main()