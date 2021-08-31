# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import math

import cv2 as cv
import numpy as np

class EAST:
    def __init__(self, modelPath, inputNames, outputNames, inputSize=[320, 320], confThreshold=0.5, nmsThreshold=0.4):
        self._inputNames = inputNames
        self._outputNames = outputNames
        self._inputHeight = inputSize[0]
        self._inputWidth = inputSize[1]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold

        self._model = cv.dnn.readNet(modelPath)

    def setBackend(self, backend):
        self._model.setPreferableBackend(backend)

    def setTarget(self, target):
        self._model.setPreferableTarget(target)

    def _preprocess(self, image, target_size):
        return cv.dnn.blobFromImage(image, 1.0, (self._inputWidth, self._inputHeight), (123.68, 116.78, 103.94), True, False)

    def infer(self, image, target_size=None):
        h, w, _ = image.shape
        original_size = [w, h]
        target_size = (self._inputWidth, self._inputHeight)

        # preprocess
        inputBlob = self._preprocess(image, target_size)

        # forward
        self._model.setInput(inputBlob)
        outputBlob = self._model.forward(self._outputNames)

        # postprocess
        results = self._postprocess(outputBlob, target_size, original_size)

        return results # n x [x0, y0, x1, y1, x2, y2, x3, y3, conf]

    def _postprocess(self, outputBlob, target_size, original_size):
        # Get scores and geometry
        scores = outputBlob[0]
        geometry = outputBlob[1]

        # Decode from scores and geometry
        [boxes, confidences] = self._decodeBoundingBoxes(scores, geometry, self._confThreshold)

        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, self._confThreshold, self._nmsThreshold)
        # box: (x1, y1)---(x2, y2)
        #         |          |
        #      (x0, y0)---(x3, y3)
        # dets: [x0, y0, x1, y1, x2, y2, x3, y3, confidence]
        dets = np.empty(shape=(0, 9), dtype=np.float32)
        rW = original_size[0] / float(target_size[0])
        rH = original_size[1] / float(target_size[1])
        for i in indices:
            # get 4 vertices of the rotated rect
            v = cv.boxPoints(boxes[i[0]])
            v[:, 0] *= rW
            v[:, 1] *= rH

            # get confidence
            c = confidences[i[0]]
            dets = np.vstack(
                (dets, [*v[0], *v[1], *v[2], *v[3], c])
            )

        return dets

    def _decodeBoundingBoxes(self, scores, geometry, scoreThresh):
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
        assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):

            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if (score < scoreThresh):
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # return results
        return [detections, confidences]

if __name__ == '__main__':
    image = cv.imread('../../images/text_detection/firstaid.jpg')

    outputNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    model = EAST('text_detection_east.pb', '', outputNames)
    res = model.forward(image)

    print(res)