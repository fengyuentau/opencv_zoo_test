# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import math

import cv2 as cv
import numpy as np

class EAST:
    def __init__(self, model, inputNames, outputNames, inputSize=[320, 320], confThreshold=0.5, nmsThreshold=0.4):
        self.inputNames = inputNames
        self.outputNames = outputNames
        self.inputHeight = inputSize[0]
        self.inputWidth = inputSize[1]
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

        self.detector = cv.dnn.readNet(model)

    def setBackend(self, backend):
        self.model.setPreferableBackend(backend)

    def setTarget(self, target):
        self.model.setPreferableTarget(target)

    def __preprocess(self, image):
        return cv.dnn.blobFromImage(image, 1.0, (self.inputWidth, self.inputHeight), (123.68, 116.78, 103.94), True, False)

    def infer(self, image):
        originalHeight, originalWidth, _ = image.shape

        # preprocess
        inputBlob = self.__preprocess(image)

        # forward
        self.detector.setInput(inputBlob)
        outputBlob = self.detector.forward(self.outputNames)

        # postprocess
        self.rW = originalWidth / float(self.inputWidth)
        self.rH = originalHeight / float(self.inputHeight)
        results = self.__postprocess(outputBlob)

        return results # n x [x0, y0, x1, y1, x2, y2, x3, y3, conf]

    def __postprocess(self, outputBlob):
        # Get scores and geometry
        scores = outputBlob[0]
        geometry = outputBlob[1]

        # Decode from scores and geometry
        [boxes, confidences] = self.__decodeBoundingBoxes(scores, geometry, self.confThreshold)

        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, self.confThreshold, self.nmsThreshold)
        # box: (x1, y1)---(x2, y2)
        #         |          |
        #      (x0, y0)---(x3, y3)
        # dets: [x0, y0, x1, y1, x2, y2, x3, y3, confidence]
        dets = np.empty(shape=(0, 9), dtype=np.float32)
        for i in indices:
            # get 4 vertices of the rotated rect
            v = cv.boxPoints(boxes[i[0]])
            v[:, 0] *= self.rW
            v[:, 1] *= self.rH

            # get confidence
            c = confidences[i[0]]
            dets = np.vstack(
                (dets, [*v[0], *v[1], *v[2], *v[3], c])
            )

        return dets

    def __decodeBoundingBoxes(self, scores, geometry, scoreThresh):
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