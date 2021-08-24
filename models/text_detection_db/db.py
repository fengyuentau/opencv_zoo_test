# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import math
from typing import NewType

import cv2 as cv
import numpy as np

class DB:
    def __init__(self, modelPath, inputSize=[736, 736], binThresh=0.3, polyThresh=0.5, maxCandidates = 200, unclipRatio = 2.0):
        self.modelPath = modelPath
        self.inputHeight = inputSize[0]
        self.inputWidth = inputSize[1]
        self.binThresh = binThresh
        self.polyThresh = polyThresh
        self.maxCandidates = maxCandidates
        self.unclipRatio = unclipRatio

        net = cv.dnn.readNet(self.modelPath)
        self.detector = cv.dnn_TextDetectionModel_DB(net)
        self.detector.setBinaryThreshold(self.binThresh)
        self.detector.setPolygonThreshold(self.polyThresh)
        self.detector.setUnclipRatio(self.unclipRatio)
        self.detector.setMaxCandidates(self.maxCandidates)

        self.detector.setInputParams(1.0/255.0, inputSize, (122.67891434, 116.66876762, 104.00698793))

    def setBackend(self, backend):
        self.model.setPreferableBackend(backend)

    def setTarget(self, target):
        self.model.setPreferableTarget(target)

    def infer(self, image):
        return self.detector.detect(image)

if __name__ == '__main__':
    image = cv.imread('../../images/text_detection/firstaid.jpg')

    model = DB('text_detection_db18.onnx')
    res = model.infer(image)

    print(res)