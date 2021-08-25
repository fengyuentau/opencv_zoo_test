# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import math

import cv2 as cv
import numpy as np

class DB:
    def __init__(self, modelPath, inputNames, outputNames, inputSize=[736, 736], binaryThreshold=0.3, polygonThreshold=0.5, maxCandidates=200, unclipRatio=2.0):
        self._modelPath = modelPath
        self._inputHeight = inputSize[0]
        self._inputWidth = inputSize[1]
        self._binaryThreshold = binaryThreshold
        self._polygonThreshold = polygonThreshold
        self._maxCandidates = maxCandidates
        self._unclipRatio = unclipRatio

        net = cv.dnn.readNet(self._modelPath)
        self._model = cv.dnn_TextDetectionModel_DB(net)
        self._model.setBinaryThreshold(self._binaryThreshold)
        self._model.setPolygonThreshold(self._polygonThreshold)
        self._model.setUnclipRatio(self._unclipRatio)
        self._model.setMaxCandidates(self._maxCandidates)

        self._model.setInputParams(1.0/255.0, inputSize, (122.67891434, 116.66876762, 104.00698793))

    def setBackend(self, backend):
        self._model.setPreferableBackend(backend)

    def setTarget(self, target):
        self._model.setPreferableTarget(target)

    def infer(self, image):
        return self._model.detect(image)

if __name__ == '__main__':
    image = cv.imread('../../images/text_detection/firstaid.jpg')

    model = DB('text_detection_db18.onnx')
    res = model.infer(image)

    print(res)