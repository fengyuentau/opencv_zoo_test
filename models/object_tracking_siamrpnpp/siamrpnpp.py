# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import cv2 as cv
import numpy as np

class SiamRPNPP:
    def __init__(self, target_net, search_net, rpn_head):
        self._targetNet = cv.dnn.readNet(target_net)
        self._searchNet = cv.dnn.readNet(search_net)
        self._rpnHead = cv.dnn.readNet(rpn_head)

        # Some fixed hyper parameters
        self._anchorStride = 8
        self._anchorRatios = [0.33, 0.5, 1, 2, 3]
        self._anchorScales = [8]
        self._trackBaseSize = 8
        self._trackExemplarSize = 127
        self._trackInstanceSize = 255
        self._trackLr = 0.4
        self._trackPenaltyK = 0.04
        self._trackWindowInfluence = 0.44
        self._scoreSize = (self._trackInstanceSize - self._trackExemplarSize) // \
                           self._anchorStride + 1 + self._trackBaseSize
        self._anchorNum = len(self._anchorRatios) * len(self._anchorScales)
        hanning = np.hanning(self._scoreSize)
        window = np.outer(hanning, hanning)
        self._window = np.tile(window.flatten(), self._anchorNum)

        # Generate anchors
        # TBD

    def setBackend(self, backendId):
        self._targetNet.setPreferableBackend(backendId)
        self._searchNet.setPreferableBackend(backendId)
        self._rpnHead.setPreferableBackend(backendId)

    def setTarget(self, targetId):
        self._targetNet.setPreferableTarget(targetId)
        self._searchNet.setPreferableTarget(targetId)
        self._rpnHead.setPreferableTarget(targetId)

    def _preprocess(self, image, target_size):
        pass

    def infer(self, image, target_size=None):
        pass

    def _postprocess(self, outputBlob, target_size, original_size):
        pass