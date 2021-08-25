# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import cv2 as cv
import numpy as np

class CRNN:
    def __init__(self, modelPath, inputNames, outputNames, inputSize=[100, 32]):
        self._model = cv.dnn.readNet(modelPath)
        self._inputNames = inputNames
        self._outputNames = outputNames
        self._inputSize = inputSize
        self.targetVertices = np.array([
            [0, self._inputSize[1] - 1],
            [0, 0],
            [self._inputSize[0] - 1, 0],
            [self._inputSize[0] - 1, self._inputSize[1] - 1]], dtype="float32")

    def setBackend(self, backend):
        self._model.setPreferableBackend(backend)

    def setTarget(self, target):
        self._model.setPreferableTarget(target)

    def _preprocess(self, image, rbbox):
        # Remove conf, reshape and ensure all is np.float32
        vertices = rbbox[:-1].reshape((4, 2)).astype(np.float32)

        rotationMatrix = cv.getPerspectiveTransform(vertices, self.targetVertices)
        cropped = cv.warpPerspective(image, rotationMatrix, self._inputSize)

        cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

        return cv.dnn.blobFromImage(cropped, size=self._inputSize, mean=127.5, scalefactor=1 / 127.5)

    def infer(self, image, rbbox):
        # Preprocess
        inputBlob = self._preprocess(image, rbbox)

        # Forward
        self._model.setInput(inputBlob)
        outputBlob = self._model.forward(self._outputNames)

        # Postprocess
        results = self._postprocess(outputBlob)

        return results

    def _postprocess(self, outputBlob):
        '''Decode charaters from outputBlob
        '''
        text = ""
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        for i in range(outputBlob.shape[0]):
            c = np.argmax(outputBlob[i][0])
            if c != 0:
                text += alphabet[c - 1]
            else:
                text += '-'

        # adjacent same letters as well as background text must be removed to get the final output
        char_list = []
        for i in range(len(text)):
            if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
                char_list.append(text[i])
        return ''.join(char_list)