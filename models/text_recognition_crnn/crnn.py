# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

import cv2 as cv
import numpy as np

class CRNN:
    def __init__(self, model, inputNames, outputNames, inputSize=[100, 32]):
        self.model = cv.dnn.readNet(model)
        self.inputNames = inputNames
        self.outputNames = outputNames
        self.inputSize = inputSize
        self.targetVertices = np.array([
            [0, self.inputSize[1] - 1],
            [0, 0],
            [self.inputSize[0] - 1, 0],
            [self.inputSize[0] - 1, self.inputSize[1] - 1]], dtype="float32")

    def setBackend(self, backend):
        self.model.setPreferableBackend(backend)

    def setTarget(self, target):
        self.model.setPreferableTarget(target)

    def __preprocess(self, image, rbbox):
        # Remove conf, reshape and ensure all is np.float32
        vertices = rbbox[:-1].reshape((4, 2)).astype(np.float32)

        rotationMatrix = cv.getPerspectiveTransform(vertices, self.targetVertices)
        cropped = cv.warpPerspective(image, rotationMatrix, self.inputSize)

        cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

        return cv.dnn.blobFromImage(cropped, size=self.inputSize, mean=127.5, scalefactor=1 / 127.5)

    def infer(self, image, rbbox):
        # Preprocess
        inputBlob = self.__preprocess(image, rbbox)

        # Forward
        self.model.setInput(inputBlob)
        outputBlob = self.model.forward(self.outputNames)

        # Postprocess
        results = self.__postprocess(outputBlob)

        return results

    def __postprocess(self, outputBlob):
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