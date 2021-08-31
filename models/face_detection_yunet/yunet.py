# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

from itertools import product

import cv2 as cv
import numpy as np

class YuNet:
    def __init__(self, modelPath, inputNames, outputNames, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, keepTopK=750):
        self._modelPath = modelPath
        self._model = cv.dnn.readNet(self._modelPath)
        self._inputNames = inputNames
        self._outputNames = outputNames
        self._inputSize = inputSize
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._keepTopK = keepTopK

        self._min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self._steps = [8, 16, 32, 64]
        self._variance = [0.1, 0.2]

    def setBackend(self, backend):
        self._model.setPreferableBackend(backend)

    def setTarget(self, target):
        self._model.setPreferableTarget(target)

    def _preprocess(self, image, target_size):
        return cv.dnn.blobFromImage(image, size=tuple(target_size))

    def infer(self, image, target_size=None):
        h, w, _ = image.shape
        original_size = (w, h)
        if target_size is None:
            target_size = (w, h)

        # Preprocess
        inputBlob = self._preprocess(image, target_size)

        # Forward
        self._model.setInput(inputBlob)
        outputBlob = self._model.forward(self._outputNames)

        # Postprocess
        results = self._postprocess(outputBlob, target_size, original_size)

        return results

    def _postprocess(self, outputBlob, target_size, original_size):
        # Generate priors
        self._priorGen(target_size)

        # Decode
        dets = self._decode(outputBlob, original_size)

        # NMS
        keepIdx = cv.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK
        ) # box_num x class_num
        if len(keepIdx) > 0:
            dets = dets[keepIdx]
            dets = np.squeeze(dets, axis=1)
        dets = dets[:self._keepTopK]

        return dets

    def _priorGen(self, target_size):
        w, h = target_size
        feature_map_2th = [int(int((h + 1) / 2) / 2),
                           int(int((w + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2),
                           int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2),
                           int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2),
                           int(feature_map_4th[1] / 2)]
        feature_map_6th = [int(feature_map_5th[0] / 2),
                           int(feature_map_5th[1] / 2)]

        feature_maps = [feature_map_3th, feature_map_4th,
                        feature_map_5th, feature_map_6th]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self._min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])): # i->h, j->w
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self._steps[k] / w
                    cy = (i + 0.5) * self._steps[k] / h

                    priors.append([cx, cy, s_kx, s_ky])
        self.priors = np.array(priors, dtype=np.float32)

    def _decode(self, outputBlob, original_size):
        loc, conf, iou = outputBlob
        # get score
        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]
        # clamp
        _idx = np.where(iou_scores < 0.)
        iou_scores[_idx] = 0.
        _idx = np.where(iou_scores > 1.)
        iou_scores[_idx] = 1.
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = np.array(original_size)

        # get bboxes
        bboxes = np.hstack((
            (self.priors[:, 0:2]+loc[:, 0:2]*self._variance[0]*self.priors[:, 2:4]) * scale,
            (self.priors[:, 2:4]*np.exp(loc[:, 2:4]*self._variance)) * scale
        ))
        # (x_c, y_c, w, h) -> (x1, y1, w, h)
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        # get landmarks
        landmarks = np.hstack((
            (self.priors[:, 0:2]+loc[:,  4: 6]*self._variance[0]*self.priors[:, 2:4]) * scale,
            (self.priors[:, 0:2]+loc[:,  6: 8]*self._variance[0]*self.priors[:, 2:4]) * scale,
            (self.priors[:, 0:2]+loc[:,  8:10]*self._variance[0]*self.priors[:, 2:4]) * scale,
            (self.priors[:, 0:2]+loc[:, 10:12]*self._variance[0]*self.priors[:, 2:4]) * scale,
            (self.priors[:, 0:2]+loc[:, 12:14]*self._variance[0]*self.priors[:, 2:4]) * scale
        ))

        dets = np.hstack((bboxes, landmarks, scores))
        return dets