# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.

from itertools import product

import cv2 as cv
import numpy as np

class YuNet:
    def __init__(self, model, inputNames, outputNames, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, keepTopK=750):
        self.model = cv.dnn.readNet(model)
        self.inputNames = inputNames
        self.outputNames = outputNames
        self.inputSize = inputSize
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.topK = topK
        self.keepTopK = keepTopK

        self.min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self.steps = [8, 16, 32, 64]
        self.variance = [0.1, 0.2]

    def __preprocess(self, image):
        return cv.dnn.blobFromImage(image)

    def infer(self, image):
        self.origHeight, self.origWidth, _ = image.shape

        # preprocess
        inputBlob = self.__preprocess(image)

        # forward
        self.model.setInput(inputBlob)
        outputBlob = self.model.forward(self.outputNames)

        # postprocess
        results = self.__postprocess(outputBlob)

        return results

    def __postprocess(self, outputBlob):
        # Generate priors
        self.__priorGen()

        # Decode
        dets = self.__decode(outputBlob[0], outputBlob[1], outputBlob[2])

        # NMS
        keepIdx = cv.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self.confThreshold,
            nms_threshold=self.nmsThreshold,
            top_k=self.topK
        ) # box_num x class_num
        if keepIdx.shape[0] > 0:
            dets = dets[keepIdx]
            dets = np.squeeze(dets, axis=1)
        dets = dets[:self.keepTopK]

        return dets

    def __priorGen(self):
        feature_map_2th = [int(int((self.origHeight + 1) / 2) / 2),
                           int(int((self.origWidth + 1) / 2) / 2)]
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

        self.priors = np.empty(shape=[0, 4])
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])): # i->origHeight, j->origWidth
                for min_size in min_sizes:
                    s_kx = min_size / self.origWidth
                    s_ky = min_size / self.origHeight

                    cx = (j + 0.5) * self.steps[k] / self.origWidth
                    cy = (i + 0.5) * self.steps[k] / self.origHeight

                    self.priors = np.vstack(
                        (self.priors, np.array([cx, cy, s_kx, s_ky]))
                    )

    def __decode(self, loc, conf, iou):
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

        # get bboxes
        bboxes = np.hstack((
            self.priors[:, 0:2]+loc[:, 0:2]*self.variance[0]*self.priors[:, 2:4],
            self.priors[:, 2:4]*np.exp(loc[:, 2:4]*self.variance)
        ))
        # (x_c, y_c, w, h) -> (x1, y1, w, h)
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2
        # scale recover
        bbox_scale = np.array([self.origWidth, self.origHeight]*2)
        bboxes = bboxes * bbox_scale

        # get landmarks
        landmarks = np.hstack((
            self.priors[:, 0:2]+loc[:,  4: 6]*self.variance[0]*self.priors[:, 2:4],
            self.priors[:, 0:2]+loc[:,  6: 8]*self.variance[0]*self.priors[:, 2:4],
            self.priors[:, 0:2]+loc[:,  8:10]*self.variance[0]*self.priors[:, 2:4],
            self.priors[:, 0:2]+loc[:, 10:12]*self.variance[0]*self.priors[:, 2:4],
            self.priors[:, 0:2]+loc[:, 12:14]*self.variance[0]*self.priors[:, 2:4]
        ))
        # scale recover
        landmark_scale = np.array([self.origWidth, self.origHeight]*5)
        landmarks = landmarks * landmark_scale

        dets = np.hstack((bboxes, landmarks, scores))
        return dets