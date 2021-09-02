# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, OpenCV China, all rights reserved.
# Third party copyrights are property of their respective owners.
# This file is modified from https://github.com/opencv/opencv/blob/master/samples/dnn/siamrpnpp.py

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
        self._trackContextAmount = 0.5
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
        self._anchors = self._generateAnchor()

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

    def _generateAnchor(self):
        anchors = np.zeros(
            shape=(len(self._anchorScales) * len(self._anchorRatios), 4),
            dtype=np.float32
        )
        size = self._anchorStride ** 2
        count = 0
        for r in self._anchorRatios:
            ws = int(np.sqrt(size * 1. / r))
            hs = int(ws * r)

            for s in self._anchorScales:
                w = ws * s
                h = hs * s
                anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1

        x1, y1, x2, y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        anchors = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        anchors = np.tile(anchors, self._scoreSize * self._scoreSize).reshape((-1, 4))
        ori = - (self._scoreSize // 2) * self._anchorStride
        xx, yy = np.meshgrid([ori + self._anchorStride * dx for dx in range(self._scoreSize)],
                             [ori + self._anchorStride * dy for dy in range(self._scoreSize)])
        xx, yy = np.tile(xx.flatten(), (self._anchorNum, 1)).flatten(), \
                 np.tile(yy.flatten(), (self._anchorNum, 1)).flatten()
        anchors[:, 0], anchors[:, 1] = xx.astype(np.float32), yy.astype(np.float32)

        return anchors

    def initialize(self, img, bbox):
        x, y, w, h = bbox
        self._targetCenter = np.array([x + (w - 1) / 2, y + (h - 1) / 2])
        self._targetH = h
        self._targetW = w
        w_z = self._targetW + self._trackContextAmount * np.add(h, w)
        h_z = self._targetH + self._trackContextAmount * np.add(h, w)
        s_z = round(np.sqrt(w_z * h_z))
        self._channelAverage = np.mean(img, axis=(0, 1))
        z_crop = self._getSubwindow(img, self._targetCenter, self._trackExemplarSize, s_z, self._channelAverage)
        self._targetNetForward(z_crop)

    def _getSubwindow(self, im, pos, model_sz, original_sz, avg_chans):
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_h, im_w, im_d = im.shape
        c = (original_sz + 1) / 2
        cx, cy = pos
        context_xmin = np.floor(cx - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(cy - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_w + 1))
        bottom_pad = int(max(0., context_ymax - im_h + 1))
        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad

        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (im_h + top_pad + bottom_pad, im_w + left_pad + right_pad, im_d)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + im_h, left_pad:left_pad + im_w, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + im_w, :] = avg_chans
            if bottom_pad:
                te_im[im_h + top_pad:, left_pad:left_pad + im_w, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, im_w + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        return im_patch

    def _targetNetForward(self, x):
        self._targetNet.setInput(x)
        outputNames = self._targetNet.getUnconnectedOutLayersNames()
        self._zfs1, self._zfs2, self._zsf3 = self._targetNet.forward(outputNames)

    def _searchNetForward(self, x):
        self._searchNet.setInput(x)
        outputNames = self._searchNet.getUnconnectedOutLayersNames()
        _xfs1, _xfs2, _xfs3 = self._searchNet.forward(outputNames)
        return _xfs1, _xfs2, _xfs3

    def _rpnHeadForward(self, xfs_1, xfs_2, xfs_3):
        self._rpnHead.setInput(np.stack([self._zfs1, self._zfs2, self._zsf3]), 'input_1')
        self._rpnHead.setInput(np.stack([xfs_1, xfs_2, xfs_3]), 'input_2')
        outputNames = self._rpnHead.getUnconnectedOutLayersNames()
        _cls, _loc = self._rpnHead.forward(outputNames)
        return {'cls': _cls, 'loc': _loc}

    def _softmax(self, x):
        """
        Softmax in the direction of the depth of the layer
        """
        x = x.astype(dtype=np.float32)
        x_max = x.max(axis=1)[:, np.newaxis]
        e_x = np.exp(x-x_max)
        div = np.sum(e_x, axis=1)[:, np.newaxis]
        y = e_x / div
        return y

    def _convertScore(self, score):
        score_transpose = np.transpose(score, (1, 2, 3, 0))
        score_con = np.ascontiguousarray(score_transpose)
        score_view = score_con.reshape(2, -1)
        score = np.transpose(score_view, (1, 0))
        score = self._softmax(score)
        return score[:,1]

    def _convertBbox(self, delta, anchor):
        delta_transpose = np.transpose(delta, (1, 2, 3, 0))
        delta_contig = np.ascontiguousarray(delta_transpose)
        delta = delta_contig.reshape(4, -1)
        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _clipBbox(self, cx, cy, width, height, boundary):
        """
        Adjusting the bounding box
        """
        bbox_h, bbox_w = boundary
        cx = max(0, min(cx, bbox_w))
        cy = max(0, min(cy, bbox_h))
        width = max(10, min(width, bbox_w))
        height = max(10, min(height, bbox_h))
        return cx, cy, width, height

    def track(self, img):
        # Run search net and rpn head
        w_z = self._targetW + self._trackContextAmount * np.add(self._targetW, self._targetH)
        h_z = self._targetH + self._trackContextAmount * np.add(self._targetW, self._targetH)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self._trackExemplarSize / s_z
        s_x = s_z * (self._trackInstanceSize / self._trackExemplarSize)
        x_crop = self._getSubwindow(img, self._targetCenter, self._trackInstanceSize, round(s_x), self._channelAverage)
        searchNetOutputs = self._searchNetForward(x_crop)
        outputs = self._rpnHeadForward(*searchNetOutputs)
        score = self._convertScore(outputs['cls'])
        pred_bbox = self._convertBbox(outputs['loc'], self._anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self._targetW * scale_z, self._targetH * scale_z)))

        # aspect ratio penalty
        r_c = change((self._targetW / self._targetH) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self._trackPenaltyK)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self._trackWindowInfluence) + \
                 self._window * self._trackWindowInfluence
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self._trackLr

        cpx, cpy = self._targetCenter
        x,y,w,h = bbox
        cx = x + cpx
        cy = y + cpy

        # smooth bbox
        width = self._targetW * (1 - lr) + w * lr
        height = self._targetH * (1 - lr) + h * lr

        # clip boundary
        cx, cy, width, height = self._clipBbox(cx, cy, width, height, img.shape[:2])

        # udpate state
        self._targetCenter = np.array([cx, cy])
        self._targetW = width
        self._targetH = height
        bbox = [cx - width / 2, cy - height / 2, width, height]
        best_score = score[best_idx]
        return {'bbox': bbox, 'best_score': best_score}