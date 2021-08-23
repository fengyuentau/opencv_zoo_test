import os

import tqdm
import numpy as np
import cv2 as cv
from scipy.io import loadmat

class WIDERFace(object):
    def __init__(self, subset='val', iouThreshold=0.5, testList=None, testSize=None, repeat=None):
        assert subset.lower() in ['val']
        self.subset = subset
        self.iou_threshold = iouThreshold
        self.test_list = testList
        self.test_size = testSize
        self.repeat = repeat

        self.parent_path = os.path.dirname(os.path.realpath(__file__))
        self.img_path = '{}/data/WIDER_{}/images'.format(self.parent_path, self.subset)

        # read all ground truth
        self.gt_dir = '{}/data/eval_tools/ground_truth'.format(self.parent_path)
        self.gt_mat = loadmat(os.path.join(self.gt_dir, 'wider_face_val.mat'))
        self.facebox_list = self.gt_mat['face_bbx_list']
        self.event_list = self.gt_mat['event_list']
        self.file_list = self.gt_mat['file_list']
        # read hard set
        self.hard_mat = loadmat(os.path.join(self.gt_dir, 'wider_hard_val.mat'))
        self.hard_gt_list = self.hard_mat['gt_list']
        # read medium set
        self.medium_mat = loadmat(os.path.join(self.gt_dir, 'wider_medium_val.mat'))
        self.medium_gt_list = self.medium_mat['gt_list']
        # read easy set
        self.easy_mat = loadmat(os.path.join(self.gt_dir, 'wider_easy_val.mat'))
        self.easy_gt_list = self.easy_mat['gt_list']

        self.settings = ['easy', 'medium', 'hard']
        self.setting_gts = [self.easy_gt_list, self.medium_gt_list, self.hard_gt_list]

    def benchmark(self, model):
        # Collect all image names
        pred = dict()
        img_list = list()

        # Forward and collect results
        if self.test_list is None:
            for idx_e, event_name in enumerate(self.event_list):
                event_name = str(event_name[0][0])
                pred[event_name] = dict()
                for idx_f, file_name in enumerate(self.file_list[idx_e][0]):
                    file_name = str(file_name[0][0]) # without suffix
                    img_list.append((event_name, file_name))
                    pred[event_name][file_name] = []

            pbar = tqdm.tqdm(img_list)
            for event_name, file_name in pbar:
                img = cv.imread(os.path.join(self.img_path, '{}/{}.jpg'.format(event_name, file_name)))
                result = model.infer(img)
                pred[event_name][file_name] = result

            self._evaluate(pred)
        else:
            tm = cv.TickMeter()
            avg_infer_time = dict()
            for img_name in self.test_list:
                img = cv.imread(os.path.join(self.img_path, img_name))
                for target_size in self.test_size:
                    infer_time = []
                    pbar = tqdm.tqdm(range(self.repeat))
                    for _ in pbar:
                        pbar.set_description('Benchmarking on {} of size {}'.format(img_name, str(target_size)))
                        tm.start()
                        result = model.infer(img, target_size)
                        tm.stop()
                        infer_time.append(tm.getTimeMilli())
                        tm.reset()

                    avg_infer_time[str(target_size)] = sum(infer_time) / self.repeat

            print(avg_infer_time)



    def _evaluate(self, pred, thresh_num=1000):
        self._norm_score(pred)

        event_num = len(self.event_list)
        aps = []
        for setting_id in range(3):
            # different setting
            gt_list = self.setting_gts[setting_id]
            count_face = 0
            pr_curve = np.zeros((thresh_num, 2)).astype('float')
            # [hard, medium, easy]
            pbar = tqdm.tqdm(range(event_num))
            for i in pbar:
                pbar.set_description('Processing {}'.format(self.settings[setting_id]))
                event_name = str(self.event_list[i][0][0])
                img_list = self.file_list[i][0]
                pred_list = pred[event_name]
                sub_gt_list = gt_list[i][0]
                gt_bbx_list = self.facebox_list[i][0]

                for j in range(len(img_list)):
                    pred_info = pred_list[str(img_list[j][0][0])]

                    gt_boxes = gt_bbx_list[j][0].astype('float')
                    keep_index = sub_gt_list[j][0]
                    count_face += len(keep_index)

                    if len(gt_boxes) == 0 or len(pred_info) == 0:
                        continue
                    ignore = np.zeros(gt_boxes.shape[0])
                    if len(keep_index) != 0:
                        ignore[keep_index-1] = 1
                    pred_recall, proposal_list = self._image_eval(pred_info, gt_boxes, ignore, self.iou_threshold)

                    _img_pr_info = self._img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                    pr_curve += _img_pr_info
            pr_curve = self._dataset_pr_info(thresh_num, pr_curve, count_face)

            propose = pr_curve[:, 0]
            recall = pr_curve[:, 1]

            ap = self._voc_ap(recall, propose)
            aps.append(ap)

        print("==================== Results ====================")
        print("Easy   Val AP: {}".format(aps[0]))
        print("Medium Val AP: {}".format(aps[1]))
        print("Hard   Val AP: {}".format(aps[2]))
        print("=================================================")
        return aps

    def _norm_score(self, pred):
        """ norm score
        pred {key: [[x1,y1,x2,y2,s]]}
        """

        max_score = 0
        min_score = 1

        for _, k in pred.items():
            for _, v in k.items():
                if len(v) == 0:
                    continue
                _min = np.min(v[:, -1])
                _max = np.max(v[:, -1])
                max_score = max(_max, max_score)
                min_score = min(_min, min_score)

        diff = max_score - min_score
        for _, k in pred.items():
            for _, v in k.items():
                if len(v) == 0:
                    continue
                v[:, -1] = (v[:, -1] - min_score)/diff

    def _bbox_overlaps(self, box_a, box_b):
        # intersection
        num_box_a = box_a.shape[0]
        num_box_b = box_b.shape[0]
        tl = np.maximum(
            np.broadcast_to(np.expand_dims(box_a[:, 0:2], 1), (num_box_a, num_box_b, 2)),
            np.broadcast_to(np.expand_dims(box_b[:, 0:2], 0), (num_box_a, num_box_b, 2))
        )
        br = np.minimum(
            np.broadcast_to(np.expand_dims(box_a[:, 2:4], 1), (num_box_a, num_box_b, 2)),
            np.broadcast_to(np.expand_dims(box_b[:, 2:4], 0), (num_box_a, num_box_b, 2))
        )
        diff = np.clip((br - tl), a_min=0, a_max=None)
        inter = diff[:, :, 0] * diff[:, :, 1]

        # union
        area_a = np.broadcast_to(np.expand_dims((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]), 1), (num_box_a, num_box_b))
        area_b = np.broadcast_to(np.expand_dims((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]), 0), (num_box_a, num_box_b))
        union = area_a + area_b - inter

        return inter / union

    def _image_eval(self, pred, gt, ignore, iou_thresh):
        """ single image evaluation
        pred: Nx5
        gt: Nx4
        ignore:
        """

        _pred = pred.copy()
        _gt = gt.copy()
        pred_recall = np.zeros(_pred.shape[0])
        recall_list = np.zeros(_gt.shape[0])
        proposal_list = np.ones(_pred.shape[0])

        _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
        _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
        _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
        _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

        # overlaps = bbox_overlaps(_pred[:, :4], _gt)
        overlaps = self._bbox_overlaps(_pred[:, :4], _gt)

        for h in range(_pred.shape[0]):

            gt_overlap = overlaps[h]
            max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
            if max_overlap >= iou_thresh:
                if ignore[max_idx] == 0:
                    recall_list[max_idx] = -1
                    proposal_list[h] = -1
                elif recall_list[max_idx] == 0:
                    recall_list[max_idx] = 1

            r_keep_index = np.where(recall_list == 1)[0]
            pred_recall[h] = len(r_keep_index)
        return pred_recall, proposal_list

    def _img_pr_info(self, thresh_num, pred_info, proposal_list, pred_recall):
        pr_info = np.zeros((thresh_num, 2)).astype('float')
        for t in range(thresh_num):

            thresh = 1 - (t+1)/thresh_num
            r_index = np.where(pred_info[:, -1] >= thresh)[0]
            if len(r_index) == 0:
                pr_info[t, 0] = 0
                pr_info[t, 1] = 0
            else:
                r_index = r_index[-1]
                p_index = np.where(proposal_list[:r_index+1] == 1)[0]
                pr_info[t, 0] = len(p_index)
                pr_info[t, 1] = pred_recall[r_index]
        return pr_info

    def _dataset_pr_info(self, thresh_num, pr_curve, count_face):
        _pr_curve = np.zeros((thresh_num, 2))
        for i in range(thresh_num):
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
            _pr_curve[i, 1] = pr_curve[i, 1] / count_face
        return _pr_curve

    def _voc_ap(self, rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

if __name__ == '__main__':
    import sys
    sys.path.append('../../../')
    import models
    model = models.YuNet('../../../models/face_detection_yunet/face_detection_yunet.onnx', '', ['loc', 'conf', 'iou'])
    model.setBackend(cv.dnn.DNN_BACKEND_CUDA)
    model.setTarget(cv.dnn.DNN_TARGET_CUDA)


    dataset = WIDERFace()
    dataset.benchmark(model=model)