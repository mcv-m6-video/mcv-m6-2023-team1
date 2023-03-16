import random
from typing import List

import numpy as np


def get_IoU(bbox_a: List, bbox_b: List):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    Args:
    - box_A bounding box a
    - boxb: bounding box b

    Returns:
    - The IoU value between the two bounding boxes.
    """

    # Get the coordinates of the bounding boxes
    x1_a, y1_a, x2_a, y2_a = bbox_a
    x1_b, y1_b, x2_b, y2_b = bbox_b

    # Compute the area of the intersection rectangle
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Compute the area of both bounding boxes
    bbox_a_area = (x2_a - x1_a + 1) * (y2_a - y1_a + 1)
    bbox_b_area = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)

    # Compute the IoU
    iou = intersection_area / float(bbox_a_area + bbox_b_area - intersection_area)

    return iou



def get_frame_mean_IoU(gt_bboxes, det_bboxes):
    used_gt_idxs = set()  # keep track of which ground truth boxes have already been used
    frame_iou = []
    for det_bbox in det_bboxes:
        max_iou = 0
        max_gt_idx = None
        for i, gt_bbox in enumerate(gt_bboxes):
            if i in used_gt_idxs:
                continue  # skip ground truth boxes that have already been used
            iou = get_IoU(gt_bbox, det_bbox[:4])
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = i
        if max_gt_idx is not None:
            used_gt_idxs.add(max_gt_idx)
            frame_iou.append(max_iou)
        else:  # False Positives (aka pred boxes that do not intersect with any gt box)
            frame_iou.append(0)
    # False negative: check if there are any ground truth boxes that were not used
    for i, gt_bbox in enumerate(gt_bboxes):
        if i not in used_gt_idxs:
            frame_iou.append(0)
    return np.mean(frame_iou)


def get_mIoU(gt_bboxes_dict, det_bboxes_dict):
    """
    Computes the mean Intersection over Union (mIoU) between two sets of bounding boxes.
    Args:
    - gt_bboxes_dict: dictionary of ground truth bounding boxes
    - det_bboxes_dict: dictionary of detected bounding boxes

    Returns:
    - The mIoU value between the two sets of bounding boxes.
    """

    # Initialize a list to hold the IoU values for each frame
    iou_list = []

    # Loop through each frame number in the ground truth dictionary
    for frame_num in gt_bboxes_dict:
        # Get the ground truth bounding boxes for the current frame
        gt_bboxes = gt_bboxes_dict[frame_num]

        # Get the detected bounding boxes for the current frame
        det_bboxes = det_bboxes_dict[frame_num]

        # Compute the IoU between the ground truth and detected bounding boxes
        frame_iou = get_frame_mean_IoU(gt_bboxes, det_bboxes)

        # Append the mean IoU value for the current frame to the list
        iou_list.append(frame_iou)

    # Compute the mean IoU value across all frames
    mIoU = np.mean(iou_list)

    return mIoU, iou_list


def ap_voc(frame_iou, total_gt, th):
    """
    Computes the Average Precision (AP) in a frame according to Pascal Visual Object Classes (VOC) Challenge.
    Args:
    - frame_iou: list with the IoU results of the detected bounding boxes
    - total_gt: int defining the number of bounding boxes in the ground truth
    - th: float defining a threshold of IoU metric. If IoU is higher than th,
          then the detected bb is a TP, otherwise is a FP.

    Returns:
    - The AP value of the bb detected.
    """
    total_det = len(frame_iou)
    # Define each detection if it is a true positive or a false positive
    tp = np.zeros(total_det)
    fp = np.zeros(total_det)
    for i in range(total_det):
        if frame_iou[i] > th:
            tp[i] = 1
        else:
            fp[i] = 1

    # Tabulate the cumulative sum of the true and false positives
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    # Compute Precision and Recall
    precision = tp / (tp + fp)  # cumulative true positives / cumulative true positive + cumulative false positives
    recall = tp / float(total_gt)  # cumulative true positives / total ground truths

    # AP measurement according to the equations 1 and 2 in page 11 of
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
    ap = 0.
    for r in np.arange(0, 1.1, 0.1):
        if any(recall >= r):
            ap += np.max(precision[recall >= r])

    ap = ap / 11
    return ap


def get_frame_ap(gt_bboxes, det_bboxes, confidence=False, n=10, th=0.5):
    """
    Computes the Average Precision (AP) in a frame according to Pascal Visual Object Classes (VOC) Challenge.
    Args:
    - gt_bboxes: dictionary of ground truth bounding boxes
    - det_bboxes: dictionary of detected bounding boxes
    - confidence: True if we have the confidence score
    - n: Number of random sorted sets.
    - th: float defining a threshold of IoU metric. If IoU is higher than th,
          then the detected bb is a TP, otherwise is a FP.

    Returns:
    - The AP value of the bb detected in the frame.
    """
    total_gt = len(gt_bboxes)

    # sort det_bboxes by confidence score in descending order
    det_bboxes.sort(reverse=True, key=lambda x: x[4])

    # Calculate the IoU of each detected bbox.
    # frame_iou = [max([get_IoU_boxa_boxb(gt_bbox, det_bbox[:4]) for gt_bbox in gt_bboxes], default=0) for det_bbox in det_bboxes]

    # V2 taking into account that each GT box can only be assigned to one predicted box
    frame_iou = [[get_IoU(gt_bbox, det_bbox[:4]) for gt_bbox in gt_bboxes] for det_bbox in det_bboxes]
    idx_assigned = []
    for i in range(len(frame_iou)):
        bb_assigned = False
        bb_num = 0  # To avoid an infinite while loop in case all the bboxes with IoU higher than 0 were assigned
        while (not bb_assigned) and (bb_num != (total_gt - 1)):
            idx_max = np.argmax(frame_iou[i])
            if any(idx_max == idx_assigned):
                frame_iou[i][idx_max] = 0
                bb_num += 1
            else:
                frame_iou[i] = frame_iou[i][idx_max]
                idx_assigned.append(idx_max)
                bb_assigned = True
        if not bb_assigned:
            frame_iou[i] = 0.

    # Compute the AP
    ap = 0.

    if confidence:
        ap = ap_voc(frame_iou, total_gt, th)
    else:
        # Generate N random sorted lists of the detections and compute the AP in each one
        ap_list = []
        for i in range(n):
            random.shuffle(frame_iou)
            ap_list.append(ap_voc(frame_iou, total_gt, th))

        # Do the average of the computed APs
        ap = np.mean(ap_list)

    return ap


def get_allFrames_ap(gt_bboxes_dict, det_bboxes_dict, confidence=False, n=10, th=0.5):
    """
    Computes the mean Average Precision (mAP) of all the detections done in the video according to Pascal Visual Object Classes (VOC) Challenge.
    Args:
    - gt_bboxes_dict: dictionary of ground truth bounding boxes
    - det_bboxes_dict: dictionary of detected bounding boxes
    - confidence: True if we have the confidence score
    - n: Number of random sorted sets.
    - th: float defining a threshold of IoU metric. If IoU is higher than th,
          then the detected bb is a TP, otherwise is a FP.

    Returns:
    - The mean of the AP in all the video frames.
    """

    # Initialize a list to hold the IoU values for each frame
    ap_list = []

    # Loop through each frame number in the ground truth dictionary
    for frame_num in gt_bboxes_dict:
        # Get the ground truth bounding boxes for the current frame
        gt_bboxes = gt_bboxes_dict[frame_num]

        # Get the detected bounding boxes for the current frame
        det_bboxes = det_bboxes_dict[frame_num]

        # Compute the mAP between the ground truth and detected bounding boxes
        frame_ap = get_frame_ap(gt_bboxes, det_bboxes, confidence, n, th)

        # Append the mean mAP value for the current frame to the list
        ap_list.append(frame_ap)

    # Compute the mean mAP value across all frames
    map = np.mean(ap_list)

    return map