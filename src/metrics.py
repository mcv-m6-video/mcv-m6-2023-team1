import random
from typing import List, Tuple, Dict

import cv2
import numpy as np

from src.background_estimation import BackgroundEstimator


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

def get_frame_IoU(gt_bboxes, det_bboxes):
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
    return frame_iou

def get_frame_mean_IoU(gt_bboxes, det_bboxes):
    if len(gt_bboxes) == 0:
        return None
    return np.mean(get_frame_IoU(gt_bboxes, det_bboxes))


def get_mIoU(gt_bboxes_dict, det_bboxes_dict):
    """
    Computes the mean Intersection over Union (mIoU) between two sets of bounding boxes.
    Args:
    - gt_bboxes_dict: dictionary or list of ground truth bounding boxes
    - det_bboxes_dict: dictionary or list of detected bounding boxes

    Returns:
    - The mIoU value between the two sets of bounding boxes.
    """
    if isinstance(gt_bboxes_dict, list):
        gt_bboxes_dict = {i: v for i, v in enumerate(gt_bboxes_dict)}
    if isinstance(det_bboxes_dict, list):
        det_bboxes_dict = {i: v for i, v in enumerate(det_bboxes_dict)}

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
        if frame_iou is not None:
            iou_list.append(frame_iou)

    # Compute the mean IoU value across all frames
    mIoU = np.mean(iou_list)

    return mIoU, iou_list


def ap_voc(frame_iou, total_det, total_gt, th):
    """
    Computes the Average Precision (AP) in a frame according to Pascal Visual Object Classes (VOC) Challenge.
    Args:
    - frame_iou: list with the IoU results of each ground truth bbox
    - total_det: int defining the number of bounding boxes detected
    - th: float defining a threshold of IoU metric. If IoU is higher than th,
          then the detected bb is a TP, otherwise is a FP.

    Returns:
    - The AP value of the bb detected.
    """
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
    precision = tp / np.maximum((tp + fp),  np.finfo(np.float64).eps)  # cumulative true positives / cumulative true positive + cumulative false positives
    recall = tp / float(total_gt)  # cumulative true positives / total ground truths

    if total_det < total_gt:
        precision = np.append(precision, 0.0)
        recall = np.append(recall, 1.0)
    
    # AP measurement according to the equations 1 and 2 in page 11 of
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
    ap = 0.0
    for r in np.arange(0.0, 1.1, 0.1):
        if any(recall >= r):
            max_precision = np.max(precision[recall >= r])
            ap = ap + max_precision

    ap = ap / 11.0
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
    total_gt = len(gt_bboxes) # Number of bboxes in the ground truth
    total_det = len(det_bboxes) # Number of bboxes in the predictions

    if total_gt == 0:
        # if we don't have any ground truth in the frame and also we don't have any prediction, we assume it's corret and we skip the frame 
        if total_det == 0:
            return None
        # if we don't have any ground truth in the frame but we have predictions, we assume they are false positives and therefore the mAP is equal to 0
        else:
            return 0.
    
    ap = 0.
    if confidence:
        # sort det_bboxes by confidence score in descending order
        det_bboxes.sort(reverse=True, key=lambda x: x[4])

        # Calculate the IoU of each detected bbox.
        frame_iou = get_frame_IoU(gt_bboxes, det_bboxes)[:total_det]

        #  Compute the AP
        ap = ap_voc(frame_iou, total_det, total_gt, th)
    else:
        # Generate N random sorted lists of the detections and compute the AP in each one
        ap_list = []
        for i in range(n):
            # sort randomly the det_bboxes
            random.shuffle(det_bboxes)

            # Calculate the IoU of each detected bbox.
            frame_iou = get_frame_IoU(gt_bboxes, det_bboxes)[:total_det]

            #  Compute the AP
            ap_list.append(ap_voc(frame_iou, total_det, total_gt, th))

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
    if isinstance(gt_bboxes_dict, list):
        gt_bboxes_dict = {i: v for i, v in enumerate(gt_bboxes_dict)}
    if isinstance(det_bboxes_dict, list):
        det_bboxes_dict = {i: v for i, v in enumerate(det_bboxes_dict)}
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
        if frame_ap is not None:
            ap_list.append(frame_ap)

    # Compute the mean mAP value across all frames
    map = np.mean(ap_list)

    return map


def evaluate_background_estimator(data: List[Tuple[str, Dict]], bg_estimator: BackgroundEstimator, plot: bool = False):
    """
    Evaluates the background estimator on the given dataset.
    :param data: List of image path and ground truth bounding boxes.
    :param bg_estimator: Background estimator to evaluate.
    :param plot: If True, plots the results.
    """
    for file_path, annotations in data:
        image = cv2.imread(file_path)
        raw_bg_estimation = bg_estimator.predict(image)
        processed_bg_estimation = process_bg_estimation(raw_bg_estimation)
        pred_bboxes = get_predicted_bboxes(processed_bg_estimation)

def mean_square_error(flow_gt, flow_pred):
    """
    Computes the mean square error between the ground truth and predicted optical flow
    """
    error = np.linalg.norm(flow_gt - flow_pred) ** 2 / flow_gt.size
    return error


def mean_absolute_error(flow_gt, flow_pred):
    """
    Computes the mean absolute error between the ground truth and predicted optical flow
    """
    error = np.sum(np.abs(flow_gt - flow_pred)) / flow_gt.size
    return error


def root_mean_square_error(flow_gt, flow_pred):
    """
    Computes the root mean square error between the ground truth and predicted optical flow
    """
    error = np.sqrt(np.linalg.norm(flow_gt - flow_pred) ** 2 / flow_gt.size)
    return error


def percentage_of_erroneous_pixels(flow_gt, flow_pred, threshold=3.0):
    """
    Computes the percentage of erroneous pixels between the ground truth and predicted optical flow
    """
    error = np.linalg.norm(flow_gt - flow_pred, axis=2)
    num_errors = np.sum(error > threshold)
    num_pixels = flow_gt.shape[0] * flow_gt.shape[1]
    pepn = num_errors / num_pixels
    return pepn