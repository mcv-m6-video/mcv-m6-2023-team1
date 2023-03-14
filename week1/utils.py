import xml.etree.ElementTree as ET
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
from typing import List, Dict
import os
from matplotlib.animation import FuncAnimation

def extract_rectangles_from_csv(path, start_frame86=True):
    """
    Parses an XML annotation file in the csv format and extracts bounding box coordinates for cars in each frame.

    Args:
        - Path to annotation csv in AI City format
    returns:
        dict[frame_num] = [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
        in top left and bottom right coordinates
    """
    ret_dict = {}

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        # next(reader)  # skip header row

        for row in reader:
            frame_num = f'f_{int(row[0]) - 1}'

            if frame_num not in ret_dict:
                ret_dict[frame_num] = []

            # convert from center, width, height to top left, bottom right
            ret_dict[frame_num].append(
                [float(row[2]), float(row[3]), float(row[2]) + float(row[4]), float(row[3]) + float(row[5]),
                 float(row[6])]
            )

    return sort_dict(ret_dict)


def extract_rectangles_from_xml(path_to_xml_file):
    """
    Parses an XML annotation file in the Pascal VOC format and extracts bounding box coordinates for cars in each frame.
    Args:
    - path_to_xml_file: path to the annotation XML file.

    Returns:
    - A dictionary of frame numbers and their corresponding car bounding box coordinates.
        example: dict = {'f_0': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]]}
        where x1, y1, x2, y2 are the coordinates of the bounding box in the top left and bottom right corners.

    """

    tree = ET.parse(path_to_xml_file)
    root = tree.getroot()

    # Initialize an empty dictionary to hold the frame numbers and their corresponding bounding box coordinates
    frame_dict = {}

    # Loop through each 'track' element in the XML file with a 'label' attribute of 'car'
    for track in root.findall(".//track[@label='car']"):

        # Loop through each 'box' element within the 'track' element to get the bounding box coordinates
        for box in track.findall(".//box"):

            # Extract the bounding box coordinates and the frame number
            x1 = float(box.attrib['xtl'])
            y1 = float(box.attrib['ytl'])
            x2 = float(box.attrib['xbr'])
            y2 = float(box.attrib['ybr'])
            frame_num = f"f_{box.attrib['frame']}"

            # If the frame number doesn't exist in the dictionary yet, add it and initialize an empty list
            if frame_num not in frame_dict:
                frame_dict[frame_num] = []

            # Append the bounding box coordinates to the list for the current frame number
            frame_dict[frame_num].append([x1, y1, x2, y2])

    return sort_dict(frame_dict)


def get_IoU_boxa_boxb(boxa, boxb):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    Args:
    - boxa: bounding box a
    - boxb: bounding box b

    Returns:
    - The IoU value between the two bounding boxes.
    """

    # Get the coordinates of the bounding boxes
    x1_a, y1_a, x2_a, y2_a = boxa
    x1_b, y1_b, x2_b, y2_b = boxb

    # Compute the area of the intersection rectangle
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Compute the area of both bounding boxes
    boxa_area = (x2_a - x1_a + 1) * (y2_a - y1_a + 1)
    boxb_area = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)

    # Compute the IoU
    iou = intersection_area / float(boxa_area + boxb_area - intersection_area)

    return iou


# def get_frame_mean_IoU(gt_bboxes, det_bboxes): #THIS ONE DID'T TAKE INTO ACCOUNT GT USED
#     frame_iou = [max([get_IoU_boxa_boxb(gt_bbox, det_bbox[:4]) for det_bbox in det_bboxes], default=0) for gt_bbox in
#                  gt_bboxes]
#     return np.mean(frame_iou)

def get_frame_mean_IoU(gt_bboxes, det_bboxes):
    used_gt_idxs = set()  # keep track of which ground truth boxes have already been used
    frame_iou = []
    for det_bbox in det_bboxes:
        max_iou = 0
        max_gt_idx = None
        for i, gt_bbox in enumerate(gt_bboxes):
            if i in used_gt_idxs:
                continue  # skip ground truth boxes that have already been used
            iou = get_IoU_boxa_boxb(gt_bbox, det_bbox[:4])
            if iou > max_iou:
                max_iou = iou
                max_gt_idx = i
        if max_gt_idx is not None:
            used_gt_idxs.add(max_gt_idx)
            frame_iou.append(max_iou)
        else: #IF NO GT BOXES ARE FOUND, THEN THE DETECTION IS A FALSE POSITIVE. Penalize it by giving it an IoU of 0
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

    # Tabulate the cummulative sum of the true and false positives
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    # Compute Precision and Recall
    precision = tp / (tp + fp)  # cummulative true positives / cummulative true positive + cummulative false positives
    recall = tp / float(total_gt)  # cummulative true positives / total ground truths

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
    #frame_iou = [max([get_IoU_boxa_boxb(gt_bbox, det_bbox[:4]) for gt_bbox in gt_bboxes], default=0) for det_bbox in det_bboxes]

    # V2 taking into account that each GT box can only be assigned to one predicted box
    frame_iou = [[get_IoU_boxa_boxb(gt_bbox, det_bbox[:4]) for gt_bbox in gt_bboxes] for det_bbox in det_bboxes]
    idx_assigned = []
    for i in range(len(frame_iou)):
        bb_assigned = False
        bb_num = 0 #To avoid an infinit while loop in case all the bboxes with IoU higher than 0 were assigned
        while ((not bb_assigned) and (bb_num != (total_gt-1))):
            idx_max = np.argmax(frame_iou[i])
            if any(idx_max == idx_assigned):
                frame_iou[i][idx_max] = 0
                bb_num +=1
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


def plot_frame(frame, gt_rects, det_rects, path_to_video, frame_iou=None, save_frame = False, file_path = None):
    """
    Plots the frame and the ground truth bounding boxes.
    Args:
    - frame: frame number
    - GT_rects: list of bounding box coordinates
    - path_to_video: path to the video file
    """
    frame_str_num = frame

    # Read the video file
    cap = cv2.VideoCapture(path_to_video)

    # Set the frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))

    # Read the frame
    ret, frame = cap.read()

    # Plot the frame in RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)

    # Plot the bounding boxes
    for rect in gt_rects:
        x1, y1, x2, y2 = rect
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=3))
    for rect in det_rects:
        x1, y1, x2, y2, conf = rect
        # plot rectangle and confidence on top
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='green', linewidth=2))
        plt.gca().text(x1, y1, 'Conf: {:.3f}'.format(conf), bbox=dict(facecolor='green', alpha=0.5), fontsize=6)

    if frame_iou is not None:
        plt.title(f"Frame {frame_str_num} IoU: {frame_iou:.3f}")
    if not save_frame:
        plt.show()
    elif save_frame:
        plt.savefig(file_path)

def plot_iou_vs_frames(ious_list:List, file_path=None, save_fig=False) -> None:
    """
    Plots the graph of iou over the frames for a sequence.
    Parameters
    ----------
    ious_list :

    Returns
    -------

    """

    #Plot the iou during the frames in a fixed single plot

    frames = range(len(ious_list))
    fig, ax = plt.subplots()
    ax.plot(frames, ious_list, linewidth=0.5)
    ax.set(xlim=(0, 2140), xticks=np.arange(0, 2140,250),
           ylim=(0, 1), yticks=np.arange(0, 1, 0.1))
    if not save_fig:
        plt.show()
    elif save_fig:
        plt.savefig(file_path)

def sort_dict(dict):
    frames_num_str = list(dict.keys())
    frames_int = sorted(int(frame[2:]) for frame in frames_num_str)
    return {i: dict["f_"+str(i)] for i in frames_int}

def make_gif(gt_bboxes_dict, det_bboxes_dict, cfg):
    ious_list = []
    ious_files_prefix = "gif_images/iou_vs_frames_plots"
    detection_files_prefix = "gif_images/detection_plots"
    if not os.path.exists("gif_images"):
        os.mkdir("gif_images")
        os.mkdir(ious_files_prefix)
        os.mkdir(detection_files_prefix)
    for frame in gt_bboxes_dict:
        ious_filepath = os.path.join(ious_files_prefix, str(frame)+".png")
        detection_filepath = os.path.join(detection_files_prefix, str(frame)+".png")

        plot_frame(frame, gt_bboxes_dict[frame], det_bboxes_dict[frame], cfg["paths"]["video"], save_frame=True, file_path=detection_filepath)
        plot_iou_vs_frames(ious_list, file_path=ious_filepath, save_fig=True )


def addNoise(gt_bboxes, res = [1920, 1080], pos=False, max_pxdisplacement = 10, size=False, max_scale = 1., min_scale = 0.1, removebbox=False, ratio_removebbox = 0.2, addbbox = False, ratio_addbbox = 0.2):
    """
    Adds noise to the ground truth bounding boxes changing their size, position and adding/removing them.
    Args:
    - gt_bboxes: dictionary of bounding boxes ground truth
    - res: 2D list that specifies the width and height of the frame.
    - pos: bool to enable the random change of position - Default value: False
    - max_pxdisplacement: int defining the maximum pixels we allow to move the bounding boxes - Default value: 10 pixels
    - size: bool to enable the random change of bounding boxes size - Default value: False
    - max_scale: float that defines the maximum scale we want to apply to the bounding boxes - Default value: 1. (no scaling)
    - min_scale: float that defines the minimum scale we want to apply to the bounding boxes - Default value: 0.1 (10 times reduced the size of the bbox)
    - removebbox: bool that enables the reduction of the quantity of bounding boxes in the ground truth - Default value: False
    - ratio_removebbox: float that defines the percentage of bounding boxes we want to remove - Default value: 0.2 (20%)
    - addbox: bool that enables the generation of new bounding boxes in the gound truth - Default value: False
    - ratio_addbbox: float that defines the percentage of bounding boxes we want to add - Default value: 0.2 (20%) 
    """
    noisy_gtbb = gt_bboxes

    for bbox in noisy_gtbb:
        #bbox format = [left, top, right, bottom]
        if pos:
            offset_x = random.randint(-max_pxdisplacement, max_pxdisplacement)
            offset_y = random.randint(-max_pxdisplacement, max_pxdisplacement)

            if bbox[2]+offset_x < res[0] and bbox[0]+offset_x > 0:
                bbox[0] += offset_x
                bbox[2] += offset_x
            if bbox[3]+offset_y < res[1] and bbox[1]+offset_y > 0:
                bbox[1] += offset_y
                bbox[3] += offset_y

        if size:
            #find center of bbox
            center_x = ((bbox[2] - bbox[0]) / 2) + bbox[0]
            center_y = ((bbox[3] - bbox[1]) / 2) + bbox[1]

            #scale bbox
            scale_x = random.uniform(min_scale, max_scale)
            scale_y = random.uniform(min_scale, max_scale)

            if ((bbox[2]-center_x) * scale_x + center_x) < res[0]:
                bbox[0] = (bbox[0]-center_x) * scale_x + center_x
                bbox[2] = (bbox[2]-center_x) * scale_x + center_x
            if ((bbox[3]-center_y) * scale_y + center_y) < res[1]:
                bbox[1] = (bbox[1]-center_y) * scale_y + center_y
                bbox[3] = (bbox[3]-center_y) * scale_y + center_y

    if removebbox:
        num_bbox = int(len(gt_bboxes)*ratio_removebbox)
        for i in range(num_bbox):
            idx = random.randint(0, len(noisy_gtbb)-1)
            del noisy_gtbb[idx]
    
    if addbbox:
        num_bbox = int(len(gt_bboxes)*ratio_addbbox)
        for i in range(num_bbox):
            width = random.randint(50, 300)
            height = int(random.uniform(0.6, 1.4)*width)
            left = random.randint(0, res[0]- width)
            top = random.randint(0, res[1]- height)
            right = left + width
            bottom = top + height            
            noisy_gtbb.append([left, top, right, bottom])
    
    return noisy_gtbb

    
