import copy
import random
from typing import Dict

import yaml


def sort_dict(dictionary: Dict):
    """
    Sorts a dictionary by the key.
    :param dictionary: dictionary to sort
    :return: sorted dictionary
    """
    frames_num_str = list(dictionary.keys())
    frames_int = sorted(int(frame[2:]) for frame in frames_num_str)
    return {i: dictionary["f_" + str(i)] for i in frames_int}


def addNoise(gt_bboxes: Dict, res=[1920, 1080], randm=True, pos=False, max_pxdisplacement=10, size=False, max_scale=1.,
             min_scale=0.1, removebbox=False, ratio_removebbox=0.2, addbbox=False, ratio_addbbox=0.2):
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
    noisy_gtbb = copy.deepcopy(gt_bboxes)

    for bbox in noisy_gtbb:
        # bbox format = [left, top, right, bottom]
        if pos:
            if randm:
                offset_x = random.randint(-max_pxdisplacement, max_pxdisplacement)
                offset_y = random.randint(-max_pxdisplacement, max_pxdisplacement)
            else:
                offset_x = max_pxdisplacement
                offset_y = max_pxdisplacement

            if bbox[2] + offset_x < res[0] and bbox[0] + offset_x > 0:
                bbox[0] += offset_x
                bbox[2] += offset_x
            if bbox[3] + offset_y < res[1] and bbox[1] + offset_y > 0:
                bbox[1] += offset_y
                bbox[3] += offset_y

        if size:
            # find center of bbox
            center_x = ((bbox[2] - bbox[0]) / 2) + bbox[0]
            center_y = ((bbox[3] - bbox[1]) / 2) + bbox[1]

            # scale bbox
            scale_x = 0.
            scale_y = 0.
            if randm:
                scale_x = random.uniform(min_scale, max_scale)
                scale_y = random.uniform(min_scale, max_scale)
            else:
                scale_x = max_scale
                scale_y = max_scale

            if ((bbox[2] - center_x) * scale_x + center_x) < res[0]:
                bbox[0] = (bbox[0] - center_x) * scale_x + center_x
                bbox[2] = (bbox[2] - center_x) * scale_x + center_x
            if ((bbox[3] - center_y) * scale_y + center_y) < res[1]:
                bbox[1] = (bbox[1] - center_y) * scale_y + center_y
                bbox[3] = (bbox[3] - center_y) * scale_y + center_y

    if removebbox:
        num_bbox = int(len(gt_bboxes) * ratio_removebbox)
        for i in range(num_bbox):
            idx = random.randint(0, len(noisy_gtbb) - 1)
            del noisy_gtbb[idx]

    if addbbox:
        num_bbox = int(len(gt_bboxes) * ratio_addbbox)
        for i in range(num_bbox):
            width = random.randint(50, 300)
            height = int(random.uniform(0.6, 1.4) * width)
            left = random.randint(0, res[0] - width)
            top = random.randint(0, res[1] - height)
            right = left + width
            bottom = top + height
            noisy_gtbb.append([left, top, right, bottom])

    return noisy_gtbb


def addNoise_all_frames(frame_dict: Dict):
    """
    Adds noise to all the frames in the dictionary.
    :param frame_dict: dictionary of frames
    :return: dictionary of frames with noise
    """
    noisy_gt = {}
    for frame in frame_dict:
        noisy_gt_bbx = addNoise(
            frame_dict[frame],
            res=[1920, 1080],
            randm=True,
            pos=True,
            max_pxdisplacement=10,
            size=True,
            max_scale=1.,
            min_scale=0.1,
            removebbox=True,
            ratio_removebbox=0.2,
            addbbox=True,
            ratio_addbbox=0.2
        )
        noisy_gt[frame] = noisy_gt_bbx
    return noisy_gt


def open_config_yaml(cfg_path):
    """
    Open config file and return it as a dictionary
    :param cfg_path: path to config file
    :return: config dictionary
    """
    # Read Config file
    with open(cfg_path, 'r') as f:
        config_yaml = yaml.safe_load(f)

    return config_yaml
