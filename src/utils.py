import copy
import random
from typing import Dict
import cv2
import tqdm
import yaml
import numpy as np

ColorSpaceConverterMethod = {"RGB": cv2.COLOR_BGR2RGB,
                                      "HSV": cv2.COLOR_BGR2HSV,
                                      "YUV": cv2.COLOR_BGR2YUV,
                                      "XYZ": cv2.COLOR_BGR2XYZ}

ChannelsUsedFromColorSpace = {"RGB": [0,1,2],
                              "HSV": [0,1,2],
                              "YUV": [1,2],
                              "XYZ":[0,2]}
def get_coco_name_from_id(num):
    id_to_name = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush'
    }

    return id_to_name[num]

def results_yolov5_to_bbox(results, classes=["car"], min_conf=0.25):
    """
    Converts the results of the yolov5 model to a list of bounding boxes.
    :param results: results of the yolov5 model
    :param classes: classes to consider
    :param min_conf: minimum confidence to consider a detection
    :return: list of bounding boxes - top left bottom right
    """
    resultsyolo = results.pandas().xyxy[0]
    resultsyolo = resultsyolo.to_numpy()

    bboxes = []
    for res in resultsyolo:
        if str(res[-1]) in classes:
            if res[4] > min_conf:
                bboxes.append([int(res[0]), int(res[1]), int(res[2]), int(res[3]), float(res[4])])
    return bboxes, resultsyolo

def results_yolov8_to_bbox(results, classes=["car"], min_conf=0.25):
    """
    Converts the results of the yolov8 model to a list of bounding boxes.
    :param results: results of the yolov8 model
    :param classes: classes to consider
    :param min_conf: minimum confidence to consider a detection
    :return: list of bounding boxes - top left bottom right
    """
    bboxes = []
    bboxes_frame = results[0].boxes.cpu().numpy()
    for res in bboxes_frame:
        if str(get_coco_name_from_id(res.cls[-1])) in classes:
            if float(res.conf[-1]) > min_conf:
                bboxes.append([int(res.boxes[0][0]), int(res.boxes[0][1]), int(res.boxes[0][2]), int(res.boxes[0][3]), float(res.conf[-1])])

    #create yolov8_np_results, wich has to be an array for each bbox, with top left right bottom coordinates, confidence, class id and class name
    yolov8_np_results = []
    for res in bboxes_frame:
        if str(get_coco_name_from_id(res.cls[-1])) in classes:
            if float(res.conf[-1]) > min_conf:
                yolov8_np_results.append([int(res.boxes[0][0]), int(res.boxes[0][1]), int(res.boxes[0][2]), int(res.boxes[0][3]), float(res.conf[-1]), int(res.cls[-1]), str(get_coco_name_from_id(res.cls[-1]))])
    return bboxes, np.array(yolov8_np_results)


def draw_img_with_yoloresults(img, results, classes=["car"], gt_bboxes=None, show_gt=False):
    """
    Draws the results of the yolov5 model on the image.
    :param img: image to draw on
    :param results: results of the yolov5 model
    :param classes: list of classes to draw, defaults to ["car"]
    :param gt_bboxes: ground truth bounding boxes
    :param show_gt: whether to show the ground truth bounding boxes
    :return: image with the results drawn
    """
    # define colors for each class
    class_colors = {}#{"car": (0, 0, 255), "person": (255, 0, 0), "bicycle": (0, 255, 0), "motorcycle": (0, 255, 255), "truck": (255, 0, 255)}

    # paint image
    imgcopy = img.copy()

    # for each bbox
    for res in results:
        if str(res[-1]) in classes:
            # get the color for the class
            color = class_colors.get(str(res[-1]), (0, 255, 0))  # defaults to black if the class isn't in the dictionary

            # draw the bbox with the class color
            cv2.rectangle(imgcopy, (int(res[0]), int(res[1])), (int(res[2]), int(res[3])), color, 2)

            # draw the class label with the class color
            cv2.putText(imgcopy, str(res[-1]) + " " + str(res[4])[:5], (int(res[0]), int(res[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if show_gt:
        for gt_bbox in gt_bboxes:
            cv2.rectangle(imgcopy, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 0, 255), 2)


    return imgcopy

def save_bboxes_to_file(bboxes: Dict, path: str):
    """
    Saves the bounding boxes to a file.
    :param bboxes: dictionary of bounding boxes
    :param path: path to save the file
    """
    with open(path, "w") as f:
        yaml.dump(bboxes, f)

def load_bboxes_from_file(path: str):
    # open the YAML file for reading
    with open(path, 'r') as file:
        # load the YAML data from the file
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_data


def transform_color_space(frames: np.ndarray, color_space:str) -> np.ndarray:
    """
    Loads the images from the given paths into a numpy array.
    :param paths_to_images: list of paths to the images
    :param grayscale: 'rgb' or 'gray'
    :return: numpy array of images
    """
    images = []
    for frame in tqdm.tqdm(frames, desc="Images being preprocessed"):
        images.append(cv2.cvtColor(frame, ColorSpaceConverterMethod[color_space]))
    return np.array(images)

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
