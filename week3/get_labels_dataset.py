import cv2
import os
import numpy as np
import torch
from numpy.linalg import inv
from tqdm import tqdm
from src.utils import save_bboxes_to_file


def cwh_to_ltrb(box, as_integer: bool = True):
    """
    Transform detection from Center, Width, Height to Left-Top, Right-Bottom.

    :param box: Detection in format [center_x, center_y, width, height]
    :param as_integer: Round detection values to nearest integer

    :return: Bounding box in format [left_x, top_y, right_x, bottom_y]
    """
    x, y, w, h = box
    b = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    if as_integer:
        b = [int(x) for x in b]
    return b


def ltrb_get_center(box):
    """
    Get center of the input bounding box.

    :param box: Detection in format [left_x, top_y, right_x, bottom_y]

    :return: Center of detection box
    """
    x1, y1, x2, y2 = box
    x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
    return x, y


def detect_yolov5(res):
    new_res = []
    for r in res:
        x1, y1, x2, y2, conf, clss, _ = r
        # x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        # w, h = int(x2 - x1), int(y2 - y1)
        new_res.append([int(clss), conf, [x1, y1, x2, y2]])
    return new_res


def cvFormattoYolo(corner, H, W):
    bbox_W = corner[2] - corner[0]
    bbox_H = corner[3] - corner[1]
    center_bbox_x = (corner[0] + corner[2]) / 2
    center_bbox_y = (corner[1] + corner[3]) / 2
    return (
        round(center_bbox_x / W, 6),
        round(center_bbox_y / H, 6),
        round(bbox_W / W, 6),
        round(bbox_H / H, 6),
    )


def main():

    model = torch.hub.load("ultralytics/yolov5", "yolov5x")

    path_dataset_to_label_txt = "../dataset/w3_frames/"
    list_imgs_to_label = os.listdir(path_dataset_to_label_txt)
    #keep only ones ending in .jpg of list_imgs_to_label
    list_imgs_to_label = [x for x in list_imgs_to_label if x[-4:] == ".jpg"]


    detections_all_frames= []
    for img_path in tqdm(list_imgs_to_label):
        det_frame = []
        img = cv2.imread(path_dataset_to_label_txt + img_path)

        results = model(img, size=1080)  # includes NMS
        res = results.pandas().xyxy[0]
        detections = detect_yolov5(res.to_numpy())  # already in cwh

        # if txt exists, delete and do again
        if os.path.isdir(path_dataset_to_label_txt + img_path[:-4] + ".txt"):

            os.remove(path_dataset_to_label_txt + img_path[:-4] + ".txt")

        # write file
        f = open(path_dataset_to_label_txt + img_path[:-4] + ".txt", "w+")
        for det in detections:
            if det[0] != 2:
                continue
            if det[1] < 0.25:
                continue
            det_frame.append([det[2][0], det[2][1],det[2][2], det[2][3]])

            centerx, centery, boxw, boxh = cvFormattoYolo(
                det[2], img.shape[0], img.shape[1]
            )
            f.write(
                str(det[0])
                + " "
                + str(centerx)
                + " "
                + str(centery)
                + " "
                + str(boxw)
                + " "
                + str(boxh)
                + "\n"
            )
        f.close()
        detections_all_frames.append(det_frame)
    save_bboxes_to_file(detections_all_frames, "bboxes_own_GT.yaml")


if __name__ == "__main__":
    main()