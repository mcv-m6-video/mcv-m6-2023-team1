import argparse
import sys
from typing import Dict
import cv2
import numpy as np
from tqdm import tqdm
import time

from src.background_estimation import get_background_estimator
from src.in_out import extract_frames_from_video, get_frames_paths_from_folder, extract_rectangles_from_xml, load_images, extract_not_parked_rectangles_from_xml
from src.utils import open_config_yaml
from src.plotting import save_results
from src.metrics import get_allFrames_ap, get_mIoU


def task1(cfg: Dict):
    """
    For this task, you will implement a background estimation algorithm using a single Gaussian model.
    """
    paths = cfg["paths"]
    bg_estimator_config = cfg["background_estimator"]

    extract_frames_from_video(video_path=paths["video"], output_path=paths["extracted_frames"])
    gt_labels = extract_rectangles_from_xml(cfg["paths"]["annotations"])
    # gt_labels = extract_not_parked_rectangles_from_xml(cfg["paths"]["annotations"])
    frames = get_frames_paths_from_folder(input_path=paths["extracted_frames"])
    print("Number of frames: ", len(frames))
    dataset = [(key, frames[key])for key in gt_labels.keys()]


    #keep only dataset of train video
    dataset = dataset[int(len(dataset)*0.25):]

    preds = []
    bboxes = []
    for i, frame in tqdm(enumerate(dataset)):

        img = cv2.imread(frame[1])

        if cfg["visualization"]["show_before"]:
            cv2.imshow('frame2', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break


        if cfg["visualization"]["show_after"]:
            cv2.imshow('frame2', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        #save all the images to preds list
        # preds.append(mask)
        # bboxes.append(bounding_boxes)


    gt_bboxes = [*gt_labels.values()]
    first_test_idx = len(gt_labels) - len(bboxes)
    gt_test_bboxes = gt_bboxes[first_test_idx:]

    # Compute mAP and mIoU
    mAP = get_allFrames_ap(gt_test_bboxes, bboxes)
    mIoU = get_mIoU(gt_test_bboxes, bboxes)
    print(f"mAP: {mAP}")
    print(f"mIoU: {mIoU}")

    #Save results
    dataset_files = [frame[1] for frame in dataset]
    if cfg["visualization"]["save"]:
        save_results(bboxes, preds, gt_test_bboxes, dataset_files, multiply255=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task3.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    task1(config)