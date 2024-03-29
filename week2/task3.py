import argparse
import sys
from typing import Dict
import cv2
import numpy as np
from tqdm import tqdm

from src.background_estimation import get_background_estimator
from src.in_out import extract_frames_from_video, get_frames_paths_from_folder, extract_rectangles_from_xml, load_images, extract_not_parked_rectangles_from_xml
from src.utils import open_config_yaml
from src.plotting import save_results
from src.metrics import get_allFrames_ap, get_mIoU


def task3(cfg: Dict):
    """
    For this task, you will implement a background estimation algorithm using a single Gaussian model.
    """
    paths = cfg["paths"]
    bg_estimator_config = cfg["background_estimator"]

    extract_frames_from_video(video_path=paths["video"], output_path=paths["extracted_frames"])
    # gt_labels = extract_rectangles_from_xml(cfg["paths"]["annotations"])
    gt_labels = extract_not_parked_rectangles_from_xml(cfg["paths"]["annotations"])
    frames = get_frames_paths_from_folder(input_path=paths["extracted_frames"])
    print("Number of frames: ", len(frames))
    dataset = [(key, frames[key])for key in gt_labels.keys()]


    if bg_estimator_config["estimator"] == "MOG2":
        estimator = cv2.createBackgroundSubtractorMOG2()
    elif bg_estimator_config["estimator"] == "KNN":
        estimator = cv2.createBackgroundSubtractorKNN()
    #pip install opencv-contrib-python
    elif bg_estimator_config["estimator"] == "LSBP":
        estimator = cv2.bgsegm.createBackgroundSubtractorLSBP()
    elif bg_estimator_config["estimator"] == "GSOC":
        estimator = cv2.bgsegm.createBackgroundSubtractorGSOC()
    elif bg_estimator_config["estimator"] == "CNT":
        estimator = cv2.bgsegm.createBackgroundSubtractorCNT()

    #keep only dataset of test video to compare with task1
    dataset = dataset[int(len(dataset)*0.25):]

    preds = []
    bboxes = []
    for i, frame in tqdm(enumerate(dataset)):

        img = cv2.imread(frame[1])
        fgmask = estimator.apply(img)

        if cfg["visualization"]["show_before"]:
            cv2.imshow('frame', fgmask)
            cv2.imshow('frame2', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        # threshold the foreground mask to get a binary image
        thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)[1]

        # perform morphological opening to remove small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        #now a closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


        # find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # create a mask of the same size as the image
        mask = np.zeros_like(img)

        # loop over contours and draw them on the mask
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 600:  # only draw contours with area greater than 1000 pixels
                cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
                # compute bounding box of contour
                x, y, w, h = cv2.boundingRect(contour)

                # draw bounding box on image
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                bounding_boxes.append((x, y, x + w, y + h))

        if cfg["visualization"]["show_after"]:
            cv2.imshow('frame', mask)
            cv2.imshow('frame2', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        #save all the images to preds list
        preds.append(mask)
        bboxes.append(bounding_boxes)


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

    task3(config)