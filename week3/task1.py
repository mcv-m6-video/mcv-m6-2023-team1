import argparse
import sys
# sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")) # comment if you are not using
# Visual Studio Code
from typing import Dict

import cv2
import torch
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
from tqdm import tqdm
from ultralytics import YOLO  # pip install ultralytics

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from src.in_out import extract_frames_from_video, get_frames_paths_from_folder, extract_rectangles_from_xml
from src.utils import open_config_yaml, results_yolov5_to_bbox, draw_img_with_yoloresults, save_bboxes_to_file, \
    results_yolov8_to_bbox, results_detectron2_to_bbox
from src.metrics import get_allFrames_ap, get_mIoU

setup_logger()


def task1(cfg: Dict):
    """
    For this task, you will implement a background estimation algorithm using a single Gaussian model.
    """
    paths = cfg["paths"]

    extract_frames_from_video(video_path=paths["video"], output_path=paths["extracted_frames"])
    gt_labels = extract_rectangles_from_xml(cfg["paths"]["annotations"])
    frames = get_frames_paths_from_folder(input_path=paths["extracted_frames"])
    print("Number of frames: ", len(frames))
    dataset = [(key, frames[key]) for key in gt_labels.keys()]

    # load model
    if cfg["model"]["name"] == "yolov5":
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
    if cfg["model"]["name"] == "yolov8":
        model = YOLO('yolov8x.pt')  # load an official model

    # if model is faster_cnn or retinaNet, we use detectron2 to load it
    if cfg["model"]["name"] == "faster_rcnn" or cfg["model"]["name"] == "retinanet":
        if cfg["model"]["name"] == "faster_rcnn":
            model_name = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
        if cfg["model"]["name"] == "retinanet":
            model_name = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'

        config = get_cfg()

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        config.merge_from_file(model_zoo.get_config_file(model_name))
        config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        model = DefaultPredictor(config)

    # keep only dataset of train video
    dataset = dataset[int(len(dataset) * 0.25):]
    gt_bboxes = [*gt_labels.values()]
    gt_test_bboxes = gt_bboxes[int(len(gt_bboxes) * 0.25):]

    bboxes = []
    results_model = []
    for i, frame in tqdm(enumerate(dataset)):

        img = cv2.imread(frame[1])
        if cfg["model"]["name"] == "yolov5" or cfg["model"]["name"] == "yolov8":
            # process frame with yolo
            results = model(img)
            if cfg["model"]["name"] == "yolov5":
                bounding_boxes, yolo_results_np = results_yolov5_to_bbox(results, ["car", "truck"],
                                                                         cfg["model"]["min_conf"])
            if cfg["model"]["name"] == "yolov8":
                bounding_boxes, yolo_results_np = results_yolov8_to_bbox(results, ["car", "truck"],
                                                                         cfg["model"]["min_conf"])

        if cfg["model"]["name"] == "faster_rcnn" or cfg["model"]["name"] == "retinanet":
            results = model(img)
            bounding_boxes, yolo_results_np = results_detectron2_to_bbox(results, ["car", "truck"],
                                                                         cfg["model"]["min_conf"])

        if cfg["visualization"]["show_detection"]:
            img_out = draw_img_with_yoloresults(img, yolo_results_np, ["car", "truck"], gt_test_bboxes[i],
                                                cfg["visualization"]["show_gt"])
            cv2.imshow('frame2', img_out)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        # save all the images to preds list
        results_model.append(yolo_results_np)
        bboxes.append(bounding_boxes)

    # save bboxes
    if cfg["bboxes"]["save"]:
        save_path = cfg["bboxes"]["path"]
        # save bboxes of yolo (List (frames) of List (each detections))
        save_bboxes_to_file(bboxes, save_path + cfg["bboxes"]["name"] + ".yaml")
        # a = load_bboxes_from_file(save_path + cfg["bboxes"]["name"] + ".yaml")

    # Compute mAP and mIoU
    if len(bboxes[0][0]) == 4:
        mAP = get_allFrames_ap(gt_test_bboxes, bboxes)
    else:
        mAP = get_allFrames_ap(gt_test_bboxes, bboxes, confidence=True)

    mIoU = get_mIoU(gt_test_bboxes, bboxes)
    print(f"mAP: {mAP}")
    print(f"mIoU: {mIoU}")

    # Save results
    dataset_files = [frame[1] for frame in dataset]
    # if cfg["visualization"]["save"]:
    #     save_results(bboxes, preds, gt_test_bboxes, dataset_files, multiply255=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1.yaml")
    # parser.add_argument("--config", default="week3/configs/task1.yaml") #comment if you are not using Visual Studio
    # Code
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    task1(config)
