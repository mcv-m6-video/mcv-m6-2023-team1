import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from src.in_out import get_frames_paths_from_folder, extract_rectangles_from_csv
from src.utils import open_config_yaml, save_bboxes_to_file, draw_img_with_yoloresults, results_yolov8_to_bbox


def main(cfg):
    paths = cfg['paths']
    viz = cfg["visualization"]
    cameras_path = paths['extracted_camera_frames']
    camera_annots_path = paths['camera_annots']

    model = YOLO('best.pt')  # load an official model
    for camera in os.listdir(cameras_path):

        gt_bboxes = extract_rectangles_from_csv(f'{camera_annots_path}/{camera}/gt/gt.txt')
        frames = get_frames_paths_from_folder(input_path=f'{cameras_path}/{camera}')
        print("Number of frames: ", len(frames))
        dataset = []
        for f_id, frame in enumerate(frames):
            dataset_tuple = (None, frame)
            if f_id in gt_bboxes:
                dataset_tuple = (f_id, frame)
            dataset.append(dataset_tuple)

        bboxes = []
        results_model = []
        for i, frame in tqdm(enumerate(dataset)):

            img = cv2.imread(frame[1])
            results = model(img)
            bounding_boxes, yolo_results_np = results_yolov8_to_bbox(results, ["person"], 0.5)
            if viz["show_detection"]:
                mapping_ft_labels_to_coco = {'person': 'car', 'bicycle': 'bicycle'}
                mapping_ft_ids_to_coco = {0: 2, 1: 1}
                yolo_results_np[:, -1] = np.vectorize(mapping_ft_labels_to_coco.get)(yolo_results_np[:, -1])
                yolo_results_np[:, -2] = np.vectorize(mapping_ft_ids_to_coco.get)(yolo_results_np[:, -2])

                img_out = draw_img_with_yoloresults(img, yolo_results_np, ["car"], gt_bboxes[i], viz["show_gt"])
                cv2.imshow('frame2', img_out)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

            # save all the images to preds list
            results_model.append(yolo_results_np)
            bboxes.append(bounding_boxes)

        save_bboxes_to_file(bboxes, f"{paths['save_path']}/{camera}_yolov8_bboxes.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bboxes_per_camera.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
