import argparse
import os
import sys
from typing import Dict
import cv2
import numpy as np

import yaml
from week1.optical_flow import read_optical_flow, plot_flow, show_field, draw_opt_flow_magnitude_and_direction


def optical_flow_viz(config: Dict):
    # Unpack config
    gt_flow_path = config['gt_flow_path']
    lk_flow_path = config['lk_flow_path']
    img_path = config['img_path']
    img_1 = config['img_1']
    img_1_gt = config['img_1_gt']
    img_2 = config['img_2']
    img_2_gt = config['img_2_gt']

    # Group all paths
    img_paths = [os.path.join(img_path, img_1_gt), os.path.join(img_path, img_2_gt)]
    img_gt_paths = [os.path.join(gt_flow_path, img_1_gt), os.path.join(gt_flow_path, img_2_gt)]
    img_pred_paths = [os.path.join(lk_flow_path, img_1), os.path.join(lk_flow_path, img_2)]

    # Visualize
    for img_path, img_gt_path, img_pred_path in zip(img_paths, img_gt_paths, img_pred_paths):
        mask_img_gt, of_img_gt = read_optical_flow(img_gt_path)
        mask_img_pred, of_img_pred = read_optical_flow(img_pred_path)
        all_of_gt = np.dstack((of_img_gt, mask_img_gt))
        all_of_pred = np.dstack((of_img_pred, mask_img_pred))
        plot_flow(all_of_gt)
        plot_flow(all_of_pred)
        draw_opt_flow_magnitude_and_direction(all_of_gt)
        draw_opt_flow_magnitude_and_direction(all_of_pred)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        show_field(all_of_gt, gray, step=20, scale=.1)
        show_field(all_of_pred, gray, step=20, scale=.1)
        if img_path == img_paths[1]:
            show_field(all_of_gt[100:250, 450:600, :], gray[100:250, 450:600], step=10, scale=0.2)
            draw_opt_flow_magnitude_and_direction(all_of_gt[50:300, 400:650, :])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task4.yaml")
    args = parser.parse_args(sys.argv[1:])
    cfg_path = args.config

    # Read Config file
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    optical_flow_viz(cfg)
