import argparse
import os
import sys
from typing import Dict
import cv2
import numpy as np

import yaml
from week1.optical_flow import read_optical_flow, plot_flow, show_field, plot_flow_to_color


def optical_flow_viz(config: Dict):
    gt_flow_path = config['gt_flow_path']
    lk_flow_path = config['lk_flow_path']
    img_path = config['img_path']
    img_1 = config['img_1']
    img_1_gt = config['img_1_gt']
    img_2 = config['img_2']
    img_2_gt = config['img_2_gt']

    img_1_path = os.path.join(img_path, img_1_gt)
    img_2_path = os.path.join(img_path, img_2_gt)
    img_1_gt_path = os.path.join(gt_flow_path, img_1_gt)
    img_2_gt_path = os.path.join(gt_flow_path, img_2_gt)
    img_1_pred_path = os.path.join(lk_flow_path, img_1)
    img_2_pred_path = os.path.join(lk_flow_path, img_2)

    mask_img_1_gt, of_img_1_gt = read_optical_flow(img_1_gt_path)
    mask_img_1_pred, of_img_1_pred = read_optical_flow(img_1_pred_path)
    all_of_gt_img_1 = np.dstack((of_img_1_gt, mask_img_1_gt))
    all_of_pred_img_1 = np.dstack((of_img_1_pred, mask_img_1_pred))
    plot_flow(all_of_gt_img_1)
    plot_flow(all_of_pred_img_1)

    mask_img_2_gt, of_img_2_gt = read_optical_flow(img_2_gt_path)
    mask_img_2_pred, of_img_2_pred = read_optical_flow(img_2_pred_path)
    all_of_gt_img_2 = np.dstack((of_img_2_gt, mask_img_2_gt))
    all_of_pred_img_2 = np.dstack((of_img_2_pred, mask_img_2_pred))
    plot_flow(all_of_gt_img_2)
    plot_flow(all_of_pred_img_2)

    gray_1 = cv2.imread(img_1_path, cv2.IMREAD_GRAYSCALE)
    show_field(all_of_gt_img_1, gray_1, step=20, scale=.1)
    show_field(all_of_pred_img_1, gray_1, step=20, scale=.1)

    gray_2 = cv2.imread(img_2_path, cv2.IMREAD_GRAYSCALE)
    show_field(all_of_gt_img_2, gray_2, step=20, scale=.1)
    show_field(all_of_pred_img_2, gray_2, step=20, scale=.1)

    show_field(all_of_gt_img_2[100:250, 450:600, :], gray_2[100:250, 450:600], step=5, scale=0.1)
    plot_flow_to_color(all_of_gt_img_2[100:250, 450:600, :2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task4.yaml")
    args = parser.parse_args(sys.argv[1:])
    cfg_path = args.config

    # Read Config file
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    optical_flow_viz(cfg)
