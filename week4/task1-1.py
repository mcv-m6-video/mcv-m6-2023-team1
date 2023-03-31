import glob
import argparse
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.optical_flow import read_optical_flow, compute_error, compute_msen, compute_pepn, visualize_error, \
    histogram_error
from typing import Dict
from src.utils import open_config_yaml
from src.optical_flow import OF_block_matching, save_plot_OF
from src.metrics import mean_square_error, root_mean_square_error, mean_absolute_error, percentage_of_erroneous_pixels


def main(cfg: Dict):
    gt_path = cfg["paths"]["gt"]  # Path of the ground truths

    if not cfg["evaluate_own_OF"]:  # if false it will evaluate the kitti made on week1
        detections_path = cfg["paths"]["kitti"]  # Path of the detections
    else:  # Compute the optical flow with the block matching algorithm
        image45_10 = cv2.imread(cfg["paths"]["sequence45"] + "/000045_10.png", cv2.IMREAD_GRAYSCALE)
        image45_11 = cv2.imread(cfg["paths"]["sequence45"] + "/000045_11.png", cv2.IMREAD_GRAYSCALE)
        # flow_det = cv2.calcOpticalFlowFarneback(image45_10, image45_11, None, 0.5, 3, 15, 3, 5, 1.2, 0) #not block matching - “Two-Frame Motion Estimation Based on Polynomial Expansion” by Gunner Farneback in 2003.

        flow_det = OF_block_matching(image45_10, image45_11)
        save_plot_OF(flow_det, cfg["paths"]["block_matching"] + "/flow_blockmatching_000045_10.png")

    # Get the detections (in our case, the optical flow is just from sequence 45)
    if not cfg["evaluate_own_OF"]:
        detections = glob.glob(detections_path + "/*.png")
    else:
        detections = glob.glob(cfg["paths"]["block_matching"] + "/*.png")
    detections = [det.replace("\\", "/") for det in detections]  # for Windows users

    # Compute the metrics and plots for the detections
    for det_filename in detections:
        if not cfg["evaluate_own_OF"]:
            seq_n = det_filename.split("/")[-1].replace("LKflow_", "")
            _, flow_det = read_optical_flow(det_filename)
        else:
            seq_n = det_filename.split("/")[-1].replace("flow_blockmatching_", "")

        mask, flow_gt = read_optical_flow(f"{gt_path}/{seq_n}")
        flow_noc_det = flow_det[mask]
        flow_noc_gt = flow_gt[mask]
        error = compute_error(flow_noc_gt, flow_noc_det)
        msen = compute_msen(error)
        pepn = compute_pepn(error)

        print(f"Image {seq_n}")
        print(f"MSEN: {msen}")
        print(f"PEPN: {pepn}\n")

        visualize_error(error, mask)
        histogram_error(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1-1.yaml")
    # parser.add_argument("--config", default="week3/configs/task1.yaml") #comment if you are not using Visual Studio
    # Code
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
