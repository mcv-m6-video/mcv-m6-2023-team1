import argparse
import os
import sys
from typing import Dict
from optical_flow_utils import *

import yaml


def optical_flow_viz(config: Dict):
    gt_flow_path = config['gt_flow_path']
    lk_flow_path = config['lk_flow_path']
    img_path = config['img_path']
    img_1 = config['img_1']
    img_2 = config['img_2']

    optical_flow = read_flow(os.path.join(lk_flow_path, img_1))
    plot_flow(optical_flow)

    optical_flow = read_flow(os.path.join(gt_flow_path, img_2))
    plot_flow(optical_flow)

    o_flow = read_flow(os.path.join(lk_flow_path, img_1))
    gray = cv2.imread(os.path.join(img_path, img_1[7:]), cv2.IMREAD_GRAYSCALE)
    show_field(o_flow, gray, step=20, scale=.1)

    o_flow = read_flow(os.path.join(gt_flow_path, img_2))
    gray = cv2.imread(os.path.join(img_path, img_2), cv2.IMREAD_GRAYSCALE)
    show_field(o_flow, gray, step=5, scale=.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task4.yaml")
    args = parser.parse_args(sys.argv[1:])
    cfg_path = args.config

    # Read Config file
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    optical_flow_viz(cfg)
