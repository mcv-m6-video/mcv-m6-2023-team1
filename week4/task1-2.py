import argparse
import sys
from src.utils import open_config_yaml
from src.optical_flow import read_optical_flow, report_errors_OF_2, visualize_error, histogram_error, draw_flow, \
    draw_hsv
import cv2


def main(cfg):
    mask, flow_gt = read_optical_flow(cfg["paths"]["gt"])
    prev = cv2.imread(cfg["paths"]["frame0"], cv2.IMREAD_GRAYSCALE)
    post = cv2.imread(cfg["paths"]["frame1"], cv2.IMREAD_GRAYSCALE)

    if cfg["method"] == "Franeback":
        flow_det = cv2.calcOpticalFlowFarneback(prev, post, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    elif cfg["method"] == "PyFlow":
        pass

    error, msen, pepn = report_errors_OF_2(flow_gt, flow_det, mask)
    visualize_error(error, mask)
    histogram_error(error)

    draw_flow(prev, flow_det)
    draw_hsv(flow_det, cfg["method"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1-2.yaml")
    # parser.add_argument("--config", default="week3/configs/task1.yaml") #comment if you are not using Visual Studio
    # Code
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
