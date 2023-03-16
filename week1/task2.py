import argparse
import random
import sys
from typing import Dict

from src.in_out import extract_rectangles_from_xml, extract_rectangles_from_csv
from src.metrics import get_frame_mean_IoU, get_frame_ap, get_mIoU, get_allFrames_ap
from src.plotting import plot_frame, plot_iou_vs_frames, make_gif
import yaml


def main(cfg: Dict):
    gt_rects_aicity_full = extract_rectangles_from_xml(cfg["paths"]["annotations"])
    det_rects_aicity_full = extract_rectangles_from_csv(cfg["paths"][cfg["settings"]["model"]])

    if cfg["settings"]["plot_random_annotation"]:
        # get a random frame labels from gt_rects_aicity_full
        frame = random.choice(list(gt_rects_aicity_full.keys()))
        # plot the frame
        frame_iou = get_frame_mean_IoU(gt_rects_aicity_full[frame], det_rects_aicity_full[frame])
        frame_map = get_frame_ap(gt_rects_aicity_full[frame], det_rects_aicity_full[frame], confidence=True, n=10,
                                 th=0.5)
        print("Frame: ", frame, " IoU: ", frame_iou, " mAP: ", frame_map)
        plot_frame(frame, gt_rects_aicity_full[frame], det_rects_aicity_full[frame], cfg["paths"]["video"], frame_iou)

    # add noise to whole dataset
    mean_iou, iou_per_frame = get_mIoU(gt_rects_aicity_full, det_rects_aicity_full)
    # print mIoU for all frames
    print("mIoU for all frames: ", mean_iou)
    print("mAP for all frames: ",
          get_allFrames_ap(gt_rects_aicity_full, det_rects_aicity_full, confidence=True, n=10, th=0.5))

    print("Plot iou vs frame")
    plot_iou_vs_frames(iou_per_frame)
    print('end')

    if cfg["settings"]["save_all_frames_plots"]:
        make_gif(gt_rects_aicity_full, det_rects_aicity_full, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1.yaml")
    args = parser.parse_args(sys.argv[1:])
    cfg_path = args.config

    # Read Config file
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)
