import argparse
import random
import sys
from typing import Dict

from utils import *
import yaml


def main(cfg: Dict):

    gt_rects_aicity_full = extract_rectangles_from_xml(cfg["paths"]["annotations"])
    det_rects_aicity_full = extract_rectangles_from_csv(cfg["paths"][cfg["settings"]["model"]])
    gt_rects_aicity_full = dict(sorted(gt_rects_aicity_full.items()))
    det_rects_aicity_full = dict(sorted(det_rects_aicity_full.items()))

    if cfg["settings"]["plot_random_annotation"]:
        # get a random frame labels from gt_rects_aicity_full
        #frame = random.choice(list(gt_rects_aicity_full.keys()))
        frame = 'f_1558'
        # plot the frame
        frame_iou = get_frame_mean_IoU(gt_rects_aicity_full[frame], det_rects_aicity_full[frame])
        frame_map = get_frame_ap(gt_rects_aicity_full[frame], det_rects_aicity_full[frame], confidence=True, n=10,
                                 th=0.5)
        print("Frame: ", frame, " IoU: ", frame_iou, " mAP: ", frame_map)
        plot_frame(frame, gt_rects_aicity_full[frame], det_rects_aicity_full[frame], cfg["paths"]["video"], frame_iou)

        # Add noise in the ground truth and measure again the IoU and mAP
        gt_bboxes = addNoise(gt_rects_aicity_full[frame], res = cfg["noise"]["res"], pos= cfg["noise"]["pos"], 
                             max_pxdisplacement = cfg["noise"]["max_pxdisplacement"], size=cfg["noise"]["size"], 
                             max_scale = cfg["noise"]["max_scale"], min_scale = cfg["noise"]["min_scale"], 
                             removebbox=cfg["noise"]["removebbox"], ratio_removebbox = cfg["noise"]["ratio_removebbox"], 
                              addbbox = cfg["noise"]["addbbox"], ratio_addbbox = cfg["noise"]["ratio_addbbox"])

        # plot the frame
        frame_iou = get_frame_mean_IoU(gt_bboxes, det_rects_aicity_full[frame])
        frame_map = get_frame_ap(gt_bboxes, det_rects_aicity_full[frame], confidence=True, n=10,
                                 th=0.5)
        print("Frame: ", frame, " IoU: ", frame_iou, " mAP: ", frame_map)
        plot_frame(frame, gt_bboxes, det_rects_aicity_full[frame], cfg["paths"]["video"], frame_iou)        


    print('end')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1.yaml")
    args = parser.parse_args(sys.argv[1:])
    cfg_path = args.config

    # Read Config file
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)
