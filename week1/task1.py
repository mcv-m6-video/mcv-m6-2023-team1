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
        frame = 1558
        
        if cfg["noise"]["size"]:
            scales = cfg["noise"]["max_scale"]
            for scale in scales:
                # Add noise in the ground truth and measure again the IoU and mAP
                gt_bboxes = addNoise(gt_rects_aicity_full[frame], randm = cfg["noise"]["random"], size=True, 
                                    max_scale = scale, min_scale = scale)
                # plot the frame
                frame_iou = get_frame_mean_IoU(gt_bboxes, det_rects_aicity_full[frame])
                frame_map = get_frame_ap(gt_bboxes, det_rects_aicity_full[frame], confidence=False, n=10,
                                        th=0.5)
                print("Scale: ", scale, " IoU: ", frame_iou, " mAP: ", frame_map)
                #plot_frame(frame, gt_bboxes, det_rects_aicity_full[frame], cfg["paths"]["video"], frame_iou)
        if cfg["noise"]["pos"]:
            offsets = cfg["noise"]["max_pxdisplacement"]
            for traslation in offsets:
                # Add noise in the ground truth and measure again the IoU and mAP
                gt_bboxes = addNoise(gt_rects_aicity_full[frame], randm = cfg["noise"]["random"], pos= True, 
                                    max_pxdisplacement = traslation)
                # plot the frame
                frame_iou = get_frame_mean_IoU(gt_bboxes, det_rects_aicity_full[frame])
                frame_map = get_frame_ap(gt_bboxes, det_rects_aicity_full[frame], confidence=False, n=10,
                                        th=0.5)
                print("Max Displacement: ", traslation, " IoU: ", frame_iou, " mAP: ", frame_map)
                #plot_frame(frame, gt_bboxes, det_rects_aicity_full[frame], cfg["paths"]["video"], frame_iou)
        if cfg["noise"]["removebbox"]:
            ratios = cfg["noise"]["ratio_removebbox"]
            for ratio in ratios:
                # Add noise in the ground truth and measure again the IoU and mAP
                gt_bboxes = addNoise(gt_rects_aicity_full[frame], removebbox=True, ratio_removebbox = ratio)
                
                # plot the frame
                frame_iou = get_frame_mean_IoU(gt_bboxes, det_rects_aicity_full[frame])
                frame_map = get_frame_ap(gt_bboxes, det_rects_aicity_full[frame], confidence=False, n=10,
                                        th=0.5)
                print("Percentage of bboxes removed: ", ratio*100, "% IoU: ", frame_iou, " mAP: ", frame_map)
                #plot_frame(frame, gt_bboxes, det_rects_aicity_full[frame], cfg["paths"]["video"], frame_iou)
        if cfg["noise"]["addbbox"]:
            ratios = cfg["noise"]["ratio_addbbox"]
            for ratio in ratios:
                # Add noise in the ground truth and measure again the IoU and mAP
                gt_bboxes = addNoise(gt_rects_aicity_full[frame], addbbox=True, ratio_addbbox = ratio)
                
                # plot the frame
                frame_iou = get_frame_mean_IoU(gt_bboxes, det_rects_aicity_full[frame])
                frame_map = get_frame_ap(gt_bboxes, det_rects_aicity_full[frame], confidence=False, n=10,
                                        th=0.5)
                print("Percentage of bboxes added: ", ratio*100, "% IoU: ", frame_iou, " mAP: ", frame_map)
                #plot_frame(frame, gt_bboxes, det_rects_aicity_full[frame], cfg["paths"]["video"], frame_iou)
                  
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
