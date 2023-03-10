import argparse
import random
import sys
from utils import *
import yaml




def main(cfg):
    #Read Config file
    with open(cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    GT_rects_aicity_full = extract_rectangles_from_xml(cfg["paths"]["annotations"])
    det_rects_aicity_full = extract_rectangles_from_csv(cfg["paths"][cfg["settings"]["model"]])
    GT_rects_aicity_full = dict(sorted(GT_rects_aicity_full.items()))
    det_rects_aicity_full = dict(sorted(det_rects_aicity_full.items()))

    if cfg["settings"]["plot_random_annotation"]:
        #get a random frame labels from GT_rects_aicity_full
        frame = random.choice(list(GT_rects_aicity_full.keys()))
        #plot the frame
        frame_iou =  get_frame_mean_IoU(GT_rects_aicity_full[frame], det_rects_aicity_full[frame])
        print("Frame: ", frame, " IoU: ",frame_iou)
        plot_frame(frame, GT_rects_aicity_full[frame], det_rects_aicity_full[frame], cfg["paths"]["video"], frame_iou)

    #print mIoU for all frames
    print("mIoU for all frames: ", get_mIoU(GT_rects_aicity_full, det_rects_aicity_full))





    print('end')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1.yaml")
    args = parser.parse_args(sys.argv[1:])
    config_name = args.config

    main(config_name)
