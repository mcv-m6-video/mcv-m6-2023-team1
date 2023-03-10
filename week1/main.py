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

    if cfg["settings"]["plot_random_annotation"]:
        #get a random frame labels from GT_rects_aicity_full
        frame = random.choice(list(GT_rects_aicity_full.keys()))
        #plot the frame
        plot_frame(frame, GT_rects_aicity_full[frame], cfg["paths"]["video"])



    print('end')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1.yaml")
    args = parser.parse_args(sys.argv[1:])
    config_name = args.config

    main(config_name)
