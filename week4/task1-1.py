import argparse
import sys
from typing import Dict
from src.utils import open_config_yaml
from src.enums import OFMethods
from src.methods_OF import evaluate_kitti_week_1, evaluate_methods_week4, evaluate_optimized_blockmatching
def main(cfg: Dict):

    if not cfg["evaluate_own_OF"]:  # if false it will evaluate the kitti made on week1
        evaluate_kitti_week_1(cfg)
    elif cfg["OF_method"]==OFMethods.OptiBlockMatching:
        evaluate_optimized_blockmatching(cfg)
    else:  # Compute the optical flow with the block matching algorithm
       evaluate_methods_week4(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="week4/configs/task1-1.yaml")
    # parser.add_argument("--config", default="week3/configs/task1.yaml") #comment if you are not using Visual Studio
    # Code
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
