"""
This code is used to train a Yolov8 model,
before executing this code it is necessary to execute create_dataset_detection.py in order to have train and validation
"""

import argparse
import sys
from ultralytics import YOLO
from src.io_utils import open_config_yaml


def main(cfg):
    model = YOLO(cfg["model"])  # load a pretrained model (recommended for training)
    # Use the model
    # cfg["train_yolo"] should be the .yaml indicating the yolo train instructions
    model.train(data=cfg["train_yolo"], epochs=cfg["epochs"], device=0, imgsz=640, workers=2, val=False,
                optimizer="Adam")
    metrics = model.val()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_detection.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
