"""
This code is used to train a Yolov8 model,
before executing this code it is necessary to execute create_dataset.py in order to have train and validation
"""

import argparse
import sys
from ultralytics import YOLO


def main(data_file):
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    # Use the model
    model.train(data=data_file, epochs=40, device=0, imgsz=640, workers=2, val=False,
                optimizer="Adam")
    metrics = model.val()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_detection.yaml")
    args = parser.parse_args(sys.argv[1:])

    config_file = args.config

    main(config_file)
