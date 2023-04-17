"""
This code is used to train a Yolov8 model,
before executing this code it is necessary to execute create_dataset.py in order to have train and validation
"""

import argparse
import sys
from ultralytics import YOLO
import numpy as np
import time
from sklearn.model_selection import ParameterSampler
from src.utils import open_config_yaml


def main(cfg):
    # DEFINITION HYPERPARAMETERS
    param_grid = cfg["param_grid"]
    n = cfg["n"]
    """ RANDOMIZED SEARCH CV """
    sampled_params = list(ParameterSampler(param_grid, n))
    for ind, params in enumerate(sampled_params):

        print('\nITERATION {} of {}'.format(ind + 1, n))

        batch_size = int(params['batch_size'])
        #limit image_size due to gpu memory limitation (6Gb)
        if batch_size == 256:
            image_size = 256
        elif batch_size == 128:
            image_size = 380 if int(params['image_size']) >= 512 else int(params['image_size'])
        else:
            image_size = int(params['image_size'])
        learning_rate = float(params['learning_rate'])

        print('batch size: {}'.format(batch_size))
        print('image_size: {}'.format(image_size))
        print('learning rate: {}'.format(learning_rate))
        name = 'batchsize_{}_image_size_{}_lr_{:.5G}'.format(
            batch_size,
            image_size,
            learning_rate
        )
        model = YOLO(cfg["model"])  # load a pretrained model (recommended for training)
        start_time = time.time()
        # Use the model
        model.train(data=cfg["train_yolo"], batch=batch_size, epochs=cfg["epochs"], device=0, imgsz=image_size, workers=0, val=True,
                    optimizer="Adam", lr0= learning_rate, pretrained=True, cache=True, cos_lr=False, patience=10,
                    name=name, save_period=15)
        final_time = time.time() - start_time
        hours = final_time // 3600
        seconds = final_time % 3600
        minutes = seconds // 60
        seconds %= 60
        print('Total training time: {:02d}hours {:02d}min {:02d}s'.format(int(hours), int(minutes), int(seconds)))
        metrics = model.val()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_detection.yaml")
    args = parser.parse_args(sys.argv[1:])

    config_file = args.config

    config = open_config_yaml(args.config)

    main(config)
