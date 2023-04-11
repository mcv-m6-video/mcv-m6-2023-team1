import argparse
import os
import sys

from src.in_out import extract_frames_from_video
from src.utils import open_config_yaml


def main(cfg):
    paths = cfg['paths']
    sequence_path = paths['sequence_path']
    extracted_frames = paths['extracted_frames']
    # Obtain all frames for each camera on the sequence
    for camera in os.listdir(sequence_path):
        video_path = sequence_path + '/' + camera + '/vdo.avi'
        extracted_frames_path = extracted_frames + '/' + camera
        extract_frames_from_video(video_path=video_path, output_path=extracted_frames_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/create_dataset.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
