import argparse
import os
import sys

from src.utils import open_config_yaml


def main(cfg):
    paths = cfg['paths']
    extracted_frames = paths['extracted_frames']
    # Obtain all frames for each camera on the sequence
    for camera in os.listdir(extracted_frames):
        frames_path = extracted_frames + '/' + camera
        for filename in os.listdir(frames_path):
            # Check if the file is a PNG image and its name starts with "frame_"
            if filename.startswith("frame_"):
                # Construct the new filename
                new_filename = camera + filename[5:]
                # Rename the file
                os.rename(os.path.join(frames_path, filename), os.path.join(frames_path, new_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rename_frames.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)