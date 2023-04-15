"""
It extracts from the selected sequences the frames and organizes them in train and validation
"""

import argparse
import sys
from src.utils import open_config_yaml
import os
import cv2


def create_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)


def read_gt(gt_path):
    # Open the file for reading
    with open(gt_path, 'r') as file:
        # Initialize an empty dictionary
        my_dict = {}
        # Loop through each line in the file
        for line in file:
            # Split the line into columns
            columns = line.strip().split(",")
            # Use the first column as the key and the rest as the values
            key = columns[0]
            values = columns[1:]
            # Add the key-value pair to the dictionary
            if key not in my_dict.keys():
                my_dict[key] = [values]
            else:
                my_dict[key].append(values)
    return my_dict


def get_yolo_format(bbox, image_width, image_height):
    xmin = int(bbox[1])
    ymin = int(bbox[2])
    width = int(bbox[3])
    height = int(bbox[4])

    # Calculate the center coordinates of the bounding box
    x_center = float(xmin + (width / 2))
    y_center = float(ymin + (height / 2))

    # Normalize the coordinates by the image width and height
    x_center /= image_width
    y_center /= image_height
    box_width = float(width) / image_width
    box_height = float(height) / image_height

    return [0, x_center, y_center, box_width, box_height]


def save_gt(txt_path, width=None, height=None, gt=None):
    if gt is not None:
        # Open the file for writing
        with open(txt_path, 'w') as file:
            # Loop through each list in the list of lists
            for my_row in gt:
                yolo_ann = get_yolo_format(my_row, width, height)
                # Convert the list to a string with each element separated by a tab character
                row_string = ' '.join(map(str, yolo_ann))
                # Write the string to the file, followed by a newline character
                file.write(row_string + '\n')
    else:
        open(txt_path, 'w')


def save_video_frames(dst_path, seq, cam, video_path, gt_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    # Read gt
    gt = read_gt(gt_path)
    # Create a folder to store the frames
    os.makedirs(dst_path, exist_ok=True)
    # Initialize frame count
    count = 0
    # Loop through the video frames
    while True:
        # Read the next frame
        ret, frame = video.read()
        # If there are no more frames, exit the loop
        if not ret:
            break
        # Save the frame as a JPEG file in the frames folder
        filename = f"{dst_path}/{seq}_{cam}_{str(count).zfill(6)}.jpg"
        cv2.imwrite(filename, frame)
        # save gt
        if str(count) in gt.keys():
            save_gt(f"{dst_path}/{seq}_{cam}_{str(count).zfill(6)}.txt", frame.shape[1], frame.shape[0], gt[str(count)])
        else:
            save_gt(f"{dst_path}/{seq}_{cam}_{str(count).zfill(6)}.txt")
        # Increment the frame count
        count += 1
    # Release the video file
    video.release()


def create_set(dataset_dir, seqs, phase):
    for seq in seqs:
        print(f"SEQ {seq}:")
        seq_dir = dataset_dir + seq
        cams = os.listdir(seq_dir)
        for i, cam in enumerate(cams):
            print(f"CAM {cam} {i}/{len(cams)}")
            video_path = f"{seq_dir}/{cam}/vdo.avi"
            gt_path = f"{seq_dir}/{cam}/gt/gt.txt"
            save_video_frames(f"data/{phase}/", seq, cam, video_path, gt_path)


def main(cfg):
    create_dirs()
    if "train_seqs" in cfg.keys():
        create_set(cfg["dataset_dir"], cfg["train_seqs"], "train")
    if "val_seqs" in cfg.keys():
        create_set(cfg["dataset_dir"], cfg["val_seqs"], "val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/create_dataset.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
