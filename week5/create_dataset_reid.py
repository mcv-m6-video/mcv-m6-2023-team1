import argparse
import sys
from src.io_utils import open_config_yaml, create_dirs, read_gt
import os
import cv2


def save_bbox_frames(dst_path, seq, cam, count, frame, boxes):
    # frame -> ["ID", "left", "top", "width", "height", "confidence", "null1", "null2", "null3"]
    for box in boxes:
        # Extract the coordinates of the bounding box
        car_id = box[0]
        x = int(box[1])
        y = int(box[2])
        width = int(box[3])
        height = int(box[4])

        os.makedirs(f"{dst_path}/{car_id}", exist_ok=True)
        # Extract the sub-image from the frame using the bounding box coordinates
        sub_image = frame[y:y + height, x:x + width]
        # Save the sub-image as a separate image file
        cv2.imwrite(f"{dst_path}/{car_id}/{seq}_{cam}_{str(count).zfill(6)}.jpg", sub_image)


def save_car_frames(dst_path, seq, cam, video_path, gt_path):
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
        # Save bounding boxes frames from this frame
        if str(count) in gt.keys():
            save_bbox_frames(dst_path, seq, cam, count, frame, gt[str(count)])
        # Increment the frame count
        count += 1
    # Release the video file
    video.release()


def create_set(dataset_dir, seqs, phase, dst_path):
    for seq in seqs:
        print(f"SEQ {seq}:")
        seq_dir = dataset_dir + seq
        cams = os.listdir(seq_dir)
        for i, cam in enumerate(cams):
            print(f"CAM {cam} {i+1}/{len(cams)}")
            video_path = f"{seq_dir}/{cam}/vdo.avi"
            gt_path = f"{seq_dir}/{cam}/gt/gt.txt"
            save_car_frames(f"{dst_path}/{phase}", seq, cam, video_path, gt_path)


def main(cfg):
    create_dirs(cfg["dst_path"])
    if "train_seqs" in cfg.keys():
        create_set(cfg["dataset_dir"], cfg["train_seqs"], "train", cfg["dst_path"])
    if "val_seqs" in cfg.keys():
        create_set(cfg["dataset_dir"], cfg["val_seqs"], "val", cfg["dst_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/create_dataset_reid.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)