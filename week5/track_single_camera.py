"""
Visualize single camera tracking
"""

from src.sort import Sort
import argparse
import sys
from ultralytics import YOLO
from src.io_utils import open_config_yaml, save_bboxes_to_file, load_bboxes_from_file
import glob
import cv2
import os
from src.plot_utils import draw_boxes
from src.io_utils import extract_frames, extract_detections
import numpy as np
import yaml
from tqdm import tqdm


def update_tracker(mot_tracker, boxes):
    np_boxes = np.array(boxes)
    track_bbs_ids = mot_tracker.update(np_boxes)[::-1]
    return track_bbs_ids


def track_sort(detections):
    tracker = Sort()

    dict_trackings = {}
    for frame_id, det in detections.items():
        track_det = update_tracker(tracker, det)
        dict_trackings[frame_id] = track_det.tolist()

    return dict_trackings


def save_tracking_videos(frames, trackings, tracker):
    os.makedirs("tracking_videos", exist_ok=True)
    # Set up the video writer
    fps = 20  # Set the frame rate
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for i, cam_id in enumerate(frames.keys()):
        print(f"Saving tracking video for CAM {cam_id} {i+1}/{len(frames.keys())}")
        frame_size = cv2.imread(frames[cam_id][0]).shape
        out = cv2.VideoWriter(f'tracking_videos/tracking_{tracker}_{cam_id}.avi', fourcc, fps, (frame_size[1], frame_size[0]))
        for frame_id, frame_path in enumerate(frames[cam_id]):
            frame = cv2.imread(frame_path)
            if frame_id in trackings[cam_id].keys():
                frame = draw_boxes(frame, trackings[cam_id][frame_id])
            out.write(frame)
        out.release()


def evaluate_trackings(trackings):
    # TODO: evaluate trackings, may need to read ground truth (use function read_gt from src.io_utils)
    pass


def main(cfg):
    frames = extract_frames(cfg['data_path'])
    detections = extract_detections(frames, cfg["model_path"])
    cam_ids = [*frames.keys()]

    trackings = {}
    for cam_id in cam_ids:
        cam_detections = detections[cam_id]

        if cfg["tracker"] == "sort":
            trackings[cam_id] = track_sort(cam_detections)
        elif cfg["tracker"] == "overlap":
            pass

    evaluate_trackings(trackings)

    # save tracking videos
    save_tracking_videos(frames, trackings, cfg["tracker"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/track_single_camera.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
