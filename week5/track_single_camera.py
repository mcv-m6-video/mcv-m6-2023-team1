"""
Visualize single camera tracking
"""
import argparse
import os
import sys
from typing import Dict

import cv2
import numpy as np

from src.io_utils import extract_frames, extract_detections
from src.io_utils import open_config_yaml
from src.plot_utils import draw_boxes
from src.sort import Sort
from week3.task2 import write_PASCAL_to_MOT_txt_w5, MOTTrackerOverlap
from week5.track_multi_camera import postprocess


def update_tracker(mot_tracker, boxes):
    np_boxes = np.array(boxes)
    track_bbs_ids = mot_tracker.update(np_boxes)[::-1]
    return track_bbs_ids


def track_sort(detections):
    tracker = Sort()

    dict_trackings = {}
    for frame_id, det in sorted(detections.items()):
        track_det = update_tracker(tracker, det)
        dict_trackings[frame_id] = track_det.tolist()

    return dict_trackings


def track_overlap(detections_dict: Dict) -> Dict:
    """
    This task consist on implementing tracking using detection overlap on top of a given detections.

    :param detections_dict: Dictionary with the detections
    :return: dictionary with the tracking results
    """
    mot_tracker_overlap = MOTTrackerOverlap()

    tracking = {}
    for frame, detections in sorted(detections_dict.items()):
        # np array where each row contains a valid bounding box and track_id (last column)
        if detections:
            if len(detections[0]) == 6:
                for i in range(len(detections)):
                    detections[i] = detections[i][:-2]
            tracking[frame] = mot_tracker_overlap.update(np.array(detections))
        else:
            tracking[frame] = mot_tracker_overlap.update(np.array([]))

    return tracking


def save_tracking_videos(frames, trackings, tracker):
    os.makedirs("tracking_videos", exist_ok=True)
    # Set up the video writer
    fps = 20  # Set the frame rate
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for i, cam_id in enumerate(sorted(frames.keys())):
        print(f"Saving tracking video for CAM {cam_id} {i + 1}/{len(frames.keys())}")
        frame_size = cv2.imread(frames[cam_id][0]).shape
        out = cv2.VideoWriter(f'tracking_videos/tracking_{tracker}_{cam_id}.avi', fourcc, fps,
                              (frame_size[1], frame_size[0]))
        for frame_id, frame_path in enumerate(sorted(frames[cam_id])):
            frame = cv2.imread(frame_path)
            if frame_id in trackings[cam_id].keys():
                frame = draw_boxes(frame, trackings[cam_id][frame_id])
            out.write(frame)
        out.release()


def main(cfg):
    frames = extract_frames(cfg['data_path'])
    detections = extract_detections(frames, cfg["model_path"])
    cam_ids = [*frames.keys()]
    frames_per_camera = {
        'c010': 2141,
        'c011': 2279,
        'c012': 2422,
        'c013': 2415,
        'c014': 2332,
        'c015': 1928,

    }

    postprocessed_detections = postprocess(detections, frames_per_camera)
    trackings = {}
    for cam_id in cam_ids:
        cam_detections = postprocessed_detections[cam_id]

        if cfg["tracker"] == "sort":
            trackings[cam_id] = track_sort(cam_detections)  # TODO: it does not track empty frames, fix pls
        elif cfg["tracker"] == "overlap":
            trackings[cam_id] = track_overlap(cam_detections)
        else:
            print("Unknown tracker")
            return
        if cfg["save_tracking"]:
            os.makedirs(f"tracking_single", exist_ok=True)
            write_PASCAL_to_MOT_txt_w5(trackings[cam_id], f"tracking_single/{cam_id}.txt")
    # save tracking videos
    if cfg["save_tracking_videos"]:
        save_tracking_videos(frames, trackings, cfg["tracker"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/track_single_camera.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
