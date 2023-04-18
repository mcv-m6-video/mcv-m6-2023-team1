import argparse
from typing import List

import cv2
import os
import sys

# sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")) # comment if you are not using VSC
import numpy as np
from tqdm import tqdm

# Import Sort for tracking using Kalman filter
from week3.sort import Sort
from src.in_out import extract_frames_from_video, get_frames_paths_from_folder, extract_rectangles_from_xml
from src.metrics import get_IoU
from src.utils import open_config_yaml, load_bboxes_from_file, draw_img_with_ids, draw_bboxes_trajectory


def write_PASCAL_to_MOT_txt(track, output_path):
    with open(output_path, "w") as file:
        for frame, frame_track in enumerate(tqdm(track)):
            for ann in frame_track:
                x1, y1, x2, y2, track_id = ann
                w = x2 - x1
                h = y2 - y1
                file.write(f"{536 + frame},{track_id + 1},{x1},{y1},{w},{h},1,-1,-1,-1\n")


def write_PASCAL_to_MOT_txt_w4(track, output_path):
    with open(output_path, "w") as file:
        for frame, frame_track in enumerate(tqdm(track)):
            for ann in frame_track:
                x1, y1, x2, y2, track_id = ann
                w = x2 - x1
                h = y2 - y1
                file.write(f"{frame + 1},{track_id + 1},{x1},{y1},{w},{h},1,-1,-1,-1\n")


def write_gt_to_MOT_txt(track, output_path):
    with open(output_path, "w") as file:
        for frame, frame_track in zip(list(track.keys()), list(track.values())):
            for ann in frame_track:
                x1, y1, x2, y2, track_id = ann
                w = x2 - x1
                h = y2 - y1
                file.write(f"{frame + 1},{track_id + 1},{x1},{y1},{w},{h},1,-1,-1,-1\n")


def draw_bboxes_and_trajectory(first_frame, last_frame, viz_config, track_bbs_ids, dataset, out_path):
    # Draw the bounding boxes and the trajectory
    overlay = np.zeros_like(cv2.imread(dataset[0][1]))
    for i, frame in tqdm(enumerate(dataset)):
        if first_frame < i < last_frame:
            if i == 0:
                continue  # Skip the first frame
            img = cv2.imread(frame[1])
            # Draw the trajectory lines
            overlay = draw_bboxes_trajectory(overlay, track_bbs_ids[i], track_bbs_ids[i - 1])
            # Draw the bounding boxes with ID number
            img_out = draw_img_with_ids(img, track_bbs_ids[i])
            # Fuse both images
            img_out = cv2.addWeighted(img_out, 1, overlay, 1, 0)

            # Show the frame with the trajectory and bounding boxes IDs
            if viz_config['show_detection']:
                cv2.imshow('frame2', img_out)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

            # Save the frame
            if viz_config['save']:
                os.makedirs(out_path, exist_ok=True)
                cv2.imwrite(os.path.join(out_path, str(i) + '.png'), img_out,
                            [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


class MOTTrackerOverlap:
    """
    This class implements a simple tracker based on detection overlap.
    """

    def __init__(self):
        self.track_id = 0
        self.tracks = {}
        self.frame_id = 0

    def update(self, dets):
        """
        This method updates the tracker with the new detections.

        :param dets: numpy array with the detections for the current frame
        :return: numpy array with the detections for the current frame and the track_id
        """
        updated_dets = []
        used_ids = set()
        for det in dets:
            max_overlap = 0
            max_track_id = None

            for track_id, track in self.tracks.items():
                if track_id in used_ids:
                    continue
                overlap = self._compute_overlap(det, track[-1])
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_track_id = track_id

            if max_overlap > 0.5:
                self.tracks[max_track_id].append(det)
                updated_dets.append(np.append(det, max_track_id))
                used_ids.add(max_track_id)
            else:
                self.tracks[self.track_id] = [det]
                updated_dets.append(np.append(det, self.track_id))
                used_ids.add(self.track_id)
                self.track_id += 1

        # Remove the tracks that are not updated
        for track_id in list(self.tracks.keys()):
            if track_id not in used_ids:
                del self.tracks[track_id]

        self.frame_id += 1
        return np.array(updated_dets)

    @staticmethod
    def _compute_overlap(det1: List, det2: List) -> float:
        """
        This method computes the overlap between two detections.

        :param det1: numpy array with the first detection
        :param det2: numpy array with the second detection
        :return: float with the overlap
        """
        return get_IoU(det1, det2)


def task2_1(out_path, dataset, bboxes, first_frame, last_frame, visualization_cfg):
    """
    This task consist on implementing tracking using detection overlap on top of a given detections.

    :param out_path: path to save the output images
    :param dataset: list of tuples (frame_id, frame_path)
    :param bboxes: list of lists of bounding boxes (frame_id, x1, y1, x2, y2, class_id, confidence)
    :param first_frame: first frame to process
    :param last_frame: last frame to process
    :param visualization_cfg: dictionary with the visualization configuration
    """
    mot_tracker_overlap = MOTTrackerOverlap()

    track_bbs_ids = []
    # update SORT
    for frame_bboxes in bboxes:
        # np array where each row contains a valid bounding box and track_id (last column)
        if len(frame_bboxes[0]) == 5:
            for i in range(len(frame_bboxes)):
                frame_bboxes[i] = frame_bboxes[i][:-1]
        track_bbs_ids.append(mot_tracker_overlap.update(np.array(frame_bboxes)))

    # Draw the bounding boxes and the trajectory
    # draw_bboxes_and_trajectory(first_frame, last_frame, visualization_cfg, track_bbs_ids, dataset, out_path)

    return track_bbs_ids


def task2_2(out_path, dataset, bboxes, first_frame, last_frame, visualization_cfg):
    """
    This task consist on implementing tracking using Kalman filters on top of a given detections.

    :param out_path: path to save the output images
    :param dataset: list of tuples (frame_id, frame_path)
    :param bboxes: list of lists of bounding boxes (frame_id, x1, y1, x2, y2, class_id, confidence)
    :param first_frame: first frame to process
    :param last_frame: last frame to process
    :param visualization_cfg: dictionary with the visualization configuration
    """

    # create instance of SORT
    mot_tracker = Sort(max_age=1,
                       min_hits=3,
                       iou_threshold=0.3)  # create instance of the SORT tracker

    track_bbs_ids = []
    # update SORT
    for frame_bboxes in bboxes:
        # np array where each row contains a valid bounding box and track_id (last column)
        track_bbs_ids.append(mot_tracker.update(np.array(frame_bboxes)))

    # Draw the bounding boxes and the trajectory
    # draw_bboxes_and_trajectory(first_frame, last_frame, visualization_cfg, track_bbs_ids, dataset, out_path)
    return track_bbs_ids


def task2_3(tracks, cfg):
    kalman_path = "data/trackers/mot_challenge/week3-train/kalman/data/Seq03.txt"
    overlap_path = "data/trackers/mot_challenge/week3-train/overlap/data/Seq03.txt"
    output_paths = [overlap_path, kalman_path]
    for track, output_path in zip(tracks, output_paths):
        write_PASCAL_to_MOT_txt(track, output_path)


def main(cfg):
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    visualization_cfg = cfg["visualization"]

    gt_labels = extract_rectangles_from_xml(paths["annotations"], add_track_id=True)
    # get detections

    if model_cfg['use_gt']:
        gt_bboxes = [*gt_labels.values()]
        gt_frames = [*gt_labels.keys()]
        bboxes = gt_bboxes[int(len(gt_labels) * 0.25):]
        frames = gt_frames[int(len(gt_labels) * 0.25):]
    else:
        bboxes = load_bboxes_from_file(paths['detected_bboxes'])
    if model_cfg['save_gt_tracking_MOT']:
        write_gt_to_MOT_txt(dict(zip(frames, bboxes)), output_path="data/gt/mot_challenge/week3-train/Seq03/gt/gt.txt")

    # Obtain all frames of the sequence
    extract_frames_from_video(video_path=paths["video"], output_path=paths["extracted_frames"])
    frames = get_frames_paths_from_folder(input_path=paths["extracted_frames"])
    dataset = [(key, frames[key]) for key in gt_labels.keys()]

    # keep only frames of selected range
    # dataset = dataset[first_frame:last_frame]
    dataset = dataset[int(len(dataset) * 0.25):]
    print("Number of frames: ", len(dataset))
    track_bbs_ids_overlap = task2_1(
        out_path=paths["output"],
        dataset=dataset,
        bboxes=bboxes,
        first_frame=model_cfg['first_frame'],
        last_frame=model_cfg['last_frame'],
        visualization_cfg=visualization_cfg
    )
    track_bbs_id_kalman = task2_2(
        out_path=paths["output"],
        dataset=dataset,
        bboxes=bboxes,
        first_frame=model_cfg['first_frame'],
        last_frame=model_cfg['last_frame'],
        visualization_cfg=visualization_cfg
    )

    if not model_cfg["use_gt"]:
        task2_3([track_bbs_ids_overlap, track_bbs_id_kalman], model_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2.yaml")
    # parser.add_argument("--config", default="week3/configs/task2_2.yaml") #comment if you are not using VSCode
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)
    main(config)
