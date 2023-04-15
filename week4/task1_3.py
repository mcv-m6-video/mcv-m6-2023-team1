import numpy as np
from src.in_out import extract_frames_from_video, get_frames_paths_from_folder, extract_rectangles_from_xml, \
    extract_of_from_dataset, get_bbox_optical_flows_from_folder
from tqdm import tqdm
from src.utils import open_config_yaml, load_bboxes_from_file, draw_img_with_ids, draw_bboxes_trajectory
import argparse
import sys
from typing import List
from src.metrics import get_IoU
import os
import cv2


def save_results_to_MOTS(tracks, cfg):
    overlap_OF_path = "data/trackers/mot_challenge/week3-train/overlap_OF_farneback/data/Seq03.txt"
    output_paths = [overlap_OF_path]
    for track, output_path in zip(tracks, output_paths):
        write_PASCAL_to_MOT_txt(track, output_path)


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


def resize_bboxes(bboxes):
    resized_bboxes = []
    for frame_bboxes in bboxes:
        resized_frame_bboxes = []
        for bbox in frame_bboxes:
            x1 = bbox[0] * 496 / 1920
            y1 = bbox[1] * 368 / 1080
            x2 = bbox[2] * 496 / 1920
            y2 = bbox[3] * 368 / 1080
            resized_frame_bboxes.append([int(x1), int(y1), int(x2), int(y2), bbox[4]])
        resized_bboxes.append(resized_frame_bboxes)
    return resized_bboxes


def resize_track_bboxes(tracks):
    resized_tracks = []
    for track in tracks:
        track[:, 0] = track[:, 0] * 1920 / 496
        track[:, 1] = track[:, 1] * 1080 / 368
        track[:, 2] = track[:, 2] * 1920 / 496
        track[:, 3] = track[:, 3] * 1080 / 368
        resized_tracks.append(track)
    return resized_tracks


class MOTTrackerOverlapOpticalFlow():
    """
    This class implements a simple tracker based on detection overlap.
    """

    def __init__(self):
        self.track_id = 0
        self.tracks = {}
        self.frame_id = 0

    def update_with_optical_flows(self, dets, flows):
        """
        This method updates the tracker with the new detections.

        :param dets: numpy array with the detections for the current frame
        :return: numpy array with the detections for the current frame and the track_id
        """
        updated_dets = []
        used_ids = set()
        for det, flow in zip(dets, flows):
            max_overlap = 0
            max_track_id = None

            for track_id, track in self.tracks.items():
                if track_id in used_ids:
                    continue
                expected_track = self._compute_expected_bbox_withOF(track[-1], flow)
                overlap = self._compute_overlap(det, expected_track)
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

        self.frame_id += 1
        return np.array(updated_dets)

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

        self.frame_id += 1
        return np.array(updated_dets)

    @staticmethod
    def _compute_expected_bbox_withOF(bbox: np.ndarray, flow: np.ndarray):
        return np.array([bbox[0] - flow[0],
                         bbox[1] - flow[1],
                         bbox[2] - flow[0],
                         bbox[3] - flow[1]])

    @staticmethod
    def _compute_overlap(det1: List, det2: List) -> float:
        """
        This method computes the overlap between two detections.

        :param det1: numpy array with the first detection
        :param det2: numpy array with the second detection
        :return: float with the overlap
        """
        return get_IoU(det1, det2)


def write_PASCAL_to_MOT_txt(track, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path, "w") as file:
        for frame, frame_track in enumerate(tqdm(track)):
            for ann in frame_track:
                x1, y1, x2, y2, track_id = ann
                w = x2 - x1
                h = y2 - y1
                file.write(f"{536 + frame},{track_id + 1},{x1},{y1},{w},{h},1,-1,-1,-1\n")


def write_gt_to_MOT_txt(track, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path, "w") as file:
        for frame, frame_track in zip(list(track.keys()), list(track.values())):
            for ann in frame_track:
                x1, y1, x2, y2, track_id = ann
                w = x2 - x1
                h = y2 - y1
                file.write(f"{frame + 1},{track_id + 1},{x1},{y1},{w},{h},1,-1,-1,-1\n")


def task1_3(out_path, dataset, bboxes, optical_flows, first_frame, last_frame, visualization_cfg):
    """
    This task consist on implementing tracking using detection overlap on top of a given detections.

    :param out_path: path to save the output images
    :param dataset: list of tuples (frame_id, frame_path)
    :param bboxes: list of lists of bounding boxes (frame_id, x1, y1, x2, y2, class_id, confidence)
    :param first_frame: first frame to process
    :param last_frame: last frame to process
    :param visualization_cfg: dictionary with the visualization configuration
    """
    mot_tracker_overlap = MOTTrackerOverlapOpticalFlow()

    track_bbs_ids = []
    # update SORT
    first_iteration_done = False
    for num_frame, frame_bboxes in enumerate(bboxes):
        # np array where each row contains a valid bounding box and track_id (last column)
        if len(frame_bboxes[0]) == 5:
            for i in range(len(frame_bboxes)):
                frame_bboxes[i] = frame_bboxes[i][:-1]
        if not first_iteration_done:
            track_bbs_ids.append(mot_tracker_overlap.update(np.array(frame_bboxes)))
        else:
            track_bbs_ids.append(mot_tracker_overlap.update_with_optical_flows(np.array(frame_bboxes),
                                                                               np.array(optical_flows[num_frame - 1])))
        first_iteration_done = True
    track_bbs_ids = resize_track_bboxes(track_bbs_ids)
    # Draw the bounding boxes and the trajectory
    if visualization_cfg:
        draw_bboxes_and_trajectory(first_frame, last_frame, visualization_cfg, track_bbs_ids, dataset, out_path)
    return track_bbs_ids


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
        write_gt_to_MOT_txt(dict(zip(frames, bboxes)), output_path="data/gt/mot_challenge/week4-train/Seq03/gt/gt.txt")

    # Obtain all frames of the sequence
    extract_frames_from_video(video_path=paths["video"], output_path=paths["extracted_frames"])
    frames = get_frames_paths_from_folder(input_path=paths["extracted_frames"])
    dataset = [(key, frames[key]) for key in gt_labels.keys()]

    # keep only frames of selected range
    # dataset = dataset[int(len(dataset) * 0.25):int(len(dataset) * 0.25)+10]
    # bboxes = bboxes[int(len(dataset) * 0.25):int(len(dataset) * 0.25)+10]
    bboxes = resize_bboxes(bboxes)
    dataset = dataset[int(len(dataset) * 0.25):]
    print("Number of frames: ", len(dataset))
    extract_of_from_dataset(dataset, paths["extracted_optical_flows_path"], model_cfg["of_method"])
    optical_flows = get_bbox_optical_flows_from_folder(bboxes, paths["extracted_optical_flows_path"])

    track_bbs_ids_overlap = task1_3(
        out_path=paths["output"],
        dataset=dataset,
        optical_flows=optical_flows,
        bboxes=bboxes,
        first_frame=model_cfg['first_frame'],
        last_frame=model_cfg['last_frame'],
        visualization_cfg=visualization_cfg
    )

    if not model_cfg["use_gt"]:
        save_results_to_MOTS([track_bbs_ids_overlap], model_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1-3.yaml")
    # parser.add_argument("--config", default="week3/configs/task1-3.yaml") #comment if you are not using VSCode
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)
    main(config)
