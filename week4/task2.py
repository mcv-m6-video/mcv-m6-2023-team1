import argparse
import sys

import numpy as np

from src.in_out import extract_frames_from_video, get_frames_paths_from_folder, extract_rectangles_from_xml, \
    extract_rectangles_from_csv, extract_rectangles_from_txt_gt, extract_of_from_dataset, \
    get_bbox_optical_flows_from_folder
from src.utils import open_config_yaml, load_bboxes_from_file
from week3.task2 import write_gt_to_MOT_txt, MOTTrackerOverlap, write_PASCAL_to_MOT_txt, task2_2, \
    draw_bboxes_and_trajectory, write_PASCAL_to_MOT_txt_w4

from task1_3 import task1_3, resize_bboxes


def get_overlap_track(out_path, dataset, bboxes, first_frame, last_frame, visualization_cfg):
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
    for bboxes_key, frame in dataset:
        # np array where each row contains a valid bounding box and track_id (last column)
        if bboxes_key is not None:
            frame_bboxes = bboxes[bboxes_key]
            if len(frame_bboxes[0]) == 5:
                for i in range(len(frame_bboxes)):
                    frame_bboxes[i] = frame_bboxes[i][:-1]
            track_bbs_ids.append(mot_tracker_overlap.update(np.array(frame_bboxes)))
        else:
            track_bbs_ids.append(mot_tracker_overlap.update(np.array([])))

    # Draw the bounding boxes and the trajectory
    """
    if visualization_cfg:
        draw_bboxes_and_trajectory(first_frame, last_frame, visualization_cfg, track_bbs_ids, dataset, out_path)
    """
    return track_bbs_ids


def save_track(track: np.ndarray, track_path: str) -> None:
    write_PASCAL_to_MOT_txt_w4(track, track_path)


def main(cfg):
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    visualization_cfg = cfg["visualization"]

    frames = get_frames_paths_from_folder(input_path=paths["extracted_frames"])

    if cfg['use_gt_bboxes']:
        bboxes = extract_rectangles_from_csv(paths['annotations'])
    else:
        bboxes = extract_rectangles_from_csv(paths['detected_bboxes'])

    dataset = []
    for f_id, frame in enumerate(frames):
        dataset_tuple = (None, frame)
        if f_id in bboxes:
            dataset_tuple = (f_id, frame)
        dataset.append(dataset_tuple)

    print("Number of frames: ", len(dataset))
    if model_cfg["use_OF"]:
        bboxes = resize_bboxes(bboxes)
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
    else:
        track_bbs_ids_overlap = get_overlap_track(
            out_path=paths["output"],
            dataset=dataset,
            bboxes=bboxes,
            first_frame=model_cfg['first_frame'],
            last_frame=model_cfg['last_frame'],
            visualization_cfg=visualization_cfg
        )
    if model_cfg["save_tracking_MOT"]:
        save_track(track_bbs_ids_overlap, paths['output'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2.yaml")
    # parser.add_argument("--config", default="week3/configs/task1.yaml") #comment if you are not using Visual Studio
    # Code
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
