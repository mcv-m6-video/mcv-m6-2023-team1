import argparse
import cv2
import os
import sys

# sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")) # comment if you are not using VSC
import numpy as np
from tqdm import tqdm

# Import Sort for tracking using Kalman filter
from sort import Sort
from src.in_out import extract_frames_from_video, get_frames_paths_from_folder, extract_rectangles_from_xml
from src.utils import open_config_yaml, load_bboxes_from_file, draw_img_with_ids, draw_bboxes_trajectory


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
    pass


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
    overlay = np.zeros_like(cv2.imread(dataset[0][1]))
    for i, frame in tqdm(enumerate(dataset)):
        if first_frame < i < last_frame:
            img = cv2.imread(frame[1])
            # Draw the trajectory lines
            overlay = draw_bboxes_trajectory(overlay, track_bbs_ids[i], track_bbs_ids[i - 1])
            # Draw the bounding boxes with ID number
            img_out = draw_img_with_ids(img, track_bbs_ids[i])
            # Fuse both images
            img_out = cv2.addWeighted(img_out, 1, overlay, 1, 0)

            # Show the frame with the trajectory and bounding boxes IDs
            if visualization_cfg['show_detection']:
                cv2.imshow('frame2', img_out)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

            # Save the frame
            if visualization_cfg['save']:
                os.makedirs(out_path, exist_ok=True)
                cv2.imwrite(os.path.join(out_path, str(i) + '.png'), img_out,
                            [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


def task2_3(cfg):
    pass


def main(cfg):
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    visualization_cfg = cfg["visualization"]

    # get detections
    gt_labels = extract_rectangles_from_xml(paths["annotations"])
    if model_cfg['use_gt']:
        gt_bboxes = [*gt_labels.values()]
        bboxes = gt_bboxes[int(len(gt_labels) * 0.25):]
    else:
        bboxes = load_bboxes_from_file(paths['detected_bboxes'])

    # Obtain all frames of the sequence
    extract_frames_from_video(video_path=paths["video"], output_path=paths["extracted_frames"])
    frames = get_frames_paths_from_folder(input_path=paths["extracted_frames"])
    dataset = [(key, frames[key]) for key in gt_labels.keys()]

    # keep only frames of selected range
    # dataset = dataset[first_frame:last_frame]
    dataset = dataset[int(len(dataset) * 0.25):]
    print("Number of frames: ", len(dataset))
    # task2_1()
    task2_2(
        out_path=paths["output"],
        dataset=dataset,
        bboxes=bboxes,
        first_frame=model_cfg['first_frame'],
        last_frame=model_cfg['last_frame'],
        visualization_cfg=visualization_cfg
    )
    # task2_3(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2.yaml")
    # parser.add_argument("--config", default="week3/configs/task2_2.yaml") #comment if you are not using VSCode
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)
    main(config)
