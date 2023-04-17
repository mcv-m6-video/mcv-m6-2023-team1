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
from src.plot_utils import Annotator, colors
import numpy as np
import yaml
from tqdm import tqdm


def update_tracker(mot_tracker, boxes):
    np_boxes = np.array(boxes)
    track_bbs_ids = mot_tracker.update(np_boxes)[::-1]
    return track_bbs_ids


def draw_boxes(im, boxes):
    annotator = Annotator(im, line_width=3)
    for *xyxy, bbox_id in boxes:
        label = f"{int(bbox_id)}"
        annotator.box_label(xyxy, label, color=colors(bbox_id, True))

    annotated_frame = annotator.result()
    return annotated_frame


def check_dir(detections_dir):
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(f"data/detection_{detections_dir}"):
        os.makedirs(f"data/detection_{detections_dir}", exist_ok=True)
        return False
    else:
        return True


def extract_detections(grouped_imgs, model_path):
    detections_dir = model_path.split("/")[-1].split(".")[0]
    dir_exists = check_dir(detections_dir)

    # if detections do not exist
    if not dir_exists:
        print("extracting detections...")

        model = YOLO(model_path)  # load a pretrained model (recommended for training)
        cam_ids = [*grouped_imgs.keys()]
        for cam_id in cam_ids:
            os.makedirs(f"data/detection_{detections_dir}/{cam_id}", exist_ok=True)

        bboxes = {}
        for cam_id, imgs in grouped_imgs.items():
            cam_boxes = []
            print(f"Extracting detections from CAM {cam_id}")
            for frame_id, img in tqdm(enumerate(imgs)):
                result = model.predict(img, conf=0.5, device=0)
                boxes = result[0].boxes.boxes
                boxes = boxes.detach().cpu().numpy()
                np_frame_id = np.ones(boxes.shape[0]) * frame_id
                boxes = np.insert(boxes, 0, np_frame_id, axis=1).tolist()
                cam_boxes = cam_boxes + boxes
            save_bboxes_to_file(cam_boxes, f"data/detection_{detections_dir}/{cam_id}/det.yaml")
            bboxes[cam_id] = cam_boxes
    # if detections already exist
    else:
        print(f"Detections of model {model_path} already exist in data/detection_{detections_dir}")
        bboxes = {}
        cams = os.listdir(f"data/detection_{detections_dir}")
        for i, cam_id in enumerate(cams):
            print(f"Loading detections from CAM {cam_id} {i+1}/{len(cams)}")
            cam_boxes = load_bboxes_from_file(f"data/detection_{detections_dir}/{cam_id}/det.yaml")
            bboxes[cam_id] = reformat_detections(cam_boxes)
    return bboxes


def reformat_detections(detections):
    # transform list os detections into diccionary where key=frame_id
    dict_detections = {}
    for det in detections:
        key = int(det[0])
        value = det[1:]
        if key in dict_detections.keys():
            dict_detections[key].append(value)
        else:
            dict_detections[key] = [value]
    return dict_detections


def extract_frames(data_path):
    img_paths = glob.glob(f"{data_path}/*.jpg")
    img_paths = [i.replace("\\", "/") for i in img_paths]
    grouped_imgs = {}
    for img_path in img_paths:
        cam_id = img_path.split("/")[-1][4:8]
        if cam_id in grouped_imgs.keys():
            grouped_imgs[cam_id].append(img_path)
        else:
            grouped_imgs[cam_id] = [img_path]
    return grouped_imgs


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
    fps = 30  # Set the frame rate
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
