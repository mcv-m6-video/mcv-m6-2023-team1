"""
Visualize single camera tracking
"""

from src.sort import Sort
import argparse
import sys
from ultralytics import YOLO
from src.io_utils import open_config_yaml
import glob
import cv2
from src.plot_utils import Annotator, colors


def init_windows(cam_ids, w, h):
    for cam_id in cam_ids:
        cv2.namedWindow(f"{cam_id}", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(f"{cam_id}", w, h)


def update_tracker(mot_tracker, boxes):
    np_boxes = boxes.cpu().detach().numpy()
    track_bbs_ids = mot_tracker.update(np_boxes)[::-1]
    return track_bbs_ids


def draw_boxes(im, boxes):
    annotator = Annotator(im, line_width=3)
    for *xyxy, bbox_id in boxes:
        label = f"{int(bbox_id)}"
        annotator.box_label(xyxy, label, color=colors(bbox_id, True))

    annotated_frame = annotator.result()
    return annotated_frame


def main(cfg):
    model = YOLO(cfg["model_path"])  # load a pretrained model (recommended for training)

    img_paths = glob.glob(f"{cfg['data_path']}/*.jpg")
    img_paths = [i.replace("\\", "/") for i in img_paths]
    img_paths.sort()
    grouped_imgs = {}
    for img_path in img_paths:
        cam_id = img_path.split("/")[-1][4:8]
        if cam_id in grouped_imgs.keys():
            grouped_imgs[cam_id].append(img_path)
        else:
            grouped_imgs[cam_id] = [img_path]

    cam_ids = [*grouped_imgs.keys()]
    init_windows(cam_ids, 640, 360)

    mot_trackers = [Sort(track_id=cam_id) for cam_id in cam_ids]

    for img_batch in zip(*list(grouped_imgs.values())):
        results = model.predict(img_batch, conf=0.5, device=0)

        for i, result in enumerate(results):
            tracking_boxes = update_tracker(mot_trackers[i], result.boxes.boxes)
            annotated_frame = draw_boxes(result.orig_img, tracking_boxes)

            # Display the annotated frame
            cv2.imshow(f"{cam_ids[i]}", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/track_single_camera.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
