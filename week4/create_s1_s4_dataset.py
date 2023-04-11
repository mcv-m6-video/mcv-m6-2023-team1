import argparse
import os
import sys

from src.in_out import extract_frames_from_video, extract_rectangles_from_xml_detection
from src.utils import open_config_yaml
from week3.yolov8.yolo_dataset import generate_rectangles_yolo


def extract_yolo_annots_from_txt(annot_path: str, dataset_path: str, camera: str):
    rectangles = extract_rectangles_from_xml_detection("../dataset/ai_challenge_s03_c010-full_annotation.xml")
    rectangles_by_frame = {}
    txt_to_voc_labels = {
        1: "car",
        0: "bike",
    }
    with open(annot_path, "r") as f:
        for line in f:
            frame, track_id, x1, y1, x2, y2, class_id, _, _, _ = map(int, line.strip().split(","))
            class_id = txt_to_voc_labels[class_id]
            if frame not in rectangles_by_frame:
                rectangles_by_frame[frame] = []
            rectangles_by_frame[frame].append([track_id, class_id, x1, y1, x2, y2])
    generate_rectangles_yolo(rectangles_by_frame, dataset_path, camera=camera)


def main(cfg):
    paths = cfg['paths']
    sequence_path = paths['sequence_path']
    extracted_frames = paths['extracted_frames']
    # Obtain all frames for each camera on the sequence
    for camera in os.listdir(sequence_path):
        video_path = f'{sequence_path}/{camera}/vdo.avi'
        extracted_frames_path = extracted_frames + '/' + camera
        extract_frames_from_video(video_path=video_path, output_path=extracted_frames_path, camera=camera)
        annot_path = f'{sequence_path}/{camera}/gt/gt.txt'
        extract_yolo_annots_from_txt(annot_path, extracted_frames_path, camera=camera)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/create_dataset.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
