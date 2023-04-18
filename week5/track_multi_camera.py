"""
This code is inspired by Team1 2022 in order to evaluate reid
"""
from typing import Dict

import numpy as np
import argparse
import sys
from src.io_utils import open_config_yaml

from pytorch_metric_learning import testers
from src.models import Embedder, HeadlessResnet
from torchvision.datasets import ImageFolder
from torchvision import transforms

from src.io_utils import extract_frames
from scipy.spatial.distance import cdist
import cv2
from src.plot_utils import draw_boxes
import os

from week3.task2 import write_PASCAL_to_MOT_txt_w5


def get_transforms():
    augmentations = {
        "val":
            transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    }
    return augmentations


def assign_ids(dist_matrix, threshold, dataset):
    # generate metadata to be able to have tracking
    img_paths = [sample[0].replace("\\", "/") for sample in dataset.samples]
    metadata = [i.split("/")[-1].split(".")[0].split("_")[1:] for i in img_paths]

    # REID, assign ids depending on distance
    num_entities = dist_matrix.shape[0]
    ids = np.zeros(num_entities, dtype=int)
    current_id = 1
    for i in range(num_entities):
        if ids[i] == 0:
            ids[i] = current_id
            for j in range(i + 1, num_entities):
                if dist_matrix[i, j] <= threshold:
                    ids[j] = current_id
            current_id += 1

    # add tracking id to metadata
    for i, id_ in enumerate(ids):
        metadata[i].append(id_)
        metadata[i][1] = int(metadata[i][1])
        metadata[i][2] = int(metadata[i][2])
        metadata[i][3] = int(metadata[i][3])
        metadata[i][4] = metadata[i][2] + int(metadata[i][4])  # x1 = x0 + width
        metadata[i][5] = metadata[i][3] + int(metadata[i][5])  # y1 = y0 + height
    metadata = sorted(metadata, key=lambda x: x[1])
    trackings = {}
    for lst in metadata:
        key = lst[0]
        if key not in trackings.keys():
            trackings[key] = {}
        subkey = lst[1]
        if subkey not in trackings[key].keys():
            trackings[key][subkey] = []
        value = lst[2:]
        trackings[key][subkey].append(value)

    return trackings


def save_tracking_videos(frames, trackings):
    os.makedirs("tracking_videos", exist_ok=True)
    # Set up the video writer
    fps = 20  # Set the frame rate
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for i, cam_id in enumerate(frames.keys()):
        print(f"Saving tracking video for CAM {cam_id} {i+1}/{len(frames.keys())}")
        frame_size = cv2.imread(frames[cam_id][0]).shape
        out = cv2.VideoWriter(f'tracking_videos/tracking_reid_{cam_id}.avi', fourcc, fps, (frame_size[1], frame_size[0]))
        for frame_id, frame_path in enumerate(frames[cam_id]):
            frame = cv2.imread(frame_path)
            if frame_id in trackings[cam_id].keys():
                frame = draw_boxes(frame, trackings[cam_id][frame_id])
            out.write(frame)
        out.release()


def postprocess(tracking_per_cam: Dict[str, Dict], max_frames_per_cam: Dict[str, int]):
    """
    This function fills the gaps in the tracking dict, by adding an empty list for each frame
    where no object was detected.
    """
    for cam_id, tracking in sorted(tracking_per_cam.items()):
        for frame_id in range(max_frames_per_cam[cam_id]):
            if frame_id not in tracking.keys():
                tracking_per_cam[cam_id][frame_id] = []
    return tracking_per_cam


def main(cfg):
    trunk_model = HeadlessResnet(cfg["trunk_weights_path"]).to(cfg["device"])
    embedder_model = Embedder(512, cfg["embedder_size"], cfg["embedder_weights_path"]).to(cfg["device"])

    # set datasets
    augmentations = get_transforms()
    dataset = ImageFolder(cfg["val_path"], transform=augmentations["val"])

    tester = testers.GlobalEmbeddingSpaceTester()
    embeddings = tester.get_all_embeddings(dataset, trunk_model=trunk_model, embedder_model=embedder_model)

    labels = embeddings[1]
    embeddings = embeddings[0]
    labels = labels.flatten()

    # %%
    labels = labels.detach().cpu().numpy()
    embeddings = embeddings.detach().cpu().numpy()

    distances = cdist(embeddings, embeddings, metric="euclidean")
    distances = np.triu(distances, k=1)
    valid = np.ones(distances.shape, dtype=bool)
    valid = np.triu(valid, k=1)

    # %%
    margin = cfg["th"]

    positives = distances < margin
    negatives = distances > margin

    same_label = labels[:, None] == labels[None, :]

    true_positives = np.count_nonzero(positives & same_label & valid)
    true_negatives = np.count_nonzero(negatives & np.logical_not(same_label) & valid)

    false_positives = np.count_nonzero(positives & np.logical_not(same_label) & valid)
    false_negatives = np.count_nonzero(negatives & same_label & valid)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1score = (2 * precision * recall) / (precision + recall)

    print(
        f"TP: {true_positives} \t  TN: {true_negatives} \n"
        f"FP: {false_positives} \t FN: {false_negatives} \n"
        f"--------------------------------------------------------------------------------\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"F1 Score: {f1score}\n"
        f"--------------------------------------------------------------------------------\n"
        f"Total Sum: {sum([true_positives, true_negatives, false_positives, false_negatives])} \n"
        f"Valid: {np.count_nonzero(valid)}"
    )
    # use reid to do tracking
    tracking = assign_ids(distances, 0.8, dataset)
    if cfg['save_tracking']:
        frames_per_camera = {
            'c010': 2141,
            'c011': 2279,
            'c012': 2422,
            'c013': 2415,
            'c014': 2332,
            'c015': 1928,

        }
        post_tracking = postprocess(tracking, frames_per_camera)
        for camera in tracking.keys():
            os.makedirs(f"tracking", exist_ok=True)
            write_PASCAL_to_MOT_txt_w5(post_tracking[camera], f"tracking/{camera}.txt")
    if cfg['save_tracking_videos']:
        frames = extract_frames(cfg['data_path'])
        save_tracking_videos(frames, tracking)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/track_multi_camera.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = open_config_yaml(config_path)

    main(config)
