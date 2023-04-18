import yaml
import os
from tqdm import tqdm
import numpy as np
import glob
from ultralytics import YOLO


def open_config_yaml(cfg_path):
    """
    Open config file and return it as a dictionary
    :param cfg_path: path to config file
    :return: config dictionary
    """
    # Read Config file
    with open(cfg_path, 'r') as f:
        config_yaml = yaml.safe_load(f)

    return config_yaml


def create_dirs(dst_path):
    os.makedirs("data", exist_ok=True)
    os.makedirs(dst_path, exist_ok=True)
    print(f"Creating dataset in: {dst_path}")
    os.makedirs(f"{dst_path}/train", exist_ok=True)
    os.makedirs(f"{dst_path}/val", exist_ok=True)


def read_gt(gt_path):
    # Open the file for reading
    with open(gt_path, 'r') as file:
        # Initialize an empty dictionary
        my_dict = {}
        # Loop through each line in the file
        for line in file:
            # Split the line into columns
            columns = line.strip().split(",")
            # Use the first column as the key and the rest as the values
            key = columns[0]
            values = columns[1:]
            # Add the key-value pair to the dictionary
            if key not in my_dict.keys():
                my_dict[key] = [values]
            else:
                my_dict[key].append(values)
    return my_dict


def save_bboxes_to_file(bboxes, path):
    """
    Saves the bounding boxes to a file.
    :param bboxes: dictionary of bounding boxes
    :param path: path to save the file
    """
    with open(path, "w") as f:
        yaml.dump(bboxes, f)


def load_bboxes_from_file(path):
    # open the YAML file for reading
    with open(path, 'r') as file:
        # load the YAML data from the file
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_data


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
