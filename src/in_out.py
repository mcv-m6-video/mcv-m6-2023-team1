import csv
import os
import xml.etree.ElementTree as elemTree
from typing import List
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
from src.utils import sort_dict


def extract_rectangles_from_csv(path):
    """
    Parses an XML annotation file in the csv format and extracts bounding box coordinates for cars in each frame.

    Args:
        - Path to annotation csv in AI City format
    returns:
        dict[frame_num] = [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
        in top left and bottom right coordinates
    """
    ret_dict = {}

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        # next(reader)  # skip header row

        for row in reader:
            frame_num = f'f_{int(row[0]) - 1}'

            if frame_num not in ret_dict:
                ret_dict[frame_num] = []

            # convert from center, width, height to top left, bottom right
            ret_dict[frame_num].append(
                [float(row[2]), float(row[3]), float(row[2]) + float(row[4]), float(row[3]) + float(row[5]),
                 float(row[6])]
            )

    return sort_dict(ret_dict)


def extract_rectangles_from_xml(path_to_xml_file):
    """
    Parses an XML annotation file in the Pascal VOC format and extracts bounding box coordinates for cars in each frame.
    Args:
    - path_to_xml_file: path to the annotation XML file.

    Returns:
    - A dictionary of frame numbers and their corresponding car bounding box coordinates.
        example: dict = {'f_0': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]]}
        where x1, y1, x2, y2 are the coordinates of the bounding box in the top left and bottom right corners.

    """

    tree = elemTree.parse(path_to_xml_file)
    root = tree.getroot()

    # Initialize an empty dictionary to hold the frame numbers and their corresponding bounding box coordinates
    frame_dict = {}

    # Loop through each 'track' element in the XML file with a 'label' attribute of 'car'
    for track in root.findall(".//track[@label='car']"):

        # Loop through each 'box' element within the 'track' element to get the bounding box coordinates
        for box in track.findall(".//box"):

            # Extract the bounding box coordinates and the frame number
            x1 = float(box.attrib['xtl'])
            y1 = float(box.attrib['ytl'])
            x2 = float(box.attrib['xbr'])
            y2 = float(box.attrib['ybr'])
            frame_num = f"f_{box.attrib['frame']}"

            # If the frame number doesn't exist in the dictionary yet, add it and initialize an empty list
            if frame_num not in frame_dict:
                frame_dict[frame_num] = []

            # Append the bounding box coordinates to the list for the current frame number
            frame_dict[frame_num].append([x1, y1, x2, y2])

    return sort_dict(frame_dict)


def extract_not_parked_rectangles_from_xml(path_to_xml_file):
    """
    Parses an XML annotation file in the Pascal VOC format and extracts bounding box coordinates for cars in each frame.
    Args:
    - path_to_xml_file: path to the annotation XML file.

    Returns:
    - A dictionary of frame numbers and their corresponding car bounding box coordinates.
        example: dict = {'f_0': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]]}
        where x1, y1, x2, y2 are the coordinates of the bounding box in the top left and bottom right corners.

    """

    tree = elemTree.parse(path_to_xml_file)
    root = tree.getroot()

    # Initialize an empty dictionary to hold the frame numbers and their corresponding bounding box coordinates
    frame_dict = {}

    # Loop through each 'track' element in the XML file with a 'label' attribute of 'car'
    for track in root.findall(".//track[@label='car']"):

        # Loop through each 'box' element within the 'track' element to get the bounding box coordinates
        for box, parked in zip(track.findall(".//box"), track.findall(".//attribute[@name='parked']")):
            # Extract the bounding box coordinates and the frame number
            x1 = float(box.attrib['xtl'])
            y1 = float(box.attrib['ytl'])
            x2 = float(box.attrib['xbr'])
            y2 = float(box.attrib['ybr'])
            frame_num = f"f_{box.attrib['frame']}"

            # If the frame number doesn't exist in the dictionary yet, add it and initialize an empty list
            if frame_num not in frame_dict:
                frame_dict[frame_num] = []

            # Append the bounding box coordinates to the list for the current frame number
            if parked.text == 'false':
                frame_dict[frame_num].append([x1, y1, x2, y2])

    return sort_dict(frame_dict)


def extract_frames_from_video(video_path: str, output_path: str) -> None:
    """
    Extract frames from a video and save them to a directory.
    :param video_path: path to the video
    :param output_path: path to the output directory
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    elif len(os.listdir(output_path)) > 0:
        print(f"Output directory {output_path} already exists and has content. Skipping extraction.")
        return

    video_capture = cv2.VideoCapture(video_path)  # Open the video file
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        output_frame_path = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_frame_path, frame)
        frame_count += 1
    video_capture.release()


def get_frames_paths_from_folder(input_path: str) -> np.ndarray:
    """
    Loads frames from a folder into a numpy array.
    """
    image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if
                   os.path.isfile(os.path.join(input_path, f)) and f.endswith('.jpg')]
    image_files.sort()
    return np.array(image_files)


def load_images(paths_to_images: List[str], grayscale: bool = True) -> np.ndarray:
    """
    Loads the images from the given paths into a numpy array.
    :param paths_to_images: list of paths to the images
    :param grayscale: 'rgb' or 'gray'
    :return: numpy array of images
    """
    images = []
    for path in tqdm(paths_to_images):
        images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR))
    return np.array(images)
def plot_3d_scatter(map_values, alpha_values, rho_values):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(alpha_values,rho_values, map_values)

    ax.set_xlabel('Alpha')
    ax.set_ylabel('Rho')
    ax.set_zlabel('mAP')

    plt.show()
