import xml.etree.ElementTree as ET

import cv2
from matplotlib import pyplot as plt


def extract_rectangles_from_xml(path_to_xml_file):
    """
    Parses an XML annotation file in the Pascal VOC format and extracts bounding box coordinates for cars in each frame.
    Args:
    - path_to_xml_file: path to the annotation XML file.

    Returns:
    - A dictionary of frame numbers and their corresponding car bounding box coordinates.
    """

    tree = ET.parse(path_to_xml_file)
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

    return frame_dict

def plot_frame(frame, GT_rects, path_to_video):
    """
    Plots the frame and the ground truth bounding boxes.
    Args:
    - frame: frame number
    - GT_rects: list of bounding box coordinates
    - path_to_video: path to the video file
    """

    # Read the video file
    cap = cv2.VideoCapture(path_to_video)

    # Set the frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame[2:]))

    # Read the frame
    ret, frame = cap.read()

    # Plot the frame in RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)

    # Plot the bounding boxes
    for rect in GT_rects:
        x1, y1, x2, y2 = rect
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=3))

    plt.show()