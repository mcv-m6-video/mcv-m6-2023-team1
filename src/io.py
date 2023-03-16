import csv
import xml.etree.ElementTree as elemTree

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
