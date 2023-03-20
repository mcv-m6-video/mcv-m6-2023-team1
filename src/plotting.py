import os
from typing import List, Dict

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.metrics import get_frame_mean_IoU
from datetime import datetime
from tqdm import tqdm


def save_results(bbox_preds, preds, gt_test_bboxes, test_imgs_paths, multiply255 = True,
                 save_just_image=False, save_GTmask=False):
    """
    Save results from background substraction

    :param bbox_preds:
    :param preds:
    :param gt_labels:
    :param test_imgs_paths:
    :return:
    """
    # Save results in outputs
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d_%H-%M-%S')
    output_path = "../outputs/"+date_string
    os.makedirs("../outputs", exist_ok=True)  # create outputs directory in case is not present
    os.mkdir(output_path)
    os.mkdir(output_path + "/bboxes")
    os.mkdir(output_path + "/masks")
    os.mkdir(output_path + "/gt_masks")

    print("Saving results")
    for i, (gt_test_bbox, pred, bbox_pred, test_img_path) in tqdm(enumerate(zip(gt_test_bboxes, preds, bbox_preds, test_imgs_paths))):
        # Draw bounding boxes on the original image for visualization
        output_img = cv2.imread(test_img_path)
        if not save_just_image:
            for bbox in bbox_pred:
                cv2.rectangle(output_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            for bbox in gt_test_bbox:
                cv2.rectangle(output_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        if save_GTmask:
            #create a mask with black pixels in bg, and white pixels in the places where there is a car
            gt_mask = np.zeros((output_img.shape[0], output_img.shape[1]))
            for bbox in gt_test_bbox:
                gt_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255
            #save gt mask
            cv2.imwrite(f"{output_path}/gt_masks/{str(i).zfill(4)}.png", gt_mask)
        #multiply by 255 to save as png if multiply255 is True
        if multiply255:
            pred = pred*255
        cv2.imwrite(f"{output_path}/masks/{str(i).zfill(4)}.png", pred)
        cv2.imwrite(f"{output_path}/bboxes/{str(i).zfill(4)}.png", output_img)


def plot_frame(
        frame: str,
        gt_rects: List,
        det_rects: List,
        path_to_video: str,
        frame_iou: float = None,
        save_frame: bool = False, file_path: str = None,
        no_confidence: bool = False
) -> None:
    """
    Plots the frame and the ground truth bounding boxes.
    :param frame: frame number
    :param gt_rects: list of ground truth bounding boxes
    :param det_rects: list of detected bounding boxes
    :param path_to_video: path to the video file
    :param frame_iou: frame IoU value
    :param save_frame: whether to save the frame
    :param file_path: path to save the frame
    :param no_confidence: whether the detection bounding boxes have confidence values
    :return: None
    """
    frame_str_num = frame

    # Read the video file
    cap = cv2.VideoCapture(path_to_video)

    # Set the frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))

    # Read the frame
    ret, frame = cap.read()

    # Plot the frame in RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)

    # Plot the bounding boxes
    for rect in gt_rects:
        x1, y1, x2, y2 = rect
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=3))
    for rect in det_rects:
        if no_confidence:
            x1, y1, x2, y2 = rect
            conf = 1
        else:
            x1, y1, x2, y2, conf = rect
        # plot rectangle and confidence on top
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='green', linewidth=2))
        plt.gca().text(x1, y1, 'Conf: {:.3f}'.format(conf), bbox=dict(facecolor='green', alpha=0.5), fontsize=6)

    if frame_iou is not None:
        plt.title(f"Frame {frame_str_num} IoU: {frame_iou:.3f}")
    if not save_frame:
        plt.show()
    elif save_frame:
        plt.savefig(file_path)
        plt.close()


def plot_iou_vs_frames(ious_list: List, file_path: str = None, save_fig: bool = False) -> None:
    """
    Plots the graph of iou over the frames for a sequence.
    :param ious_list: list of ious for each frame
    :param file_path: path to save the plot
    :param save_fig: whether to save the plot or not
    :return: None
    """

    # Plot the iou during the frames in a fixed single plot

    frames = range(len(ious_list))
    fig, ax = plt.subplots()
    ax.plot(frames, ious_list, linewidth=0.5)
    ax.set(xlim=(0, 2140), xticks=np.arange(0, 2140, 250),
           ylim=(0, 1), yticks=np.arange(0, 1, 0.1))
    if not save_fig:
        plt.show()
    elif save_fig:
        plt.savefig(file_path)
        plt.close()


def make_gif(gt_bboxes_dict: Dict, det_bboxes_dict: Dict, cfg: Dict) -> None:
    """
    Creates a gif of the detections and the IoU over the frames.
    :param gt_bboxes_dict: dictionary of ground truth bounding boxes for each frame
    :param det_bboxes_dict: dictionary of detected bounding boxes for each frame
    :param cfg: config dictionary
    :return: None
    """
    ious_list = []
    ious_files_prefix = "gif_images/iou_vs_frames_plots"
    detection_files_prefix = "gif_images/detection_plots"
    if not os.path.exists("gif_images"):
        os.mkdir("gif_images")
        os.mkdir(ious_files_prefix)
        os.mkdir(detection_files_prefix)
    for frame in list(gt_bboxes_dict.keys()):
        ious_list.append(get_frame_mean_IoU(gt_bboxes_dict[frame], det_bboxes_dict[frame]))
        if frame % 5 == 0:
            ious_filepath = os.path.join(ious_files_prefix, str(frame) + ".png")
            detection_filepath = os.path.join(detection_files_prefix, str(frame) + ".png")
            plot_frame(frame, gt_bboxes_dict[frame], det_bboxes_dict[frame], cfg["paths"]["video"], save_frame=True,
                       file_path=detection_filepath, no_confidence=True)
            plot_iou_vs_frames(ious_list, file_path=ious_filepath, save_fig=True)
