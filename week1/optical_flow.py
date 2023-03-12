import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_optical_flow(path):
    flow = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype("float32")

    # flow without occlusions
    new_flow = np.zeros((flow.shape[0], flow.shape[1], 2), dtype="float32")

    u_flow = (flow[:, :, 2] - 2 ** 15) / 64
    v_flow = (flow[:, :, 1] - 2 ** 15) / 64
    mask = flow[:, :, 0].astype('bool')

    new_flow[:, :, 0] = u_flow
    new_flow[:, :, 1] = v_flow

    return mask, new_flow


def compute_error(flow_gt, flow_det):
    """
    compute squared difference (error) between gt and detection
    """
    diff = flow_det - flow_gt
    error = np.sqrt(np.sum(diff ** 2, axis=1))
    return error


def compute_pepn(error, th=3):
    return np.sum(error > th) / len(error)


def compute_msen(error):
    return np.mean(error)


def visualize_error(error, mask):
    error_image = np.zeros(mask.shape)
    error_image[mask] = error
    plt.imshow(error_image, cmap="hot")
    plt.title("Squared error")
    plt.show()


def histogram_error(error):
    plt.hist(error)
    plt.title("Histogram of error for non-occluded areas")
    plt.xlabel("Squared error")
    plt.ylabel("Number of pixels")
    plt.show()
