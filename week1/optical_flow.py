import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from week1.flow_vis import flow_to_color


def read_optical_flow(path: str) -> (np.ndarray, np.ndarray):
    flow = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype("float32")

    # flow without occlusions
    new_flow = np.zeros((flow.shape[0], flow.shape[1], 2), dtype="float32")

    u_flow = (flow[:, :, 2] - 2 ** 15) / 64
    v_flow = (flow[:, :, 1] - 2 ** 15) / 64
    mask = flow[:, :, 0].astype('bool')

    new_flow[:, :, 0] = u_flow
    new_flow[:, :, 1] = v_flow

    return mask, new_flow


def compute_error(flow_gt: np.ndarray, flow_det: np.ndarray) -> float:
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


def plot_flow(flow_array: np.ndarray):
    """
    Plots the flow vectors in a 3x3 grid.

    :param flow_array: numpy array of shape (H, W, 3) containing the flow vectors
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    images = np.dsplit(flow_array, 3)
    titles = ['u_flow', 'v_flow', 'v_mask']

    for ax, image, title in zip(axes.flatten(), images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def show_field(flow, gray, step=30, scale=0.5):
    plt.figure(figsize=(16, 8))
    plt.imshow(gray, cmap='gray')

    u = flow[:, :, 0]
    v = flow[:, :, 1]
    h = np.hypot(u, v)

    (height, w) = flow.shape[0:2]
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, height))

    x = x[::step, ::step]
    y = y[::step, ::step]
    u = u[::step, ::step]
    v = v[::step, ::step]
    h = h[::step, ::step]

    plt.quiver(x, y, u, v, h, scale_units='xy', angles='xy', scale=scale)

    plt.axis('off')
    plt.show()


def plot_flow_to_color(flow: np.ndarray):
    plt.figure(figsize=(16, 8))
    plt.axis('off')
    plt.imshow(flow_to_color(flow))
    plt.show()
