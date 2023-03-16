import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    titles = ['u_flow', 'v_flow', 'mask']

    for ax, image, title in zip(axes.flatten(), images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def show_field(flow: np.ndarray, gray: np.ndarray, step: int = 30, scale: float = 0.5):
    """
    Plots the flow vectors on top of the gray image.

    :param flow: flow field, only the first two channels are used
    :param gray: gray image
    :param step: step size for the flow vectors, every step-th vector is plotted
    :param scale: scale of the flow vectors, the length of the vectors is multiplied by this value
    """
    plt.figure(figsize=(16, 8))
    plt.imshow(gray, cmap='gray')

    u = flow[:, :, 0]
    v = flow[:, :, 1]
    hyp = np.hypot(u, v)

    (h, w) = flow.shape[0:2]
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    # For every step-th vector, take the x, y, u, v, hyp values
    x, y, u, v, hyp = (a[::step, ::step] for a in (x, y, u, v, hyp))

    plt.quiver(x, y, u, v, hyp, scale_units='xy', angles='xy', scale=scale, cmap='magma')

    plt.axis('off')
    plt.show()


def draw_opt_flow_magnitude_and_direction(flow: np.ndarray):
    """
    Function to plot the magnitude and direction of an optical flow field.
    :param flow: np.array
    """
    # Compute the magnitude and direction of the flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)

    # Clip the highest magnitude values according to the 0.95 quantile
    clip_th = np.quantile(magnitude, 0.95)
    magnitude = np.clip(magnitude, 0, clip_th)

    # Normalize
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert direction to HSV color space
    hsv = np.zeros_like(flow, dtype=np.uint8)
    hsv[..., 0] = angle / 2 / np.pi * 179
    hsv[..., 1] = magnitude
    hsv[..., 2] = 255
    direction = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Plots
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].set_title('Optical Flow Magnitude')
    axs[0].imshow(magnitude, cmap='gray')
    axs[1].set_title('Optical Flow Direction')
    axs[1].imshow(direction)
    plt.show()
