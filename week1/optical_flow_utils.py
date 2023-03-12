import cv2
import numpy as np
from matplotlib import pyplot as plt


def read_flow(path: str) -> np.ndarray:
    """
    Reads a flow file and returns the flow vectors as a numpy array of shape (H, W, 3).

    The flow file is expected to be a 3-channel uint16 PNG image where the first channel contains the valid mask
    (where 1 denotes a valid flow vector), the second channel the v-component and the third channel the u-component.

    :param path: path to the flow file
    :return: numpy array of shape (H, W, 3) containing the flow vectors
    """
    # cv2 flips the order of reading channels
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)

    # valid mask
    v_mask = img[:, :, 0]

    # get flow vectors
    u_flow = (img[:, :, 2] - 2 ** 15) / 64
    v_flow = (img[:, :, 1] - 2 ** 15) / 64

    v_mask[v_mask > 1] = 1

    # remove invalid flow values
    u_flow[v_mask == 0] = 0
    v_flow[v_mask == 0] = 0

    # return image in correct order
    return np.dstack((u_flow, v_flow, v_mask))


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
