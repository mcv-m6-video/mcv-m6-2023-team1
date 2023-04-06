import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.metrics import mean_absolute_error, compute_distance

def report_errors_OF(gt_path, seq_n, flow_det):
    mask, flow_gt = read_optical_flow(f"{gt_path}/{seq_n}")
    flow_noc_det = flow_det[mask]
    flow_noc_gt = flow_gt[mask]
    error = compute_error(flow_noc_gt, flow_noc_det)
    msen = compute_msen(error)
    pepn = compute_pepn(error)

    return mask, error, msen, pepn


def report_errors_OF_2(flow_gt, flow_det, mask):
    flow_noc_det = flow_det[mask]
    flow_noc_gt = flow_gt[mask]
    error = compute_error(flow_noc_gt, flow_noc_det)
    msen = compute_msen(error)
    pepn = compute_pepn(error)

    return error, msen, pepn


def read_optical_flow(path: str) -> (np.ndarray, np.ndarray):
    flow = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype("float32")

    u_flow = (flow[:, :, 2] - 2 ** 15) / 64
    v_flow = (flow[:, :, 1] - 2 ** 15) / 64
    mask = flow[:, :, 0].astype('bool')

    new_flow = np.zeros((flow.shape[0], flow.shape[1], 2), dtype="float32")
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
    plt.colorbar()
    plt.show()

def histogram_error(error):
    plt.hist(error, bins=50)
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


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    plt.imshow(vis)
    plt.show()


def draw_hsv(flow, method, scale=4):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    if method != 'pyflow':
        # scales if not pyflow xq no se ve una kk
        hsv[..., 2] = np.minimum(mag * scale, 255)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    plt.imshow(rgb)
    plt.show()


def save_plot_OF(OF, path):
    # convert the flow to a three-channel image with hue for direction and value for magnitude
    mag, ang = cv2.cartToPolar(OF[:,:,0], OF[:,:,1])
    hsv = np.zeros((OF.shape[0], OF.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 1] = 255

    # convert the HSV image to BGR for display and saving
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # save the image as a PNG file
    cv2.imwrite(path, bgr)


def OF_block_matching(img1, img2, area_of_search=(60,60), block_size=(8,8), step_size=(4,4), error_function=mean_absolute_error):
    """
    Block matching algorithm for optical flow estimation.
    Args:
        img1 and img2: the two consecutive frames of the video sequence.
        area_of_search: the size of the search area around each block.
        block_size: the size of the blocks used for matching.
        step_size: the stride used for sliding the blocks over the images.
        error_function: the function used to compute the error between two blocks.

    Returns:
        A 3D numpy array of size (h, w, 2), where the third dimension corresponds to the horizontal and vertical components of the optical flow vectors.
    """
    h, w = img1.shape
    num_blocks_y = h // block_size[0]
    num_blocks_x = w // block_size[1]

    flow = np.zeros((h, w, 2), dtype=np.float32)

    for y in range(0, h - block_size[0], step_size[0]):
        for x in range(0, w - block_size[1], step_size[1]):
            # Define the search area for the current block
            min_y = max(0, y - area_of_search[0])
            max_y = min(h - block_size[0], y + area_of_search[0])
            min_x = max(0, x - area_of_search[1])
            max_x = min(w - block_size[1], x + area_of_search[1])

            # Extract the block from the first image
            block1 = img1[y:y+block_size[0], x:x+block_size[1]]

            # Initialize variables for the best match
            min_error = float('inf')
            best_dx, best_dy = 0, 0

            # Search for the best match within the search area
            for dy in range(min_y, max_y, step_size[0]):
                for dx in range(min_x, max_x, step_size[1]):
                    # Extract the block from the second image
                    block2 = img2[dy:dy+block_size[0], dx:dx+block_size[1]]

                    # Compute the error between the two blocks
                    error = error_function(block1, block2)
                    # #draw the block1 over the img2
                    # img2copy = img2.copy()
                    # img2copy[dy:dy+block_size[0], dx:dx+block_size[1]] = block1
                    # plt.imshow(img2copy)
                    # plt.show()
                    # plt.tight_layout()

                    # Update the best match if the error is lower
                    if error < min_error:
                        min_error = error
                        best_dx, best_dy = dx - x, dy - y

            # Store the flow vectors for the current block
            for i in range(y, y+block_size[0]):
                for j in range(x, x+block_size[1]):
                    flow[i, j, 0] = best_dx
                    flow[i, j, 1] = best_dy

    return flow


def smooth_flow(flow, sigma):
    # Apply Gaussian smoothing to the flow vectors
    flow_smooth = np.zeros_like(flow)
    flow_smooth[..., 0] = cv2.GaussianBlur(flow[..., 0], (0, 0), sigma)
    flow_smooth[..., 1] = cv2.GaussianBlur(flow[..., 1], (0, 0), sigma)

    return flow_smooth


def median_smoothing(flow, ksize=3):
    """
    Applies median filtering to the optical flow to reduce noise and improve accuracy.
    Args:
        flow: a 3D numpy array of size (h, w, 2), where the third dimension corresponds to the horizontal and vertical components of the optical flow vectors.
        ksize: the size of the median filter kernel.

    Returns:
        A 3D numpy array of size (h, w, 2), where the third dimension corresponds to the horizontal and vertical components of the smoothed optical flow vectors.
    """
    h, w = flow.shape[:2]
    smoothed_flow = np.zeros_like(flow)

    # Pad the flow image with zeros
    padded_flow = np.pad(flow, ((ksize//2, ksize//2), (ksize//2, ksize//2), (0,0)), mode='constant')

    # Apply median filter to each pixel in the flow image
    for y in range(ksize//2, h+ksize//2):
        for x in range(ksize//2, w+ksize//2):
            # Extract the kernel around the current pixel
            kernel = padded_flow[y-ksize//2:y+ksize//2+1, x-ksize//2:x+ksize//2+1]

            # Compute the median of the flow vectors in the kernel
            median_flow = np.median(kernel, axis=(0,1))

            # Store the smoothed flow vector for the current pixel
            smoothed_flow[y-ksize//2, x-ksize//2] = median_flow

    return smoothed_flow


def block_matching(img1, img2, block_size, search_border, method, step=1):
    h, w = img1.shape
    of = np.zeros((h, w, 2), dtype=float)

    # Iterate over rows with given step size
    for i in tqdm(range(0, h - block_size, block_size)):
        # Iterate over cols with given step size
        for j in range(0, w - block_size, block_size):

            # Crop reference block and target area to search
            first_i = max(i - search_border, 0)
            first_j = max(j - search_border, 0)
            last_i = min(i + block_size + search_border, h)
            last_j = min(j + block_size + search_border, w)
            reference = img1[i: i + block_size, j: j + block_size]
            area_of_search = img2[first_i: last_i, first_j:last_j]

            # Initialize variables to hold minimum distance and its corresponding index
            min_dist = np.inf
            min_idx = None

            # Search over blocks with given step size
            for ii in range(0, area_of_search.shape[0] - block_size + 1, step):
                for jj in range(0, area_of_search.shape[1] - block_size + 1, step):
                    target = area_of_search[ii: ii + block_size, jj: jj + block_size]
                    dist = compute_distance(reference, target, method)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = (ii, jj)

            # Save the optical flow (u and v)
            of[i:i + block_size, j:j + block_size] = [min_idx[1] - (j - first_j), min_idx[0] - (i - first_i)]

    return of
