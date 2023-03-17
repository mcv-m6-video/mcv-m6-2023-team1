from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2


def get_background_estimator(config: dict):
    """
    Get the background estimator from the config.
    """
    if config["estimator"] == "single_gaussian":
        return SingleGaussianBackgroundEstimator()
    else:
        raise ValueError(f"Unknown background estimator {config['estimator']}")


class BackgroundEstimator(ABC):
    """
    Abstract class for background estimation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, frames):
        pass

    @abstractmethod
    def predict(self, frames):
        pass


class SingleGaussianBackgroundEstimator(BackgroundEstimator):
    """
    Background estimation using a single Gaussian model.
    """

    def __init__(self):
        super().__init__()
        self.mu = None
        self.sigma = None

    def fit(self, frames: np.ndarray, mu="mean"):
        """
        Estimate the background using a single Gaussian model.
        Assuming batch of frames of shape (n_frames, height, width).
        :param mu:
        :param frames: frames of the video
        :return: None
        """
        assert len(frames.shape) == 3, "The frames are not of shape (n_frames, height, width)"
        if mu == "mean":
            self.mu = np.mean(frames, axis=0)
        elif mu == "median":
            self.mu = np.median(frames, axis=0)
        self.sigma = np.std(frames, axis=0)

    def predict(self, frame: np.ndarray, alpha: float = 1):
        """
        Predict the background of the frames.
        :param frame: frames of the video
        :param alpha: regularization term
        :return: background of the frames
        """
        assert len(frame.shape) == 2, "The frames are not of shape (height, width)"
        pred = np.zeros(frame.shape, dtype="uint8")
        pred[np.abs(frame - self.mu) >= alpha * (self.sigma + 2)] = 1
        return pred

    def batch_prediction(self, frames_path: np.ndarray, alpha: float = 1):
        """
        Predict the background of the frames.
        :param frames_path: frames paths of the video
        :param alpha: regularization term
        :return: background of the frames
        """
        preds = []
        for frame_path in tqdm(frames_path):
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            pred = self.predict(frame, alpha)
            pred = self.post_process(pred)
            preds.append(pred)
        return preds

    def plot_model(self):
        """
        This function plot the mean and standard deviation, the current background model.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(self.mu, cmap='gray')
        ax1.set_title('Mean')
        ax2.imshow(self.sigma, cmap='gray')
        ax2.set_title('Standard Deviation')
        plt.show()

    @staticmethod
    def get_bboxes(preds):
        bbox_preds = []
        for pred in tqdm(preds):
            pred = pred.astype("uint8")
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred)
            bounding_boxes = []
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                bounding_boxes.append((x, y, x + w, y + h))
            bbox_preds.append(bounding_boxes)
        return bbox_preds

    @staticmethod
    def post_process(pred):
        # Perform opening to remove small objects
        kernel = np.ones((11, 11), np.uint8)
        opening = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
        # Perform closing to connect big objects
        kernel = np.ones((41, 41), np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        return closing
