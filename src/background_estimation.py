from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
from src.utils import ColorSpaceConverterMethod, ChannelsUsedFromColorSpace


def get_background_estimator(config: dict):
    """
    Get the background estimator from the config.
    """
    if config["estimator"] == "single_gaussian":
        return SingleGaussianBackgroundEstimator()
    elif config["estimator"] == "adaptive_single_gaussian":
        return AdaptiveSingleGaussianBackgroundEstimator()
    elif config["estimator"] == "multi_gaussian":
        return MultiGaussianBackgroundEstimator(config["method"], config["color_space"])
    else:
        raise ValueError(f"Unknown background estimator {config['estimator']}")


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


def post_process(pred):
    # Perform opening to remove small objects
    kernel = np.ones((5, 5), np.uint8)
    x = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)

    # Perform closing to connect big objects
    kernel = np.ones((2, 80), np.uint8)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((80, 2), np.uint8)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((2, 80), np.uint8)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((10, 10), np.uint8)
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel)

    return x


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
            pred = post_process(pred)
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


class AdaptiveSingleGaussianBackgroundEstimator(BackgroundEstimator):
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

    def predict(self, frame: np.ndarray, alpha: float = 1, rho: float = 1):
        """
        Predict the background of the frames.
        :param rho:
        :param frame: frames of the video
        :param alpha: regularization term
        :return: background of the frames
        """
        assert len(frame.shape) == 2, "The frames are not of shape (height, width)"
        pred = np.zeros(frame.shape, dtype="uint8")
        pred[np.abs(frame - self.mu) >= alpha * (self.sigma + 2)] = 1

        new_mu = rho * frame + (1 - rho) * self.mu
        new_variance = rho * (frame - self.mu) ** 2 + (1 - rho) * self.sigma**2
        new_sigma = np.sqrt(new_variance)

        self.mu = np.where(pred == 0, new_mu, self.mu)
        self.sigma = np.where(pred == 0, new_sigma, self.sigma)

        return pred

    def batch_prediction(self, frames_path: np.ndarray, alpha: float = 1, rho: float = 1):
        """
        Predict the background of the frames.
        :param rho:
        :param frames_path: frames paths of the video
        :param alpha: regularization term
        :return: background of the frames
        """
        preds = []
        for frame_path in tqdm(frames_path):
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            pred = self.predict(frame, alpha, rho)
            pred = post_process(pred)
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

class MultiGaussianBackgroundEstimator(BackgroundEstimator):

    def __init__(self, method:str, color_space:str):
        super().__init__()
        self.color_space_transformation = ColorSpaceConverterMethod[color_space]
        self.color_space_chanels = ChannelsUsedFromColorSpace[color_space]
        self.method = method
        if method == "adaptative":
            self.estimators = [AdaptiveSingleGaussianBackgroundEstimator() for i in range(len(self.color_space_chanels))]
        if method == "non-adaptative":
            self.estimators = [SingleGaussianBackgroundEstimator() for i in range(len(self.color_space_chanels))]


    def fit(self, frames: np.ndarray, mu="mean"):
        """
        Estimate the background using a single Gaussian model.
        Assuming batch of frames of shape (n_frames, height, width).
        :param mu:
        :param frames: frames of the video (4 dimensions frame, height, width, channels)
        :return: None
        """
        for channel, estimator in tqdm(zip(self.color_space_chanels, self.estimators), desc="Fitting gaussian curves"):
            estimator.fit(frames[:,:,:,channel], mu)

    def predict(self, frame: np.ndarray, alpha: float = 1, rho: float = 1, estimator = None):
        """
        Predict the background of the frames.
        :param rho:
        :param frame: frames of the video
        :param alpha: regularization term
        :return: background of the frames
        """
        if self.method == "non-adaptative":
            parameters = {"alpha":alpha}
        elif self.method == "adaptative":
            parameters = {"alpha":alpha, "rho":rho}
        else:
            raise ValueError("Method incorrect select from [adaptative, non-adaptative]")

        return estimator.predict(frame,**parameters)

    def multi_predict(self, frame: np.ndarray, alpha: float = 1, rho: float = 1):
        predictions = np.zeros((frame.shape[0], frame.shape[1], len(self.color_space_chanels)))
        for i , estimator in enumerate(self.estimators):
            predictions[:,:,i] = self.predict(frame[:,:,i], alpha, rho, estimator)

        return np.logical_and.reduce(predictions, axis=-1).astype(np.uint8)

    def batch_prediction(self, frames_path: np.ndarray, alpha: float = 1, rho: float = 1):
        """
        Predict the background of the frames.
        :param rho:
        :param frames_path: frames paths of the video
        :param alpha: regularization term
        :return: background of the frames
        """
        preds = []
        for frame_path in tqdm(frames_path, desc="making predictions"):
            frame = cv2.imread(frame_path)
            transf_frame = cv2.cvtColor(frame, self.color_space_transformation)
            pred = self.multi_predict(transf_frame, alpha, rho)
            pred = post_process(pred)
            preds.append(pred*255)
        return preds


    def plot_model(self):
        """
        This function plot the mean and standard deviation, the current background model.
        """
        for estimator in self.estimators:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(estimator.mu, cmap='gray')
            ax1.set_title('Mean')
            ax2.imshow(estimator.sigma, cmap='gray')
            ax2.set_title('Standard Deviation')
            plt.show()

