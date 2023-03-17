import argparse
import sys
from typing import Dict

from src.background_estimation import get_background_estimator
from src.in_out import extract_frames_from_video, get_frames_paths_from_folder, extract_rectangles_from_xml, load_images
from src.utils import open_config_yaml
from src.plotting import save_results
from src.metrics import get_allFrames_ap, get_mIoU


def task1(cfg: Dict):
    """
    For this task, you will implement a background estimation algorithm using a single Gaussian model.
    """
    paths = cfg["paths"]
    bg_estimator_config = cfg["background_estimator"]
    estimator = get_background_estimator(bg_estimator_config)

    extract_frames_from_video(video_path=paths["video"], output_path=paths["extracted_frames"])
    gt_labels = extract_rectangles_from_xml(cfg["paths"]["annotations"])
    frames = get_frames_paths_from_folder(input_path=paths["extracted_frames"])
    print("Number of frames: ", len(frames))
    dataset = [(key, frames[key])for key in gt_labels.keys()]
    train_data = dataset[:int(len(dataset)*0.25)]
    test_data = dataset[int(len(dataset)*0.25):]

    train_imgs_paths = [frame[1] for frame in train_data]
    train_imgs = load_images(train_imgs_paths, grayscale=True)
    print("Train images loaded ", len(train_imgs))
    estimator.fit(train_imgs)
    del train_imgs
    estimator.plot_model()

    # Background substraction
    test_imgs_paths = [frame[1] for frame in test_data]
    print("Test images loaded ", len(test_imgs_paths))

    preds = estimator.batch_prediction(test_imgs_paths, alpha=3)
    print("Computed all predictions")

    bboxes = estimator.get_bboxes(preds)
    print("Bounding boxes extracted")

    # Get test images ground truth
    gt_bboxes = [*gt_labels.values()]
    first_test_idx = len(gt_labels) - len(bboxes)
    gt_test_bboxes = gt_bboxes[first_test_idx:]
    # Save results
    # save_results(bboxes, preds, gt_test_bboxes, test_imgs_paths)

    # Compute mAP and mIoU
    mAP = get_allFrames_ap(gt_test_bboxes, bboxes)
    mIoU = get_mIoU(gt_test_bboxes, bboxes)
    print(f"mAP: {mAP}")
    print(f"mIoU: {mIoU}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1.yaml")
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    task1(config)
