"""
This code is inspired by Team1 2022 in order to evaluate reid
"""

import numpy as np
import argparse
import sys
from src.io_utils import open_config_yaml

from pytorch_metric_learning import testers
from src.models import Embedder, HeadlessResnet
from torchvision.datasets import ImageFolder
from torchvision import transforms

from scipy.spatial.distance import cdist


def get_transforms():
    augmentations = {
        "val":
            transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    }
    return augmentations


def main(cfg):
    trunk_model = HeadlessResnet(cfg["trunk_weights_path"]).to(cfg["device"])
    embedder_model = Embedder(512, cfg["embedder_size"], cfg["embedder_weights_path"]).to(cfg["device"])

    # set datasets
    augmentations = get_transforms()
    dataset = ImageFolder(cfg["val_path"], transform=augmentations["val"])

    tester = testers.GlobalEmbeddingSpaceTester()
    embeddings = tester.get_all_embeddings(dataset, trunk_model=trunk_model, embedder_model=embedder_model)

    labels = embeddings[1]
    embeddings = embeddings[0]
    labels = labels.flatten()

    # %%
    labels = labels.detach().cpu().numpy()
    embeddings = embeddings.detach().cpu().numpy()

    distances = cdist(embeddings, embeddings, metric="euclidean")
    distances = np.triu(distances, k=1)
    valid = np.ones(distances.shape, dtype=bool)
    valid = np.triu(valid, k=1)

    # %%
    margin = cfg["th"]

    positives = distances < margin
    negatives = distances > margin

    same_label = labels[:, None] == labels[None, :]

    true_positives = np.count_nonzero(positives & same_label & valid)
    true_negatives = np.count_nonzero(negatives & np.logical_not(same_label) & valid)

    false_positives = np.count_nonzero(positives & np.logical_not(same_label) & valid)
    false_negatives = np.count_nonzero(negatives & same_label & valid)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1score = (2 * precision * recall) / (precision + recall)

    print(
        f"TP: {true_positives} \t  TN: {true_negatives} \n"
        f"FP: {false_positives} \t FN: {false_negatives} \n"
        f"--------------------------------------------------------------------------------\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"F1 Score: {f1score}\n"
        f"--------------------------------------------------------------------------------\n"
        f"Total Sum: {sum([true_positives, true_negatives, false_positives, false_negatives])} \n"
        f"Valid: {np.count_nonzero(valid)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test_reid.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_path = args.config

    config = open_config_yaml(config_path)

    main(config)
