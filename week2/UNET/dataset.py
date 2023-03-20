from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from torch import nn
import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import cv2
import spectral as sp
import torch.nn.functional as F

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.gt_dir = os.path.join(root_dir, 'gt')

        self.rgb_filenames = sorted(os.listdir(self.rgb_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))
        self.gt_filenames = sorted(os.listdir(self.gt_dir))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])
        # Adding a new transform to load grayscale mask as single channel tensor
        self.mask_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        ])

    def __len__(self):
        return len(self.rgb_filenames)

    def pad_image(self, image):
        h, w = image.shape[-2:]
        new_h = int(np.ceil(h / 32) * 32)
        new_w = int(np.ceil(w / 32) * 32)
        pad_h = new_h - h
        pad_w = new_w - w
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        padded_image = F.pad(image, padding)
        return padded_image

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_filenames[idx])

        rgb = Image.open(rgb_path)
        rgb = np.array(rgb.resize((rgb.width//4, rgb.height//4)))
        mask = Image.open(mask_path)
        mask = mask.resize((mask.width//4, mask.height//4))
        mask = np.array(mask)[:,:,0] #[:,:,0]
        mask = np.expand_dims(mask, axis=2)
        rgb = np.concatenate([rgb, mask], axis=2)
        rgb = rgb.transpose((2, 0, 1))
        img = self.pad_image(torch.tensor(rgb, dtype=torch.float32))

        # Load the grayscale target image and convert it to a one hot encoded version, knowing it is a 0-255 mask, and at the end, convert it to a tensor
        gt = Image.open(gt_path)
        gt = np.array(gt.resize((gt.width//4, gt.height//4)))
        gt = np.where(gt == 255, 1, 0)
        gt = self.pad_image(torch.tensor(gt, dtype=torch.torch.int64))


        return img, gt