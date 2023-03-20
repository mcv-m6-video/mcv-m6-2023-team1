from dataset import *
from train import *
from UNET import *
import segmentation_models_pytorch as smp
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from sklearn.model_selection import StratifiedKFold
import segmentation_models_pytorch as smp
import torch
import os
from torchvision.utils import save_image
import numpy as np



unet = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)
unet.load_state_dict(torch.load("./best.pt"))
dataset = CustomDataset('../dataset/inference/')
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

unet.eval()  # set model to evaluation mode
with torch.no_grad():  # disable gradient computation for inference
    all_preds = []
    for xb, yb in tqdm(dataloader):
        preds = unet(xb)
        all_preds.append(preds)

    # concatenate the predicted masks for all batches into a single tensor
    all_preds = torch.cat(all_preds, dim=0)

print('a')
# iterate over the predicted masks and save each one as a separate image
for i, preds in enumerate(all_preds):
    # use the sigmoid function to convert the output logits to probabilities
    probs = torch.sigmoid(preds)
    # convert the probabilities to binary masks by thresholding at 0.5
    masks = torch.argmax(probs, dim=0)
    masks = masks.to("cpu", torch.uint8).numpy()
    masks = masks*255
    cv2.imwrite(f"../dataset/inference/predictions/{i}.jpg", masks)
