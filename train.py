import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from rooms_pano_utils import panorama_to_plane
from rooms_dataset import RoomsDataset
import random
from datasets import load_dataset

from share import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

class RandomPerspectiveTransform():
    def __call__(self, item):
        # Converts the panorama image into a perspective image of size 512x512.
        # The perspective itself is at a random angle (yaw) of the 360 panorama.
        fov = 120
        output_size = (512, 512)
        yaw = random.randint(0, 360)
        pitch = 90
        
        item['hint'] = panorama_to_plane(item['hint'], fov, output_size, yaw, pitch)
        item['jpg'] = panorama_to_plane(item['jpg'], fov, output_size, yaw, pitch)
  
        return item

class NormalizeTransform():
    def __call__(self, item):
        # Normalize according to what controlnet wants.
        
        # Normalize source images to [0, 1].
        item["hint"] = item["hint"].astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        item["jpg"] = (item["jpg"].astype(np.float32) / 127.5) -1.0

        return item

class MLSLDTransform():
    def __call__(self, item):
        apply_mlsd = MLSDdetector()

        item["hint"] = item["hint"].astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        item["jpg"] = (item["jpg"].astype(np.float32) / 127.5) -1.0

        return item


# Create a dataset
transform = transforms.Compose([
    RandomPerspectiveTransform(),
    MLSDTransform(),
    NormalizeTransform(),
])

# Configs
resume_path = '/mnt/shared/models/room-train-run-2.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-6
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

dataset = RoomsDataset(transform, only_room_types=['living room'])
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
img_logger = ImageLogger(batch_frequency=logger_freq)

wandb_logger = pl.loggers.WandbLogger(save_dir='/mnt/shared/wandb_logs/')
checkpoint_callback = ModelCheckpoint(dirpath='/mnt/shared/wandb_logs/ckpts/')

trainer = pl.Trainer(gpus=1, precision=32, callbacks=[img_logger, checkpoint_callback], logger=wandb_logger)

# Train!
trainer.fit(model, data_loader)
