import os
import time
import glob
import numpy as np

import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
    
# NeRF dataset
import json


class NeRFDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        self.path = path

        # load cameras
        transform_path = os.path.join(self.path, 'transforms.json')
        with open(transform_path, 'r') as f:
            transform = json.load(f)

        self.images = []
        self.cameras = []
        self.intrinsics = []

        



    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):



        results = {

        }

        return results