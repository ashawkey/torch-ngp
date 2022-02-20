from operator import index
import os
import time
import glob
import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scipy.spatial.transform import Slerp, Rotation

# NeRF dataset
import json


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ])
    return new_pose


class NeRFDataset(Dataset):
    def __init__(self, path, type='train', downscale=1, radius=1, n_test=10):
        super().__init__()
        # path: the json file path.

        self.root_path = os.path.dirname(path)
        self.type = type
        self.downscale = downscale
        self.radius = radius # TODO: generate custom views for test?

        # load nerf-compatible format data.
        with open(path, 'r') as f:
            transform = json.load(f)
        
        # load image size
        self.H = int(transform['h']) // downscale
        self.W = int(transform['w']) // downscale

        # load intrinsics
        self.intrinsic = np.eye(3, dtype=np.float32)
        self.intrinsic[0, 0] = transform['fl_x'] / downscale
        self.intrinsic[1, 1] = transform['fl_y'] / downscale
        self.intrinsic[0, 2] = transform['cx'] / downscale
        self.intrinsic[1, 2] = transform['cy'] / downscale

        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])

        if type == 'test':
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32)) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32)) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            if type == 'train':
                frames = frames[1:]
            elif type == 'valid':
                frames = frames[:1]

            self.poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
            
            self.poses = np.stack(self.poses, axis=0).astype(np.float32)
        

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):

        results = {
            'pose': self.poses[index],
            'intrinsic': self.intrinsic,
            'index': index,
        }

        if self.type == 'test':
            # only string can bypass the default collate, so we don't need to call item: https://github.com/pytorch/pytorch/blob/67a275c29338a6c6cc405bf143e63d53abe600bf/torch/utils/data/_utils/collate.py#L84
            results['H'] = str(self.H)
            results['W'] = str(self.W)
            return results
        else:
            results['image'] = self.images[index]
            return results