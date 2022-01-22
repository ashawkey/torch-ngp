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

# NeRF dataset
import json


def normalize(v):
    """Normalize a vector."""
    return v / (np.linalg.norm(v) + 1e-8)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 4, 4)
    Outputs:
        pose_avg: (4, 4) the average pose
    """
    # 1. Compute the center
    center = poses[:, :3, 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[:, :3, 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[:, :3, 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    pose_avg_homo = np.eye(4, dtype=np.float32)
    pose_avg_homo[:3] = pose_avg

    return pose_avg_homo


def center_poses(poses_homo):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 4, 4)
    Outputs:
        poses_centered: (N_images, 4, 4) the centered poses
        pose_avg: (4, 4) the average pose
    """

    pose_avg_homo = average_poses(poses_homo) # (4, 4)
    inv_pos_avg_homo = np.linalg.inv(pose_avg_homo)

    poses_centered = inv_pos_avg_homo @ poses_homo # (N_images, 4, 4)

    return poses_centered, inv_pos_avg_homo

def create_spiral_poses(radius, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3
    Inputs:
        radius: (3) radius of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path
    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radius

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        pose = np.stack([x, y, z, center], 1)

        #pose = nerf_matrix_to_ngp(pose, 1)

        print(pose)

        poses_spiral += [pose] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)

def create_spheric_poses(radius, n_poses=30):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [nerf_matrix_to_ngp(spheric_pose(th, np.pi/4, radius), scale=1)] # 36 degree view downwards
        
    return np.stack(spheric_poses, 0)


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
    def __init__(self, path, type='train', downscale=1, radius=4):
        super().__init__()

        self.path = path
        self.type = type
        self.downscale = downscale
        self.radius = radius # TODO: how to determine?

        # load ngp-format fox dataset
        transform_path = os.path.join(self.path, 'transforms.json')
        with open(transform_path, 'r') as f:
            transform = json.load(f)
        
        self.intrinsic = np.eye(3, dtype=np.float32)
        self.intrinsic[0, 0] = transform['fl_x'] / downscale
        self.intrinsic[1, 1] = transform['fl_y'] / downscale
        self.intrinsic[0, 2] = transform['cx'] / downscale
        self.intrinsic[1, 2] = transform['cy'] / downscale

        self.H = int(transform['h']) // downscale
        self.W = int(transform['w']) // downscale

        if type == 'test':
            self.poses = create_spheric_poses(radius=radius).astype(np.float32)
            #self.poses = create_spiral_poses(radius=radius, focus_depth=1).astype(np.float32)
        else:

            frames = transform["frames"]
            frames = sorted(frames, key=lambda d: d['file_path'])

            if type == 'train':
                frames = frames[1:]
            elif type == 'valid':
                frames = frames[:1]

            self.poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.path, f['file_path'])

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose) # TODO: check

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                image = image.astype(np.float32) / 255 # [H, W, 3]

                self.poses.append(pose)
                self.images.append(image)
            
            self.poses = np.stack(self.poses, axis=0).astype(np.float32)

        # center poses?
        #self.poses, _ = center_poses(self.poses)
        

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):

        results = {
            'pose': self.poses[index],
            'intrinsic': self.intrinsic,
            'index': index,
        }

        if self.type == 'test':
            results['shape'] = (self.H, self.W)
            return results
        else:
            results['image'] = self.images[index]
            return results