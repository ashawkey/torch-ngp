from audioop import avg
import os
import glob
import numpy as np
import math
import json

import argparse

# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db): 
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="root directory to the LLFF dataset (contains images/ and pose_bounds.npy)")
    parser.add_argument('--images', type=str, default='images', help="images folder")
    parser.add_argument('--downscale', type=float, default=1, help="image size down scale")

    opt = parser.parse_args()
    print(f'[INFO] process {opt.path}')
    
    # load data
    images = [f[len(opt.path):] for f in sorted(glob.glob(os.path.join(opt.path, opt.images, "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
    
    poses_bounds = np.load(os.path.join(opt.path, 'poses_bounds.npy'))
    N = poses_bounds.shape[0]

    print(f'[INFO] loaded {len(images)} images, {N} poses_bounds as {poses_bounds.shape}')

    assert N == len(images)

    poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N, 3, 5)
    bounds = poses_bounds[:, -2:] # (N, 2)

    H, W, fl = poses[0, :, -1] 

    H = H // opt.downscale
    W = W // opt.downscale
    fl = fl / opt.downscale

    print(f'[INFO] H = {H}, W = {W}, fl = {fl} (downscale = {opt.downscale})')

    # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
    poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1) # (N, 3, 4)

    # to homogeneous 
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N, 1, 4)
    poses = np.concatenate([poses, last_row], axis=1) # (N, 4, 4) 

    # the following stuff are from colmap2nerf... 
    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    poses = poses[:, [1, 0, 2, 3], :] # swap y and z
    poses[:, 2, :] *= -1 # flip whole world upside down

    up = poses[:, 0:3, 1].sum(0)
    up = up / np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    poses = R @ poses

    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for i in range(N):
        mf = poses[i, :3, :]
        for j in range(i + 1, N):
            mg = poses[j, :3, :]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            #print(i, j, p, w)
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print(f'[INFO] totp = {totp}')
    poses[:, :3, 3] -= totp

    avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()

    poses[:, :3, 3] *= 4.0 / avglen

    print(f'[INFO] average radius = {avglen}')

    # construct frames
    frames = []
    for i in range(N):
        frames.append({
            'file_path': images[i],
            'transform_matrix': poses[i].tolist(),
        })

    # construct a transforms.json
    transforms = {
        'w': W,
        'h': H,
        'fl_x': fl,
        'fl_y': fl,
        'cx': W // 2,
        'cy': H // 2,
        'aabb_scale': 2,
        'frames': frames,
    }

    # write
    output_path = os.path.join(opt.path, 'transforms.json')
    print(f'[INFO] write to {output_path}')
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)

