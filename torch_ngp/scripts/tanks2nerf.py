import os
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
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="root directory to the Tanks&Temple dataset (contains rgb/, pose/, intrinsics.txt)")

    opt = parser.parse_args()
    print(f'[INFO] process {opt.path}')

    # load data

    intrinsics = np.loadtxt(os.path.join(opt.path, "intrinsics.txt"))
    fl_x = intrinsics[0, 0]
    fl_y = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    H = 1080
    W = 1920

    pose_files = sorted(os.listdir(os.path.join(opt.path, 'pose')))
    img_files  = sorted(os.listdir(os.path.join(opt.path, 'rgb')))

    # read in all poses, and do transform
    poses = []
    for pose_f in pose_files:
        pose = np.loadtxt(os.path.join(opt.path, 'pose', pose_f)) # [4, 4]
        poses.append(pose)
    
    poses = np.stack(poses, axis=0) # [N, 4, 4]
    N = poses.shape[0]

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

    # process three splits
    for split, prefix in zip(['train', 'val', 'test'], ['0_', '1_', '2_']):

        print(f'[INFO] process split = {split}')

        split_poses = [poses[i] for i, x in enumerate(pose_files) if x.startswith(prefix)]
        split_images = [x for x in img_files if x.startswith(prefix)]

        if len(split_poses) == 0:
            print(f'[INFO] No test data found, use valid as test')
            split_poses = [poses[i] for i, x in enumerate(pose_files) if x.startswith('1_')]
            split_images = [x for x in img_files if x.startswith('1_')]

        print(f'[INFO] loaded {len(split_images)} images, {len(split_poses)} poses.')

        assert len(split_poses) == len(split_images)

        # construct a transforms.json
        frames = []
        for image, pose in zip(split_images, split_poses):
            frames.append({
                'file_path': os.path.join('rgb', image),
                'transform_matrix': pose.tolist(),
            })
            
        transforms = {
            'w': W,
            'h': H,
            'fl_x': fl_x,
            'fl_y': fl_y,
            'cx': cx,
            'cy': cy,
            'aabb_scale': 2,
            'frames': frames,
        }

        # write
        output_path = os.path.join(opt.path, f'transforms_{split}.json')
        print(f'[INFO] write to {output_path}')
        with open(output_path, 'w') as f:
            json.dump(transforms, f, indent=2)

