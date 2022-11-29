#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

    parser.add_argument("--video", default="", help="input path to the video")
    parser.add_argument("--images", default="", help="input path to the images folder, ignored if --video is provided")
    parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")

    parser.add_argument("--dynamic", action="store_true", help="for dynamic scene, extraly save time calculated from frame index.")
    parser.add_argument("--estimate_affine_shape", action="store_true", help="colmap SiftExtraction option, may yield better results, yet can only be run on CPU.")
    parser.add_argument('--hold', type=int, default=8, help="hold out for validation every $ images")

    parser.add_argument("--video_fps", default=3)
    parser.add_argument("--time_slice", default="", help="time (in seconds) in the format t1,t2 within which the images should be generated from the video. eg: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video")

    parser.add_argument("--colmap_matcher", default="exhaustive", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    parser.add_argument("--skip_early", default=0, help="skip this many images from the start")

    parser.add_argument("--colmap_text", default="colmap_text", help="input path to the colmap text files (set automatically if run_colmap is used)")
    parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")

    args = parser.parse_args()
    return args

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def run_ffmpeg(args):
    video = args.video
    images = args.images
    fps = float(args.video_fps) or 1.0

    print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
    if (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
        sys.exit(1)

    try:
        shutil.rmtree(images)
    except:
        pass

    do_system(f"mkdir {images}")

    time_slice_value = ""
    time_slice = args.time_slice
    if time_slice:
        start, end = time_slice.split(",")
        time_slice_value = f",select='between(t\,{start}\,{end})'"

    do_system(f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" {images}/%04d.jpg")

def run_colmap(args):
    db = args.colmap_db
    images = args.images
    text = args.colmap_text
    flag_EAS = int(args.estimate_affine_shape) # 0 / 1

    db_noext = str(Path(db).with_suffix(""))
    sparse = db_noext + "_sparse"

    print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
    if (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
        sys.exit(1)
    if os.path.exists(db):
        os.remove(db)
    do_system(f"colmap feature_extractor --ImageReader.camera_model OPENCV --SiftExtraction.estimate_affine_shape {flag_EAS} --SiftExtraction.domain_size_pooling {flag_EAS} --ImageReader.single_camera 1 --database_path {db} --image_path {images}")
    do_system(f"colmap {args.colmap_matcher}_matcher --SiftMatching.guided_matching {flag_EAS} --database_path {db}")
    try:
        shutil.rmtree(sparse)
    except:
        pass
    do_system(f"mkdir {sparse}")
    do_system(f"colmap mapper --database_path {db} --image_path {images} --output_path {sparse}")
    do_system(f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
    try:
        shutil.rmtree(text)
    except:
        pass
    do_system(f"mkdir {text}")
    do_system(f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

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

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
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

if __name__ == "__main__":
    args = parse_args()

    if args.video != "":
        root_dir = os.path.dirname(args.video)
        args.images = os.path.join(root_dir, "images") # override args.images
        run_ffmpeg(args)
    else:
        args.images = args.images[:-1] if args.images[-1] == '/' else args.images # remove trailing / (./a/b/ --> ./a/b)
        root_dir = os.path.dirname(args.images)
    
    args.colmap_db = os.path.join(root_dir, args.colmap_db)
    args.colmap_text = os.path.join(root_dir, args.colmap_text)

    if args.run_colmap:
        run_colmap(args)

    SKIP_EARLY = int(args.skip_early)
    TEXT_FOLDER = args.colmap_text

    with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
        angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

    with open(os.path.join(TEXT_FOLDER, "images.txt"), "r") as f:
        i = 0

        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])

        frames = []

        up = np.zeros(3)
        for line in f:
            line = line.strip()

            if line[0] == "#":
                continue

            i = i + 1
            if i < SKIP_EARLY*2:
                continue

            if i % 2 == 1:
                elems = line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)

                name = '_'.join(elems[9:])
                full_name = os.path.join(args.images, name)
                rel_name = full_name[len(root_dir) + 1:]

                b = sharpness(full_name)
                # print(name, "sharpness =",b)

                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                
                c2w[0:3, 2] *= -1 # flip the y and z axis
                c2w[0:3, 1] *= -1
                c2w = c2w[[1, 0, 2, 3],:] # swap y and z
                c2w[2, :] *= -1 # flip whole world upside down

                up += c2w[0:3, 1]

                frame = {
                    "file_path": rel_name, 
                    "sharpness": b, 
                    "transform_matrix": c2w
                }

                frames.append(frame)

    N = len(frames)
    up = up / np.linalg.norm(up)

    print("[INFO] up vector was", up)

    R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    for f in frames:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

    # find a central point they are all looking at
    print("[INFO] computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in frames:
        mf = f["transform_matrix"][0:3,:]
        for g in frames:
            mg = g["transform_matrix"][0:3,:]
            p, weight = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if weight > 0.01:
                totp += p * weight
                totw += weight
    totp /= totw
    for f in frames:
        f["transform_matrix"][0:3,3] -= totp
    avglen = 0.
    for f in frames:
        avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
    avglen /= N
    print("[INFO] avg camera distance from origin", avglen)
    for f in frames:
        f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    # sort frames by id
    frames.sort(key=lambda d: d['file_path'])

    # add time if scene is dynamic
    if args.dynamic:
        for i, f in enumerate(frames):
            f['time'] = i / N

    for f in frames:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    # construct frames

    def write_json(filename, frames):

        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "frames": frames,
        }

        output_path = os.path.join(root_dir, filename)
        print(f"[INFO] writing {len(frames)} frames to {output_path}")
        with open(output_path, "w") as outfile:
            json.dump(out, outfile, indent=2)

    # just one transforms.json, don't do data split
    if args.hold <= 0:

        write_json('transforms.json', frames)
        
    else:
        all_ids = np.arange(N)
        test_ids = all_ids[::args.hold]
        train_ids = np.array([i for i in all_ids if i not in test_ids])

        frames_train = [f for i, f in enumerate(frames) if i in train_ids]
        frames_test = [f for i, f in enumerate(frames) if i in test_ids]

        write_json('transforms_train.json', frames_train)
        write_json('transforms_val.json', frames_test[::10])
        write_json('transforms_test.json', frames_test)