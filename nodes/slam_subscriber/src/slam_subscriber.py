#!/usr/bin/env python
from __future__ import print_function

import sys, os
import json
import rospy
import cv2
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


#########################################################
# HELPER FUNCIONS
#########################################################

def quat2euler(q):
    """Convert quaternion to Euler angles
    Parameters
    ----------
    q : array-like (4)
        Quaternion in form (qx, qy, qz, qw)
    
    Returns
    -------
    array-like (3)
        x,y,z Euler angles in radians (extrinsic)
    """
    r = R.from_quat(q)
    return r.as_euler('XYZ')


def euler2quat(x, y, z):
    """Convert Euler angles to quaternion
    
    Parameters
    ----------
    x, y, z : float
        x,y,z Euler angles in radians (extrinsic)
    
    Returns
    -------
    array-like (4)
        Quaternion in form (qx, qy, qz, qw)
    """
    r = R.from_euler('XYZ', [x, y, z])
    return r.as_quat()


def euler2mat(pose): #TODO: @Gadi write this :)
    """
    Convert given pose in format (x, y, z, eu_ang) to the
    tranformation matrix format that NeRF wants.
    Return a 4x4 np array.
    """
    x, y, z, eu_ang = pose # unpack given pose
    print(eu_ang)
    r = R.from_euler('XYZ', eu_ang)
    transform = np.eye(4)
    transform[:3,:3] = r.as_matrix()
    transform[0,-1] = x
    transform[1,-1] = y
    transform[2,-1] = z

    print(transform)

    return transform

#########################################################
# Subscriber Class
#########################################################

class SLAM_Subscriber:
    def __init__(self, config):
        self.config = config
        self.counter = 0
        self.rate = config['pub_rate']
        self.bridge = CvBridge()

        # pose parameters before and after transformation
        self.slam_pose = None
        self.trans_pose = None

        # image store path
        self.img_subpath = "images/{:05d}.jpg"
        self.img_fullpath = None

        # image sharpness
        self.sharpness = None
        
        # camera info parameters
        self.cam_D = None
        self.h = None
        self.w = None

        # setup publisher to publish transformed pose data
        self.pose_pub = rospy.Publisher(config['pub_topic_name'], PoseStamped, queue_size=config['queue_size'])



    def pose_callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "   " + str(self.counter) + " Got pose data from orb_slam.")
        self.counter += 1
        # read position and orientation from pose message
        x = data.pose.position.x
        y = data.pose.position.y
        z = data.pose.position.z
        q0 = data.pose.orientation.w
        q1 = data.pose.orientation.x
        q2 = data.pose.orientation.y
        q3 = data.pose.orientation.z
        eu_ang = quat2euler([q1, q2, q3, q0])
        # set member attribute to store this pose data
        self.slam_pose = (x, y, z, eu_ang)
        


    def img_callback(self, data):
        rospy.loginfo(rospy.get_caller_id() + "   " + str(self.counter) + " Got image data from orb_slam.")
        self.img_fullpath = self.config['data_dir'] + self.img_subpath.format(self.counter)
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            cv2.imwrite(self.img_fullpath, cv2_img)
            # get sharpness of image,
            # see https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
            cv2_img = cv2.GaussianBlur(cv2_img, (3, 3), 0)
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY) # convert to grayscale
            out_img = cv2.Laplacian(cv2_img, cv2.CV_16S, ksize=3) # use uint16 to avoid overflow
            out_img = cv2.convertScaleAbs(out_img) # convert back to uint8
            self.sharpness = np.mean(out_img)


    def cam_info_callback(self, data):
        # TODO:
        self.cam_D = data.D
        self.h = data.height
        self.w = data.width
    
    # transform slam pose to frame expected by NeRF
    def transform_pose(self):
        # TODO: 

        # slam_pose: (x, y, z, eu_ang)
        self.trans_pose = self.slam_pose
        #self.trans_pose = euler2mat(self.slam_pose)
        
    def pub_pose_msg(self, pose):
        # given pose is in the format (x, y, z, eu_ang)
        p = PoseStamped()
        p.pose.position.x = pose[0]
        p.pose.position.y = pose[1]
        p.pose.position.z = pose[2]
        eu_ang = pose[3]
        q = euler2quat(eu_ang[0], eu_ang[1], eu_ang[2])
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        p.pose.orientation.x = qx
        p.pose.orientation.y = qy
        p.pose.orientation.z = qz
        p.pose.orientation.w = qw
        self.pose_pub.publish(p)

    def run(self):
        # creating a subscriber for input and publisher for output
        rospy.init_node(self.config['node_name'], anonymous=True)
        rospy.Subscriber(self.config['pose_sub_topic_name'], PoseStamped, self.pose_callback)
        rospy.Subscriber(self.config['img_sub_topic_name'], Image, self.img_callback)
        rospy.Subscriber(self.config['cam_info_sub_topic_name'], CameraInfo, self.cam_info_callback)
        
        rate = rospy.Rate(self.rate)

        frames = [] # to store list of data required for transforms.json

        # loop through all orb slam images, storing the output pose
        # transformation matrix each time
        while not rospy.is_shutdown():
            try:
                if self.slam_pose is not None:
                    self.transform_pose() # updates self.trans_pose
                    self.pub_pose_msg(self.trans_pose) # publish transformed pose
                    mat = euler2mat(self.trans_pose) # convert to transformation matrix
                    frame_dict = {
                        "file_path": self.img_subpath.format(self.counter),
                        "sharpness": self.sharpness,
                        "transform_matrix": mat.tolist(),
                    }
                    frames.append(frame_dict)

                    rospy.loginfo("Processed slam pose and image ".format(id=self.counter))
                else:
                    rospy.loginfo("Subscribed SLAM pose is currently None")
            except CvBridgeError as e:
                print(e)

            rate.sleep()


        ### write all data to file ###
        data_dict = { # TODO: fill with proper values
           "camera_angle_x": 1, # TODO: update
           "camera_angle_y": 2, # TODO: update
           "fl_x": 3, # TODO: update
           "fl_y": 4, # TODO: update
           "k1": self.cam_D[0],
           "k2": self.cam_D[1],
           "p1": self.cam_D[2],
           "p2": self.cam_D[3],
           "cx": 5, # TODO: update
           "cy": 6, # TODO: update
           "w": self.w,
           "h": self.h,
           "aabb_scale": 4, #TODO: 16?
           "frames": frames,
        }
        json_obj = json.dumps(data_dict, indent=4)
        trans_file = self.config['data_dir'] + "tranforms.json"
        with open(trans_file, "w") as jsonfile:
           jsonfile.write(json_obj)
        
        print("\nComplete.\n")


if __name__ == '__main__':
    # Populate the config dictionary with any
    # configuration parameters that you need
    config = {
        'pose_sub_topic_name': "/orb_slam2_stereo/pose",
        'img_sub_topic_name': "/cam_front/right/image_rect_color",
        'cam_info_sub_topic_name': "/image_right/camera_info",
        # 'img_sub_topic_name': "/cam_front/left/image_rect_color",
        # 'cam_info_sub_topic_name': "/image_left/camera_info",
        'pub_topic_name': "/slam_subscriber/pose",
        'node_name': "slam_subscriber",
        'queue_size': 10,
        'pub_rate': 2,
        'data_dir': "/home/chris/Project/catkin_ws/src/AA275_NeRF_SLAM_Project/data/village_kitti/",
    }
    try:
        slam_sub = SLAM_Subscriber(config)
        slam_sub.run()
    except rospy.ROSInterruptException:
        pass

