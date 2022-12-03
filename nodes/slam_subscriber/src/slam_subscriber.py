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
    r = R.from_euler('XYZ', eu_ang)
    transform = np.eye(4)
    transform[:3,:3] = r.as_matrix()
    transform[-1,0] = x
    transform[-1,1] = y
    transform[-1,2] = z

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
        self.img_path = None
        
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
        self.img_path = self.config['data_dir'] + "images/{:05d}.jpg".format(self.counter)
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            cv2.imwrite(self.img_path, cv2_img)


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
        
    def pub_pose_msg(self, pose):
        # given pose is format (x, y, z, eu_ang)
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
        rospy.Subscriber(self.config[])
        
        rate = rospy.Rate(self.rate)

        transforms = [] # to store list of 4x4 matrices for transforms.json

        # TODO: store data to write to transforms.json
        # TODO: Or, continously write data to file as we recieve it (I think this is the move)
        while not rospy.is_shutdown():
            try:
                if self.slam_pose is not None:
                    self.transform_pose() # updates self.trans_pose
                    self.pub_pose_msg(self.trans_pose)
                    mat = euler2mat(self.trans_pose) # convert to 
                    transforms.append(mat)

                    rospy.loginfo("Processed slam pose and image ".format(id=self.count))
                else:
                    rospy.loginfo("Subscribed SLAM pose is currently None")
            except CvBridgeError as e:
                print(e)

            rate.sleep()


        ### write all data to file ###
        data_dict = { # TODO: fill with proper values
            "camera_angle_x": ?,
            "camera_angle_y": ?,
            "fl_x": ?,
            "fl_y": ?,
            "k1": self.cam_D[0],
            "k2": self.cam_D[1],
            "p1": self.cam_D[2],
            "p2": self.cam_D[3],
            "cx": ?,
            "cy": ?,
            "w": self.w
            "h": self.h,
            "aabb_scale": 4, #TODO: 16?
            "frames": transforms, #TODO: FIX this
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

