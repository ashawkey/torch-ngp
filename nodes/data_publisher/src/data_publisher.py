#!/usr/bin/env python3
import sys, os
import json
import argparse
import configparser

import rospy
import cv2
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class DatasetPublisher:
    def __init__(self, config):

        # save config just in case for verification
        self.config = config

        # get data from specified dataset path (assumes kitti-datset structure)
        self.data_dir = config['data_dir']
        self.lpath = self.data_dir + '/image_02/data/'
        self.rpath = self.data_dir + '/image_03/data/'
        self.lfiles = sorted(os.listdir(self.lpath))
        self.rfiles = sorted(os.listdir(self.rpath))
        self.count = 0
        self.bridge = CvBridge()
        self.left_img_dims = config['left_img_dims']
        self.right_img_dims = config['right_img_dims']

        # initialize left camera calibration messages
        self.left_cam_info = CameraInfo()
        self.left_cam_info.header = Header()
        self.left_cam_info.width = config['left_img_dims'][0]
        self.left_cam_info.height = config['left_img_dims'][1]
        self.left_cam_info.distortion_model = config['left_distortion_model']
        self.left_cam_info.K = config['left_K']
        self.left_cam_info.D = config['left_D']
        self.left_cam_info.R = config['left_R']
        self.left_cam_info.P = config['left_P']

        # initialize right camera calibration messages
        self.right_cam_info = CameraInfo()
        self.right_cam_info.header = Header()
        self.right_cam_info.width = config['right_img_dims'][0]
        self.right_cam_info.height = config['right_img_dims'][1]
        self.right_cam_info.distortion_model = config['right_distortion_model']
        self.right_cam_info.K = config['right_K']
        self.right_cam_info.D = config['right_D']
        self.right_cam_info.R = config['right_R']
        self.right_cam_info.P = config['right_P']

        # setup publishers to publish recorded data
        self.left_pub = rospy.Publisher(config['topic_name_img_left'], Image, queue_size=config['queue_size'])
        self.left_pub_info = rospy.Publisher(config['topic_name_info_left'], CameraInfo, queue_size=config['queue_size'])
        self.right_pub = rospy.Publisher(config['topic_name_img_right'], Image, queue_size=config['queue_size'])
        self.right_pub_info = rospy.Publisher(config['topic_name_info_right'], CameraInfo, queue_size=config['queue_size'])

        # initialize ros publisher node name
        rospy.init_node(config['node_name'], anonymous=True)

        #save publishing rate
        self.rate = config['pub_rate']

    def generate_image_message(self):
        
        # generate image message and ends if no new images are available
        if self.count >= len(self.lfiles) or self.count >= len(self.rfiles):
            return None
        else:
            limg = cv2.imread(self.lpath + self.lfiles[self.count])
            rimg = cv2.imread(self.rpath + self.rfiles[self.count])
            lmsg = self.bridge.cv2_to_imgmsg(np.array(limg), "bgr8")
            rmsg = self.bridge.cv2_to_imgmsg(np.array(rimg), "bgr8")

            self.count += 1
            return [lmsg, rmsg]


    def publish(self):
        
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            try:
                images = self.generate_image_message()

                if images is not None:
                    # update camera info headers
                    self.left_cam_info.header.stamp = rospy.Time.now()
                    self.right_cam_info.header.stamp = rospy.Time.now()

                    # publish image data
                    self.left_pub.publish(images[0])
                    self.right_pub.publish(images[1])
                    
                    # publish camera intrinics
                    self.left_pub_info.publish(self.left_cam_info)
                    self.right_pub_info.publish(self.right_cam_info)
                    
                    rospy.loginfo("Published Images {id} and Info".format(id=self.count))
            
            except CvBridgeError as e:
                print(e)

            rate.sleep()


if __name__ == '__main__':

    #config = rospy.get_param('/data_publisher/config')

    args = rospy.myargv(argv=sys.argv)
    config_path = args[1]

    print(config_path)

    parser = configparser.ConfigParser()
    parser.read(config_path)


    config = {}
    config['data_dir'] = parser['General']['data_dir']
    config['left_img_dims'] = json.loads(parser.get('Left Camera', 'left_img_dims'))
    config['right_img_dims'] = json.loads(parser.get('Right Camera', 'right_img_dims'))
    config['left_distortion_model'] = parser['Left Camera']['left_distortion_model']
    config['right_distortion_model'] = parser['Right Camera']['right_distortion_model']
    config['left_K'] = json.loads(parser.get('Left Camera', 'left_K'))
    config['left_D'] = json.loads(parser.get('Left Camera', 'left_D'))
    config['left_R'] = json.loads(parser.get('Left Camera', 'left_R'))
    config['left_P'] = json.loads(parser.get('Left Camera', 'left_P'))
    config['right_K'] = json.loads(parser.get('Right Camera', 'right_K'))
    config['right_D'] = json.loads(parser.get('Right Camera', 'right_D'))
    config['right_R'] = json.loads(parser.get('Right Camera', 'right_R'))
    config['right_P'] = json.loads(parser.get('Right Camera', 'right_P'))
    config['topic_name_img_left'] = parser['ROS']['topic_name_img_left']
    config['topic_name_img_right'] = parser['ROS']['topic_name_img_right']
    config['topic_name_info_left'] = parser['ROS']['topic_name_info_left']
    config['topic_name_info_right'] = parser['ROS']['topic_name_info_right']
    config['queue_size'] = parser.getint('ROS', 'queue_size')
    config['pub_rate'] = float(parser['ROS']['pub_rate'])
    config['node_name'] = parser['ROS']['node_name']

    print("HI")
    print(config)
    print(config['pub_rate'])
    data_pub = DatasetPublisher(config)
    data_pub.publish()
    #try:
    #    data_pub.publish()
    #except rospy.ROSInterruptException:
    #    pass 
