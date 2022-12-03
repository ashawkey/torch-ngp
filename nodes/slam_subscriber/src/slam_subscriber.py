#!/usr/bin/env python
from __future__ import print_function

import sys, os
import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

#########################################################
# CONFIG AND GLOBAL VARIABLES
#########################################################

# Populate the config dictionary with any
# configuration parameters that you need
config = {
    'sub_topic_name': "/orb_slam2_stereo/pose", 
    'pub_topic_name': "/slam_subscriber/pose",
    'node_name': "slam_subscriber",
}

# TODO: do we need a callback for image data too?
def callback(data):
    # TODO: Read position and orientation from Pose message,

    # TODO: return proper information to user to be stored in transforms.json


def subscribe(config):
    # creating a subscriber for input and publisher for output
    rospy.init_node(config['node_name'], anonymous=True)
    rospy.Subscriber(config['sub_topic_name'], PoseStamped, callback)
    pose_pub = rospy.Publisher(config['pub_topic_name'], PoseStamped, queue_size=10)

    # TODO: how do we want to set rate?
    rate = rospy.Rate(config['pub_rate'])

    # TODO: store data to write to transforms.json
    # TODO: Or, continously write data to file as we recieve it (I think this is the move)
    while not rospy.is_shutdown():
        pass


if __name__ == '__main__':
    try:
        return_val = subscribe(config)
    except rospy.ROSInterruptException:
        pass

