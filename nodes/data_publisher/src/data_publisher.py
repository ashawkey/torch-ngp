import sys, os
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
        self.img_dims = config['img_dims']

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
            return lmsg, rmsg


    def generate_camera_info_message(self):
        self.left_cam_info.header.stamp = rospy.Time.now()
        self.right_cam_info.header.stamp = rospy.Time.now()

    
    def publish(self):
        
        


        while not rospy.is_shutdown():
            try:
                images = self.generate_image_message()

                if images is None:
                    lmsg_info, rmsg_info = dataset.build_camera_info()
                    self.left_pub.publish(images[0])
                    self.right_pub.publish(images[1])
                    self.left_pub_info.publish(lmsg_info)
                    self.right_pub_info.publish(rmsg_info)
                    rospy.loginfo("Published Images {id} and Info".format(id=dataset.count))
            
            except CvBridgeError as e:
                print(e)

            rate.sleep()

