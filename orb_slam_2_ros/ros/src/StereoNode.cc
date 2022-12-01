#include "StereoNode.h"

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Stereo");
    ros::start();

    if(argc > 1) {
        ROS_WARN ("Arguments supplied via command line are neglected.");
    }

    ros::NodeHandle node_handle;
    image_transport::ImageTransport image_transport (node_handle);

    // initialize
    StereoNode node (ORB_SLAM2::System::STEREO, node_handle, image_transport);

    node.Init();

    ros::spin();

    return 0;
}


StereoNode::StereoNode (const ORB_SLAM2::System::eSensor sensor, ros::NodeHandle &node_handle, image_transport::ImageTransport &image_transport) : Node (sensor, node_handle, image_transport) {
    left_sub_ = new message_filters::Subscriber<sensor_msgs::Image> (node_handle, "image_left/image_color_rect", 1);
    right_sub_ = new message_filters::Subscriber<sensor_msgs::Image> (node_handle, "image_right/image_color_rect", 1);
    camera_info_topic_ = "image_left/camera_info";

    sync_ = new message_filters::Synchronizer<sync_pol> (sync_pol(10), *left_sub_, *right_sub_);
    sync_->registerCallback(boost::bind(&StereoNode::ImageCallback, this, _1, _2));
}


StereoNode::~StereoNode () {
    delete left_sub_;
    delete right_sub_;
    delete sync_;
}


void StereoNode::ImageCallback (const sensor_msgs::ImageConstPtr& msgLeft, const sensor_msgs::ImageConstPtr& msgRight) {
  cv_bridge::CvImageConstPtr cv_ptrLeft;
  try {
      cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
  } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
  }

  cv_bridge::CvImageConstPtr cv_ptrRight;
  try {
      cv_ptrRight = cv_bridge::toCvShare(msgRight);
  } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
  }

  current_frame_time_ = msgLeft->header.stamp;

  orb_slam_->TrackStereo(cv_ptrLeft->image, cv_ptrRight->image, cv_ptrLeft->header.stamp.toSec());

  Update ();
}
