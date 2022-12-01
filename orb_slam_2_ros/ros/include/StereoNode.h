#pragma once

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types.hpp>


#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <tf/transform_broadcaster.h>

#include "System.h"
#include "Node.h"

class StereoNode : public Node
{
  public:
    StereoNode (const ORB_SLAM2::System::eSensor sensor, ros::NodeHandle &node_handle, image_transport::ImageTransport &image_transport);
    ~StereoNode ();
    void ImageCallback (const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight);

private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Subscriber<sensor_msgs::Image> *left_sub_;
    message_filters::Subscriber<sensor_msgs::Image> *right_sub_;
    message_filters::Synchronizer<sync_pol> *sync_;

    int resize_horizontal;
    int resize_vertical;
};

