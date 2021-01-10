/** @file vision/service.hpp
 *  @brief Wrapper class to handle the different vision callbacks.
 *
 *  The main purpose of this class is to ensure that the images read from the
 *  ROS image publisher can be used for object detection. It keeps the images in
 *  one location and allows ros::spin() to update them as needed.
 *
 *  @author David Zhang
 *  @author Emil Tu
 *  @author Vincent Wang
 */
#ifndef VISION_SERVICE_HPP
#define VISION_SERVICE_HPP

#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/msg/image.hpp"
#include "sub_vision/observation.hpp"
#include "sub_vision/model.hpp"
#include "sub_vision/tensor.hpp"
#include "sub_vision/log.hpp"
#include <iostream>
#include <string>

//#include "sub_vision_interfaces/msg/task.hpp"
#include "sub_vision_interfaces/srv/vision.hpp"

//using namespace sub_vision_interfaces::msg;
using namespace sub_vision_interfaces::srv;

class VisionService
{
	public:
		cv::Mat front, down;
		Model model;
		Task task;

		bool detectCallback(std::shared_ptr<Vision::Request>,
                std::shared_ptr<Vision::Response>);
		void frontCaptureCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
		void downCaptureCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
		Observation findGate(const cv::Mat &);
		Observation findGateML(cv::Mat);
		Observation findTarget(const cv::Mat &);
		Observation findTargetML(cv::Mat);
		Observation findSecondTargetML(cv::Mat);
		Observation findBins(const cv::Mat &);
		Observation findBinsML(cv::Mat);
};

void setResponse(const Observation &, std::shared_ptr<Vision::Response>);

#endif
