/** @file mock_camera.cpp
 *  TODO: do we have to port this?
 *  @brief Main node runner to simulate acquisition_node with a test image.
 *
 *  @author David Zhang
 */
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vision/filters.hpp>
#include <vision/log.hpp>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

int main(int argc, char** argv)
{
	ros::init(argc, argv, "vision_mock_camera_node");
    auto node = rclcpp::Node::make_shared("vision_mock_camera_node");

	// Rather than using Spinnaker to publish an image, read test_image.png and
	// publish that instead.
	image_transport::ImageTransport it(node);
	image_transport::Publisher pub = it.advertise("front_camera", 1);
	image_transport::Publisher down_pub = it.advertise("down_camera", 1);

	cv::Mat image = cv::imread("test_images/test_image.png", cv::IMREAD_COLOR);
	cv::Mat down_image = cv::imread("test_images/test_image_down.png", cv::IMREAD_COLOR);
	resize(down_image, down_image, cv::Size(512,383));

	sensor_msgs::ImagePtr msg =
		cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
	sensor_msgs::ImagePtr down_msg =
		cv_bridge::CvImage(std_msgs::Header(), "bgr8", down_image).toImageMsg();

	while (node.ok())
	{
        std::cout << "Published image." << std::endl;
        std::this_thread::sleep_for(2s);
		pub.publish(msg);
		down_pub.publish(down_msg);
		rclcpp::spin_some(node);
	}
}
