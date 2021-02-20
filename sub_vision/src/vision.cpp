/* @file vision.cpp
 *
 *  @brief Main node runner for vision.
 *
 *  @author David Zhang
 *  @author Suhas Nagar
 */
#include "rclcpp/rclcpp.hpp"

#include <functional>
#include <image_transport/image_transport.h>
#include "sub_vision/service.hpp"
#include "sub_vision/config.hpp"
// #include "sub_vision_interfaces/msg/task.hpp"
#include "sub_vision_interfaces/srv/vision.hpp"

//using namespace sub_vision_interfaces::msg;
using namespace std::placeholders;
using namespace sub_vision_interfaces::srv;


bool VISION_SIM;

int main(int argc, char** argv)
{
	srand((unsigned) time(0));
    rclcpp::init(argc, argv);
	auto node = rclcpp::Node::make_shared("vision_node");

	// Check if user has said if sim is on, default is false
	node->declare_parameter<bool>("SIM", false);
	node->get_parameter("SIM", VISION_SIM);

	// Set no task in the beginning so the first model is loaded.
	VisionService service;
	service.task = Task::NONE;

	// Setup observation request.
	auto server = node->create_service<Vision>("vision",
			std::bind(&VisionService::detectCallback, &service, _1, _2));

	// Setup front camera to receive images.
    rmw_qos_profile_t qos = rmw_qos_profile_default;
	image_transport::Subscriber front_sub, down_cam;

	if (!VISION_SIM)
	{
		front_sub = image_transport::create_subscription(node.get(), "front_camera",
			std::bind(&VisionService::frontCaptureCallback, &service, _1), "raw", qos);
		down_cam = image_transport::create_subscription(node.get(), "down_camera",
			std::bind(&VisionService::downCaptureCallback, &service, _1), "raw", qos);
	}
	else
	{
		front_sub = image_transport::create_subscription(node.get(), "/nemo/front_camera/image",
			std::bind(&VisionService::frontCaptureCallback, &service, _1), "raw", qos);
		down_cam = image_transport::create_subscription(node.get(), "/nemo/down_camera/image",
			std::bind(&VisionService::downCaptureCallback, &service, _1), "raw", qos);
	}


	// Setup down camera to receive images.
	// Deprecated with Spinnaker.
	// Camera down;
	// bool isdown = down.init(1);

	// Create directory to log images.
	init();

	while (rclcpp::ok())
	{
		switch (CAMERA_MODE)
		{
			case CameraMode::MOCK:
				service.front = cv::imread("mock.png", 1);
				service.down = cv::imread("mock.png", 1);
				break;
			case CameraMode::LIVE:

				// Updates front and down camera.
				rclcpp::spin_some(node);

				// Read down camera without using ROS.
				// Deprecated with Spinnaker.
				// if (isdown) service.down = down.capture(false);

				break;
		}
	}
}
