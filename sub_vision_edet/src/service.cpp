/** @file vision/src/service.cpp
 *  @brief Wrapper definitions to handle the different vision callbacks.
 *
 *  @author David Zhang
 *  @author Emil Tu
 */
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "sub_vision/service.hpp"
#include "sub_vision/filters.hpp"
#include "sub_vision/config.hpp"
//#include "sub_vision_interfaces/msg/task.hpp"
#include "sub_vision_interfaces/srv/vision.hpp"

//using namespace sub_vision_interfaces::msg;
using namespace sub_vision_interfaces::srv;


void VisionService::frontCaptureCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
	// Read front camera data from ROS Spinnaker publisher.
	try
	{
		cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
		image.copyTo(this->front);
		if (LOG_FRONT && !FAST_LOG)
			log(this->front, 'f');
		if (VISION_SIM && SIM_FILTER_FRONT)
			underwaterEnhance(this->front);
	}
	catch (cv_bridge::Exception &e)
	{
        std::cerr << "Could not read image from Spinnaker publisher." << std::endl;
	}
}

void VisionService::downCaptureCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
	// Read front camera data from ROS Spinnaker publisher.
	try
	{
		cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
		image.copyTo(this->down);
		if (LOG_DOWN && !FAST_LOG)
			log(this->down, 'd');
		if (VISION_SIM && SIM_FILTER_DOWN)
			underwaterEnhance(this->down);
	}
	catch (cv_bridge::Exception &e)
	{
        std::cerr << "Could not read image from Spinnaker publisher." << std::endl;
	}
}

bool VisionService::detectCallback(
        std::shared_ptr<Vision::Request> req,
        std::shared_ptr<Vision::Response> res)
{
	// Load new vision model if new task is different and requires a machine
	// learning model.
	if (req->task != this->task)
	{
		// Get filepath to the model directory
		if (req->task == Task::GATE_ML)
		{
            std::cout << "Starting to setup gate model." << std::endl;
			this->model.setup("models/gpu_gate.pb");
			this->task = Task::GATE_ML;
            std::cout << "Done setting up gate model." << std::endl;
		}
		else if (req->task == Task::BINS_ML)
		{
            std::cout << "Starting to setup bins model." << std::endl;
			this->model.setup("models/pool_bins.pb");
			this->task = Task::BINS_ML;
            std::cout << "Done settings up bins model." << std::endl;
		}
		else if (req->task == Task::TARGET_ML)
		{
            std::cout << "Starting to setup target model." << std::endl;
			this->model.setup("models/gpu_target.pb");
			this->task = Task::TARGET_ML;
            std::cout << "Done settings up target model." << std::endl;
		}
	}

	// Get new observation from vision functions.
	if (req->task == Task::GATE)
	{
		std::cout << "Received detection request for GATE" << std::endl;
		Observation obs = this->findGate(this->front);
		obs.calcAngles(FRONT);
		std::cout << "Sending GATE observation @ " << obs.text().c_str() << std::endl;
		setResponse(obs, res);
		return true;
	}
	else if (req->task == Task::GATE_ML)
	{
		std::cout << "Received detection request for GATE_ML" << std::endl;
		Observation obs = this->findGateML(this->front);
		obs.calcAngles(FRONT);
		std::cout << "Sending GATE_ML observation @ " << obs.text().c_str() << std::endl;
		setResponse(obs, res);
		return true;
	}
	else if (req->task == Task::TARGET)
	{
		std::cout << "Received detection request for TARGET" << std::endl;
		Observation obs = this->findTarget(this->front);
		obs.calcAngles(FRONT);
		std::cout << "Sending TARGET observation @ " << obs.text().c_str() << std::endl;
		setResponse(obs, res);
		return true;
	}
	else if (req->task == Task::TARGET_ML)
	{
		std::cout << "Received detection request for TARGET_ML" << std::endl;
		Observation obs = this->findTargetML(this->front);
		obs.calcAngles(FRONT);
		std::cout << "Sending TARGET_ML observation @ " << obs.text().c_str() << std::endl;
		setResponse(obs, res);
		return true;
	}
	else if (req->task == Task::SECOND_TARGET_ML)
	{
		std::cout << "Received detection request for SECOND_TARGET_ML" << std::endl;
		Observation obs = this->findSecondTargetML(this->front);
		obs.calcAngles(FRONT);
		std::cout << "Sending SECOND_TARGET_ML observation @ " << obs.text().c_str() << std::endl;
		setResponse(obs, res);
		return true;
	}
	else if (req->task == Task::BINS)
	{
		std::cout << "Received detection request for BINS" << std::endl;
		Observation obs = this->findBins(this->down);
		obs.calcAngles(DOWN);
		std::cout << "Sending BINS observation @ " << obs.text().c_str() << std::endl;
		setResponse(obs, res);
		return true;
	}
	else if (req->task == Task::BINS_ML)
	{
		std::cout << "Received detection request for BINS_ML" << std::endl;
		Observation obs = this->findBinsML(this->down);
		obs.calcAngles(DOWN);
		std::cout << "Sending BINS_ML observation @ " << obs.text().c_str() << std::endl;
		setResponse(obs, res);
		return true;
	}
	else if (req->task == Task::OCTAGON)
	{

	}
    std::cout << "Finished detection request." << std::endl;
	return false;
}

void setResponse(const Observation &obs, std::shared_ptr<Vision::Response> res)
{
	res->confidence = obs.prob;
	res->r = obs.r;
	res->c = obs.c;
	res->dist = obs.dist;
	res->hangle = obs.hangle;
	res->vangle = obs.vangle;
}
