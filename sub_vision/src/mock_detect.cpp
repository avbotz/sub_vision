/** @file mock_detect.cpp
 *  @brief Main node runner to simulate getting a response from vision.
 *  TODO: do we have to port this?
 *
 *  @author David Zhang
 */
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vision/Vision.h>
#include "vision/config.hpp"


int main(int argc, char** argv)
{
	ros::init(argc, argv, "vision_mock_detect_node");
	ros::NodeHandle node;
	ros::ServiceClient client =
		node.serviceClient<vision::Vision>("vision");

	// This node is just for reading what the vision program is outputting. Use
	// it as an alternative to the mission node when the sub shouldn't move.
	vision::Vision srv;
	srv.request.task = Task::GATE_ML;
	srv.request.camera = FRONT;
	if (client.call(srv))
	{
		ROS_INFO("task: %f\t r: %f\t c: %f\t", srv.request.task, srv.response.r,
				srv.response.c);
	}
}
