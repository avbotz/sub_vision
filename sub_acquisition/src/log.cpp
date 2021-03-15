/** @file log.cpp
 *  @brief Logging functions for images or text.
 *
 *  @author David Zhang
 *  @author Vincent Wang
 */
#include "sub_acquisition/log.hpp"
#include "rclcpp/rclcpp.hpp"
#include <iostream>
#include <string>
#include <ctime>
#include <unistd.h>
#include <sys/stat.h>
#include "sub_acquisition/config.hpp"


void log(const cv::Mat &img, char ending)
{
	// Get date and time to form directory and file name.
    // file path will be "YYYY_MM_DD_H_M_S_{nanoseconds since epoch}.{bmp, png}"
	time_t t = std::time(0);
	struct tm now = *std::localtime(&t);
	char date[80], name[80];
	std::strftime(date, sizeof(date), "%Y_%m_%d", &now);
	std::strftime(name, sizeof(name), "%H_%M_%S", &now);

	std::string loc = "images/" + std::string(date) + "/" + std::string(name) +
		"_" + std::string(1, ending);

	if (FAST_LOG)
		loc += ".bmp";
	else
		loc += ".png";

    // TODO: figure out logging
	// RCLCPP_INFO("%s", loc.c_str());

	// Create directory if it doesn't exist.
	mkdir(std::string("images/" + std::string(date)).c_str(), ACCESSPERMS);

	// Write the images to the correct path, but make sure they exist first.
	if (!img.data)
		RCLCPP_ERROR(rclcpp::get_logger("vision_logger"), "Could not find logging image data.");
	if (!cv::imwrite(loc, img))
		RCLCPP_ERROR(rclcpp::get_logger("vision_logger"), "Could not log image.");
}

void init()
{
	// Get date and time to form directory and file name.
	time_t t = std::time(0);
	struct tm now = *std::localtime(&t);
	char date[80];
	std::strftime(date, sizeof(date), "%Y_%m_%d", &now);

	// Create directory if it doesn't exist.
	mkdir(std::string("logs/text/" + std::string(date)).c_str(), ACCESSPERMS);
}
