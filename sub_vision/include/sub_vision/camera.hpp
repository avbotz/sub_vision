/** @file camera.hpp
 *  @brief Camera struct for FlyCapture based cameras.
 *
 *  @author Surya Ramesh
 */
#ifndef CAPTURE_CAMERA_HPP
#define CAPTURE_CAMERA_HPP

#include <opencv2/core/core.hpp>
#include <flycapture/FlyCapture2.h>

struct Camera 
{
	unsigned int numCameras;
	unsigned int idx;
	FlyCapture2::PGRGuid guid;
	FlyCapture2::Camera cam;

	bool init(unsigned int);
	bool quit();
	void flipImage(cv::Mat &);
	cv::Mat capture(bool);
};

#endif
