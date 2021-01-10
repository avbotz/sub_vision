/** @file camera.cpp
 *  @brief Camera function definitions for FlyCapture based cameras.
 *
 *  This code is deprecated because the new Spinnaker SDK is
 *  backwards-compatible.
 *
 *  @author Surya Ramesh
 */
#include "sub_vision/camera.hpp"


bool Camera::init(unsigned int i)
{
	FlyCapture2::Error error;
	FlyCapture2::BusManager busMgr;

	error = busMgr.GetNumOfCameras(&numCameras);
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}

	if (numCameras < 1)
	{
		fprintf(stderr, "error: no camera detected at index %d\n", i);
		return false;
	}
	idx = i;
	error = busMgr.GetCameraFromIndex(i, &guid);

	if (error != FlyCapture2::PGRERROR_OK)
	{
		fprintf(stderr, "error: failed to connect to the specified camera at index %d\n", i);
		error.PrintErrorTrace();
		return false;
	}

	error = cam.Connect(&guid);
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}

	error = cam.StartCapture();
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}
	error = cam.StopCapture();
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}
	error = cam.Disconnect();
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}
	error = cam.Connect(&guid);
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}

	FlyCapture2::Format7ImageSettings format;
	unsigned int packetSize;
	float percentage;
	error = cam.GetFormat7Configuration(&format, &packetSize, &percentage);
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}

	format.mode = FlyCapture2::MODE_1;
	format.offsetX = 0;
	format.offsetY = 0;
	format.width = 640;
	format.height = 480;
	format.pixelFormat = FlyCapture2::PIXEL_FORMAT_RGB8;
	bool valid;
	FlyCapture2::Format7PacketInfo packetInfo;
	error = cam.ValidateFormat7Settings(&format, &valid, &packetInfo);
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}
	if (!valid)
	{
		fprintf(stderr, "error: invalid format\n");
		return false;
	}

	error = cam.SetFormat7Configuration(&format, packetInfo.recommendedBytesPerPacket);
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}

	error = cam.GetFormat7Configuration(&format, &packetSize, &percentage);
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}

	error = cam.StartCapture();
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}

	return true;
}

void Camera::flipImage(cv::Mat& img)
{
	int rows = img.rows;
	int cols = img.cols;
	unsigned char* ptr = img.ptr();
	for (size_t i = 0; i < rows*cols / 2; i++)
	{
		size_t o = rows*cols - i - 1;
		int p = ptr[3*i + 0];
		ptr[3*i + 0] = ptr[3*o + 0];
		ptr[3*o + 0] = p;
		p = ptr[3*i + 1];
		ptr[3*i + 1] = ptr[3*o + 1];
		ptr[3*o + 1] = p;
		p = ptr[3*i + 2];
		ptr[3*i + 2] = ptr[3*o + 2];
		ptr[3*o + 2] = p;
	}

	return;
}

cv::Mat Camera::capture(bool flip)
{
	FlyCapture2::Error error;
	FlyCapture2::Image fcImage;
	error = cam.RetrieveBuffer(&fcImage);
	if (error != FlyCapture2::PGRERROR_OK)
	{
		fprintf(stderr, "error: getting image\n");
		error.PrintErrorTrace();
		return cv::Mat::ones(2, 2, CV_8UC3);
	}

	FlyCapture2::Image bgr;
	error = fcImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &bgr);
	if (error != FlyCapture2::PGRERROR_OK)
	{
		fprintf(stderr, "error: converting image\n");
		error.PrintErrorTrace();
		return cv::Mat::ones(2, 2, CV_8UC3);
	}

	cv::Mat image(bgr.GetRows(), bgr.GetCols(), CV_8UC3, bgr.GetData(), bgr.GetStride());
	if (flip)
		flipImage(image);

	cv::Mat out;
	image.copyTo(out);

	return out;
}

bool Camera::quit()
{
	FlyCapture2::Error error;
	error = cam.StopCapture();
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}

	error = cam.Disconnect();
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}
	return true;
}

