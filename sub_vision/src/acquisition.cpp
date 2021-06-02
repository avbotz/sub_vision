/** @file acquisition.cpp
 *  @brief Main node runner for acquiring front camera images with Spinnaker.
 *
 *  @author David Zhang
 *  @author Suhas Nagar
 */
#include <thread>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "sub_vision/acquisition.hpp"
#include "sub_vision/log.hpp"
#include "sub_vision/config.hpp"
#include "sub_vision/filters.hpp"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;


void setupContinuousAcquisition(CameraPtr camera)
{
	try
	{
		INodeMap &nm_device = camera->GetTLDeviceNodeMap();
		camera->Init();
		INodeMap &nm = camera->GetNodeMap();

		// Check if front camera is writable.
		CEnumerationPtr acq_mode = nm.GetNode("AcquisitionMode");
		if (!IsAvailable(acq_mode) || !IsWritable(acq_mode))
		{
            RCLCPP_ERROR(rclcpp::get_logger("acquisition"),
                    "Front camera is either unavailable and/or unwritable");
			return;
		}

		// Check if front camera can be set to continuous acquisition.
		CEnumEntryPtr acq_mode_cont = acq_mode->GetEntryByName("Continuous");
		if (!IsAvailable(acq_mode_cont) || !IsReadable(acq_mode_cont))
		{
            RCLCPP_ERROR(rclcpp::get_logger("acquisition"),
                    "Front camera cannot be set to continuous.");
			return;
		}
	}
	catch(Spinnaker::Exception &e)
	{
        std::cerr << "Error: " << e.what() << std::endl;
	}
}

void setupContinuousAcquisition(CameraPtr camera, int exposure_time)
{
	try
	{
		INodeMap &nm_device = camera->GetTLDeviceNodeMap();
		camera->Init();
		INodeMap &nm = camera->GetNodeMap();

		// Check if front camera is writable.
		CEnumerationPtr acq_mode = nm.GetNode("AcquisitionMode");
		if (!IsAvailable(acq_mode) || !IsWritable(acq_mode))
		{
            std::cerr << "Front camera is either unavailable and/or unwritable" << std::endl;
			return;
		}

		// Check if front camera can be set to continuous acquisition.
		CEnumEntryPtr acq_mode_cont = acq_mode->GetEntryByName("Continuous");
		if (!IsAvailable(acq_mode_cont) || !IsReadable(acq_mode_cont))
		{
            std::cerr << "Front camera cannot be set to continuous." << std::endl;
			return;
		}

		// Set exposure settings.
		Spinnaker::GenApi::CEnumerationPtr auto_exp_ =
			camera->GetNodeMap().GetNode("ExposureAuto");
		auto_exp_->SetIntValue(auto_exp_->GetEntryByName("Off")->GetValue());
		Spinnaker::GenApi::CEnumerationPtr exp_mode_ =
			camera->GetNodeMap().GetNode("ExposureMode");
		exp_mode_->SetIntValue(exp_mode_->GetEntryByName("Timed")->GetValue());
		Spinnaker::GenApi::CFloatPtr exp_time_ =
			camera->GetNodeMap().GetNode("ExposureTime");
		exp_time_->SetValue(exposure_time);

		// Set acquisition mode to continuous.
		int64_t acq_cont_val = acq_mode_cont->GetValue();
		acq_mode->SetIntValue(acq_cont_val);
	}
	catch (Spinnaker::Exception &e)
	{
        std::cerr << "Error: " << e.what() << std::endl;
	}
}

void setupFramerate(CameraPtr camera, float framerate)
{
	try
	{
		INodeMap &nm = camera->GetNodeMap();
		CBooleanPtr framerate_option = nm.GetNode("AcquisitionFrameRateEnable");
		if (!IsAvailable(framerate_option) || !IsReadable(framerate_option))
		{
            std::cerr << "Unable to set framerate." << std::endl;
			return;
		}
		framerate_option->SetValue(1);
		camera->AcquisitionFrameRate.SetValue(framerate);
	}
	catch (Spinnaker::Exception &e)
	{
        RCLCPP_ERROR(rclcpp::get_logger("acquisition"), "Error: %s", e.what());
	}
}

void runCamera(CameraPtr camera, std::string channel)
{
    auto nh = rclcpp::Node::make_shared("acquisition");
	image_transport::ImageTransport it(nh);
	image_transport::Publisher pub = it.advertise(channel, 1);

	// Begin acquisition for camera.
	camera->BeginAcquisition();
	RCLCPP_INFO(rclcpp::get_logger("acquisition"),
            "Beginning acquisition for %s channel.", channel.c_str());

	// Setup FPS logger.
	rclcpp::Time tracker = nh->now();

	while (rclcpp::ok())
	{
		try
		{
			// Camera will hang here if buffer has nothing.
			// ROS_INFO("Attempt for %s channel.", channel.c_str());
            ImagePtr img_ptr = camera->GetNextImage();
			// ROS_INFO("Completed attempt for %s channel.", channel.c_str());

			// Ensure image completion.
			if (img_ptr->IsIncomplete())
			{
	            RCLCPP_ERROR(rclcpp::get_logger("acquisition"),
                        "Incomplete image for %s channel.", channel.c_str());
			}
			else
			{
				// Get image information.
				size_t width = img_ptr->GetWidth();
				size_t height = img_ptr->GetHeight();

				// Convert to OpenCV.
                ImagePtr ip = img_ptr->Convert(Spinnaker::PixelFormat_BGR8,
						Spinnaker::HQ_LINEAR);
				cv::Mat img(ip->GetHeight(), ip->GetWidth(),
						CV_8UC3, ip->GetData(), ip->GetStride());
				cv::Mat out;
				img.copyTo(out);

				// Log images as needed.
				if (LOG_DOWN && FAST_LOG && channel == "down_camera")
					log(out, 'd');
				if (LOG_FRONT && FAST_LOG && channel == "front_camera")
					log(out, 'f');

				// Correct the image before publishing
				resize(out, maxdim);
				if (FILTER_ON)
					underwaterEnhance(out);
					
				// Publish image.
                auto msg = cv_bridge::CvImage(std_msgs::msg::Header(),
					"bgr8", out).toImageMsg();
				pub.publish(msg);
			}

			// Release image to prevent buffer overflow.
			img_ptr->Release();

			// Calculate FPS.
			rclcpp::Time temp = nh->now();
			float dt = (temp-tracker).nanoseconds() / 1e9;
			float fps = 1./dt;
			tracker = nh->now();
//			ROS_INFO("FPS for %s: %f", channel.c_str(), fps);
		}
		catch (Spinnaker::Exception &e)
		{
            RCLCPP_ERROR(rclcpp::get_logger("acquisition"),
                    "Error: %s", e.what());
		}
	}

	// End acquisition ensures that cameras do not have to be powered cycled
	// to run again.
	camera->EndAcquisition();
}

int main(int argc, char** argv)
{
	rclcpp::init(argc, argv);

	// Setup log folder if needed.
	init();

	// Find connected cameras from system.
	SystemPtr system = System::GetInstance();
	CameraList cameras = system->GetCameras();
	int num_cameras = cameras.GetSize();
    std::cout << "Number of cameras: " << num_cameras << std::endl;

	// Exit if no cameras are detected.
	if (num_cameras == 0)
	{
		cameras.Clear();
		system->ReleaseInstance();
        std::cout << "Not enough cameras. Exiting now." << std::endl;
		getchar();
		return 0;
	}

	/*
	 * The camera serial numbers make this code a bit of a headache. Since the
	 * down camera SN is less than the front camera, it shows up first.
	 * However, if the down camera isn't connected, which happens often, then
	 * the front camera has the lowest SN.
	 */
	CameraPtr down_cam = NULL;
	CameraPtr front_cam = NULL;
	if (num_cameras == 1)
	{
		down_cam = cameras.GetByIndex(0);
		//CHANGED FOR 2 pool test
		setupContinuousAcquisition(down_cam);
		runCamera(down_cam, "down_camera");
	}
	if (num_cameras == 2)
	{
		down_cam = cameras.GetByIndex(0);
		front_cam = cameras.GetByIndex(1);
		setupContinuousAcquisition(down_cam);
		// setupFramerate(down_cam, 6.);
		setupContinuousAcquisition(front_cam);
		std::thread t1(runCamera, down_cam, "down_camera");
		std::thread t2(runCamera, front_cam, "front_camera");
		t1.join();
		t2.join();
	}

	// Deinitialize cameras.
	if (down_cam != NULL)
		down_cam->DeInit();
	front_cam->DeInit();
	down_cam = NULL;
	front_cam = NULL;
	cameras.Clear();
	system->ReleaseInstance();
    std::cout << "Press enter to exit." << std::endl;
	getchar();
}

