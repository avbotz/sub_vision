/** @file gate.cpp
 *  @brief Vision functions to detect the gate.
 *
 *  @author David Zhang
 *  @author Emil Tu
 *  @author Suhas Nagar
 *  @author Craig Wang
 */
#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include "sub_vision/service.hpp"
#include "sub_vision/filters.hpp"


Observation VisionService::findGate(const cv::Mat &img)
{
	// Check that image isn't null.
	if (!img.data)
	{
        // TODO: Implement logging properly
        std::cout << "No image data for gate." << std::endl;
		return Observation(0, 0, 0, 0);
	}

	// Illuminate image using filter.
	// cv::Mat illum = illumination(img);

	// Strong blur to remove noise from image.
	cv::Mat blur;
	cv::blur(img, blur, cv::Size(17, 17));

	// Canny edge detection with low threshold.
	cv::Mat can, cdst;
	cv::Canny(blur, can, 20, 60, 3);
	cv::cvtColor(can, cdst, cv::COLOR_GRAY2BGR);

	// Get lines using OpenCV Hough Lines algorithm, and store probable lines to
	// use later. The last three parameters for HoughLinesP are threshold,
	// minLength, and maxGap.
	std::vector<cv::Vec4i> lines;
	std::vector<cv::Vec4i> probable_lines;
	int ac=0, bc=0, ar=0, br=0;
	cv::Vec4i a_line, b_line;
	cv::HoughLinesP(can, lines, 2, CV_PI/180, 50, 80, 30);
	for (int i = 0; i < lines.size(); i++)
	{
		cv::Vec4i line = lines[i];
		int x1=line[0],y1=line[1],x2=line[2],y2=line[3];
		float dist = std::sqrt(std::pow(std::abs(x1-x2), 2) +
				std::pow(std::abs(y1-y2), 2));
		float rotation = std::abs(std::acos(std::abs(y1-y2)/dist)*180/CV_PI);
		cv::line(cdst, cv::Point(x1, y1), cv::Point(x2, y2),
				cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
		if (rotation <= 30 && dist > 100. && y1 < 2500 && y2 < 2500)
		{
			if (ac == 0)
			{
				ac = (x1+x2)/2;
				ar = (y1+y2)/2;
				a_line = cv::Vec4i(x1, y1, x2, y2);
			}
			else if (std::abs(x1-ac) > 150 && br == 0)
			{
				bc = (x1+x2)/2;
				br = (y1+y2)/2;
				b_line = cv::Vec4i(x1, y1, x2, y2);
			}
			else
			{
				cv::line(cdst, cv::Point(x1, y1), cv::Point(x2, y2),
						cv::Scalar(0, 255, 255), 3, cv::LINE_AA);
			}
		}
	}
	cv::line(cdst, cv::Point(a_line[0], a_line[1]), cv::Point(a_line[2],
				a_line[3]), cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
	cv::line(cdst, cv::Point(b_line[0], b_line[1]), cv::Point(b_line[2],
				b_line[3]), cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
	cv::circle(cdst, cv::Point((ac+bc)/2, (ar+br)/2), 50,
			cv::Scalar(255, 255, 255), cv::FILLED, 8, 0);
	log(cdst, 'e');

	// Calculate midpoint and return observation if valid.
	if (ac == 0 && bc == 0)
		return Observation(0, 0, 0, 0);
	return Observation(0.8, (ar+br)/2, (ac+bc)/2, 0);
}

Observation VisionService::findGateML(cv::Mat img)
{
    std::cout << "Starting machine learning detection for GATE." << std::endl;

	log(img, 'f');

	//Preprocess the image and place the filter
	cv::Mat temp;
	cv::Mat inp;
	img.copyTo(temp);
	cv::cvtColor(img, inp, CV_BGR2RGB);
	std::vector<float> img_data;
	preprocess(inp,img_data,maxdim);

	auto inpName = new Tensor(model, "input_1");
	auto out_boxes = new Tensor(model, "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3");
	auto out_scores = new Tensor(model, "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3");
	auto out_labels = new Tensor(model, "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3");

	// Put image in tensor.
	inpName->set_data(img_data, { 1, maxdim, maxdim, 3 });

	model.run(inpName, { out_boxes, out_scores, out_labels });
	// Store output in variables so don't have to keep calling get_data()
	auto boxes = out_boxes->get_data<float>();
	auto scores = out_scores->get_data<float>();
	auto labels = out_labels->get_data<int>();

	// Visualize detected bounding boxes.
	for (int i = 0; i < scores.size(); i++)
	{
		int class_id = labels[i];
		float score = scores[i];
		std::vector<float> bbox = { boxes[i*4], boxes[i*4+1],
			boxes[i*4+2], boxes[i*4+3] };
		if (score > 0.5)
		{
			float x = bbox[0];
			float y = bbox[1];
			float right = bbox[2];
			float bottom = bbox[3];
			if (class_id == 0){
                std::cout << "Gate Found" << std::endl;
				cv::rectangle(temp, {(int)x, (int)y}, {(int)right, (int)bottom},
						{255, 0, 255}, 5);
				//log(temp, 'e');

				// Calculate position to pass through on left or right side of gate
				float target_x;
				float target_y;
				float gate_width = std::fabs(right-x);
				float gate_height = std::fabs(bottom-y);
				float x_offset = (float)gate_width * 0.25;
				float y_offset = (float)gate_height * 0.4;

				// Fix this, it was meant to read from mission config.hpp but it caused errors during catkin make
				// Because vision is built before mission
				bool GATE_LEFT = false;
				if (GATE_LEFT) target_x = (x+right)/2 - x_offset;
				else target_x = (x+right)/2 + x_offset;
				target_y = bottom - y_offset;
				cv::circle(temp, cv::Point((int)target_x, (int)target_y), 10, {255, 0, 255}, cv::FILLED);
				log(temp, 'e');

				// Calculate distance based on camera parameters
				float det_height = std::fabs(bottom-y);
				float det_width = std::fabs(right-x);
				float dist = calcDistance(FRONT_FOCAL_LENGTH, GATE_HEIGHT_MM, 
					FIMG_DIM_RES[0], det_height, FRONT_SENSOR_SIZE);

				return Observation(score, target_y, target_x, dist);
			}
		}
		else break;
	}

	return Observation(0, 0, 0, 0);
}
