/** @file target.cpp
 *  @brief Vision functions to detect the target.
 *
 *  @author David Zhang
 *  @author Suhas Nagar
 *  @author Craig Wang
 */
#include <rclcpp/rclcpp.hpp>
#include "sub_vision/service.hpp"
#include "sub_vision/filters.hpp"


Observation VisionService::findTarget(const cv::Mat &input)
{
	/*
	 * This target code is meant to be run at Suhas' pool, with a black outline.
	 * Do not use for competition.
	 *
	 * Ok, it looks like we're using this at competition.
	 */
	log(input, 'f');

	// Illuminate image using filter.
	// cv::Mat illum = illumination(input);
	cv::Mat illum = input;

	// Strong blur to remove noise from image.
	cv::Mat blur;
	cv::blur(illum, blur, cv::Size(9, 9));
	log(illum, 'f');

	// Threshold for black.
	cv::Mat thresh;
	cv::Mat cdst;
	cv::inRange(blur, cv::Scalar(120, 120, 140), cv::Scalar(255, 255, 255), thresh);
	cv::cvtColor(thresh, cdst, cv::COLOR_GRAY2BGR);
	// cv::cvtColor(thresh, thresh, cv::COLOR_BGR2GRAY);

	// Contour detection.
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(thresh, contours, hierarchy, CV_RETR_TREE,
			CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	for (int i = 0; i < contours.size(); i++)
	{
		cv::drawContours(cdst, contours, i, cv::Scalar(0, 255, 0), 2, 8,
				hierarchy, 0, cv::Point());
	}

	// Approximate and convert to rectangles.
	std::vector<std::vector<cv::Point>> contour_polygons (contours.size());
	std::vector<cv::Rect> rectangles (contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		cv::approxPolyDP(cv::Mat(contours[i]), contour_polygons[i],
				0.01*cv::arcLength(cv::Mat(contours[i]), true), true);
		rectangles[i] = cv::boundingRect(cv::Mat(contour_polygons[i]));
	}
	for (int i = 0; i < contours.size(); i++)
	{
		cv::drawContours(cdst, contour_polygons, i, cv::Scalar(255, 0, 0), 1, 8,
				std::vector<cv::Vec4i>(), 0, cv::Point());
		cv::rectangle(cdst, rectangles[i].tl(), rectangles[i].br(),
				cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	// Choose largest rectangle that is in an appropriate ratio.
	std::sort(rectangles.begin(), rectangles.end(), [](const cv::Rect &a,
				const cv::Rect &b) -> bool { return a.area() > b.area(); });
	for (int i = 0; i < rectangles.size(); i++)
	{
		float height = rectangles[i].height;
		float width = rectangles[i].width;
		float rect_ratio = height/width;
		if (rect_ratio < 4. && rect_ratio > 0.25 &&
				rectangles[i].tl().x+width/2. > 456 &&
				rectangles[i].br().x+height/2. < 5016)
		{
			cv::rectangle(cdst, rectangles[i].tl(), rectangles[i].br(),
					cv::Scalar(255, 0, 255), 3, 8, 0);
			int x = (rectangles[i].tl().x+rectangles[i].br().x)/2;
			int y = (rectangles[i].tl().y+rectangles[i].br().y)/2;
			cv::circle(cdst, cv::Point(x, y), 4, cv::Scalar(255, 0, 255), 3);
			log(cdst, 'e');
			float dist = 2250./rectangles[i].width;
			if (dist > 10.) dist = 10.;
			return Observation(0.8, y, x, dist);
		}
	}

	// No contours found. I doubt this will run often because the sub is more
	// prone to picking up noise instead of nothing at all.
	return Observation(0, 0, 0, 0);
}

Observation VisionService::findTargetML(cv::Mat img)
{
	std::cout << "Starting machine learning detection for TARGET_ML." << std::endl;

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

	// Put image in tensor and run model.
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
		if (score > 0.3)
		{
			float x = bbox[0];
			float y = bbox[1];
			float right = bbox[2];
			float bottom = bbox[3];
			if (class_id != 2)
			{
				std::cout << "Target Found" << std::endl;
				cv::rectangle(temp, {(int)x, (int)y}, {(int)right, (int)bottom},
						{255, 0, 255}, 5);
				log(temp, 'e');

				// Calculate distance based on camera parameters
				float det_height = std::fabs(bottom - y);
				float dist = calcDistance(FRONT_FOCAL_LENGTH, GATE_HEIGHT_MM, 
					FIMG_DIM_RES[0], det_height, FRONT_SENSOR_SIZE);

				return Observation(score, (y+bottom)/2, (x+right)/2, dist);
			}
		}
		else break;
	}

	return Observation(0, 0, 0, 0);
}

Observation VisionService::findSecondTargetML(cv::Mat img)
{
	std::cout << "Starting machine learning detection for SECOND_TARGET_ML." << std::endl;

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
		if (score > 0.3)
		{
			float x = bbox[0];
			float y = bbox[1];
			float right = bbox[2];
			float bottom = bbox[3];
			if (class_id == 2){
	            std::cout << "Second target found" << std::endl;
				cv::rectangle(temp, {(int)x, (int)y}, {(int)right, (int)bottom},
						{255, 0, 255}, 5);
				log(temp, 'e');

				// Calculate distance based on camera parameters
				float det_height = std::fabs(bottom-y);
				float dist = calcDistance(FRONT_FOCAL_LENGTH, TARGET_HEIGHT_MM, 
					FIMG_DIM_RES[0], det_height, FRONT_SENSOR_SIZE);

				return Observation(score, (y+bottom)/2, (x+right)/2, dist);
			}
		}
		else break;
	}

	return Observation(0, 0, 0, 0);
}
