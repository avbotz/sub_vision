/** @file filters.cpp
 *  @brief Vision filtering functions.
 *
 *  These are meant for standard OpenCV vision, but could be good for
 *  preprocessing for ML too.
 *
 *  @author David Zhang
 *  @author Suhas Nagar
 *  @author Arnav Garg
 *  @author Craig Wang
 *  @author Phoebe Tang
 */
#include <rclcpp/rclcpp.hpp>
#include "sub_vision/filters.hpp"
#include "sub_vision/service.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
#include <stdio.h>
using namespace std;
using namespace cv;

cv::Mat equalizeHist(bool b, bool g, bool r, const cv::Mat &input)
{
	std::vector<cv::Mat> channels;
	cv::split(input, channels);

	if (b)
		cv::equalizeHist(channels[0], channels[0]);
	if (g)
		cv::equalizeHist(channels[1], channels[1]);
	if (r)
		cv::equalizeHist(channels[2], channels[2]);

	cv::Mat corrected;
	cv::merge(channels, corrected);

	return corrected;
}

cv::Mat illumination(const cv::Mat &input)
{
	// Convert to diff colorspace and split.
	cv::Mat img;
	std::vector<cv::Mat> channels;
	cv::cvtColor(input, img, cv::COLOR_BGR2Lab);
	cv::split(img, channels);

	// Use CLAHE.
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	cv::Mat temp0, temp1, temp2;
	clahe->setClipLimit(2);
	clahe->apply(channels[0], temp0);
	clahe->apply(channels[1], temp1);
	clahe->apply(channels[2], temp2);
	temp0.copyTo(channels[0]);
	temp1.copyTo(channels[1]);
	temp2.copyTo(channels[2]);
	cv::merge(channels, img);

	// Convert back to BGR.
	cv::Mat dst;
	cv::cvtColor(img, dst, cv::COLOR_Lab2BGR);

	return dst;
}

cv::Mat homomorphic(const cv::Mat &src)
{
	std::vector<cv::Mat> hlsimg;
	cv::Mat tmphls;
	cv::cvtColor(src, tmphls, cv::COLOR_BGR2HLS);
	cv::split(tmphls, hlsimg);
	cv::Mat img = hlsimg[0];

	// Apply FFT.
	cv::Mat fftimg;
	fft(img, fftimg);

	// Apply Butterworth HPS.
	cv::Mat filter = butterworth(fftimg, 10, 4, 100, 30);
	cv::Mat bimg;
	cv::Mat bchannels[] = {cv::Mat_<float>(filter),
		cv::Mat::zeros(filter.size(), CV_32F)};
	cv::merge(bchannels, 2, bimg);
	cv::mulSpectrums(fftimg, bimg, fftimg, 0);

	// Apply inverse FFT.
	cv::Mat ifftimg;
	cv::idft(fftimg, ifftimg, 32);
	cv::Mat expimg;
	cv::exp(ifftimg, expimg);

	cv::Mat final;
	hlsimg[0] = cv::Mat(expimg, cv::Rect(0, 0, src.cols, src.rows));
	hlsimg[0].convertTo(hlsimg[0], CV_8U);

	merge(&hlsimg[0], 3, img);
	cv::cvtColor(img, final, cv::COLOR_HLS2BGR);
	return final;
}

void fft(const cv::Mat &src, cv::Mat &dst)
{
	// Convert to a 32F mat and take log.
	cv::Mat logimg;
	src.convertTo(logimg, CV_32F);
	cv::log(logimg+1, logimg);

	// Resize to optimal fft size.
	cv::Mat padded;
	int m = cv::getOptimalDFTSize(src.rows);
	int n = cv::getOptimalDFTSize(src.cols);
	cv::copyMakeBorder(logimg, padded, 0, m-logimg.rows, 0, n-logimg.cols,
			cv::BORDER_CONSTANT, cv::Scalar::all(0));

	// Add imaginary column to mat and apply fft.
	cv::Mat plane[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(),
			CV_32F)};
	cv::Mat imgComplex;
	cv::merge(plane, 2, imgComplex);
	cv::dft(imgComplex, dst);
}

cv::Mat butterworth(const cv::Mat &img, int d0, int n, int high, int low)
{
	cv::Mat single(img.rows, img.cols, CV_32F);
	int cx = img.rows / 2;
	int cy = img.cols / 2;
	float upper = high * 0.01;
	float lower = low * 0.01;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			float radius = sqrt(pow(i-cx, 2) + pow(j-cy, 2));
			single.at<float>(i, j) = (upper-lower)*(1/pow(d0/radius, 2*n))
				+ lower;
		}
	}
	return single;
}

void underwaterEnhance(cv::Mat &mat)
{
	double discard_ratio = 0.05;
	int hists[3][256];
	memset(hists, 0, 3 * 256 * sizeof(int));

	for (int y = 0; y < mat.rows; ++y)
	{
		uchar *ptr = mat.ptr<uchar>(y);
		for (int x = 0; x < mat.cols; ++x)
		{
			for (int j = 0; j < 3; ++j)
			{
				hists[j][ptr[x * 3 + j]] += 1;
			}

		}

	}

	int total = mat.cols * mat.rows;
	int vmin[3], vmax[3];
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 255; ++j)
		{
			hists[i][j + 1] += hists[i][j];

		}
        vmin[i] = 0;
		vmax[i] = 255;
		while (hists[i][vmin[i]] < discard_ratio * total)
			vmin[i] += 1;
		while (hists[i][vmax[i]] > (1 - discard_ratio) * total)
			vmax[i] -= 1;
		if (vmax[i] < 255 - 1)
			vmax[i] += 1;
	}
	for (int y = 0; y < mat.rows; ++y)
	{
		uchar *ptr = mat.ptr<uchar>(y);
		for (int x = 0; x < mat.cols; ++x)
		{
			for (int j = 0; j < 3; ++j)
			{
				int val = ptr[x * 3 + j];
				if (val < vmin[j])
					val = vmin[j];
				if (val > vmax[j])
					val = vmax[j];
				ptr[x * 3 + j] = static_cast<uchar>((val - vmin[j]) * 255.0 / (vmax[j] - vmin[j]));
			}
		}
	}
}

void resize(cv::Mat &image, int image_size)
{
    /*
     * Resize images while preserving the aspect ratio.
     */

    int image_height = image.rows;
    int image_width = image.cols;
    int resized_height, resized_width;
    float scale;

    // calculates what height and width to resize to preserve height/width ratios
    if (image_height > image_width)
    {
        scale = (float)image_size / image_height;
        resized_height = image_size;
        resized_width = (int)(image_width * scale);
    }
    else
    {
        scale = (float)image_size / image_width;
        resized_height = (int)(image_height * scale);
        resized_width = image_size;
    }

    cv::resize(image, image, cv::Size(resized_width, resized_height));
}

void preprocess(cv::Mat image, std::vector<float> &img_data, int image_size)
{
	/*
	 * Normalizes the image and prepares image data for efficientdet model input
	 * into an std vector that is passed by reference.
	 */
	if (image.cols != image_size) resize(image, image_size);

	// Initialize new image
	image.convertTo(image, CV_32FC3);                                          // converts to float matrix so we can multiply and divide
	cv::Mat temp(image_size, image_size, CV_32FC3, cv::Scalar(128,128,128));   // makes temporary mat with shape (image_size, image_size, 3) filled with 128s
	image.copyTo(temp(cv::Rect(0, 0, image.cols, image.rows)));                // pastes the image on top left corner (point 0, 0) of empty cv mat

	// Normalize image data
	cv::divide(temp, cv::Scalar(255.0, 255.0, 255.0), temp);          //convert to values from 0-1
	temp -= cv::Scalar(0.485, 0.456, 0.406);                          //subtract the mean from each channel
	cv::divide(temp, cv::Scalar(0.229, 0.224, 0.225), temp);          //divide each channel by standard deviation

	// Put the mat inside an std vector
	img_data.assign((float*)temp.data, (float*)temp.data + temp.total()*temp.channels());   // copies the the processed mat to the vector
}

float calcDistance(float focal_length, float obj_height, float image_height, float det_height, float sensor_height)
{
	// Inputs are millimeters and pixels
	float meterDist = (focal_length * obj_height * image_height) / (det_height * sensor_height) / 1000.;
	return meterDist;
}
