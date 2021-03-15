/** @file filters.hpp
 *  @brief Vision filtering function definitions.
 *  
 *  @author David Zhang
 *  @Author Suhas Nagar
 *  @author Arnav Garg
 */
#ifndef VISION_FILTERS_HPP
#define VISION_FILTERS_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat equalizeHist(bool, bool, bool, const cv::Mat &);
cv::Mat illumination(const cv::Mat &);
void underwaterEnhance(cv::Mat &);
cv::Mat homomorphic(const cv::Mat &);
cv::Mat butterworth(const cv::Mat &, int, int, int, int);
void fft(const cv::Mat &, cv::Mat &);
void resize(cv::Mat &, int);
void preprocess(cv::Mat, std::vector<float> &, int);
float calcDistance(float, float, float, float, float);
#endif
