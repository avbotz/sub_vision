
/** @file observation.hpp
 *  @brief Observation struct definition that represents the results from an
 *         object detection function.
 *
 *  @author David Zhang
 *  @author Suhas Nagar
 */
#ifndef OBSERVATION_HPP
#define OBSERVATION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "sub_vision/config.hpp"

struct Observation
{
	float prob;
	float r, c;
	float dist;
	float hangle, vangle;
	bool valid;

	Observation(float _prob, float _r, float _c, float _dist)
	{
		this->prob = _prob;
		this->r = _r;
		this->c = _c;
		this->dist = _dist;
	}

	Observation(float _prob, float _r, float _c, float _dist, bool _valid)
	{
		this->prob = _prob;
		this->r = _r;
		this->c = _c;
		this->dist = _dist;
		this->valid = _valid;
	}

	Observation(float _prob, float _r, float _c, float _dist, float _hangle,
			float _vangle)
	{
		this->prob = _prob;
		this->r = _r;
		this->c = _c;
		this->dist = _dist;
		this->hangle = _hangle;
		this->vangle = _vangle;
	}

	void calcAngles(int camera)
    {
		if (camera == FRONT)
		{
			// this->vangle = (r-FIMG_DIM[0]/2.0)/FIMG_DIM[0]*VFOV;
			// this->hangle = (c-FIMG_DIM[1]/2.0)/FIMG_DIM[1]*HFOV;
			this->vangle=std::atan((2*r-FIMG_DIM_RES[0])*(std::tan(VFOV*M_PI/360))/(FIMG_DIM_RES[0]))*180/M_PI;
			this->hangle=std::atan((2*c-FIMG_DIM_RES[1])*(std::tan(HFOV*M_PI/360))/(FIMG_DIM_RES[1]))*180/M_PI;
		}
		else if (camera == DOWN)
		{
			// this->vangle = (DIMG_DIM[0]/2.0-r)/DIMG_DIM[0]*DOWN_VFOV;
			// this->hangle = (c-DIMG_DIM[1]/2.0)/DIMG_DIM[1]*DOWN_HFOV;
			this->vangle=std::atan((DIMG_DIM_RES[0]-2*r)*(std::tan(DOWN_VFOV*M_PI/360))/(DIMG_DIM_RES[0]))*180/M_PI;
			this->hangle=std::atan((2*c-DIMG_DIM_RES[1])*(std::tan(DOWN_HFOV*M_PI/360))/(DIMG_DIM_RES[1]))*180/M_PI;
		}
	}

	std::string text()
	{
		std::ostringstream os;
		os.precision(2);
		os << std::fixed;
		os << "(" << hangle << " " << vangle << ", " << r << " " << c << ", " <<
			dist << ", " << prob << ")";
		return os.str();
	}
};

#endif
