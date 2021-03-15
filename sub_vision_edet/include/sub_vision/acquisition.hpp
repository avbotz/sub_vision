/** @file acquisition.hpp
 *  @brief Function definitions for image acquisition from BlackFly cameras.
 *
 *  @author David Zhang
 */
#ifndef VISION_ACQUISITION_HPP
#define VISION_ACQUISITION_HPP

#include <string>
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

void setupContinuousAcquisition(Spinnaker::CameraPtr camera);

void setupContinuousAcquisition(Spinnaker::CameraPtr camera, int exposure_time);

void runCamera(Spinnaker::CameraPtr camera, std::string channel);

#endif
