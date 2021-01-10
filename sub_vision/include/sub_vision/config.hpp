/** @file sub_vision/include/vision/config.hpp
 *  @brief Vision configuration that is used in other packages as well.
 *
 *  @author David Zhang
 *  @author Suhas Nagar
 */
#ifndef VISION_CONFIG_HPP
#define VISION_CONFIG_HPP

/*
 * TODO Rewrite config using parameters.
 */

/*
 * Deprecated. Use mock_camera_node instead.
 */
enum class CameraMode { MOCK, LIVE };

enum Task
{
    NONE,
    GATE,
    GATE_ML,
    TARGET,
    TARGET_ML,
    SECOND_TARGET_ML,
    BINS,
    BINS_ML,
    OCTAGON
};

const bool VISION_SIM = false;

const CameraMode CAMERA_MODE = CameraMode::LIVE;
const int FRONT_EXPOSURE = 7500;
const bool LOG_FRONT = false;
const bool LOG_DOWN = false;
const bool FAST_LOG = false;
const bool FILTER_ON = true;
const bool SIM_FILTER_FRONT = true;
const bool SIM_FILTER_DOWN = false;
/*
const float HFOV = 83;
const float VFOV = 90;
const float DOWN_HFOV = 135;
const float DOWN_VFOV = 60;
*/
const float HFOV = 77.3;
const float VFOV = 62;
const float DOWN_HFOV = 135;
const float DOWN_VFOV = 122;
const int FRONT = 0;
const int DOWN = 1;
const int maxdim = 512;
const float FIMG_DIM[2] = { 3648, 5472 };
const float FIMG_DIM_RES[2] = {(int)((3648*maxdim)/5472),maxdim};
const float DIMG_DIM[2] = { 964, 1288 };
const float DIMG_DIM_RES[2] = {(int)((964*maxdim)/1288),maxdim};
#endif
