#ifndef hirt_tensor_transform_hpp
#define hirt_tensor_transform_hpp

#include "Macro.h"
#include <iostream>
#include <stdint.h>
#include <string>
#include <cassert>

#ifdef USE_OPENCV
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  

void _NC8HW8_TO_NCHW_Uint8(const uint8_t* src, uint8_t* dst, int B, int C, int H, int W);
void _NHWC_TO_NC8HW8_Uint8(const uint8_t* src, uint8_t* dst, int B, int C, int H, int W);
void _NCHW_TO_NC8HW8_Uint8(const uint8_t* src, uint8_t* dst, int B, int C, int H, int W);
void _NC8HW8_TO_NC4HW4_Uint8(const uint8_t* src, uint8_t* dst, int B, int C, int H, int W);
void _NC4HW4_TO_NC8HW8_Uint8(const uint8_t* src, uint8_t* dst, int B, int C, int H, int W);
void _IMAGE_TO_NC8HW8(const uint8_t* src, uint8_t* dst, int H, int W, int stride);
void _WEIGHT_TO_NC8HW8(const uint8_t* src, uint8_t* dst, int B, int C, int H, int W, int stride);


#if 1
int  CHW_calcsize_HIPUMEM(const int dims[4]);
void BIAS_copyto_HIPUMEM(const uint32_t*, const int srcdim, uint32_t* destData);
void WEIGHT_copyto_HIPUMEM(const uint8_t* srcData, const int SrcDims[4], uint8_t* destData);
void DWEIGHT_copyto_HIPUMEM(const uint8_t* srcData, const int SrcDims[4], uint8_t* destData);
#endif

#endif