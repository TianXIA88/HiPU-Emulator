/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-30 14:31:23
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-03-29 11:04:56
 */
#ifndef __LIBHI_KRNL_PARAM_NETWORK_LANE_EFFICIENTNET_H__
#define __LIBHI_KRNL_PARAM_NETWORK_LANE_EFFICIENTNET_H__

#include "hi_krnl_param.h"
#include "hi_krnl_param_conv2d.h"
#include "hi_krnl_param_conv1s1_dwc3s1_conv1s1_add.h"
#include "hi_krnl_param_conv1s1_dwc3s2_conv1s1.h"
#include "hi_krnl_param_conv3s2_dwc3s1_conv1s1.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


typedef struct
{
    hirtKernelParamTableBase_t infoBase;
    int count;

    paramTableConv3s2_dwc3s1_conv1s1_Entry_t        param_blk1;
    paramTableConv1s1_dwc3s2_conv1s1_Entry_t        param_blk2;
    paramTableConv1s1_dwc3s1_conv1s1_add_Entry_t    param_blk3;
    paramTableConv1s1_dwc3s2_conv1s1_Entry_t        param_blk4;
    paramTableConv1s1_dwc3s1_conv1s1_add_Entry_t    param_blk5;
    paramTableConv1s1_dwc3s2_conv1s1_Entry_t        param_blk6;
    paramTableConv1s1_dwc3s1_conv1s1_add_Entry_t    param_blk7;
    paramTableConv1s1_dwc3s2_conv1s1_Entry_t        param_blk8;
    paramTableConv1s1_dwc3s1_conv1s1_add_Entry_t    param_blk9;
    paramTableConv2d_Entry_t                        param_blk10;
    paramTableConv2d_Entry_t                        param_blk29;
    paramTableConv2d_Entry_t                        param_blk30;
} paramTable_net_lane_efficientnet_t;

#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_KRNL_PARAM_NETWORK_LANE_EFFICIENTNET_H__*/
