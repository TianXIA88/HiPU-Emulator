/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-30 14:31:23
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-03-29 11:04:56
 */
#ifndef __LIBHI_KRNL_PARAM_NETWORK_OBSTACLE_H__
#define __LIBHI_KRNL_PARAM_NETWORK_OBSTACLE_H__

#include "hi_krnl_param.h"
#include "hi_krnl_param_conv2d.h"
#include "hi_krnl_param_conv1s1_dwc3s1_conv1s1_add.h"
#include "hi_krnl_param_conv1s1_dwc3s2_conv1s1.h"
#include "hi_krnl_param_conv1s1_upsmp2x_add.h"
#include "hi_krnl_param_conv3s2_dwc3s1_conv1s1.h"
#include "hi_krnl_param_conv1s1_conv3s1_conv3s1.h"

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
    paramTableConv1s1_upsmp2x_add_Entry_t           param_blk8;
    paramTableConv1s1_upsmp2x_add_Entry_t           param_blk9;
    paramTableConv1s1_upsmp2x_add_Entry_t           param_blk10;
    paramTableConv1s1_conv3s1_conv3s1_Entry_t       param_blk11;
    paramTableConv2d_Entry_t                        param_blk11_1a_s2;
    paramTableConv2d_Entry_t                        param_blk11_1a_c3;
    paramTableConv2d_Entry_t                        param_blk11_1b_s2;
    paramTableConv2d_Entry_t                        param_blk11_1b_r3;
    paramTableConv1s1_conv3s1_conv3s1_Entry_t       param_blk12;
    paramTableConv2d_Entry_t                        param_blk12_1a_s2;
    paramTableConv2d_Entry_t                        param_blk12_1a_c3;
    paramTableConv2d_Entry_t                        param_blk12_1b_s2;
    paramTableConv2d_Entry_t                        param_blk12_1b_r3;
    paramTableConv1s1_conv3s1_conv3s1_Entry_t       param_blk13;
    paramTableConv2d_Entry_t                        param_blk13_1a_s2;
    paramTableConv2d_Entry_t                        param_blk13_1a_c3;
    paramTableConv2d_Entry_t                        param_blk13_1b_s2;
    paramTableConv2d_Entry_t                        param_blk13_1b_r3;
} paramTable_net_obstacle_t;

#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_KRNL_PARAM_NETWORK_OBSTACLE_H__*/
