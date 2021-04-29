/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-30 14:31:23
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-03-29 11:04:56
 */
#ifndef __LIBHI_KRNL_PARAM_CONV1S1_DWC3S2_CONV1S1_H__
#define __LIBHI_KRNL_PARAM_CONV1S1_DWC3S2_CONV1S1_H__

#include "hi_krnl_param.h"
#include "hi_krnl_param_conv2d.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct
{
    conv2d_params_t conv1;
    conv2d_params_t conv2;
    conv2d_params_t conv3;

    hikl_addr_t wt_addr_conv1;
    hikl_addr_t bias_addr_conv1;
    hikl_addr_t shift_addr_conv1;
    hikl_addr_t ifm_addr_conv1;
    hikl_addr_t ofm_addr_conv1;

    hikl_addr_t wt_addr_conv2;
    hikl_addr_t bias_addr_conv2;
    hikl_addr_t shift_addr_conv2;
    hikl_addr_t ifm_addr_conv2;
    hikl_addr_t ofm_addr_conv2;

    hikl_addr_t wt_addr_conv3;
    hikl_addr_t bias_addr_conv3;
    hikl_addr_t shift_addr_conv3;
    hikl_addr_t ifm_addr_conv3;
    hikl_addr_t ofm_addr_conv3;
}paramTableConv1s1_dwc3s2_conv1s1_Entry_t;

typedef struct
{
    hirtKernelParamTableBase_t infoBase;
    int count;
    paramTableConv1s1_dwc3s2_conv1s1_Entry_t param;
} paramTableConv1s1_dwc3s2_conv1s1_t;


#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_KRNL_PARAM_CONV1S1_DWC3S2_CONV1S1_H__*/
