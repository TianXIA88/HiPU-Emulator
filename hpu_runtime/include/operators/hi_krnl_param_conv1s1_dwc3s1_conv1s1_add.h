
/*
 * @Descripttion: 
 * @version: 
 * @Author: wangyz
 * @Date: 
 * @LastEditors: wangyz
 * @LastEditTime: 
 */
#ifndef __LIBHI_KRNL_PARAM_CONV1S1_DWC3S1_CONV1S1_ADD_H__
#define __LIBHI_KRNL_PARAM_CONV1S1_DWC3S1_CONV1S1_ADD_H__

#include "hi_krnl_param.h"
#include "hi_krnl_param_conv2d.h"
#include "hi_krnl_param_add.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct
{
    conv2d_params_t conv1;
    conv2d_params_t conv2;
    conv2d_params_t conv3;
    add_params_t add1;

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

    hikl_addr_t ofm_addr_add1;
} paramTableConv1s1_dwc3s1_conv1s1_add_Entry_t;

typedef struct
{
    hirtKernelParamTableBase_t infoBase;
    int count;
    paramTableConv1s1_dwc3s1_conv1s1_add_Entry_t param;
} paramTableConv1s1_dwc3s1_conv1s1_add_t;


#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_KRNL_PARAM_CONV1S1_DWC3S1_CONV1S1_ADD_H__*/
