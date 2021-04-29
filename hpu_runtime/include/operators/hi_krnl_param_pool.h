/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-30 14:31:23
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-02-06 12:22:55
 */
#ifndef __LIBHI_KRNL_PARAM_Pool_H__
#define __LIBHI_KRNL_PARAM_Pool_H__

#include "hi_krnl_param.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


typedef struct{
    uint8 x_pos;
    uint8 y_pos;
    uint32 lcaddr;
}hikl_addr_t;

typedef struct{
    uint32 ifm_h;
    uint32 ifm_w;
    uint32 ifm_c;
    uint32 ofm_c;
    uint32 k_h;
    uint32 k_w;
}conv_shape_t;

typedef struct{
    uint8 top;
    uint8 bottom;
    uint8 side;
}pad_shape_t;

typedef struct{
    uint8 w_strd;
    uint8 h_strd;
}stride_shape_t;

typedef struct{
    uint8 w_dilat;
    uint8 h_dilat;
}dilate_shape_t;

typedef struct{
    uint32 grp_num;
}chn_group_t;

typedef struct{
    conv_shape_t cshape;
    pad_shape_t pshape;
    stride_shape_t  strd_shape;
    dilate_shape_t  dilat_shape;
    chn_group_t cgroup_num;
}Pool_params_t;

typedef struct
{
    Pool_params_t Pool;
    hikl_addr_t wt_addr;
    hikl_addr_t bias_addr;
    hikl_addr_t shift_addr;
    hikl_addr_t ifm_addr;
    hikl_addr_t ofm_addr;
}paramTablePool_Entry_t;

typedef struct
{
    hirtKernelParamTableBase_t infoBase;
    int count;
    paramTablePool_Entry_t param[0];
} paramTablePool_t;


#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_KRNL_PARAM_Pool_H__*/
