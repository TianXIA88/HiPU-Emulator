/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-30 14:31:23
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-16 17:19:17
 */
#ifndef __LIBHI_KRNL_PARAM_CONV2D_H__
#define __LIBHI_KRNL_PARAM_CONV2D_H__

#include "hi_krnl_param.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define CONV_TYPE_CLASSIC 1
#define CONV_TYPE_DEPTH_WISE 2

#define CONV_SINGLE -1
#define CONV_HEAD 0
#define CONV_TAIL 1
#define CONV_BODY 100


#define TIMER_TICK (0x6000)

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

//以下两种情况只向左、上补padding
// if(ofm_w % 2 == 0 && w_strd == 2 && k_w == 3)
// {
//     pshape.top = 0;
//     pshape.left = 1;
//     pshape.bottom = 0;
//     pshape.right = 0;
// }

// if(ofm_h % 2 == 0 && h_strd == 2 && k_h == 3)
// {
//     pshape.top = 1;
//     pshape.left = 0;
//     pshape.bottom = 0;
//     pshape.right = 0;
// }
typedef struct{
    uint8 top;
    uint8 bottom;
    uint8 left;
    uint8 right;
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
    uint32 relu;              //true: conv + bn + relu(0/1); false: conv + bn
}conv2d_params_t;

typedef struct
{
    conv2d_params_t conv2d;
    hikl_addr_t wt_addr;
    hikl_addr_t bias_addr;
    hikl_addr_t shift_addr;
    hikl_addr_t ifm_addr;
    hikl_addr_t ofm_addr;
}paramTableConv2d_Entry_t;

typedef struct
{
    hirtKernelParamTableBase_t infoBase;
    int count;
    paramTableConv2d_Entry_t param[0];
} paramTableConv2d_t;


#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_KRNL_PARAM_CONV2D_H__*/
