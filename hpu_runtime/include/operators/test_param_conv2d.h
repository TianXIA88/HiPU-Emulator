/*
 * @Descripttion: 
 * @version: 
 * @Author: wangyz
 */
#ifndef __LIBHI_KRNL_PARAM_CONV2D_H__
#define __LIBHI_KRNL_PARAM_CONV2D_H__

// #include "hi_krnl_param.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#include<string> 

typedef struct
{
    int index;
    std::string layername;
    float ks_h;
    float ks_w  ;
    float ks_if ;
    float ks_of ;
    float if_h  ;
    float if_w  ;
    float if_ch ;
    float of_h  ;
    float of_w  ;
    float of_ch ;
    float stride;
    float ele_en;
    float ele_shift;
    float ele_add_shift;
    float relu;
    float base;
    float params;
    float ops;
} OpParamTable;


#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_KRNL_PARAM_CONV2D_H__*/
