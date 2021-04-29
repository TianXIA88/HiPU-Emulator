/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-30 14:31:23
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-10 15:39:03
 */
#ifndef __LIBHI_KRNL_PARAM_ADD_H__
#define __LIBHI_KRNL_PARAM_ADD_H__

#include "hi_krnl_param.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct{
    uint32 shift;
    uint32 clip;
}add_params_t;


#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_KRNL_PARAM_ADD_H__*/
