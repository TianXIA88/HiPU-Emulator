/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-30 14:31:23
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-01-11 11:29:36
 */
#ifndef __LIBHI_KRNL_PARAM_VADD_H__
#define __LIBHI_KRNL_PARAM_VADD_H__

#include "hi_krnl_param.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct
{
    u32_t srcNocnod_A;
    u32_t srcOffset_A;
    u32_t srcNocnod_B;
    u32_t srcOffset_B;
    u32_t dstNocnod;
    u32_t dstOffset;
    u32_t len;

} paramTableVAdd_Entry_t;

typedef struct
{
    hirtKernelParamTableBase_t infoBase;
    paramTableVAdd_Entry_t param;
} paramTableVAdd_t;



#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_KRNL_PARAM_VADD_H__*/
