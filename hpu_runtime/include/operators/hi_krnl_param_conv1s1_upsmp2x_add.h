/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-30 14:31:23
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-03-29 11:04:56
 */
#ifndef __LIBHI_KRNL_PARAM_CONV1S1_UPSMP2X_ADD_H__
#define __LIBHI_KRNL_PARAM_CONV1S1_UPSMP2X_ADD_H__

#include "hi_krnl_param.h"
#include "hi_krnl_param_conv2d.h"
#include "hi_krnl_param_add.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


                              
//                      ifm_add1 
//          /              /     
//          |              |     
//          |              |     
//          |              |     
//    /-----\-----\        |     
//    |           |        |     
//    |   conv1   |        |     
//    |    1s1    |        |     
//    |           |        |     
//    \-----/-----/        |     
//          |              |     
//          |              |     
//          |              |     
//          |              |     
//          |              |     
//    /-----\-----\        |     
//    |           |        |     
//    |  upsmp2x  |        |     
//    |           |        |     
//    |           |        |     
//    \------,----/        |     
//            `.           |     
//              `.         |     
//                `.       |     
//                  `.     |     
//                /---'----\--\  
//                |           |  
//                |   add     |  
//                |           |  
//                |           |  
//                \-----/-----/  
//                      |        
//                      \        
//                   ofm_add1    

typedef struct
{
    conv2d_params_t conv1;
    add_params_t add1;

    hikl_addr_t wt_addr_conv1;
    hikl_addr_t bias_addr_conv1;
    hikl_addr_t shift_addr_conv1;
    hikl_addr_t ifm_addr_conv1;
    hikl_addr_t ofm_addr_conv1;

    hikl_addr_t ifm_addr_add1; /*shortcut input for add operation*/
    hikl_addr_t ofm_addr_add1;
}paramTableConv1s1_upsmp2x_add_Entry_t;

typedef struct
{
    hirtKernelParamTableBase_t infoBase;
    int count;
    paramTableConv1s1_upsmp2x_add_Entry_t param;
} paramTableConv1s1_upsmp2x_add_t;


#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_KRNL_PARAM_CONV1S1_UPSMP2X_ADD_H__*/
