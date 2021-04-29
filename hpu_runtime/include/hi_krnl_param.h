/*
 * @Descripttion:
 * @version:
 * @Author: AlonzoChen
 * @Date: 2020-12-30 14:31:23
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-27 13:46:19
 */
#ifndef __LIBHI_KRNL_PARAM_H__
#define __LIBHI_KRNL_PARAM_H__

#include "hi_addr_def.h"
#include "hisdk_config.h"
#include "hisdk_type.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct {
  u32_t op_type;
  u32_t table_size;
  u32_t task_dim; // the number of cores involved in this task
  u32_t task_cores[HIPU200_SOC_CORE_NUM]; // the ID of each core, for example:
                                          // multi layers conv: Table[0] is
                                          // head, Table[1] is tail,
                                          // Table[2,3,...] are body
} hirtKernelParamTableBase_t;

/*solo operator*/
#define KERNEL_OP_RELU                                (1000)
#define KERNEL_OP_CONV2D                              (1001)
#define KERNEL_OP_BN                                  (1002)
#define KERNEL_OP_VADD                                (1003)
#define KERNEL_OP_DWCONV                              (1004)

/*compound operator*/
#define KERNEL_OP_CONV3S2_DWC3S1_CONV1S1              (1005)
#define KERNEL_OP_CONV1S1_DWC3S2_CONV1S1              (1006)
#define KERNEL_OP_CONV1S1_DWC3S1_CONV1S1_ADD          (1007)
#define KERNEL_OP_CONV1S1_UPSMP2X_ADD                 (1008)
#define KERNEL_OP_CONV1S1_CONV3S1_CONV3S1             (1009)

/*network*/
#define KERNEL_OP_NET_TRAFFICLIGHT                    (1010)
#define KERNEL_OP_NET_OBSTACLE                        (1011)
#define KERNEL_OP_NET_LANEDET                         (1012)

#define KERNEL_OP_FC                                  (10001)

#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_KRNL_PARAM_H__*/
