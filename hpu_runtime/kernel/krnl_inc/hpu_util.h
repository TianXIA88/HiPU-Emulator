/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 15:02:23
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-02 17:00:18
 */
// ---------------------------------------------------------------------------------------------------------------------
// Copyright (c) 1986 - 2020, CAG team, Institute of AI and Robotics, Xi'an Jiaotong University. Proprietary and
// Confidential All Rights Reserved.
// ---------------------------------------------------------------------------------------------------------------------
// NOTICE: All information contained herein is, and remains the property of CAG team, Institute of AI and Robotics,
// Xi'an Jiaotong University. The intellectual and technical concepts contained herein are proprietary to CAG team, and
// may be covered by P.R.C. and Foreign Patents, patents in process, and are protected by trade secret or copyright law.
//
// This work may not be copied, modified, re-published, uploaded, executed, or distributed in any way, in any time, in
// any medium, whether in whole or in part, without prior written permission from CAG team, Institute of AI and
// Robotics, Xi'an Jiaotong University.
//
// The copyright notice above does not evidence any actual or intended publication or disclosure of this source code,
// which includes information that is confidential and/or proprietary, and is a trade secret, of CAG team.
// ---------------------------------------------------------------------------------------------------------------------
// FILE NAME  : hpu_api.h
// DEPARTMENT : Architecture
// AUTHOR     : wenzhe
// AUTHOR'S EMAIL : wenzhe@xjtu.edu.cn
// ---------------------------------------------------------------------------------------------------------------------
// Ver 1.0  2020--08--01 initial version.
// ---------------------------------------------------------------------------------------------------------------------

#ifndef _HPU_UTIL_H
#define _HPU_UTIL_H

#include "hisdk_type.h"
#include "hihw.h"

#ifdef __cplusplus
extern "C" {
#endif

/**< Code will be compiled for running on HPU */
#ifndef __R_HPU
#define __R_HPU
#endif /*__R_HPU */

typedef enum {
  KERNEL_HPUMEM_TYPE_MMA = 0,     /**< MMA*/
  KERNEL_HPUMEM_TYPE_MMB,         /**< MMB */
} KrnlHPUMemType_t;

extern int u_i_mod(unsigned long value, unsigned long base);
extern uint32_t bitwise_div_32 (uint32_t dividend, uint32_t divisor);
//read from ddr, not from cache: flush first and then read    
extern void read_from_ddr(void* addr, unsigned int size, void *data);

__R_HPU
int Krnl_hpu_malloc(uint32_t *pDevAddr, size_t nBytes, KrnlHPUMemType_t memType);

__R_HPU
void buf_print(uint32_t buf_addr, uint32_t buf_len);


#ifdef __cplusplus
}
#endif

#endif

