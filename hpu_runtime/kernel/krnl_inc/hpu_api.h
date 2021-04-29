/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 15:02:23
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2020-12-30 08:52:14
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

#ifndef _HPU_API_H
#define _HPU_API_H

#ifdef __cplusplus
extern "C" {
#endif

// system function
extern int get_hpuid(void);
// extern void enable_timer_intr(void);
// extern void disable_timer_intr(void);
extern void _timer_handle(void);
// --NDMA
extern void enable_ndma_intr(void);
extern void disable_ndma_intr(void);
// extern void _ndma_handle(void);
extern void ndma_save_data(int loc_addr, int rmt_addr, int len, int node_xy);
extern void ndma_load_data(int loc_addr, int rmt_addr, int len, int node_xy);
extern void ndma_atomic_swap(int loc_addr, int rmt_addr, int node_xy);
extern void ndma_wait(void);
extern void claim_sim_complete(int fail);

#ifdef __cplusplus
}
#endif

#endif

