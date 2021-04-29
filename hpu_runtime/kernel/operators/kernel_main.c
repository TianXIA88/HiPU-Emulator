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
// FILE NAME  : main.c
// DEPARTMENT : Architecture
// AUTHOR     : wenzhe
// AUTHOR'S EMAIL : wenzhe@xjtu.edu.cn
// ---------------------------------------------------------------------------------------------------------------------
// Ver 1.0  2020--08--01 initial version.
// ---------------------------------------------------------------------------------------------------------------------
#include "hpu_def.h"
#include "hpu_api.h"
#include "hihw.h"
#include "int.h"
#include "hi_addr_def.h"
#include "hi_krnl_param.h"
#include "hisdk_config.h"
#include "krnl_log.h"
// #include "qemu.h"
#include "operators/hi_krnl_param_conv2d.h"
#include "operators/hi_krnl_param_conv1s1_dwc3s2_conv1s1.h"

typedef void (*op_func)();
typedef struct
{
  u32_t op_type;
  char op_name[50]; 
  op_func op_func;
} HiKernelOpEntry;

//custom defined op_funcs 
extern void conv2d_multi_layers();
extern void fc_multi_layers();
extern void flush_l2_cache();
extern void kernel_conv3s2_dwc3s1_conv1s1();
extern void kernel_conv1s1_dwc3s2_conv1s1();
extern void kernel_conv1s1_dwc3s1_conv1s1_add();
extern void kernel_conv1s1_upsmp2x_add();
extern void kernel_conv1s1_conv3s1_conv3s1();

void relu(){};
void bn(){};
extern void vadd();
extern void kernel_network_trafficlight();
#ifdef QEMU_ENV
    extern void qemu_arch_setup();
#endif

#define BASE_INDEX 1000

static HiKernelOpEntry gOpEntryList[] = {
    {KERNEL_OP_RELU, "relu", &relu},
    {KERNEL_OP_CONV2D, "conv2d_multi_layers", &conv2d_multi_layers},
    {KERNEL_OP_BN, "bn", &bn},
    {KERNEL_OP_VADD, "vadd", &vadd},
    {KERNEL_OP_DWCONV, "depthwise_conv", &conv2d_multi_layers},
    {KERNEL_OP_CONV3S2_DWC3S1_CONV1S1, "conv3s2_dwc3s1_conv1s1", &kernel_conv3s2_dwc3s1_conv1s1},
    {KERNEL_OP_CONV1S1_DWC3S2_CONV1S1, "conv1s1_dwc3s2_conv1s1", &kernel_conv1s1_dwc3s2_conv1s1},
    {KERNEL_OP_CONV1S1_DWC3S1_CONV1S1_ADD, "conv1s1_dwc3s1_conv1s1_add", &kernel_conv1s1_dwc3s1_conv1s1_add},
    {KERNEL_OP_CONV1S1_UPSMP2X_ADD, "conv1s1_upsmp2x_add", &kernel_conv1s1_upsmp2x_add},
    {KERNEL_OP_CONV1S1_CONV3S1_CONV3S1, "conv1s1_conv3s1_conv3s1", &kernel_conv1s1_conv3s1_conv3s1},
    // {KERNEL_OP_FC,"fc_multi_layers",&fc_multi_layers},
    {KERNEL_OP_NET_TRAFFICLIGHT, "net_trafficlight",  &kernel_network_trafficlight},
    {KERNEL_OP_NET_OBSTACLE, "op_net_obstacle", NULL},
    {KERNEL_OP_NET_LANEDET, "op_net_lanedet", NULL}
     };

void _kernel_sync(int rootCoreNum, int idx);

void kernel_entry(void) {
#ifdef QEMU_ENV
  qemu_arch_setup();
  printf("func: kernel_entry\n");
#endif
  amo_struct *amo_mem = (amo_struct *)AMO_ADDR_S;
  amo_mem->busy_ndma_channel_count = 0;
  // initialize the interrupt configuration
  // init_intr();         //set at the beginning of _init
  enable_intr();
  unsigned int *_ssig = (unsigned int *)HIPU200_MEM_SSIG_ADDR;
  // int _coreid = get_hpuid(); /*get core id number*/
  unsigned int _coreid = amo_mem->_coreid;
  KRNL_LOG_INFO(LOG_SYSTEM, "enter into kernel_entry for core: %d\n", _coreid);

  // unsigned int _taskid;
  // int *_rtcode = (int *)HIPU200_KNL_RTCODE_ADDR; /*kernel return code to host
  // runtime*/
  // *_rtcode = HIPU200_KNL_RTCODE_BEGIN;
  amo_mem->return_code = HIPU200_KNL_RTCODE_BEGIN;

#ifdef QEMU_ENV
  uint32_t PTABLE_ADDR =  0x8c299000;
  amo_mem->param_table = PTABLE_ADDR;
  paramTableConv1s1_dwc3s2_conv1s1_t *_pParamTable = (hirtKernelParamTableBase_t *)amo_mem->param_table;

  hikl_addr_t ddr_ifm1_addr = {0, 3, 0xc0000000};
  hikl_addr_t ddr_ifm2_addr = {0, 3, 0xc0100000};
  hikl_addr_t ddr_ifm3_addr = {0, 3, 0xc0200000};
  hikl_addr_t ddr_wt1_addr = {0, 3, 0xc0300000};
  hikl_addr_t ddr_wt2_addr = {0, 3, 0xc0400000};
  hikl_addr_t ddr_wt3_addr = {0, 3, 0xc0500000};
  hikl_addr_t ddr_bias1_addr = {0, 3, 0xc0600000};
  hikl_addr_t ddr_bias2_addr = {0, 3, 0xc0700000};
  hikl_addr_t ddr_bias3_addr = {0, 3, 0xc0800000};
  hikl_addr_t ddr_shift1_addr = {0, 3, 0xc0900000};
  hikl_addr_t ddr_shift2_addr = {0, 3, 0xc0a00000};
  hikl_addr_t ddr_shift3_addr = {0, 3, 0xc0b00000};
  hikl_addr_t ddr_ofm1_addr = {0, 3, 0xc0c00000};
  hikl_addr_t ddr_ofm2_addr = {0, 3, 0xc0d00000};
  hikl_addr_t ddr_ofm3_addr = {0, 3, 0xc0e00000};
  _pParamTable->param.ifm_addr_conv1 = ddr_ifm1_addr;
  _pParamTable->param.ifm_addr_conv2 = ddr_ifm2_addr;
  _pParamTable->param.ifm_addr_conv3 = ddr_ifm3_addr;
  _pParamTable->param.wt_addr_conv1 = ddr_wt1_addr;
  _pParamTable->param.wt_addr_conv2 = ddr_wt2_addr;
  _pParamTable->param.wt_addr_conv3 = ddr_wt3_addr;
  _pParamTable->param.bias_addr_conv1 = ddr_bias1_addr;
  _pParamTable->param.bias_addr_conv2 = ddr_bias2_addr;
  _pParamTable->param.bias_addr_conv3 = ddr_bias3_addr;
  _pParamTable->param.shift_addr_conv1 = ddr_shift1_addr;
  _pParamTable->param.shift_addr_conv2 = ddr_shift2_addr;
  _pParamTable->param.shift_addr_conv3 = ddr_shift3_addr;
  _pParamTable->param.ofm_addr_conv1 = ddr_ofm1_addr;
  _pParamTable->param.ofm_addr_conv2 = ddr_ofm2_addr;
  _pParamTable->param.ofm_addr_conv3 = ddr_ofm3_addr;

  // hikl_addr_t ddr_ifm1_addr = {0, 3, 0xc0000000};
  // hikl_addr_t ddr_wt1_addr = {0, 3, 0xc0300000};
  // hikl_addr_t ddr_bias1_addr = {0, 3, 0xc0600000};
  // hikl_addr_t ddr_shift1_addr = {0, 3, 0xc0900000};
  // hikl_addr_t ddr_ofm1_addr = {0, 3, 0xc0c00000};
  // _pParamTable->param[0].ifm_addr = ddr_ifm1_addr;
  // _pParamTable->param[0].wt_addr = ddr_wt1_addr;
  // _pParamTable->param[0].bias_addr = ddr_bias1_addr;
  // _pParamTable->param[0].shift_addr = ddr_shift1_addr;
  // _pParamTable->param[0].ofm_addr = ddr_ofm1_addr;
#endif

  // hirtKernelParamTableBase_t *_ptable = *((hirtKernelParamTableBase_t
  // **)HIPU200_KNL_PTABLE_ADDR);/*get kernel param table from runtime*/
  hirtKernelParamTableBase_t *_ptable =
      (hirtKernelParamTableBase_t *)amo_mem->param_table;

  int paramTableLen = _ptable->table_size;
  int parallelCoreNum = _ptable->task_dim;

  KRNL_LOG_INFO(LOG_SYSTEM, "task_dim: %d", _ptable->task_dim);
  /*judge if the parallelism is bigger than the maxcorenum*/
#ifdef QEMU_ENV
  printf("the addr of table: %x\n", _ptable);
  printf("parallelCoreNum: %d\n", parallelCoreNum);
  printf("parallelLen: %d\n", _ptable->table_size);
  printf("op_type: %d\n", _ptable->op_type);
  // wchar_t *script = "../lanuch.py";
#endif
  if ((parallelCoreNum > HIPU200_SOC_CORE_NUM) || (parallelCoreNum < 1)) {
    amo_mem->return_code = 0xdead0001;
    goto fail;
  }

  if (paramTableLen < 1) {
    amo_mem->return_code = 0xdead0003;
    goto fail;
  }
  // enter into op func
  ((HiKernelOpEntry *)&gOpEntryList[_ptable->op_type - BASE_INDEX])->op_func();
  // synchronize all cores within current task group
  _kernel_sync(0, 0);
  KRNL_LOG_INFO(LOG_DEBUG, "%s"," --------------kernel sync over ------------ ");			
  // return code = ok
  // *_rtcode = HIPU200_KNL_RTCODE_SUCCESS;
  amo_mem->return_code = HIPU200_KNL_RTCODE_SUCCESS;
  #ifdef QEMU_ENV
 	printf("QEMU KERNEL OVER\n\r"); 
  	qemu_fprint(QEMU_LOG_MEM, ddr_ofm3_addr.lcaddr, 1280*64); 
  #endif
  disable_intr();
fail:
  flush_l2_cache();
  _end();
}

// multicore synchronization with rootcore dtcm
void _kernel_sync(int rootCoreNum, int idx) {}

void timer_callback(void) { return; }

void exception_handle() {
  unsigned int *p_mcause = (unsigned int *)HIPU200_KNL_MCAUSE_ADDR;
  unsigned int *p_mepc = (unsigned int *)HIPU200_KNL_MEPC_ADDR;
  unsigned int *p_mtval = (unsigned int *)HIPU200_KNL_MTVAL_ADDR;
  unsigned int *p_mstatus = (unsigned int *)HIPU200_KNL_MSTATUS_ADDR;
  asm volatile("csrr %[mcause_dest], mcause\n"
               : [mcause_dest] "=r"(*p_mcause)::);
  asm volatile("csrr %[mepc_dest], mepc\n" : [mepc_dest] "=r"(*p_mepc)::);
  asm volatile("csrr %[mtval_dest], mtval\n" : [mtval_dest] "=r"(*p_mtval)::);
  asm volatile("csrr %[mstatus_dest], mstatus\n"
               : [mstatus_dest] "=r"(*p_mstatus)::);

  _end();
}
