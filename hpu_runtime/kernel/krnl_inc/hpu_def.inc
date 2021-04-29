## ---------------------------------------------------------------------------------------------------------------------
## Copyright (c) 1986 - 2020, CAG team, Institute of AI and Robotics, Xi'an Jiaotong University. Proprietary and
## Confidential All Rights Reserved.
## ---------------------------------------------------------------------------------------------------------------------
## NOTICE: All information contained herein is, and remains the property of CAG team, Institute of AI and Robotics,
## Xi'an Jiaotong University. The intellectual and technical concepts contained herein are proprietary to CAG team, and
## may be covered by P.R.C. and Foreign Patents, patents in process, and are protected by trade secret or copyright law.
##
## This work may not be copied, modified, re-published, uploaded, executed, or distributed in any way, in any time, in
## any medium, whether in whole or in part, without prior written permission from CAG team, Institute of AI and
## Robotics, Xi'an Jiaotong University.
##
## The copyright notice above does not evidence any actual or intended publication or disclosure of this source code,
## which includes information that is confidential and/or proprietary, and is a trade secret, of CAG team.
## ---------------------------------------------------------------------------------------------------------------------
## FILE NAME  : hpu_def.inc
## DEPARTMENT : Architecture
## AUTHOR     : wenzhe
## AUTHOR'S EMAIL : wenzhe@xjtu.edu.cn
## ---------------------------------------------------------------------------------------------------------------------
## Ver 1.0  2020--08--01 initial version.
## ---------------------------------------------------------------------------------------------------------------------

# ==========
# global definition
.equ CSR_HPU_ID,                0xf15
# -- Scalar region
# -- NoC DMA
.equ CSR_NDMA_CTRL,             0x7c0
.equ CSR_NDMA_STATUS,           0x7c1
.equ CSR_NDMA_LCADDR,           0x7c2
.equ CSR_NDMA_RTADDR,           0x7c3
.equ CSR_NDMA_SIZE,             0x7c4
.equ CSR_NDMA_DESTX,            0x7c5
.equ CSR_NDMA_DESTXY,           0x7c5
.equ CSR_NDMA_DESTY,            0x7c6
# -- VMU
.equ CSR_VMU_STATUS,            0x7cf

# -- VCSR region
# -- Vec
.equ CSR_ROUND_TYPE,            0xbc0
.equ CSR_FADD_SHIFT_NUM,        0xbc1
.equ CSR_FADD_PROT_HIGH,        0xbc2
.equ CSR_FADD_PROT_LOW,         0xbc3
.equ CSR_FSUB_SHIFT_NUM,        0xbc4
.equ CSR_FSUB_PROT_HIGH,        0xbc5
.equ CSR_FSUB_PROT_LOW,         0xbc6
.equ CSR_FMUL_SHIFT_NUM,        0xbc7
.equ CSR_FMUL_PROT_HIGH,        0xbc8
.equ CSR_FMUL_PROT_LOW,         0xbc9
# -- Mtx
.equ CSR_MTX_CLUSTER_START,     0xbd0
.equ CSR_MTX_CLUSTER_END,       0xbd1
.equ CSR_MTX_REGION_START,      0xbd2
.equ CSR_MTX_REGION_END,        0xbd3
.equ CSR_MTX_BLK_SIZE,          0xbd4
.equ CSR_MTX_BLK_NUM,           0xbd5
.equ CSR_MTX_CLS_NUM,           0xbd6
.equ CSR_MTXRW_BLK_STRIDE,      0xbd7
.equ CSR_MTXRW_CLS_STRIDE,      0xbd8
.equ CSR_MTXRO_BLK_STRIDE,      0xbd9
.equ CSR_MTXRO_CLS_STRIDE,      0xbda
.equ CSR_MTX_PAD_TYPE,          0xbdb
# -- LUT
.equ CSR_VMU_WR_LUT_FIRST,      0xbe0
.equ CSR_VMU_WR_LUT_INC,        0xbe1

# -- MIE
# | 15  | 14 | 13 | 12 | 11 | 10 | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
# | NDMA| L2C| DC | IC | ME | HE | SE| UE| MT| HT| ST| UT| MS| HS| SS| US|

# Memory map
.equ CLINT_BASE_ADDR,           0x02000000
.equ DTCM_BASE_ADDR,            0x02010000
.equ LMRW_BASE_ADDR,            0x02100000
.equ LMRO_BASE_ADDR,            0x02140000

.equ LCM_TIME_ADDR,             0x02000000
.equ LCM_TIMEH_ADDR,            0x02000004
.equ LCM_TIMECMP_ADDR,          0x02000008
.equ LCM_TIMECMPH_ADDR,         0x0200000c
.equ LCM_CSIP_ADDR,             0x02000010

.equ STACK_BASE_ADDR,           0x80003f00

.equ l2c_csr_start,        0x7f0
.equ l2c_csr_finish,       0x7f1
.equ l2c_csr_addr_set_way, 0x7f2
.equ l2c_csr_mode,         0x7f3

