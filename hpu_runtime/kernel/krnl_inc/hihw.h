/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 14:33:45
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-27 17:16:51
 */
#ifndef HIPU200CLIB_HIHW_H__
#define HIPU200CLIB_HIHW_H__

// #include "config.h"

// #define QEMU_ENV

// #ifdef QEMU_ENV
// #include "stdio.h"
// #include "qemu.h"
// #else
// #define printf(x, ...)      ;;
// #endif

#include "hihw_csr_def.h"
#include "hihw_ext_isa.h"
#include "hihw_reg_def.h"
#include "krnl_log.h"

#define MMA_BEGIN       (0)             // in bytes, 0 - 256 KB
#define MMB_BEGIN       (1 << 18)       // in bytes, 256 - 512 KB
#define MMA_BYTE_SIZE   (1 << 18)       // 256KB
#define MMB_BYTE_SIZE   (1 << 18)       // 256KB

#define MEM_CLINT_ADDR_S (0x02000000)
#define MEM_DTCM_ADDR_S  (0x02010000)
#define AMO_ADDR_S       (0x02010000)     // total 64 Bytes
#define MEM_LCMEM_ADDR_S (0x02100000)

#define MMA_START_ADDR        (MEM_LCMEM_ADDR_S + MMA_BEGIN)
#define MMA_BANK_SIZE  (MMA_BYTE_SIZE >> 3)    // MMA is divided into 8 separate banks      32KB:32768Bytes, 512 HPU index
#define MMA_BANK0_START_ADDR     (MMA_START_ADDR)
#define MMA_BANK1_START_ADDR     (MMA_START_ADDR + MMA_BANK_SIZE * 1)
#define MMA_BANK2_START_ADDR     (MMA_START_ADDR + MMA_BANK_SIZE * 2)
#define MMA_BANK3_START_ADDR     (MMA_START_ADDR + MMA_BANK_SIZE * 3)
#define MMA_BANK4_START_ADDR     (MMA_START_ADDR + MMA_BANK_SIZE * 4)
#define MMA_BANK5_START_ADDR     (MMA_START_ADDR + MMA_BANK_SIZE * 5)
#define MMA_BANK6_START_ADDR     (MMA_START_ADDR + MMA_BANK_SIZE * 6)
#define MMA_BANK7_START_ADDR     (MMA_START_ADDR + MMA_BANK_SIZE * 7)

#define MMB_START_ADDR        (MEM_LCMEM_ADDR_S + MMB_BEGIN)

#define DIVIDE_BY_32BYTES(x)    (x >> 5)        // (x / 32 Bytes)
#define HPU64BytesWordToByte(x)     ((x)<<6)        // Each MM word = 64 Bytes
#define ByteToHPU64BytesWord(x)     ((x)>>6)
#define W64ToByte(x)                ((x)<<6)        // Each MM word = 64 Bytes
#define ByteToW64(x)                ((x)>>6)
#define align_4K(x)     (((((x)-1)>>12)+1)<<12)

#define HPU200_NDMA_CHANNEL_NUM (8)       //读写共用8个通道index

#define _set_mmac_region_start(x)      csrw(CSR_MTX_REGION_START, x)
#define _set_mmac_region_end(x)        csrw(CSR_MTX_REGION_END, x)
#define _set_mmac_fm_blk_size(x)       csrw(CSR_MTX_BLK_SIZE, x)
#define _set_mmac_fm_blk_num(x)        csrw(CSR_MTX_BLK_NUM, x)
#define _set_mmac_fm_cluster_start(x)      csrw(CSR_MTX_CLUSTER_START, x)
#define _set_mmac_fm_cluster_end(x)        csrw(CSR_MTX_CLUSTER_END, x)
#define _set_mmac_fm_cluster_num(x)        csrw(CSR_MTX_CLS_NUM, x)
#define _set_mmac_fm_blk_stride(x)     csrw(CSR_MTXRW_BLK_STRIDE, x)
#define _set_mmac_fm_cluster_stride(x)     csrw(CSR_MTXRW_CLS_STRIDE, x)
#define _set_mmac_wt_blk_stride(x)     csrw(CSR_MTXRO_BLK_STRIDE, x)
#define _set_mmac_wt_cluster_stride(x)     csrw(CSR_MTXRO_CLS_STRIDE, x)
//pad_type == 1: 64Byte, fm 平均分成8份，存放顺序是：第一块放到 0~7 Byte， 第二块放到8~15
//pad_type == 2: 64Byte, fm 平均分成8份，存放顺序是：第一块放到 56~63 Byte， 第二块放到48~55 Byte
//pad_type == 0: batch mode, 一次算8幅图像
#define MTX_PAD_TYPE_1                  (1)
#define _set_mtx_pad_type(x)            csrw(CSR_MTX_PAD_TYPE, x)

//round type: 0: 直接取整， 1: 四舍五入， 2: 向上取整
// type: | 1.2 | 1.5 | 1.7 | -1.2 | -1.5 | -1.7 | 2.5 | -2.5
// 0   : | 1.0 | 1.0 | 1.0 | -2.0 | -2.0 | -2.0 | 2.0 | -3.0
// 1   : | 1.0 | 2.0 | 2.0 | -1.0 | -1.0 | -2.0 | 3.0 | -2.0
// 2   : | 2.0 | 2.0 | 2.0 | -1.0 | -1.0 | -1.0 | 3.0 | -2.0
#define _set_mmac_round_type(x)        csrw(CSR_ROUND_TYPE, x)
#define _set_mmac_fadd_shift_num(x)    csrw(CSR_FADD_SHIFT_NUM, x)
#define _set_mmac_fadd_prot_high(x)    csrw(CSR_FADD_PROT_HIGH, x)
#define _set_mmac_fadd_prot_low(x)     csrw(CSR_FADD_PROT_LOW, x)
#define _set_mmac_fsub_shift_num(x)    csrw(CSR_FSUB_SHIFT_NUM, x)
#define _set_mmac_fsub_prot_high(x)    csrw(CSR_FUSB_PROT_HIGH, x)
#define _set_mmac_fsub_prot_low(x)     csrw(CSR_FUSB_PROT_LOW, x)
#define _set_mmac_fmul_shift_num(x)    csrw(CSR_FMUL_SHIFT_NUM, x)
#define _set_mmac_fmul_prot_high(x)    csrw(CSR_FMUL_PROT_HIGH, x)
#define _set_mmac_fmul_prot_low(x)     csrw(CSR_FMUL_PROT_LOW, x)

#define _set_intr_mtvec(x)             asm("csrw mtvec,     %0"::"r"(x):)
#define _set_bit_intr_mstatus(x)       asm("csrs mstatus,   %0"::"r"(x):)
#define _clr_bit_intr_mstatus(x)       asm("csrc mstatus,   %0"::"r"(x):)
#define _set_intr_mscratch(x)          asm("csrw mscratch,  %0"::"r"(x):)
#define _set_intr_mie(x)               asm("csrw mie,       %0"::"r"(x):)
#define _set_bit_intr_mie(x)           asm("csrs mie,       %0"::"r"(x):)
#define _clr_bit_intr_mie(x)           asm("csrc mie,       %0"::"r"(x):)

#define _clr_vreg(x)                 vsub_vv(0,x,x,x)

#define LIBHIKL_NASSERT(x)       if(x){ printf("ASSERT ERROR: line: %d in func: %s()\n\r", __LINE__, __FUNCTION__); __abort(); }
#define LIBHIKL_ASSERT(x)       if(!x){ printf("ASSERT ERROR: line: %d in func: %s()\n\r", __LINE__, __FUNCTION__); __abort(); }
#define __abort()               {_eexit();}
extern void _end();
extern void _eexit();
extern void _intrpt_vec_table();

#endif /*HIPU200CLIB_HIHW_H__*/
