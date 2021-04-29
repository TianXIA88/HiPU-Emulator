/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2021-02-24 09:27:21
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-03-29 15:29:45
 */
#ifndef __QEMU_H__
#define __QEMU_H__

#include "hihw.h"
// #include "hikl_param.h"

#ifdef QEMU_ENV

#define QEMU_LOG_MEM     0
#define QEMU_LOG_MMAB    1

void qemu_echo();
void qemu_fprint(int type, int addr, int len);
void qemu_wr_mma(uint32 *ptr, uint32 val);
uint32 qemu_rd_mma(uint32 *ptr);
void qemu_arch_setup();

#endif

#endif /* __QEMU_H__ */