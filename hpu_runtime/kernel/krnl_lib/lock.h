/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 14:35:14
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2020-12-31 10:28:28
 */
#ifndef LIBLOCK_H
#define LIBLOCK_H

#include "hisdk_type.h"
#include "hihw.h"
#include "libconv.h"
#include "dma.h"

void init_local_lock(local_fm* fm);

// void acquire_remote_lock(remote_fm* fm, uint32 row_num, uint8 x, uint8 y);
// void release_remote_lock(remote_fm* fm, uint32 row_num, uint8 x, uint8 y);
// void acquire_local_lock(local_fm* fm, uint32 row_num);
// void release_local_lock(local_fm* fm, uint32 row_num);

void acquire_remote_lock(uint32* val, uint32 lock_addr, uint8 x, uint8 y);
void release_remote_lock(uint32* val, uint32 lock_addr, uint8 x, uint8 y);
void acquire_local_lock(uint32 lock_addr);
void release_local_lock(uint32 lock_addr);

void wait(int input_var);

#endif /*LIBLOCK_H*/