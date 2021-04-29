/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 14:35:14
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2020-12-31 08:59:07
 */
#include "hihw.h"
#include "libconv.h"
#include "dma.h"
#include "lock.h"

void acquire_remote_lock(uint32* val, uint32 lock_addr, uint8 x, uint8 y){
    *val = 1;
    do{  
        __swp_rmt_lock_blocking(val, x, y, lock_addr);
    }while(*val);
}

void release_remote_lock(uint32* val, uint32 lock_addr, uint8 x, uint8 y){
    *val = 0;
    __swp_rmt_lock_blocking(val, x, y, lock_addr);
}

void acquire_local_lock(uint32 lock_addr){
    int state = 1;
    do{  
        asm volatile("amoswap.w.aq %0, %0, (%1)":"+r"(state):"r"(lock_addr));
    }while(state);
}

void release_local_lock(uint32 lock_addr){
    int state = 0;
    asm volatile("amoswap.w.rl %0, %0, (%1)":"+r"(state):"r"(lock_addr));
}

void wait(int input_var){
    asm volatile(
                    "mv a0, %0\n"
                    "li a1, 0\n"
                    "ag: nop\n"
                    "addi a0, a0, -1\n"
                    "bne a0, a1, ag\n"
                    :
                    :"r"(input_var)
                    :
                );
}