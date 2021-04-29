/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 14:35:14
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-07 10:31:52
 */
#include "hihw.h"
#include "int.h"
#include "libconv.h"

#ifdef QEMU_ENV
#define VEC_TABLE_ADDR      0x20400000
#else
// #define VEC_TABLE_ADDR      0xa0000000
#endif
#define LMRO_BASE_ADDR      0x02140000
#define LCM_TIME_ADDR       0x0200bff8
#define LCM_TIMEH_ADDR      0x0200bffc
#define LCM_TIMECMP_ADDR    0x02004000
#define LCM_TIMECMPH_ADDR   0x02004004
#define LCM_MSIP_ADDR       0x02000000

// Function: void init_intr(void)
void init_intr(){
    uint32 mie = 0;
    _set_intr_mie(mie); // disable all intr src
    // _set_intr_mtvec(VEC_TABLE_ADDR + 1); // set vector table
    _set_intr_mtvec(_intrpt_vec_table + 1); // set vector table
    return;
}

void enable_intr(){
    uint32 mstatus = 8;
    uint32 mie = 0x800;     //enable external interrupt
    _set_bit_intr_mstatus(mstatus);
    _set_intr_mie(mie); 

}

void disable_intr(){
    uint32 mstatus = 8;
    _clr_bit_intr_mstatus(mstatus);
}

//  Function: void enable_timer_intr(void)
void enable_timer_intr(){
    uint32 mie = 0x80;
    _set_bit_intr_mie(mie);
    return;
}

//  Function: void disable_timer_intr(void)
void disable_timer_intr(){
    uint32 mie = 0x80;
    _clr_bit_intr_mie(mie);
    return;
}

//  Function: void enable_swi_intr(void)
void enable_swi(){
    uint32 mie = 0x8;
    _set_bit_intr_mie(mie);
    return;
}

//  Function: void disable_swi_intr(void)
void disable_swi(){
    uint32 mie = 0x8;
    _clr_bit_intr_mie(mie);
    return;
}

void gen_swi(){
    uint32 *msip = (uint32 *)LCM_MSIP_ADDR;
    *msip = 1;
    return;
}

void clr_swi(){
    uint32 *msip = (uint32 *)LCM_MSIP_ADDR;
    *msip = 0;
    return;
}

//  Function: void cfg_timer(int timecmp, int timecmph)
void set_timer_tick(int add_time_l, int add_time_h){
    uint32 *time  = (uint32 *)LCM_TIME_ADDR; 
    uint32 *timecmp = (uint32 *)LCM_TIMECMP_ADDR;
    uint32 max_val = 0xffffffff;
    uint32 tl = time[0];
    uint32 th = time[1];
    if(max_val - tl > add_time_l){
        timecmp[0] = tl + add_time_l;
        timecmp[1] = th + add_time_h;
    }
    else{
        timecmp[0] = tl - max_val + add_time_l;
        timecmp[1] = th + add_time_h + 1;
    }
    return;
}

void _swi_print(void) {
    printf("!!=========software interrupt========\n");
    int x =1;
    return;
}

//  Function: void enable_ndma_intr(void)
void enable_ndma_intr(){
    uint32 mie = 0x8000;
    _set_bit_intr_mie(mie);
    return;
}

//  Function: void disable_ndma_intr(void)
void disable_ndma_intr(){
    uint32 mie = 0x8000;
    _clr_bit_intr_mie(mie);
    return;
}
