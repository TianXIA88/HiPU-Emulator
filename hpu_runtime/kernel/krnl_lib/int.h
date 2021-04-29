/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 14:35:14
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2020-12-31 10:28:19
 */
#include "hisdk_type.h"
#include "hihw.h"

void init_intr();
void enable_intr();
void disable_intr();
void enable_timer_intr();
void disable_timer_intr();
void enable_swi_intr();
void disable_swi_intr();
void gen_swi();
void clr_swi();
void set_timer_tick(int add_time_l, int add_time_h);
void _swi_print(void);
void enable_ndma_intr();
void disable_ndma_intr();