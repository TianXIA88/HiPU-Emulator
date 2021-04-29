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
## FILE NAME  : init.s 
## DEPARTMENT : Architecture
## AUTHOR     : wenzhe
## AUTHOR'S EMAIL : wenzhe@xjtu.edu.cn
## ---------------------------------------------------------------------------------------------------------------------
## Ver 1.0  2020--08--01 initial version.
## ---------------------------------------------------------------------------------------------------------------------

.include "krnl_inc/hpu_def.inc"

.section .text
.align 2

.globl _start
.global _end
.global _eexit
.global _intrpt_vec_table

.weak kernel_entry
.weak exception_handle
.weak _timer_handle
.weak _wfi_handle
.weak _ndma_handle


# interrupt vector table
_start:
	jal invalidate_cache
	# li t0, 0x1234					# debug purpose
	# li t1, 0x02010008				# debug purpose
	# sw t0, (t1)						# debug purpose
    jal _init                       # power on / reset, Jump to kernel_entry function
_intrpt_vec_table:
    j exception_handle            # U-Mode soft intr: 
    mret                            # S-Mode soft intr:
    mret                            # H-Mode soft intr:
    mret                            # M-Mode soft intr:
    mret                            # U-Mode timer intr:
    mret                            # S-Mode timer intr:
    mret                            # H-Mode timer intr:
    j _timer_handle                 # M-Mode timer intr:
    mret                            # U-Mode external intr:
    mret                            # S-Mode external intr:
    mret                            # H-Mode external intr:
    j _wfi_handle                   # M-Mode external intr/WFI awake: Jump to WFI awake handler
    mret                            # Instr cache intr:
    mret                            # Data cache intr:
    mret                            # L2 cache intr:
    j _ndma_handle                  # NDMA intr: Jump to NDMA intr handler

_init:
	# li t0, 0x5555					# debug purpose
	# li t1, 0x02010008				# debug purpose
	# sw t0, (t1)						# debug purpose

	li t0, 0
	csrw mie, t0
	li t0, 0x84000009				# _set_intr_mtvec(_intrpt_vec_table + 1); // set vector table
	csrw mtvec, t0

    la sp, __stack_bottom__         # set the stacke top pointer

	# li t0, 0x6666 					# debug purpose
	# li t1, 0x02010008				# debug purpose
	# sw t0, (t1)						# debug purpose

	/* Load data section */

	la a0, __data_load_start__
	la a1, __data_start__
	la a2, __data_end__
	bgeu a1, a2, 2f
1:
	lw t0, (a0)
	sw t0, (a1)
	addi a0, a0, 4
	addi a1, a1, 4
	bltu a1, a2, 1b
2:

	/* Load sdata section */
	la a0, __sdata_load_start__
	la a1, __sdata_start__
	la a2, __sdata_end__
	bgeu a1, a2, 2f
1:
	lw t0, (a0)
	sw t0, (a1)
	addi a0, a0, 4
	addi a1, a1, 4
	bltu a1, a2, 1b
2:
	/* Zero bss section */
	la a0, __bss_start__
	la a1, __bss_end__
	bgeu a0, a1, 2f
1:
	sw zero, (a0)
	addi a0, a0, 4
	bltu a0, a1, 1b
2:

	/* Zero sbss section */
	la a0, __sbss_start__
	la a1, __sbss_end__
	bgeu a0, a1, 2f
1:
	sw zero, (a0)
	addi a0, a0, 4
	bltu a0, a1, 1b
2:
	# li t0, 0x7777 					# debug purpose
	# li t1, 0x02010008				# debug purpose
	# sw t0, (t1)						# debug purpose

    j kernel_entry

_end:
 	wfi
	#nop
_eexit:
 	la t1, _eexit
 	jr t1
