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
## FILE NAME  : hpu_api.s
## DEPARTMENT : Architecture
## AUTHOR     : wenzhe
## AUTHOR'S EMAIL : wenzhe@xjtu.edu.cn
## ---------------------------------------------------------------------------------------------------------------------
## Ver 1.0  2020--08--01 initial version.
## ---------------------------------------------------------------------------------------------------------------------

.include "krnl_inc/hpu_def.inc"

.section .text
.align 2

.globl get_hpuid
.globl init_enable_intr
.globl enter_wfi
.globl _wfi_handle
.globl cfg_timer
# .globl enable_timer_intr
# .globl disable_timer_intr
.globl _timer_handle
#.globl enable_ndma_intr
#.globl disable_ndma_intr
.globl _ndma_handle
.globl ndma_save_data
.globl ndma_load_data
.globl ndma_atomic_swap
.globl ndma_wait
.globl claim_sim_complete
.globl flush_l2_cache
.globl invalidate_cache

.weak ndma_callback

.equ l2c_csr_start,			0x7f0
.equ l2c_csr_finish,		0x7f1
.equ l2c_csr_addr_set_way,	0x7f2
.equ l2c_csr_mode,			0x7f3

.equ dc_csr_start,			0x7e0
.equ dc_csr_status,			0x7e1
.equ dc_csr_addr,			0x7e2
.equ dc_csr_mode,			0x7e3
.equ ic_csr_addr,  			0x7d2
.equ ic_csr_start,			0x7d0

.equ loop_times,			200     # may change bigger when flush_l2_cached is called too frequently

# Function: int get_hpuid(void)
get_hpuid:
    csrr a0, CSR_HPU_ID             # Read HPU_ID from CSR_HPU_ID
    ret

_swi_handle:
    csrw mscratch, sp
    la sp, __interrupt_stack_bottom__
    addi sp, sp, -116
    sw a0, 4(sp)
    sw a1, 8(sp)
    sw a2, 12(sp)
    sw a3, 16(sp)
    sw a4, 20(sp)
    sw a5, 24(sp)
    sw a6, 28(sp)
    sw a7, 32(sp)
    sw t0, 36(sp)
    sw t1, 40(sp)
    sw t2, 44(sp)
    sw t3, 48(sp)
    sw t4, 52(sp)
    sw t5, 56(sp)
    sw t6, 60(sp)
    sw ra, 64(sp)
    sw s0, 68(sp)
    sw s1, 72(sp)
    sw s2, 76(sp)
    sw s3, 80(sp)
    sw s4, 84(sp)
    sw s5, 88(sp)
    sw s6, 92(sp)
    sw s7, 96(sp)
    sw s8, 100(sp)
    sw s9, 104(sp)
    sw s10, 108(sp)
    sw s11, 112(sp)
    call swi_handler
    lw a0, 4(sp)               # restore context
    lw a1, 8(sp)
    lw a2, 12(sp)
    lw a3, 16(sp)
    lw a4, 20(sp)
    lw a5, 24(sp)
    lw a6, 28(sp)
    lw a7, 32(sp)
    lw t0, 36(sp)
    lw t1, 40(sp)
    lw t2, 44(sp)
    lw t3, 48(sp)
    lw t4, 52(sp)
    lw t5, 56(sp)
    lw t6, 60(sp)
    lw ra, 64(sp)
    lw s0, 68(sp)
    lw s1, 72(sp)
    lw s2, 76(sp)
    lw s3, 80(sp)
    lw s4, 84(sp)
    lw s5, 88(sp)
    lw s6, 92(sp)
    lw s7, 96(sp)
    lw s8, 100(sp)
    lw s9, 104(sp)
    lw s10, 108(sp)
    lw s11, 112(sp)
    addi sp, sp, 116
    csrr sp, mscratch
    mret

# Function: void intr::_timer_handle(void)
_timer_handle:
    csrw mscratch, sp
    la sp, __interrupt_stack_bottom__
    addi sp, sp, -116
    sw a0, 4(sp)
    sw a1, 8(sp)
    sw a2, 12(sp)
    sw a3, 16(sp)
    sw a4, 20(sp)
    sw a5, 24(sp)
    sw a6, 28(sp)
    sw a7, 32(sp)
    sw t0, 36(sp)
    sw t1, 40(sp)
    sw t2, 44(sp)
    sw t3, 48(sp)
    sw t4, 52(sp)
    sw t5, 56(sp)
    sw t6, 60(sp)
    sw ra, 64(sp)
    sw s0, 68(sp)
    sw s1, 72(sp)
    sw s2, 76(sp)
    sw s3, 80(sp)
    sw s4, 84(sp)
    sw s5, 88(sp)
    sw s6, 92(sp)
    sw s7, 96(sp)
    sw s8, 100(sp)
    sw s9, 104(sp)
    sw s10, 108(sp)
    sw s11, 112(sp)
    call timer_handler
    lw a0, 4(sp)               # restore context
    lw a1, 8(sp)
    lw a2, 12(sp)
    lw a3, 16(sp)
    lw a4, 20(sp)
    lw a5, 24(sp)
    lw a6, 28(sp)
    lw a7, 32(sp)
    lw t0, 36(sp)
    lw t1, 40(sp)
    lw t2, 44(sp)
    lw t3, 48(sp)
    lw t4, 52(sp)
    lw t5, 56(sp)
    lw t6, 60(sp)
    lw ra, 64(sp)
    lw s0, 68(sp)
    lw s1, 72(sp)
    lw s2, 76(sp)
    lw s3, 80(sp)
    lw s4, 84(sp)
    lw s5, 88(sp)
    lw s6, 92(sp)
    lw s7, 96(sp)
    lw s8, 100(sp)
    lw s9, 104(sp)
    lw s10, 108(sp)
    lw s11, 112(sp)
    addi sp, sp, 116
    csrr sp, mscratch
    mret

# Function: void enter_wfi(void)
enter_wfi:
    addi sp, sp, -4                 # Allocate the stack frame
    sw ra, 4(sp)                    # Save the return address on the stack frame
    # set external interrupt
    li t0, 0x800
    csrs mie, t0                    # csr_mie[ME] = 1'b1;
    # Enter WFI
    wfi                             # enter sleep mode.
    # exit this function
    lw ra, 4(sp)                    # Restore the return address on the stack frame
    addi sp, sp, 4                  # Deallocate the stack frame
    ret

# Function: void intr::_wfi_handle(void)
_wfi_handle:
    csrrw a0, mscratch, a0          # swap a0 <=> csr_mscratch
    sw t0, 4(a0)
    # clear external interrupt
    li t0, 0x800
    csrc mie, t0                    # csr_mie[ME] = 1'b0;
    csrrw a0, mscratch, a0          # swap csr_mscratch <=> a0
    mret

# Function: void enable_ndma_intr(void)
#enable_ndma_intr:
#    li t0, 0x8000
#    csrs mie, t0                    # csr_mie[NDMA] = 1'b1;
#    ret

# Function: void disable_ndma_intr(void)
#disable_ndma_intr:
#    li t0, 0x8000
#    csrc mie, t0                    # csr_mie[NDMA] = 1'b0;
#    ret

# Function: void intr::_ndma_handle(void) check and only clear one ndma channel index
# _ndma_handle:
#     csrrw s0, mscratch, s0          # swap s0 <=> csr_mscratch
#     sw a0, 4(s0)                    # Save context
#     sw a1, 8(s0)
#     sw a2, 12(s0)
#     sw a3, 16(s0)
#     sw a4, 20(s0)
#     sw a5, 24(s0)
#     sw a6, 28(s0)
#     sw a7, 32(s0)
#     sw t0, 36(s0)
#     sw t1, 40(s0)
#     sw t2, 44(s0)
#     sw t3, 48(s0)
#     sw t4, 52(s0)
#     sw t5, 56(s0)
#     sw t6, 60(s0)
#     sw ra, 64(s0)
#     csrr t0, CSR_NDMA_STATUS        # clear the insterrupt signal
#     li t2, 0x80000000
#     li t3, 0x01000000
#     li a0, 7
# 0:  and t1, t0, t2
#     bnez t1, 1f
#     slli t0, t0, 1
#     addi a0, a0, -1
#     bge a0, zero, 0b
# 1:  sll t6, t3, a0
#     addi t6, t6, 3
#     csrw CSR_NDMA_CTRL, t6          # clear the interrupt signal
#     call ndma_callback              # call back function in C code
#     lw a0, 4(s0)                    # Restore context
#     lw a1, 8(s0)
#     lw a2, 12(s0)
#     lw a3, 16(s0)
#     lw a4, 20(s0)
#     lw a5, 24(s0)
#     lw a6, 28(s0)
#     lw a7, 32(s0)
#     lw t0, 36(s0)
#     lw t1, 40(s0)
#     lw t2, 44(s0)
#     lw t3, 48(s0)
#     lw t4, 52(s0)
#     lw t5, 56(s0)
#     lw t6, 60(s0)
#     lw ra, 64(s0)
#     csrrw s0, mscratch, s0          # swap csr_mscratch <=> s0
#     mret

#Function: void intr::_ndma_handle(void) return ndma_status and clear in ndma_callback
_ndma_handle:
    csrw mscratch, sp
    la sp, __interrupt_stack_bottom__
    addi sp, sp, -116
    sw a0, 4(sp)
    sw a1, 8(sp)
    sw a2, 12(sp)
    sw a3, 16(sp)
    sw a4, 20(sp)
    sw a5, 24(sp)
    sw a6, 28(sp)
    sw a7, 32(sp)
    sw t0, 36(sp)
    sw t1, 40(sp)
    sw t2, 44(sp)
    sw t3, 48(sp)
    sw t4, 52(sp)
    sw t5, 56(sp)
    sw t6, 60(sp)
    sw ra, 64(sp)
    sw s0, 68(sp)
    sw s1, 72(sp)
    sw s2, 76(sp)
    sw s3, 80(sp)
    sw s4, 84(sp)
    sw s5, 88(sp)
    sw s6, 92(sp)
    sw s7, 96(sp)
    sw s8, 100(sp)
    sw s9, 104(sp)
    sw s10, 108(sp)
    sw s11, 112(sp)
    # csrci CSR_NDMA_STATUS, 1        # clear the insterrupt signal
    csrr a0, CSR_NDMA_STATUS        
    call ndma_callback              # call back function in C code
    lw a0, 4(sp)               # restore context
    lw a1, 8(sp)
    lw a2, 12(sp)
    lw a3, 16(sp)
    lw a4, 20(sp)
    lw a5, 24(sp)
    lw a6, 28(sp)
    lw a7, 32(sp)
    lw t0, 36(sp)
    lw t1, 40(sp)
    lw t2, 44(sp)
    lw t3, 48(sp)
    lw t4, 52(sp)
    lw t5, 56(sp)
    lw t6, 60(sp)
    lw ra, 64(sp)
    lw s0, 68(sp)
    lw s1, 72(sp)
    lw s2, 76(sp)
    lw s3, 80(sp)
    lw s4, 84(sp)
    lw s5, 88(sp)
    lw s6, 92(sp)
    lw s7, 96(sp)
    lw s8, 100(sp)
    lw s9, 104(sp)
    lw s10, 108(sp)
    lw s11, 112(sp)
    addi sp, sp, 116
    csrr sp, mscratch
    mret

# Function: void ndma_save_data(int loc_addr, int rmt_addr, int len, int node_x, int node_y)
ndma_save_data:
    csrw CSR_NDMA_LCADDR, a0        # set local address of ndma
    csrw CSR_NDMA_RTADDR, a1        # set remote address of ndma
    csrw CSR_NDMA_SIZE,   a2        # set transfer length of ndma
    csrw CSR_NDMA_DESTXY, a3        # set node.x location of ndma
    fence
    csrwi CSR_NDMA_CTRL, 0          # set read command to ndma
    ret
 
# Function: void ndma_load_data(int loc_addr, int rmt_addr, int len, int node_x, int node_y)
ndma_load_data:
    csrw CSR_NDMA_LCADDR, a0        # set local address of ndma
    csrw CSR_NDMA_RTADDR, a1        # set remote address of ndma
    csrw CSR_NDMA_SIZE,   a2        # set transfer length of ndma
    csrw CSR_NDMA_DESTXY, a3        # set node.x location of ndma
    fence
    csrwi CSR_NDMA_CTRL, 1          # set read command to ndma
    ret
 
# Function: void ndma_atomic_swap(int loc_addr, int rmt_addr, int node_x, int node_y)
ndma_atomic_swap:
    csrw CSR_NDMA_LCADDR, a0        # set local address of ndma
    csrw CSR_NDMA_RTADDR, a1        # set remote address of ndma
    csrwi CSR_NDMA_SIZE,   4        # set transfer length of ndma
    csrw CSR_NDMA_DESTXY, a2        # set node.x location of ndma
    fence
    csrwi CSR_NDMA_CTRL, 2          # set read command to ndma
    ret
 
# function: void ndma_wait(void)
ndma_wait:
0:  csrr t0, CSR_NDMA_STATUS        # read NDMA status
    lui	t1,0xff000
    and t0, t0, t1                  # only pay attention to bit[0]
    beqz t0, 0b                     # do-while(s0 != 0)
    li   t0, 0xFFFFFFFF
    csrw CSR_NDMA_CTRL, t0
    ret

# function: void claim_sim_complete(int fail)
claim_sim_complete:
    csrw mie, zero                  # close all interrupt source.
    csrw mstatus, zero              # clean mstatus
    beqz a0, mk                     # Judge whether it is sucess
    li t0, 0xfff
    csrw mstatus, t0                # mstatus = 0xfff
mk: csrwi mie, 1                    # Just for simulation, current simulation is finished.
0:  j 0b

# function: void flush_l2_cache()
flush_l2_cache:
    addi sp, sp, -28                # Allocate the stack frame
    sw ra, 4(sp)                    # Save the return address on the stack frame
    sw s1, 8(sp)                    # Save s1
    sw s2, 12(sp)                   # Save s2
    sw s3, 16(sp)                   # Save s3
    sw s4, 20(sp)                   # Save s4
    sw s5, 24(sp)                   # Save s5

    li s1, 0x00000000                # s1 configure set/way info
    li s2, 0x00000004                # s2 model set to write back line
    li s3, 0x10000                   # in total 16 lines, each 0x1000
    li s4, 0x1000                    # add every time

    li t0, loop_times               # patch: wait for other hardware modules idle
    li t1, 0                        # patch: wait for other hardware modules idle
delay_while:                        # patch: wait for other hardware modules idle
    addi t1, t1, 1                  # patch: wait for other hardware modules idle
    bge t0, t1, delay_while         # patch: wait for other hardware modules idle

    fence
    csrw  l2c_csr_mode, s2           # configure model = s2

__t_flushl2:
    csrw  l2c_csr_addr_set_way, s1   # configure set/way = s1
    fence
    csrwi l2c_csr_start, 0x1         # start
    fence

    # l2c_csr_start 一设置为1，l2c_csr_finish就自动被设置为0，直到l2cache flush/invalidate 结束时，
    # l2c_csr_finish 被设置为1, 直到下次再设置l2c_csr_start 为1时，开始启动l2cache flush/invalidate
    # 与此同时l2c_csr_finish 才会被再次设置为0 , invalidate 1 拍就完成，flush要多拍       
__t_flushl2_wait:
    csrr  s5, l2c_csr_finish         # wait for finish
    beqz  s5, __t_flushl2_wait       # if 0 == t3, jump to__t_flushl2_wait
    fence

    add   s1, s1, s4                 # s1 = s1 + s4, add 0x1000
    blt   s1, s3, __t_flushl2        # if s1 < s3, jump to __t_flusl2
    fence

    lw ra, 4(sp)                    # Restore the return address on the stack frame
    lw s1, 8(sp)                    # Restore s1
    lw s2, 12(sp)                   # Restore s2
    lw s3, 16(sp)                   # Restore s3
    lw s4, 20(sp)                   # Restore s4
    lw s5, 24(sp)                   # Restore s5
    addi sp, sp, 28                 # Deallocate the stack frame
    ret

invalidate_cache:
/*invalidate L2 cache*/
	fence
	# li t0,	0x8003a110
	# csrw l2c_csr_addr_set_way, t0
	li t0,  0x00000000
	csrw l2c_csr_mode,  t0
	# li t0,  0x80000001
	li t0,  0x1
	csrw l2c_csr_start, t0
	fence

# __t_flushl2_wait:
#   	csrr  t3, l2c_csr_finish
#   	beqz  t3, __t_flushl2_wait      # if 0 == t3, jump to__t_flushl2_wait 
#   	fence 

	/*invalidate L1 dcache*/
	csrw dc_csr_addr,  0x00000000
	csrw dc_csr_mode,  0x00000010
	csrw dc_csr_start, 0x00000001
	fence

	/*invalidate L1 icache*/
	# fence.i            # 这种fence.i的用法也可以
	csrw ic_csr_addr,  0x00000001			#0x7d2
	csrw ic_csr_start, 0x00000001			#0x7d0
	fence

	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
    
    ret


