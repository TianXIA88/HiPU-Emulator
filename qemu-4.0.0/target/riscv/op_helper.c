/*
 * RISC-V Emulation Helpers for QEMU.
 *
 * Copyright (c) 2016-2017 Sagar Karandikar, sagark@eecs.berkeley.edu
 * Copyright (c) 2017-2018 SiFive, Inc.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2 or later, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "cpu.h"
#include "qemu/main-loop.h"
#include "exec/exec-all.h"
#include "exec/helper-proto.h"
#include "exec/cpu_ldst.h"

#include <stdio.h>
#include <libexplain/mmap.h>

/* Exceptions processing helpers */
void QEMU_NORETURN riscv_raise_exception(CPURISCVState *env,
                                         uint32_t exception, uintptr_t pc)
{
    CPUState *cs = CPU(riscv_env_get_cpu(env));
    qemu_log_mask(CPU_LOG_INT, "%s: %d\n", __func__, exception);
    cs->exception_index = exception;
    cpu_loop_exit_restore(cs, pc);
}

void helper_raise_exception(CPURISCVState *env, uint32_t exception)
{
    riscv_raise_exception(env, exception, 0);
}

target_ulong helper_csrrw(CPURISCVState *env, target_ulong src,
                          target_ulong csr)
{
    target_ulong val = 0;
    if (riscv_csrrw(env, csr, &val, src, -1) < 0)
    {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    return val;
}

target_ulong helper_csrrs(CPURISCVState *env, target_ulong src,
                          target_ulong csr, target_ulong rs1_pass)
{
    target_ulong val = 0;
    if (riscv_csrrw(env, csr, &val, -1, rs1_pass ? src : 0) < 0)
    {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    return val;
}

target_ulong helper_csrrc(CPURISCVState *env, target_ulong src,
                          target_ulong csr, target_ulong rs1_pass)
{
    target_ulong val = 0;
    if (riscv_csrrw(env, csr, &val, 0, rs1_pass ? src : 0) < 0)
    {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    return val;
}

#ifndef CONFIG_USER_ONLY

target_ulong helper_sret(CPURISCVState *env, target_ulong cpu_pc_deb)
{
    if (!(env->priv >= PRV_S))
    {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    target_ulong retpc = env->sepc;
    if (!riscv_has_ext(env, RVC) && (retpc & 0x3))
    {
        riscv_raise_exception(env, RISCV_EXCP_INST_ADDR_MIS, GETPC());
    }

    if (env->priv_ver >= PRIV_VERSION_1_10_0 &&
        get_field(env->mstatus, MSTATUS_TSR))
    {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    target_ulong mstatus = env->mstatus;
    target_ulong prev_priv = get_field(mstatus, MSTATUS_SPP);
    mstatus = set_field(mstatus,
                        env->priv_ver >= PRIV_VERSION_1_10_0 ? MSTATUS_SIE : MSTATUS_UIE << prev_priv,
                        get_field(mstatus, MSTATUS_SPIE));
    mstatus = set_field(mstatus, MSTATUS_SPIE, 0);
    mstatus = set_field(mstatus, MSTATUS_SPP, PRV_U);
    riscv_cpu_set_mode(env, prev_priv);
    env->mstatus = mstatus;

    return retpc;
}

target_ulong helper_mret(CPURISCVState *env, target_ulong cpu_pc_deb)
{
    if (!(env->priv >= PRV_M))
    {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    target_ulong retpc = env->mepc;
    if (!riscv_has_ext(env, RVC) && (retpc & 0x3))
    {
        riscv_raise_exception(env, RISCV_EXCP_INST_ADDR_MIS, GETPC());
    }

    target_ulong mstatus = env->mstatus;
    target_ulong prev_priv = get_field(mstatus, MSTATUS_MPP);
    mstatus = set_field(mstatus,
                        env->priv_ver >= PRIV_VERSION_1_10_0 ? MSTATUS_MIE : MSTATUS_UIE << prev_priv,
                        get_field(mstatus, MSTATUS_MPIE));
    mstatus = set_field(mstatus, MSTATUS_MPIE, 0);
    mstatus = set_field(mstatus, MSTATUS_MPP, PRV_U);
    riscv_cpu_set_mode(env, prev_priv);
    env->mstatus = mstatus;

    return retpc;
}

void helper_wfi(CPURISCVState *env)
{
    CPUState *cs = CPU(riscv_env_get_cpu(env));

    if (env->priv == PRV_S &&
        env->priv_ver >= PRIV_VERSION_1_10_0 &&
        get_field(env->mstatus, MSTATUS_TW))
    {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    else
    {
        cs->halted = 1;
        cs->exception_index = EXCP_HLT;
        cpu_loop_exit(cs);
    }
}

void helper_tlb_flush(CPURISCVState *env)
{
    RISCVCPU *cpu = riscv_env_get_cpu(env);
    CPUState *cs = CPU(cpu);
    if (env->priv == PRV_S &&
        env->priv_ver >= PRIV_VERSION_1_10_0 &&
        get_field(env->mstatus, MSTATUS_TVM))
    {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    else
    {
        tlb_flush(cs);
    }
}

#endif /* !CONFIG_USER_ONLY */

#ifdef CONFIG_HIPUISA

static int32_t helper_type_round_up(int32_t old_val, int32_t shift_num)
{
    int32_t shift_val = (old_val >> shift_num);
    int32_t mask = 0x1;
    int32_t fracpart, frac_half = 0x1;
    for (int i=0;i<shift_num-1;i++)
    {
        frac_half = frac_half << 1;
        mask = mask << 1;
        mask = mask + 1;
    }
    fracpart = old_val & mask;
    if(fracpart != 0)
        return (shift_val + 1);
    else return shift_val;
}

static int32_t helper_type_round_low(int32_t old_val, int32_t shift_num)
{
    int32_t shift_val = (old_val >> shift_num);
    return shift_val;
}

static int32_t helper_type_round_mid(int32_t old_val, int32_t shift_num)
{
    int32_t shift_val = (old_val >> shift_num);
    int32_t mask = 0x1;
    int32_t fracpart, frac_half = 0x1;
    for (int i=0;i<shift_num-1;i++)
    {
        frac_half = frac_half << 1;
        mask = mask << 1;
        mask = mask + 1;
    }
    fracpart = old_val & mask;
    if (fracpart > frac_half)
        return (shift_val + 1);
    if ((fracpart == frac_half) && (old_val > 0))
        return (shift_val + 1);
    else 
        return shift_val;
}

// static int32_t helper_type_toolchain(int32_t old_val, int32_t shift_num)
// {
//     int32_t shift_val = (old_val >> shift_num);
//     return shift_val;
// }
static int32_t helper_round_type(int type, int32_t old_val, int32_t shift_num)
{ 
    //DBG_PRINT("roundtype:%d\n", type);
    if (shift_num == 0)
        return old_val;
    switch(type)
    {
        case 0:return helper_type_round_up(old_val, shift_num); 
                break;  //near_zero
        case 1:return helper_type_round_mid(old_val, shift_num);
                break;  //far_zero
        case 2:return helper_type_round_low(old_val, shift_num);
                break;  //near_even
        default:return helper_type_round_mid(old_val, shift_num);
    }

}

static int32_t helper_add(int32_t a, int32_t b) { return a + b; }
static int32_t helper_sub(int32_t a, int32_t b) { return a - b; }
static int32_t helper_max(int32_t a, int32_t b) { return (a > b) ? a : b; }
static int32_t helper_min(int32_t a, int32_t b) { return (a < b) ? a : b; }
static int32_t helper_mul(int32_t a, int32_t b) { return a * b; }
static int32_t helper_and(int32_t a, int32_t b) { return a & b; }
static int32_t helper_or(int32_t a, int32_t b)  { return a | b; }
static int32_t helper_xor(int32_t a, int32_t b) { return a ^ b; }
static int32_t helper_sll(int32_t a, int32_t b) { return a << b;}
static int32_t helper_srl(int32_t a, int32_t b) { return (uint32_t)a >> b; }
static int32_t helper_sra(int32_t a, int32_t b) { return a >> b;}

int32_t (*va_func[HPU_VEC_ALGO_CNT])(int32_t, int32_t) = {
    helper_add,
    helper_sub,
    helper_max,
    helper_min,
    helper_mul,
    helper_and,
    helper_or,
    helper_xor,
    helper_sll,
    helper_srl,
    helper_sra,
};

static bool helper_peq(int32_t a, int32_t b) { return a == b; }
static bool helper_pneq(int32_t a, int32_t b) { return a != b; }
static bool helper_plt(int32_t a, int32_t b) { return a < b; }
static bool helper_pge(int32_t a, int32_t b) { return a >= b; }
static bool helper_pand(bool a, bool b) { return a || b; }
static bool helper_por(bool a, bool b) { return a && b; }
static bool helper_pxor(bool a, bool b) { return a ^ b; }
static bool helper_pnot(bool a, bool b) { return !a; }

bool (*vp_func[VPGE + 1])(int32_t, int32_t) = {
    helper_peq,
    helper_pneq,
    helper_plt,
    helper_pge};

bool (*vpp_func[VPNOT - VPGE])(bool, bool) = {
    helper_pand,
    helper_por,
    helper_pxor,
    helper_pnot};

static int32_t helper_vfadd(CPURISCVState *env, int32_t a, int32_t b)
{
    int32_t type, val, shift_num, prot_high, prot_low;
    val = a + b;
    type = env->round_type;
    shift_num = env->fadd_shift_num & 0x001f;
    prot_high = env->fadd_prot_high;
    prot_low = env->fadd_prot_low;
    val = helper_round_type(type, val, shift_num);
    if(val > prot_high)
        val = prot_high;
    if(val < prot_low)
        val = prot_low;
    return val;
}
static int32_t helper_vfsub(CPURISCVState *env,int32_t a, int32_t b)
{   
    int32_t type, val, shift_num, prot_high, prot_low;
    val = a - b;
    type = env->round_type;
    shift_num = env->fsub_shift_num & 0x001f;
    prot_high = env->fsub_prot_high;
    prot_low = env->fsub_prot_low;
    val = helper_round_type(type, val, shift_num);
    if(val > prot_high)
        val = prot_high;
    if(val < prot_low)
        val = prot_low;
    return val;
}

static int32_t helper_vfmul(CPURISCVState *env,int32_t a, int32_t b)
{   
    int32_t type, val, shift_num, prot_high, prot_low;
    val = a * b;
    type = env->round_type;
    shift_num = env->fmul_shift_num & 0x001f;
    prot_high = env->fmul_prot_high;
    prot_low = env->fmul_prot_low;
    val = helper_round_type(type, val, shift_num);
    if(val > prot_high)
        val = prot_high;
    if(val < prot_low)
        val = prot_low;
    return val;
}

int32_t (*vf_func[HPU_F_OP_CNT])(CPURISCVState *, int32_t, int32_t)={
    helper_vfadd,
    helper_vfsub,
    helper_vfmul
};

static HpuVPR vpr_ones = {{{1, 1, 1, 1, 1, 1, 1, 1},
                           {1, 1, 1, 1, 1, 1, 1, 1},
                           {1, 1, 1, 1, 1, 1, 1, 1},
                           {1, 1, 1, 1, 1, 1, 1, 1},
                           {1, 1, 1, 1, 1, 1, 1, 1},
                           {1, 1, 1, 1, 1, 1, 1, 1},
                           {1, 1, 1, 1, 1, 1, 1, 1},
                           {1, 1, 1, 1, 1, 1, 1, 1}}};

#ifdef HIPUISA_DEBUG_INFO
static void tool_dump_stack(uint8_t *stack, uint32_t offset, uint32_t len)
{
    int i;
    int a = offset;
    printf("\ndump_stack: offset=0x%x, len=%d(Bytes)\n", offset, len);
    for (i = 0; i < len; i = i + 8)
    {
        printf("%x: %x %x %x %x %x %x %x %x\n", a, stack[a], stack[a + 1], stack[a + 2], stack[a + 3],
               stack[a + 4], stack[a + 5], stack[a + 6], stack[a + 7]);
        a += 8;
    }
    printf("\n");
}

static void tool_dump_mm(HpuMM *mm, uint32_t addr, uint32_t len, int fmt)
{
    printf("\nDump mm: addr=0x%x len=%d\n", addr, len);
    int a = addr;
    for (int k = 0; k < len; k++)
    {
        a = addr + k;
        printf("\n0x%x: BLOCK %d\n", a, k);
        for (int i = 0; i < HPU_MX_HIGHT; i++)
        {
            for (int j = 0; j < HPU_MX_WIDTH; j++)
            {
                if (fmt == 1)
                    printf("%d(%x) ", (int8_t)mm->mblks[a].bytes[i][j], mm->mblks[a].bytes[i][j]);
                else
                    printf("%x ", mm->mblks[a].bytes[i][j]);
            }
            printf("\n");
        }
    }
}

static void tool_dump_vr(CPURISCVState *env, uint32_t vrn)
{
    printf("\nDump VR n=%d\n", vrn);
    HpuVR *r = &env->vregs[vrn];
    int32_t *w;
    for (int i = 0; i < HPU_MX_HIGHT; i++)
    {
        for (int j = 0; j < HPU_MX_WIDTH; j++)
        {
            w = (int32_t *)&r->words[i][j];
            printf("%d-%d: %d(0x%x)\n", i, j, w[0], w[0]);
        }
    }
}

static void tool_dump_vpr(CPURISCVState *env, uint32_t vprn)
{
    printf("\nDump VPR n=%d\n", vprn);
    HpuVPR *r = &env->vpregs[vprn];
    for (int i = 0; i < HPU_MX_HIGHT; i++)
    {
        for (int j = 0; j < HPU_MX_WIDTH; j++)
        {
            printf("%d ", r->flags[i][j]);
        }
        printf("\n");
    }
}
#endif

// size is in mm blocks
static bool illegal_mm(uint32_t addr, uint32_t size)
{
    int addr_e = addr + size;
    if (addr_e >= (HPU_MM_LENGTH * 2))
    {
        printf("ERROR: DMA OVERFLOW\n");
        return true;
    }
    if ((addr < HPU_MM_LENGTH) & (addr_e >= HPU_MM_LENGTH))
    {
        printf("ERROR: DMA CROSS BOUNDARY\n");
        return true;
    }
    return false;
}

static bool illegal_vr(uint32_t vr)
{
    if (vr >= HPU_VR_CNT)
    {
        printf("ERROR: VR INVALID INDEX\n");
        return true;
    }
    return false;
}

static int32_t uint8x2toint(uint8_t s1, uint8_t s2)
{
    uint16_t uu = (s2 << 8) + s1;
    int32_t ss = (int16_t)uu;
    return ss;
}

static int32_t uint8x4toint(uint8_t s1, uint8_t s2, uint8_t s3, uint8_t s4)
{
    uint32_t uu = (s4 << 24) + (s3 << 16) + (s2 << 8) + s1;
    int32_t ss = (uint32_t)uu;
    return ss;
}

typedef struct
{
    int8_t bytes[HPU_MX_HIGHT - 1][HPU_MX_WIDTH];
} mm_subblk;

static void int8_matrix_shift(int8_t *tmp, int8_t *a, hipu_shift_type_e sh)
{
    int8_t *subblk, *cpystrt;
    if (sh == SHUP)
    {
        subblk = &a[HPU_MX_WIDTH];
        memcpy(&tmp[0], subblk, sizeof(mm_subblk));
        for (int i = 0; i < HPU_MX_WIDTH; i++)
        {
            tmp[(HPU_MX_HIGHT - 1) * HPU_MX_WIDTH + i] = 0;
        }
    }
    else if (sh == SHDOWN)
    {
        subblk = &a[0];
        cpystrt = &tmp[HPU_MX_WIDTH];
        memcpy(&cpystrt[0], &subblk[0], sizeof(mm_subblk));
        for (int i = 0; i < HPU_MX_WIDTH; i++)
        {
            tmp[i] = 0;
        }
    }
    else
    {
        exit(-1);
    }
    return;
}

static int32_t int8_vec_dot_prod(int8_t *a, int8_t *b, int size)
{
    int32_t sum = 0;
    for (int k = 0; k < size; k++)
    {
        sum += (int32_t)(a[k] * b[k]);
    }
    return sum;
}

static void int8_matrix_product(int32_t *p, int8_t *a, int8_t *b, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            p[size * i + j] += int8_vec_dot_prod(&a[size * i], &b[size * j], size);
        }
    }
    return;
}

static void int8_matrix_dot_product(int32_t *p, int8_t *a, int8_t *b, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            p[size * i + j] += a[size * i + j] * b[size * i + j];
        }
    }
    return;
}

static void int8_vec_matrix_product(int32_t *p, int8_t *a, int8_t *b, int size)
{
    for(int k = 0; k < 8; k++)
        for(int i = 0;i < size/8; i++)
            for(int j = 0; j < size; j++)
            {
                p[j] += (int32_t)(a[8 * k + i] * b[64 * 8 * k + 8 * j + i]);
            }
    return;
}

static void int8_vec_bundle_matrix_product(int32_t *p, int8_t *a, int8_t *b, int size)
{
    return;
}

void (*mops[HPU_M_OP_CNT])(int32_t *, int8_t *, int8_t *, int) = {
    int8_matrix_product,
    int8_matrix_dot_product,
    int8_vec_matrix_product,
    int8_vec_bundle_matrix_product};

static int mop_size[4] = {8, 8, 64, 8};

#define ACC_ITERATE_OFFSET(i, size, stride) (uint32_t)((i / size) * stride + (i % size))

static bool *get_vpr(CPURISCVState *env, uint32_t vpr)
{
    int vpr_i;
    bool vpr_enable = vpr >> 1;
    if (vpr_enable)
    {
        vpr_i = vpr  & 0x1;
        return &env->vpregs[vpr_i].flags[0][0];
    }
    else
    {
        return &vpr_ones.flags[0][0];
    }
}

void helper_hipu_ldvr(CPURISCVState *env, uint32_t vrd, uint32_t mm_addr, uint32_t type)
{
    int index;
    uint8_t zeros[HPU_MM_BLOCK_LENGTH] = {0};
    uint8_t *buf1, *buf2, *buf3, *buf4;
    //DBG_PRINT("\n=========\nhipu_ldvr from 0x%x to VR%d in (%dBytes)\n", mm_addr, vrd, 1 << type)

    if (unlikely(illegal_mm(mm_addr, 1 << type)))
        exit(1);

    if (unlikely(illegal_vr(vrd)))
        exit(1);

    switch (type)
    {
    case HIPU_BYTE:
        buf1 = (uint8_t *)&env->mmab.mblks[mm_addr];
        buf2 = (uint8_t *)&zeros;
        buf3 = (uint8_t *)&zeros;
        buf4 = (uint8_t *)&zeros;
        break;
    case HIPU_HALF:
        buf1 = (uint8_t *)&env->mmab.mblks[mm_addr];
        buf2 = (uint8_t *)&env->mmab.mblks[mm_addr + 1];
        buf3 = (uint8_t *)&zeros;
        buf4 = (uint8_t *)&zeros;
        break;
    case HIPU_WORD:
    default:
        buf1 = (uint8_t *)&env->mmab.mblks[mm_addr];
        buf2 = (uint8_t *)&env->mmab.mblks[mm_addr + 1];
        buf3 = (uint8_t *)&env->mmab.mblks[mm_addr + 2];
        buf4 = (uint8_t *)&env->mmab.mblks[mm_addr + 3];
        break;
    }
    for (int i = 0; i < HPU_MX_HIGHT; i++)
    {
        for (int j = 0; j < HPU_MX_WIDTH; j++)
        {
            index = i * HPU_MX_HIGHT + j;
            switch (type)
            {
            case HIPU_BYTE:
                env->vregs[vrd].words[i][j] = (int8_t)(buf1[index]);
                break;
            case HIPU_HALF:
                env->vregs[vrd].words[i][j] = uint8x2toint(buf1[index], buf2[index]);
                break;
            case HIPU_WORD:
            default:
                env->vregs[vrd].words[i][j] = uint8x4toint(buf1[index], buf2[index], buf3[index], buf4[index]);
                break;
            }
        }
    }
}

void helper_hipu_stvr(CPURISCVState *env, uint32_t vrs, uint32_t mm_addr, uint32_t type)
{
    uint32_t index;
    index = 1 << type;
    // DBG_PRINT("\n=========\nhipu_stvr: VR%d -> 0x%x (at width %d Bytes)\n", vrs, mm_addr, index)
    for (int k = 0; k < index; k++)
    {
        for (int i = 0; i < HPU_MX_HIGHT; i++)
        {
            for (int j = 0; j < HPU_MX_WIDTH; j++)
            {
                
                env->mmab.mblks[mm_addr + k].bytes[i][j] = (uint8_t)((env->vregs[vrs].words[i][j] >> (k * 8)) & 0x000000ff);
            }
        }
    }
    // DUMP_VR(env, vrs);
    // DUMP_MM(&env->mmab, mm_addr, 1, 0);
}

void helper_hipu_va(CPURISCVState *env, uint32_t vpr, uint32_t vrd, uint32_t vrsa, uint32_t vrsb, uint32_t op)
{
    int *pd, *psa, *psb;
    bool *vpr_r;
    vpr_r = get_vpr(env, vpr);
    // DBG_PRINT("\n=========\nOP=%d: VR[%d] x VR[%d] => VR[%d]\n", op, vrsa, vrsb, vrd)
    pd = (int *)&env->vregs[vrd].words[0][0];
    psa = (int *)&env->vregs[vrsa].words[0][0];
    psb = (int *)&env->vregs[vrsb].words[0][0];
    for (int k = 0; k < HPU_MX_LENGTH; k++)
    {
        if (vpr_r[k])
            pd[k] = (*va_func[op])(psa[k], psb[k]);
        else
            continue;
    }
    // DUMP_VR(env, vrd);
}

void helper_hipu_vai(CPURISCVState *env, uint32_t vpr, uint32_t vrd, uint32_t vrsa, uint32_t imm, uint32_t op)
{
    int *pd, *psa;
    bool *vpr_r;
    vpr_r = get_vpr(env, vpr);
    // DBG_PRINT("\n=========\nOP=%d: VR[%d] x (imm=%d) => VR[%d]\n", op, vrsa, imm, vrd)
    pd = (int *)&env->vregs[vrd].words[0][0];
    psa = (int *)&env->vregs[vrsa].words[0][0];
    for (int k = 0; k < HPU_MX_LENGTH; k++)
    {
        if (vpr_r[k])
            pd[k] = (*va_func[op])(psa[k], imm);
        else
            continue;
    }
    //DUMP_VR(env, vrd);
}

void helper_hipu_valh(CPURISCVState *env, uint32_t vpr, uint32_t vrd, uint32_t vrs, uint32_t pos)
{
    HpuVR tmp_vr;
    uint32_t pos_col;
    HpuVPR *vpr_ptr;
    vpr_ptr = (HpuVPR *)get_vpr(env, vpr);
    // DBG_PRINT("\n=========\nSwitch colomns of VR[%d] => VR[%d] with order %x\n", vrs, vrd, pos)
    for (int j = 0; j < HPU_MX_WIDTH; j++)
    {
        pos_col = (pos >> (j * 3)) & 0x7;
        for (int i = 0; i < HPU_MX_HIGHT; i++)
        {
            if (vpr_ptr->flags[i][j])
                tmp_vr.words[i][j] = env->vregs[vrs].words[i][pos_col];
            else
                tmp_vr.words[i][j] = env->vregs[vrd].words[i][j];
        }
    }
    memcpy(&env->vregs[vrd].words[0][0], &tmp_vr.words[0][0], sizeof(tmp_vr));
    DUMP_VR(env, vrd);
}

void helper_hipu_valv(CPURISCVState *env, uint32_t vpr, uint32_t vrd, uint32_t vrs, uint32_t pos)
{
    HpuVR tmp_vr;
    uint32_t pos_lin;
    HpuVPR *vpr_ptr;
    vpr_ptr = (HpuVPR *)get_vpr(env, vpr);
    DBG_PRINT("\n=========\nSwitch lines of VR[%d] => VR[%d] with order %x\n", vrs, vrd, pos)
    for (int i = 0; i < HPU_MX_HIGHT; i++)
    {
        pos_lin = (pos >> (i * 3)) & 0x7;
        for (int j = 0; j < HPU_MX_WIDTH; j++)
        {
            if (vpr_ptr->flags[i][j]){
                tmp_vr.words[i][j] = env->vregs[vrs].words[pos_lin][j];}
            else
                tmp_vr.words[i][j] = env->vregs[vrd].words[i][j];
        }
    }
    memcpy(&env->vregs[vrd].words[0][0], &tmp_vr.words[0][0], sizeof(tmp_vr));
    //DUMP_VR(env, vrd);
}

void helper_hipu_vp(CPURISCVState *env, uint32_t vprd, uint32_t vrs1, uint32_t vrs2, uint32_t op)
{
    int32_t *vr1, *vr2;
    bool *vpd;
    DBG_PRINT("\n=========\nOP=%d: VR[%d] x VR[%d] => VPR[%d]\n", op, vrs1, vrs2, vprd)
    vr1 = &env->vregs[vrs1].words[0][0];
    vr2 = &env->vregs[vrs2].words[0][0];
    vpd = &env->vpregs[vprd].flags[0][0];
    for (int k = 0; k < HPU_MX_LENGTH; k++)
    {
        vpd[k] = (*vp_func[op])(vr1[k], vr2[k]);
    }
    DUMP_VPR(env, vprd);
    return;
};

void helper_hipu_vpi(CPURISCVState *env, uint32_t vprd, uint32_t vrs1, uint32_t imm, uint32_t op)
{
    int32_t *vr1;
    bool *vpd;
    DBG_PRINT("\n=========\nOP=%d: VR[%d] x [imm=%d] => VPR[%d]\n", op, vrs1, imm, vprd)
    vr1 = &env->vregs[vrs1].words[0][0];
    vpd = &env->vpregs[vprd].flags[0][0];
    for (int k = 0; k < HPU_MX_LENGTH; k++)
    {
        vpd[k] = (*vp_func[op])(vr1[k], imm);
    }
    DUMP_VPR(env, vprd);
    return;
};

void helper_hipu_vpp(CPURISCVState *env, uint32_t vprd, uint32_t vrs1, uint32_t vrs2, uint32_t op)
{
    bool *vp1, *vp2, *vpd;
    DBG_PRINT("\n=========\nOP=%d: VPR[%d] x VPR[%d] => VPR[%d]\n", op, vrs1, vrs2, vprd)
    vp1 = &env->vpregs[vrs1].flags[0][0];
    vp2 = &env->vpregs[vrs2].flags[0][0];
    vpd = &env->vpregs[vprd].flags[0][0];
    for (int k = 0; k < HPU_MX_LENGTH; k++)
    {
        vpd[k] = (*vpp_func[op - VPAND])(vp1[k], vp2[k]);
    }
    DUMP_VPR(env, vprd);
    return;
};

void helper_hipu_vpswap(CPURISCVState *env, uint32_t vpr1, uint32_t vpr2)
{
    HpuVPR *vp1, *vp2, tmp;
    DBG_PRINT("\n=========\nVPSWAP: VPR[%d] <-> VPR[%d]\n", vpr1, vpr2)
    vp1 = &env->vpregs[vpr1];
    vp2 = &env->vpregs[vpr2];
    int s = sizeof(tmp);
    memcpy(&tmp, vp2, s);
    memcpy(vp2, vp1, s);
    memcpy(vp1, &tmp, s);
    DUMP_VPR(env, vpr1);
    DUMP_VPR(env, vpr2);
    return;
}

void helper_hipu_vf(CPURISCVState *env, uint32_t vpr, uint32_t vrsa, uint32_t vrsb, uint32_t vrd, uint32_t op){
    int *pd, *psa,*psb;
    bool *vpr_r;
    vpr_r = get_vpr(env, vpr);
    // DBG_PRINT("\n=========\nOP=%d: (VR[%d] x VR[%d])(shift)=> VR[%d]\n", op, vrsa, vrsb, vrd)
    pd  = (int *)&env->vregs[vrd].words[0][0];
    psa = (int *)&env->vregs[vrsa].words[0][0];
    psb = (int *)&env->vregs[vrsb].words[0][0];
    for(int k=0; k < HPU_MX_LENGTH; k++){
        if(vpr_r[k]){
            pd[k] = (*vf_func[op])(env, psa[k], psb[k]);
            //R shift
        }
        else
            continue;
    }
    DUMP_VR(env, vrd);
}

void helper_hipu_vfi(CPURISCVState *env, uint32_t vpr, uint32_t vrsa, uint32_t imm, uint32_t vrd, uint32_t op){
    int *pd, *psa;
    bool *vpr_r;
    vpr_r = get_vpr(env, vpr);
    // DBG_PRINT("\n=========\nOP=%d: (VR[%d] x (imm=%d))(shift)=> VR[%d]\n", op, vrsa, imm, vrd)
    pd  = (int *)&env->vregs[vrd].words[0][0];
    psa = (int *)&env->vregs[vrsa].words[0][0];
    for(int k=0; k < HPU_MX_LENGTH; k++){
        if(vpr_r[k]){
            pd[k] = (*vf_func[op])(env, psa[k], imm);
            //R shift
        }
        else
            continue;
    }
    // DUMP_VR(env, vrd);
}

/* misc = | sh | ce | op | vpr | */
void helper_hipu_matrix_op(CPURISCVState *env, uint32_t vrd, uint32_t addr_a, uint32_t addr_b,  uint32_t vrs, uint32_t misc){
    uint32_t vpr = misc & 0xFF;
    uint32_t op  = (misc >> 8 ) & 0xFF;
    uint32_t ce  = (misc >> 16) & 0xFF;
    bool *vpr_r;
    vpr_r = get_vpr(env, vpr);
    // DBG_PRINT("\n=========\nOP=%d: MM[0x%x] x MM[0x%x] + VR[%d] => VR[%d] misc=0x%x\n", op, addr_a, addr_b, vrs, vrd, misc);
    int8_t *m1, *m2, *ms;
    int32_t *md, *mv, *pd;
    int32_t mac_size, mac_stride, mac_region_end, mac_region_start, mtx_cluster_start, mtx_cluster_end, mtx_cluster_num, mtx_cluster_stride;
    int32_t wt_blk_stride, wt_cluster_stride;
    HpuVR tmp_vr = {{{0}}};
    HpuMMBlock tmp_ms;
    int size = mop_size[op];
    int step = size * size / HPU_MM_BLOCK_LENGTH;
    int32_t ans;
    md = (int32_t *)&tmp_vr.words[0][0];
    ms = (int8_t  *)&tmp_ms.bytes[0][0];

    int mac_num, ofa, ofb=0;
    mtx_cluster_start = riscv_csr_read(env, 0xBD0);
    mtx_cluster_end = riscv_csr_read(env, 0xBD1);
    mtx_cluster_stride = riscv_csr_read(env, 0xBD8);
    wt_cluster_stride = riscv_csr_read(env, 0xBDA);
    // printf("cls_start = %x, cls_end = %x, cls_stride = %x \n", mtx_cluster_start,  mtx_cluster_end, mtx_cluster_stride);
    if(ce)
        mac_num = riscv_csr_read(env, 0xBD5);
    else
        mac_num = 0;
    
    mtx_cluster_num = riscv_csr_read(env, 0xBD6);
    if(ce)
        mac_size = riscv_csr_read(env, 0xBD4);

    else
        mac_size = 0;
    mac_stride = riscv_csr_read(env, 0xBD7);
    wt_blk_stride = riscv_csr_read(env, 0xBD9);
    mac_region_end = riscv_csr_read(env, 0xBD3);
    mac_region_start = riscv_csr_read(env, 0xBD2);
    
    // printf("mac_num = %x, mac_size = %x, mac_stride = %x, mac_region_start = %x, mac_region_end = %x, cls_num = %x\n", mac_num, mac_size, mac_stride, mac_region_start, mac_region_end, mtx_cluster_num);
    int start_to_end = mtx_cluster_end - mtx_cluster_start;
    int iter;

    for(int cls_iter=0; cls_iter < mtx_cluster_num + 1; cls_iter++){
        if(addr_a + 1> mac_region_end){
            addr_a = (addr_a - mac_region_end) % (mac_region_end - mac_region_start) + mac_region_start;
        }
        if(mtx_cluster_start + 1> mac_region_end){
            mtx_cluster_start = (mtx_cluster_start - mac_region_end) % (mac_region_end - mac_region_start) + mac_region_start;
            mtx_cluster_end = (mtx_cluster_end - mac_region_end) % (mac_region_end - mac_region_start) + mac_region_start;
        }
        for(int num_iter=0; num_iter < mac_num + 1; num_iter++){
            for(int size_iter=0; size_iter < mac_size + 1; size_iter++){
                iter = num_iter * (mac_size + 1)+ size_iter;
                if(ce){
                    ofa = ACC_ITERATE_OFFSET(iter, (mac_size + 1), mac_stride);
                    ofb = ACC_ITERATE_OFFSET(iter, (mac_size + 1)*step, wt_blk_stride);
                    // ofa += cls_iter * mtx_cluster_stride;
                    // ofb += cls_iter * wt_cluster_stride;
                }
                else{
                    ofa = ACC_ITERATE_OFFSET(num_iter * (mac_size + 1) + size_iter, 1, 2);
                    ofb = ACC_ITERATE_OFFSET(num_iter * (mac_size + 1) + size_iter, 1, 2);
                }

                ans = addr_a + ofa;

                m2 = (int8_t  *)&env->mmab.mblks[addr_b + ofb].bytes[0][0];

                if(ans < mtx_cluster_start){
                    m1 = (int8_t  *)&env->mmab.mblks[(ans + start_to_end)].bytes[0][0];
                    int8_matrix_shift(ms, m1, SHDOWN);
 
                }
                else if(ans + 1> mtx_cluster_end){
                    m1 = (int8_t  *)&env->mmab.mblks[(ans - start_to_end)].bytes[0][0];
                    int8_matrix_shift(ms, m1, SHUP);
                }
                else{
                   ms = (int8_t  *)&env->mmab.mblks[ans].bytes[0][0];
 
                }                
                (*mops[op])(md, ms, m2, size);
                ms = (int8_t  *)&tmp_ms.bytes[0][0];
            }
        }
        addr_a +=  mtx_cluster_stride;
        addr_b +=  wt_cluster_stride;
        mtx_cluster_end += mtx_cluster_stride;
        mtx_cluster_start += mtx_cluster_stride;
    }
    mv = (int32_t *)&env->vregs[vrs].words[0][0];
    pd = (int32_t *)&env->vregs[vrd].words[0][0];
    for(int k=0; k < HPU_MX_LENGTH; k++){
        if(vpr_r[k]){
            pd[k] = md[k] + mv[k];
            // DBG_PRINT("%d: %d = %d + %d\n", k, pd[k], md[k], mv[k]);
        }
        else
            continue;
    }
    // DUMP_VR(env, vrd);
    // DUMP_MM(&env->mmab, 0x0, 4, 0);
}

/* misc = | sh | ce | op | vpr | */
void helper_hipu_matrix_op_sh(CPURISCVState *env, uint32_t vrd, uint32_t addr_a, uint32_t addr_b, uint32_t vrs,  uint32_t misc){
    uint32_t vpr = misc & 0xFF;
    uint32_t op  = (misc >> 8 ) & 0xFF;
    uint32_t ce  = (misc >> 16) & 0xFF;
    uint32_t sh  = (misc >> 24) & 0xFF;

    bool *vpr_r;
    vpr_r = get_vpr(env, vpr);
    
    DBG_PRINT("\n=========\nOP=%d: MM[0x%x](SHIFT%d) x MM[0x%x] + VR[%d] => VR[%d] misc=0x%x\n", op, addr_a, sh, addr_b, vrs, vrd, misc)
    int8_t *m1, *m2, *ms;
    int32_t *md, *mv, *pd;
    int32_t mac_size, mac_stride, mac_region_end, mac_region_offset;
    HpuVR tmp_vr = {{{0}}};
    HpuMMBlock tmp_ms;
    int size = mop_size[op];
    int step = size * size / HPU_MM_BLOCK_LENGTH;
    int flag = 0;
    md = (int32_t *)&tmp_vr.words[0][0];
    ms = (int8_t  *)&tmp_ms.bytes[0][0];

    int mac_num, ofa, ofb=0;
    if(ce)
        mac_num = env->mac_blk_num;
    else
        mac_num = 0;

    if(ce)
        mac_size = env->mac_blk_size;
    else 
        mac_size = 0;

    mac_stride = env->mac_mma_blk_stride;
    // mac_region_end = riscv_csr_read(env, 0xBD3);
    // mac_region_offset = riscv_csr_read(env, 0xBD4);
    
    //DBG_PRINT("acc_num= %d, acc_size= %d, acc_stride= %d\n", acc_num, acc_size, acc_stride);
    for(int iter=0; iter < ((mac_num + 1) * (mac_size+ 1)); iter++){
        if(ce)
            ofa = ACC_ITERATE_OFFSET(iter, mac_size + 1, mac_stride);
        else
            ofa = ACC_ITERATE_OFFSET(iter, 1, 2);
        m1 = (int8_t  *)&env->mmab.mblks[addr_a + ofa].bytes[0][0];
        m2 = (int8_t  *)&env->mmab.mblks[addr_b + ofb].bytes[0][0];
        int8_matrix_shift(ms, m1, sh);
        (*mops[op])(md, ms, m2, size);
        ofb += step;
    }

    mv = (int32_t *)&env->vregs[vrs].words[0][0];
    pd = (int32_t *)&env->vregs[vrd].words[0][0];
    for(int k=0; k < HPU_MX_LENGTH; k++){
        if(vpr_r[k]){
            pd[k] = md[k] + mv[k];
            // DBG_PRINT("%d: %d = %d + %d\n", k, pd[k], md[k], mv[k]);
        }
        else
            continue;
    }
    
    DUMP_VR(env, vrd);
}

void helper_hipu_vlut(CPURISCVState *env, uint32_t vpr, uint32_t vrd, uint32_t vrs, uint32_t imm){
    int *pd, *ps;
    bool *vpr_r = get_vpr(env, vpr);;
    int index = 0;

    ps = (int32_t *)&env->vregs[vrs].words[0][0];
    pd = (int32_t *)&env->vregs[vrd].words[0][0];
    for (int i = 0; i < HPU_MX_LENGTH; i++)
    {
        if (vpr_r[i])
        {
            index = ps[i] & 0x000001FF; // 9-bit valid before activate
            pd[i] = env->vlut_table[index];
        }
    }

    // DUMP_VR(env, vrs);
    // DUMP_VR(env, vrd);
}
int print_num_mmab = 0;
void helper_print_mmab(CPURISCVState *env){
    uint32_t addr = env->fprint_addr;
    uint32_t len = env->fprint_len;
    unsigned char *mm_ptr; //= &env->mmab.mblks[addr];
    
    printf("Printing local MMAB at %x (%d Bytes)\n", addr, len);
    char output_file_name[40] = "../output/mmab";
    print_num_mmab++;
    char print_num_str[10] = {'0'+print_num_mmab};
    strcat(print_num_str, ".txt");
    strcat(output_file_name, print_num_str);
    FILE *fp = fopen(output_file_name, "w");
    for(int i=0; i<len/64; i++){ //for vsb
        mm_ptr = &env->mmab.mblks[addr + i];
        for(int j=63; j>=0; j--){
            fprintf(fp,"%02hhx", mm_ptr[j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return;
}
int print_num_mem = 0;
void helper_print_mem(CPURISCVState *env){
    uint32_t addr = env->fprint_addr;
    uint32_t len = env->fprint_len;
    char tmp;
    printf("Printing memory at %x (%d Bytes)\n", addr, len);
    char output_file_name[40] = "../output/mem";
    print_num_mem++;
    char print_num_str[10] = {'0'+print_num_mem};
    strcat(print_num_str, ".txt");
    strcat(output_file_name, print_num_str);
    FILE *fp = fopen(output_file_name, "w");

    for(int i=0; i<len/64; i++){ //for vsb
        for(int j=63; j>=0; j--){
            tmp = cpu_ldsb_data(env, addr+i*64+j);
            fprintf(fp, "%02hhx",tmp);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return;
}

void helper_sort(CPURISCVState *env, uint32_t src, uint32_t dst, uint32_t len){
    uint32_t mm_src, mm_dst;
    uint8_t *p1, *p2;
    mm_src = src;
    mm_dst = dst;

    if(dst == 0x2500)
        for(int l=0;l<len;l++)
        {    
            p1 = (uint8_t *)&env->mmab.mblks[mm_src+l*0x200];
            p2 = (uint8_t *)&env->mmab.mblks[mm_dst+l*0x200];
            for(int h=0; h<8; h++)
                for(int c=0; c<8; c++)
                    for(int in=0; in<8; in++)
                        for(int n=0; n<8; n++)
                            for(int w=0; w<8; w++)
                            {
                                p2[4096*h+512*in+64*n+8*w+c] = p1[4096*in+512*h+64*w+8*n+c];
                            }
        }
    
    if(dst == 0x2900)
        for(int l=0;l<len;l++)
        {    
            p1 = (uint8_t *)&env->mmab.mblks[mm_src+l*0x200];
            p2 = (uint8_t *)&env->mmab.mblks[mm_dst+l*0x200];
            for(int iic=0;iic<2;iic++)
                for(int ic=0;ic<8;ic++)
                    for(int c=0; c<8; c++)
                        for(int in=0; in<8; in++)
                            for(int n=0; n<8; n++)
                            {
                                p2[4096*iic+512*in+64*n+8*ic+c] = p1[1024*in+512*iic+64*ic+8*n+c];
                            }
        }
}

void helper_vpr(CPURISCVState *env, uint32_t dst, uint32_t len){
    uint32_t mm_addr;
    uint8_t *p;
    if( unlikely(illegal_mm(dst, len/HPU_MM_BLOCK_LENGTH)) )
        exit(1);
    mm_addr = dst;
    p = (uint8_t *)&env->mmab.mblks[mm_addr];
    for(int i=0; i<64*8; i++){
        
        p[i] = 0;
    }
    for(int k=0;k<8;k++)
        for(int i=64*k+8*k;i<8*(k+1)+64*k;i++)
            p[i]=1;
}


void helper_trace_insn(uint32_t x)
{
    printf("trace_insn %d\n", x);
}

void helper_hipu_insn(void)
{
    printf("hipu_insn\n");
}

void helper_check_pc(CPURISCVState *env)
{
    printf("\n check-pc=%x t0=%x\n\n", env->pc, env->gpr[5]);
}

#endif
