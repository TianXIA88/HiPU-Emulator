/*
 * RISC-V translation routines for the RVXI Base Integer Instruction Set.
 *
 * Copyright (c) 2016-2017 Sagar Karandikar, sagark@eecs.berkeley.edu
 * Copyright (c) 2018 Peer Adelt, peer.adelt@hni.uni-paderborn.de
 *                    Bastian Koppelmann, kbastian@mail.uni-paderborn.de
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

static bool trans_lui(DisasContext *ctx, arg_lui *a)
{
    if (a->rd != 0)
    {
        tcg_gen_movi_tl(cpu_gpr[a->rd], a->imm);
    }
    return true;
}

static bool trans_auipc(DisasContext *ctx, arg_auipc *a)
{
    if (a->rd != 0)
    {
        tcg_gen_movi_tl(cpu_gpr[a->rd], a->imm + ctx->base.pc_next);
    }
    return true;
}

static bool trans_jal(DisasContext *ctx, arg_jal *a)
{
    gen_jal(ctx, a->rd, a->imm);
    return true;
}

static bool trans_jalr(DisasContext *ctx, arg_jalr *a)
{
    /* no chaining with JALR */
    TCGLabel *misaligned = NULL;
    TCGv t0 = tcg_temp_new();

    gen_get_gpr(cpu_pc, a->rs1);
    tcg_gen_addi_tl(cpu_pc, cpu_pc, a->imm);
    tcg_gen_andi_tl(cpu_pc, cpu_pc, (target_ulong)-2);

    if (!has_ext(ctx, RVC))
    {
        misaligned = gen_new_label();
        tcg_gen_andi_tl(t0, cpu_pc, 0x2);
        tcg_gen_brcondi_tl(TCG_COND_NE, t0, 0x0, misaligned);
    }

    if (a->rd != 0)
    {
        tcg_gen_movi_tl(cpu_gpr[a->rd], ctx->pc_succ_insn);
    }
    tcg_gen_lookup_and_goto_ptr();

    if (misaligned)
    {
        gen_set_label(misaligned);
        gen_exception_inst_addr_mis(ctx);
    }
    ctx->base.is_jmp = DISAS_NORETURN;

    tcg_temp_free(t0);
    return true;
}

static bool gen_branch(DisasContext *ctx, arg_b *a, TCGCond cond)
{
    TCGLabel *l = gen_new_label();
    TCGv source1, source2;
    source1 = tcg_temp_new();
    source2 = tcg_temp_new();
    gen_get_gpr(source1, a->rs1);
    gen_get_gpr(source2, a->rs2);

    tcg_gen_brcond_tl(cond, source1, source2, l);
    gen_goto_tb(ctx, 1, ctx->pc_succ_insn);
    gen_set_label(l); /* branch taken */

    if (!has_ext(ctx, RVC) && ((ctx->base.pc_next + a->imm) & 0x3))
    {
        /* misaligned */
        gen_exception_inst_addr_mis(ctx);
    }
    else
    {
        gen_goto_tb(ctx, 0, ctx->base.pc_next + a->imm);
    }
    ctx->base.is_jmp = DISAS_NORETURN;

    tcg_temp_free(source1);
    tcg_temp_free(source2);

    return true;
}

static bool trans_beq(DisasContext *ctx, arg_beq *a)
{
    return gen_branch(ctx, a, TCG_COND_EQ);
}

static bool trans_bne(DisasContext *ctx, arg_bne *a)
{
    return gen_branch(ctx, a, TCG_COND_NE);
}

static bool trans_blt(DisasContext *ctx, arg_blt *a)
{
    return gen_branch(ctx, a, TCG_COND_LT);
}

static bool trans_bge(DisasContext *ctx, arg_bge *a)
{
    return gen_branch(ctx, a, TCG_COND_GE);
}

static bool trans_bltu(DisasContext *ctx, arg_bltu *a)
{
    return gen_branch(ctx, a, TCG_COND_LTU);
}

static bool trans_bgeu(DisasContext *ctx, arg_bgeu *a)
{
    return gen_branch(ctx, a, TCG_COND_GEU);
}

static bool gen_load(DisasContext *ctx, arg_lb *a, TCGMemOp memop)
{
    TCGv t0 = tcg_temp_new();
    TCGv t1 = tcg_temp_new();
    gen_get_gpr(t0, a->rs1);
    tcg_gen_addi_tl(t0, t0, a->imm);

    tcg_gen_qemu_ld_tl(t1, t0, ctx->mem_idx, memop);
    gen_set_gpr(a->rd, t1);
    tcg_temp_free(t0);
    tcg_temp_free(t1);
    return true;
}

static bool trans_lb(DisasContext *ctx, arg_lb *a)
{
    return gen_load(ctx, a, MO_SB);
}

static bool trans_lh(DisasContext *ctx, arg_lh *a)
{
    return gen_load(ctx, a, MO_TESW);
}

static bool trans_lw(DisasContext *ctx, arg_lw *a)
{
    return gen_load(ctx, a, MO_TESL);
}

static bool trans_lbu(DisasContext *ctx, arg_lbu *a)
{
    return gen_load(ctx, a, MO_UB);
}

static bool trans_lhu(DisasContext *ctx, arg_lhu *a)
{
    return gen_load(ctx, a, MO_TEUW);
}

static bool gen_store(DisasContext *ctx, arg_sb *a, TCGMemOp memop)
{
    TCGv t0 = tcg_temp_new();
    TCGv dat = tcg_temp_new();
    gen_get_gpr(t0, a->rs1);
    tcg_gen_addi_tl(t0, t0, a->imm);
    gen_get_gpr(dat, a->rs2);

    tcg_gen_qemu_st_tl(dat, t0, ctx->mem_idx, memop);
    tcg_temp_free(t0);
    tcg_temp_free(dat);
    return true;
}

static bool trans_sb(DisasContext *ctx, arg_sb *a)
{
    return gen_store(ctx, a, MO_SB);
}

static bool trans_sh(DisasContext *ctx, arg_sh *a)
{
    return gen_store(ctx, a, MO_TESW);
}

static bool trans_sw(DisasContext *ctx, arg_sw *a)
{
    return gen_store(ctx, a, MO_TESL);
}

#ifdef TARGET_RISCV64
static bool trans_lwu(DisasContext *ctx, arg_lwu *a)
{
    return gen_load(ctx, a, MO_TEUL);
}

static bool trans_ld(DisasContext *ctx, arg_ld *a)
{
    return gen_load(ctx, a, MO_TEQ);
}

static bool trans_sd(DisasContext *ctx, arg_sd *a)
{
    return gen_store(ctx, a, MO_TEQ);
}
#endif

static bool trans_addi(DisasContext *ctx, arg_addi *a)
{
    return gen_arith_imm(ctx, a, &tcg_gen_add_tl);
}

static void gen_slt(TCGv ret, TCGv s1, TCGv s2)
{
    tcg_gen_setcond_tl(TCG_COND_LT, ret, s1, s2);
}

static void gen_sltu(TCGv ret, TCGv s1, TCGv s2)
{
    tcg_gen_setcond_tl(TCG_COND_LTU, ret, s1, s2);
}

static bool trans_slti(DisasContext *ctx, arg_slti *a)
{
    return gen_arith_imm(ctx, a, &gen_slt);
}

static bool trans_sltiu(DisasContext *ctx, arg_sltiu *a)
{
    return gen_arith_imm(ctx, a, &gen_sltu);
}

static bool trans_xori(DisasContext *ctx, arg_xori *a)
{
    return gen_arith_imm(ctx, a, &tcg_gen_xor_tl);
}
static bool trans_ori(DisasContext *ctx, arg_ori *a)
{
    return gen_arith_imm(ctx, a, &tcg_gen_or_tl);
}
static bool trans_andi(DisasContext *ctx, arg_andi *a)
{
    return gen_arith_imm(ctx, a, &tcg_gen_and_tl);
}
static bool trans_slli(DisasContext *ctx, arg_slli *a)
{
    if (a->shamt >= TARGET_LONG_BITS)
    {
        return false;
    }

    if (a->rd != 0)
    {
        TCGv t = tcg_temp_new();
        gen_get_gpr(t, a->rs1);

        tcg_gen_shli_tl(t, t, a->shamt);

        gen_set_gpr(a->rd, t);
        tcg_temp_free(t);
    } /* NOP otherwise */
    return true;
}

static bool trans_srli(DisasContext *ctx, arg_srli *a)
{
    if (a->shamt >= TARGET_LONG_BITS)
    {
        return false;
    }

    if (a->rd != 0)
    {
        TCGv t = tcg_temp_new();
        gen_get_gpr(t, a->rs1);

        tcg_gen_shri_tl(t, t, a->shamt);
        gen_set_gpr(a->rd, t);
        tcg_temp_free(t);
    } /* NOP otherwise */
    return true;
}

static bool trans_srai(DisasContext *ctx, arg_srai *a)
{
    if (a->shamt >= TARGET_LONG_BITS)
    {
        return false;
    }

    if (a->rd != 0)
    {
        TCGv t = tcg_temp_new();
        gen_get_gpr(t, a->rs1);

        tcg_gen_sari_tl(t, t, a->shamt);
        gen_set_gpr(a->rd, t);
        tcg_temp_free(t);
    } /* NOP otherwise */
    return true;
}

static bool trans_add(DisasContext *ctx, arg_add *a)
{
    return gen_arith(ctx, a, &tcg_gen_add_tl);
}

static bool trans_sub(DisasContext *ctx, arg_sub *a)
{
    return gen_arith(ctx, a, &tcg_gen_sub_tl);
}

static bool trans_sll(DisasContext *ctx, arg_sll *a)
{
    return gen_shift(ctx, a, &tcg_gen_shl_tl);
}

static bool trans_slt(DisasContext *ctx, arg_slt *a)
{
    return gen_arith(ctx, a, &gen_slt);
}

static bool trans_sltu(DisasContext *ctx, arg_sltu *a)
{
    return gen_arith(ctx, a, &gen_sltu);
}

static bool trans_xor(DisasContext *ctx, arg_xor *a)
{
    return gen_arith(ctx, a, &tcg_gen_xor_tl);
}

static bool trans_srl(DisasContext *ctx, arg_srl *a)
{
    return gen_shift(ctx, a, &tcg_gen_shr_tl);
}

static bool trans_sra(DisasContext *ctx, arg_sra *a)
{
    return gen_shift(ctx, a, &tcg_gen_sar_tl);
}

static bool trans_or(DisasContext *ctx, arg_or *a)
{
    return gen_arith(ctx, a, &tcg_gen_or_tl);
}

static bool trans_and(DisasContext *ctx, arg_and *a)
{
    return gen_arith(ctx, a, &tcg_gen_and_tl);
}

#ifdef TARGET_RISCV64
static bool trans_addiw(DisasContext *ctx, arg_addiw *a)
{
    return gen_arith_imm(ctx, a, &gen_addw);
}

static bool trans_slliw(DisasContext *ctx, arg_slliw *a)
{
    TCGv source1;
    source1 = tcg_temp_new();
    gen_get_gpr(source1, a->rs1);

    tcg_gen_shli_tl(source1, source1, a->shamt);
    tcg_gen_ext32s_tl(source1, source1);
    gen_set_gpr(a->rd, source1);

    tcg_temp_free(source1);
    return true;
}

static bool trans_srliw(DisasContext *ctx, arg_srliw *a)
{
    TCGv t = tcg_temp_new();
    gen_get_gpr(t, a->rs1);
    tcg_gen_extract_tl(t, t, a->shamt, 32 - a->shamt);
    /* sign-extend for W instructions */
    tcg_gen_ext32s_tl(t, t);
    gen_set_gpr(a->rd, t);
    tcg_temp_free(t);
    return true;
}

static bool trans_sraiw(DisasContext *ctx, arg_sraiw *a)
{
    TCGv t = tcg_temp_new();
    gen_get_gpr(t, a->rs1);
    tcg_gen_sextract_tl(t, t, a->shamt, 32 - a->shamt);
    gen_set_gpr(a->rd, t);
    tcg_temp_free(t);
    return true;
}

static bool trans_addw(DisasContext *ctx, arg_addw *a)
{
    return gen_arith(ctx, a, &gen_addw);
}

static bool trans_subw(DisasContext *ctx, arg_subw *a)
{
    return gen_arith(ctx, a, &gen_subw);
}

static bool trans_sllw(DisasContext *ctx, arg_sllw *a)
{
    TCGv source1 = tcg_temp_new();
    TCGv source2 = tcg_temp_new();

    gen_get_gpr(source1, a->rs1);
    gen_get_gpr(source2, a->rs2);

    tcg_gen_andi_tl(source2, source2, 0x1F);
    tcg_gen_shl_tl(source1, source1, source2);

    tcg_gen_ext32s_tl(source1, source1);
    gen_set_gpr(a->rd, source1);
    tcg_temp_free(source1);
    tcg_temp_free(source2);
    return true;
}

static bool trans_srlw(DisasContext *ctx, arg_srlw *a)
{
    TCGv source1 = tcg_temp_new();
    TCGv source2 = tcg_temp_new();

    gen_get_gpr(source1, a->rs1);
    gen_get_gpr(source2, a->rs2);

    /* clear upper 32 */
    tcg_gen_ext32u_tl(source1, source1);
    tcg_gen_andi_tl(source2, source2, 0x1F);
    tcg_gen_shr_tl(source1, source1, source2);

    tcg_gen_ext32s_tl(source1, source1);
    gen_set_gpr(a->rd, source1);
    tcg_temp_free(source1);
    tcg_temp_free(source2);
    return true;
}

static bool trans_sraw(DisasContext *ctx, arg_sraw *a)
{
    TCGv source1 = tcg_temp_new();
    TCGv source2 = tcg_temp_new();

    gen_get_gpr(source1, a->rs1);
    gen_get_gpr(source2, a->rs2);

    /*
     * first, trick to get it to act like working on 32 bits (get rid of
     * upper 32, sign extend to fill space)
     */
    tcg_gen_ext32s_tl(source1, source1);
    tcg_gen_andi_tl(source2, source2, 0x1F);
    tcg_gen_sar_tl(source1, source1, source2);

    gen_set_gpr(a->rd, source1);
    tcg_temp_free(source1);
    tcg_temp_free(source2);

    return true;
}
#endif

static bool trans_fence(DisasContext *ctx, arg_fence *a)
{
    /* FENCE is a full memory barrier. */
    tcg_gen_mb(TCG_MO_ALL | TCG_BAR_SC);
    return true;
}

static bool trans_fence_i(DisasContext *ctx, arg_fence_i *a)
{
    /*
     * FENCE_I is a no-op in QEMU,
     * however we need to end the translation block
     */
    tcg_gen_movi_tl(cpu_pc, ctx->pc_succ_insn);
    tcg_gen_exit_tb(NULL, 0);
    ctx->base.is_jmp = DISAS_NORETURN;
    return true;
}

#define RISCV_OP_CSR_PRE                            \
    do                                              \
    {                                               \
        source1 = tcg_temp_new();                   \
        csr_store = tcg_temp_new();                 \
        dest = tcg_temp_new();                      \
        rs1_pass = tcg_temp_new();                  \
        gen_get_gpr(source1, a->rs1);               \
        tcg_gen_movi_tl(cpu_pc, ctx->base.pc_next); \
        tcg_gen_movi_tl(rs1_pass, a->rs1);          \
        tcg_gen_movi_tl(csr_store, a->csr);         \
        gen_io_start();                             \
    } while (0)

#define RISCV_OP_CSR_POST                           \
    do                                              \
    {                                               \
        gen_io_end();                               \
        gen_set_gpr(a->rd, dest);                   \
        tcg_gen_movi_tl(cpu_pc, ctx->pc_succ_insn); \
        tcg_gen_exit_tb(NULL, 0);                   \
        ctx->base.is_jmp = DISAS_NORETURN;          \
        tcg_temp_free(source1);                     \
        tcg_temp_free(csr_store);                   \
        tcg_temp_free(dest);                        \
        tcg_temp_free(rs1_pass);                    \
    } while (0)

static bool trans_csrrw(DisasContext *ctx, arg_csrrw *a)
{
    TCGv source1, csr_store, dest, rs1_pass;
    RISCV_OP_CSR_PRE;
    gen_helper_csrrw(dest, cpu_env, source1, csr_store);
    RISCV_OP_CSR_POST;
    return true;
}

static bool trans_csrrs(DisasContext *ctx, arg_csrrs *a)
{
    TCGv source1, csr_store, dest, rs1_pass;
    RISCV_OP_CSR_PRE;
    gen_helper_csrrs(dest, cpu_env, source1, csr_store, rs1_pass);
    RISCV_OP_CSR_POST;
    return true;
}

static bool trans_csrrc(DisasContext *ctx, arg_csrrc *a)
{
    TCGv source1, csr_store, dest, rs1_pass;
    RISCV_OP_CSR_PRE;
    gen_helper_csrrc(dest, cpu_env, source1, csr_store, rs1_pass);
    RISCV_OP_CSR_POST;
    return true;
}

static bool trans_csrrwi(DisasContext *ctx, arg_csrrwi *a)
{
    TCGv source1, csr_store, dest, rs1_pass;
    RISCV_OP_CSR_PRE;
    gen_helper_csrrw(dest, cpu_env, rs1_pass, csr_store);
    RISCV_OP_CSR_POST;
    return true;
}

static bool trans_csrrsi(DisasContext *ctx, arg_csrrsi *a)
{
    TCGv source1, csr_store, dest, rs1_pass;
    RISCV_OP_CSR_PRE;
    gen_helper_csrrs(dest, cpu_env, rs1_pass, csr_store, rs1_pass);
    RISCV_OP_CSR_POST;
    return true;
}

static bool trans_csrrci(DisasContext *ctx, arg_csrrci *a)
{
    TCGv source1, csr_store, dest, rs1_pass;
    RISCV_OP_CSR_PRE;
    gen_helper_csrrc(dest, cpu_env, rs1_pass, csr_store, rs1_pass);
    RISCV_OP_CSR_POST;
    return true;
}

#ifdef CONFIG_HIPUISA

#define TRANS_VA(x, y)                                   \
    static bool trans_##x(DisasContext *ctx, arg_##x *a) \
    {                                                    \
        hipu_vec_algo_e algo = y;                        \
        if (a->mod)                                      \
            return gen_vas(a, algo);                     \
        else                                             \
            return gen_va(a, algo);                      \
    }

#define TRANS_VAI(x, y)                                  \
    static bool trans_##x(DisasContext *ctx, arg_##x *a) \
    {                                                    \
        hipu_vec_algo_e algo = y;                        \
        if (a->mod == 001)                               \
            return gen_vas(a, algo);                     \
        else if (a->mod == 010)                          \
            return gen_vai(a, algo);                     \
        else                                             \
            return gen_va(a, algo);                      \
    }

#define TRANS_VP(x, y)                                   \
    static bool trans_##x(DisasContext *ctx, arg_##x *a) \
    {                                                    \
        hipu_p_op_e po = y;                              \
        return gen_vp(a, po);                            \
    }

#define TRANS_VPS(x, y)                                  \
    static bool trans_##x(DisasContext *ctx, arg_##x *a) \
    {                                                    \
        hipu_p_op_e po = y;                              \
        return gen_vps(a, po);                           \
    }

#define TRANS_VPP(x, y)                                  \
    static bool trans_##x(DisasContext *ctx, arg_##x *a) \
    {                                                    \
        hipu_p_op_e po = y;                              \
        return gen_vpp(a, po);                           \
    }

#define TRANS_VF(x, y)                                   \
    static bool trans_##x(DisasContext *ctx, arg_##x *a) \
    {                                                    \
        hipu_f_op_e po = y;                              \
        return gen_vf(a, po);                            \
    }

#define TRANS_VFS(x, y)                                  \
    static bool trans_##x(DisasContext *ctx, arg_##x *a) \
    {                                                    \
        hipu_f_op_e po = y;                              \
        return gen_vfs(a, po);                           \
    }

#define TRANS_MOP(x, y)                                  \
    static bool trans_##x(DisasContext *ctx, arg_##x *a) \
    {                                                    \
        hipu_m_op_e po = y;                              \
        return gen_mop(a, po, 0);                        \
    }

#define TRANS_MOPC(x, y)                                 \
    static bool trans_##x(DisasContext *ctx, arg_##x *a) \
    {                                                    \
        hipu_m_op_e po = y;                              \
        return gen_mop(a, po, 1);                        \
    }

#define TRANS_MOPS(x, y)                                 \
    static bool trans_##x(DisasContext *ctx, arg_##x *a) \
    {                                                    \
        hipu_m_op_e po = y;                              \
        return gen_mop_sh(a, po, 0);                     \
    }

#define TRANS_MOPCS(x, y)                                \
    static bool trans_##x(DisasContext *ctx, arg_##x *a) \
    {                                                    \
        hipu_m_op_e po = y;                              \
        return gen_mop_sh(a, po, 1);                     \
    }

TRANS_VA(vadd, VADD)
TRANS_VA(vsub, VSUB)
TRANS_VA(vmax, VMAX)
TRANS_VA(vmin, VMIN)
TRANS_VA(vmul, VMUL)
TRANS_VA(vand, VAND)
TRANS_VA(vor, VOR)
TRANS_VA(vxor, VXOR)
TRANS_VAI(vsll, VSLL)
TRANS_VAI(vsrl, VSRL)
TRANS_VAI(vsra, VSRA)

TRANS_VP(vpeq, VPEQ)
TRANS_VP(vpneq, VPNEQ)
TRANS_VP(vplt, VPLT)
TRANS_VP(vpge, VPGE)
TRANS_VPS(vpeqs, VPEQ)
TRANS_VPS(vpneqs, VPNEQ)
TRANS_VPS(vplts, VPLT)
TRANS_VPS(vpges, VPGE)
TRANS_VPP(vpand, VPAND)
TRANS_VPP(vpor, VPOR)
TRANS_VPP(vpxor, VPXOR)
TRANS_VPP(vpnot, VPNOT)

TRANS_VF(vfadd,VFADD)
TRANS_VF(vfsub,VFSUB)
TRANS_VF(vfmul,VFMUL)
TRANS_VFS(vfadds,VFADD)
TRANS_VFS(vfsubs,VFSUB)
TRANS_VFS(vfmuls,VFMUL)

TRANS_MOP(mmad, MMAD)
TRANS_MOP(mdmad, MDMAD)
TRANS_MOP(mvmad, MVMAD)
TRANS_MOP(mvbmad, MVBMAD)
TRANS_MOPC(mmadc, MMAD)
TRANS_MOPC(mdmadc, MDMAD)
TRANS_MOPC(mvmadc, MVMAD)
TRANS_MOPC(mvbmadc, MVBMAD)

TRANS_MOPS(mmads, MMAD)
TRANS_MOPS(mdmads, MDMAD)
TRANS_MOPS(mvmads, MVMAD)
TRANS_MOPS(mvbmads, MVBMAD)
TRANS_MOPCS(mmadcs, MMAD)
TRANS_MOPCS(mdmadcs, MDMAD)
TRANS_MOPCS(mvmadcs, MVMAD)
TRANS_MOPCS(mvbmadcs, MVBMAD)

static bool trans_vlb(DisasContext *ctx, arg_vlb *a)
{
    hipu_type_e typ = HIPU_BYTE;
    return gen_ldvr(a, typ);
}

static bool trans_vlh(DisasContext *ctx, arg_vlh *a)
{
    hipu_type_e typ = HIPU_HALF;
    return gen_ldvr(a, typ);
}

static bool trans_vlw(DisasContext *ctx, arg_vlw *a)
{
    hipu_type_e typ = HIPU_WORD;
    return gen_ldvr(a, typ);
}

static bool trans_vsb(DisasContext *ctx, arg_vsb *a)
{
    hipu_type_e typ = HIPU_BYTE;
    return gen_stvr(a, typ);
}

static bool trans_vsh(DisasContext *ctx, arg_vsh *a)
{
    hipu_type_e typ = HIPU_HALF;
    return gen_stvr(a, typ);
}

static bool trans_vsw(DisasContext *ctx, arg_vsw *a)
{
    hipu_type_e typ = HIPU_WORD;
    return gen_stvr(a, typ);
}

static bool trans_valh(DisasContext *ctx, arg_valh *a)
{
    TCGv_i32 vpr, vrd, vrs1, position;
    vpr = tcg_const_i32(a->vpr);
    vrd = tcg_const_i32(a->vrd);
    vrs1 = tcg_const_i32(a->vrs1);
    position = tcg_temp_new_i32();
    tcg_gen_mov_i32(position, cpu_gpr[a->vrs2]);
    gen_helper_hipu_valh(cpu_env, vpr, vrd, vrs1, position);
    return true;
}

static bool trans_valv(DisasContext *ctx, arg_valv *a)
{
    TCGv_i32 vpr, vrd, vrs1, position;
    vpr = tcg_const_i32(a->vpr);
    vrd = tcg_const_i32(a->vrd);
    vrs1 = tcg_const_i32(a->vrs1);
    position = tcg_temp_new_i32();
    tcg_gen_mov_i32(position, cpu_gpr[a->vrs2]);
    gen_helper_hipu_valv(cpu_env, vpr, vrd, vrs1, position);
    return true;
}

static bool trans_vpswap(DisasContext *ctx, arg_vpswap *a)
{
    TCGv_i32 vpr1, vpr2;
    vpr1 = tcg_const_i32(a->vrs1);
    vpr2 = tcg_const_i32(a->vrs2);
    gen_helper_hipu_vpswap(cpu_env, vpr1, vpr2);
    return true;
}

static bool trans_vlut(DisasContext *ctx, arg_vlut *a)
{
    TCGv_i32 vpr, vrd, vrs1, imm;
    vpr = tcg_const_i32(a->vpr);
    vrd = tcg_const_i32(a->vrd);
    vrs1 = tcg_const_i32(a->vrs1);
    imm = tcg_const_i32(a->imm);
    gen_helper_hipu_vlut(cpu_env, vpr, vrd, vrs1, imm);
    return true;
}


#endif