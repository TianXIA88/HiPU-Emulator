#ifndef HIPU200CLIB_EXTEND_INSTR_GEN_H__
#define HIPU200CLIB_EXTEND_INSTR_GEN_H__

#define MSHIFT(val, msk, shft)  ((val & msk) << shft)
#define _TOSTR(x)               #x
#define TOSTR(x)                _TOSTR(x)

#define vlb_code                (0)  //000000b
#define vlh_code                (1)  //000001b
#define vlw_code                (2)  //000010b
#define vsb_code                (4)  //000100b
#define vsh_code                (5)  //000101b
#define vsw_code                (6)  //000110b

#define vadd_code               (8)  //001000b  
#define vsub_code               (9)  //001001b  
#define vmax_code               (10) //001010b  
#define vmin_code               (11) //001011b  
#define vmul_code               (12) //001100b  
#define vand_code               (16) //010000b  
#define vor_code                (17) //010001b  
#define vxor_code               (18) //010010b  
#define vnor_code               (19) //010011b  
#define vsll_code               (20) //010100b  
#define vsrl_code               (21) //010101b  
#define vsra_code               (22) //010110b  
#define valh_code               (24) //011000b  
#define valv_code               (25) //011001b  
#define vlut_code               (28) //011100b  
#define vpeq_code               (32) //100000b  
#define vpneq_code              (33) //100001b  
#define vplt_code               (34) //100010b  
#define vpge_code               (35) //100011b  
#define vpand_code              (36) //100100b  
#define vpor_code               (37) //100101b  
#define vpxor_code              (38) //100110b  
#define vpnot_code              (39) //100111b  
#define vpswap_code             (40) //101000b  
#define vfadd_code              (48) //110000b  
#define vfsub_code              (49) //110001b  
#define vfmul_code              (50) //110010b  

#define mmad_code               (0)  //000b
#define mmac_code               (1)  //001b
#define mdmad_code              (2)  //010b
#define mdmac_code              (3)  //011b
#define mvmad_code              (4)  //100b
#define mvmac_code              (5)  //101b
#define mvbmad_code             (6)  //110b
#define mvbmac_code             (7)  //111b

#define GEN_VECTOR_INSTR1(imm9_0, bit29_28, SRs1, bit14_11, VRd, code) \
        MSHIFT(imm9_0,   0x300, (30 - 8)) |                            \
        MSHIFT(bit29_28, 0x3,    28)      |                            \
        MSHIFT(imm9_0,   0xE0,  (25 - 5)) |                            \
        MSHIFT(imm9_0,   0x1F,   20)      |                            \
        MSHIFT(SRs1,     0x1F,   15)      |                            \
        MSHIFT(bit14_11, 0xF,    11)      |                            \
        MSHIFT(VRd,      0xF,    7)       |                            \
        MSHIFT(code,     0x7F,   0)

#define GEN_VECTOR_INSTR2(imm9_0, bit29_28, VRs2, SRs1, bit14_11, code) \
        MSHIFT(imm9_0,   0x300, (30 - 8)) |                             \
        MSHIFT(bit29_28, 0x3,    28)      |                             \
        MSHIFT(imm9_0,   0xF0,  (24 - 4)) |                             \
        MSHIFT(VRs2,     0xF,    20)      |                             \
        MSHIFT(SRs1,     0x1F,   15)      |                             \
        MSHIFT(bit14_11, 0xF,    11)      |                             \
        MSHIFT(imm9_0,   0x0F,   7)       |                             \
        MSHIFT(code,     0x7F,   0)

#define GEN_VECTOR_INSTR3(VRP, bit29_28, bit27_25, VSRs2, VRs1, bit14_11, VRd, code)  \
        MSHIFT(VRP,      0x3,    30)      |                                           \
        MSHIFT(bit29_28, 0x3,    28)      |                                           \
        MSHIFT(bit27_25, 0x7,    25)      |                                           \
        MSHIFT(VSRs2,    0x1F,   20)      |                                           \
        MSHIFT(VRs1,     0x1F,   15)      |                                           \
        MSHIFT(bit14_11, 0x1F,   11)      |                                           \
        MSHIFT(VRd,      0x1F,   7)       |                                           \
        MSHIFT(code,     0x7F,   0)

#define GEN_MATRIX_INSTR(VRP, bit29, VRs, SRs2, SRs1, bit14_11, VRd, code)   \
        MSHIFT(VRP,      0x3,    30) |                                       \
        MSHIFT(bit29,    0x1,    29) |                                       \
        MSHIFT(VRs,      0xF,    25) |                                       \
        MSHIFT(SRs2,     0x1F,   20) |                                       \
        MSHIFT(SRs1,     0x1F,   15) |                                       \
        MSHIFT(bit14_11, 0x1F,   11) |                                       \
        MSHIFT(VRd,      0x1F,   7)  |                                       \
        MSHIFT(code,     0x7F,   0)

#define GEN_CSR_INSTR(csradd, rs1_imm, bit14_12, rd, code) \
        MSHIFT(csradd,   0xFFF,  20) |                     \
        MSHIFT(rs1_imm,  0x1F,   15) |                     \
        MSHIFT(bit14_12, 0x7,    12) |                     \
        MSHIFT(rd,       0x1F,   7)  |                     \
        MSHIFT(code,     0x7F,   0)

#define VECTOR_INSTR1(imm9_0, bit29_28, SRs1, bit14_11, VRd,  code)               asm(".long "TOSTR(GEN_VECTOR_INSTR1(imm9_0, bit29_28, SRs1, bit14_11, VRd,  code)))
#define VECTOR_INSTR2(imm9_0, bit29_28, VRs2, SRs1, bit14_11, code)               asm(".long "TOSTR(GEN_VECTOR_INSTR2(imm9_0, bit29_28, VRs2, SRs1, bit14_11, code)))
#define VECTOR_INSTR3(VRP, bit29_28, bit27_25, VSRs2,  VRs1, bit14_11, VRd, code) asm(".long "TOSTR(GEN_VECTOR_INSTR3(VRP, bit29_28, bit27_25, VSRs2,  VRs1, bit14_11, VRd, code)))
#define MATRIX_INSTR(VRP, bit29, VRs, SRs2, SRs1, bit14_11, VRd, code)            asm(".long "TOSTR(GEN_MATRIX_INSTR(VRP, bit29, VRs, SRs2, SRs1, bit14_11, VRd, code)))
#define CSR_INSTR(csradd, rs1_imm, bit14_12, rd, code)                            asm(".long "TOSTR(GEN_CSR_INSTR(csradd, rs1_imm, bit14_12, rd, code)))

#define vlb(VRd, SRs, imm9_0)                   VECTOR_INSTR1(imm9_0, 0x0, SRs, 0x0, VRd, 0x2B)
#define vlh(VRd, SRs, imm9_0)                   VECTOR_INSTR1(imm9_0, 0x1, SRs, 0x0, VRd, 0x2B)
#define vlw(VRd, SRs, imm9_0)                   VECTOR_INSTR1(imm9_0, 0x2, SRs, 0x0, VRd, 0x2B)
#define vsb(VRs, SRd, imm9_0)                   VECTOR_INSTR2(imm9_0, 0x0, VRs, SRd, 0x1, 0x2B)
#define vsh(VRs, SRd, imm9_0)                   VECTOR_INSTR2(imm9_0, 0x1, VRs, SRd, 0x1, 0x2B)
#define vsw(VRs, SRd, imm9_0)                   VECTOR_INSTR2(imm9_0, 0x2, VRs, SRd, 0x1, 0x2B)

#define vadd_vv(VPRs, VRd, VRs1, VRs2)          VECTOR_INSTR3(VPRs, 0x0, 0x0, VRs2, VRs1, 0x2, VRd, 0x2B)
#define vadd_vs(VPRs, VRd, VRs1, SRs2)          VECTOR_INSTR3(VPRs, 0x0, 0x1, SRs2, VRs1, 0x2, VRd, 0x2B)
#define vsub_vv(VPRs, VRd, VRs1, VRs2)          VECTOR_INSTR3(VPRs, 0x1, 0x0, VRs2, VRs1, 0x2, VRd, 0x2B)
#define vsub_vs(VPRs, VRd, VRs1, SRs2)          VECTOR_INSTR3(VPRs, 0x1, 0x1, SRs2, VRs1, 0x2, VRd, 0x2B)
#define vmax_vv(VPRs, VRd, VRs1, VRs2)          VECTOR_INSTR3(VPRs, 0x2, 0x0, VRs2, VRs1, 0x2, VRd, 0x2B)
#define vmax_vs(VPRs, VRd, VRs1, SRs2)          VECTOR_INSTR3(VPRs, 0x2, 0x1, SRs2, VRs1, 0x2, VRd, 0x2B)
#define vmin_vv(VPRs, VRd, VRs1, VRs2)          VECTOR_INSTR3(VPRs, 0x3, 0x0, VRs2, VRs1, 0x2, VRd, 0x2B)
#define vmin_vs(VPRs, VRd, VRs1, SRs2)          VECTOR_INSTR3(VPRs, 0x3, 0x1, SRs2, VRs1, 0x2, VRd, 0x2B)
#define vmul_vv(VPRs, VRd, VRs1, VRs2)          VECTOR_INSTR3(VPRs, 0x0, 0x0, VRs2, VRs1, 0x3, VRd, 0x2B)
#define vmul_vs(VPRs, VRd, VRs1, SRs2)          VECTOR_INSTR3(VPRs, 0x0, 0x1, SRs2, VRs1, 0x3, VRd, 0x2B)
#define vand_vv(VPRs, VRd, VRs1, VRs2)          VECTOR_INSTR3(VPRs, 0x0, 0x0, VRs2, VRs1, 0x4, VRd, 0x2B)
#define vand_vs(VPRs, VRd, VRs1, SRs2)          VECTOR_INSTR3(VPRs, 0x0, 0x1, SRs2, VRs1, 0x4, VRd, 0x2B)
#define vor_vv(VPRs, VRd, VRs1, VRs2)           VECTOR_INSTR3(VPRs, 0x1, 0x0, VRs2, VRs1, 0x4, VRd, 0x2B)
#define vor_vs(VPRs, VRd, VRs1, SRs2)           VECTOR_INSTR3(VPRs, 0x1, 0x1, SRs2, VRs1, 0x4, VRd, 0x2B)
#define vxor_vv(VPRs, VRd, VRs1, VRs2)          VECTOR_INSTR3(VPRs, 0x2, 0x0, VRs2, VRs1, 0x4, VRd, 0x2B)
#define vxor_vs(VPRs, VRd, VRs1, SRs2)          VECTOR_INSTR3(VPRs, 0x2, 0x1, SRs2, VRs1, 0x4, VRd, 0x2B)
#define vnxor_vv(VPRs, VRd, VRs1, VRs2)         VECTOR_INSTR3(VPRs, 0x3, 0x0, VRs2, VRs1, 0x4, VRd, 0x2B)
#define vnxor_vs(VPRs, VRd, VRs1, SRs2)         VECTOR_INSTR3(VPRs, 0x3, 0x1, SRs2, VRs1, 0x4, VRd, 0x2B)
#define vsll_vv(VPRs, VRd, VRs1, VRs2)          VECTOR_INSTR3(VPRs, 0x0, 0x0, VRs2, VRs1, 0x5, VRd, 0x2B)

#define vsll_vi(VPRs, VRd, VRs1, imm4_0)        VECTOR_INSTR3(VPRs, 0x0, 0x2, imm4_0, VRs1, 0x5, VRd, 0x2B)
#define vsll_vs(VPRs, VRd, VRs1, SRs2)          VECTOR_INSTR3(VPRs, 0x0, 0x1, SRs2,   VRs1, 0x5, VRd, 0x2B)
#define vsrl_vv(VPRs, VRd, VRs1, VRs2)          VECTOR_INSTR3(VPRs, 0x1, 0x0, VRs2,   VRs1, 0x5, VRd, 0x2B)
#define vsrl_vi(VPRs, VRd, VRs1, imm4_0)        VECTOR_INSTR3(VPRs, 0x1, 0x2, imm4_0, VRs1, 0x5, VRd, 0x2B)
#define vsrl_vs(VPRs, VRd, VRs1, SRs2)          VECTOR_INSTR3(VPRs, 0x1, 0x1, SRs2,   VRs1, 0x5, VRd, 0x2B)
#define vsra_vv(VPRs, VRd, VRs1, VRs2)          VECTOR_INSTR3(VPRs, 0x3, 0x0, VRs2,   VRs1, 0x5, VRd, 0x2B)
#define vsra_vi(VPRs, VRd, VRs1, imm4_0)        VECTOR_INSTR3(VPRs, 0x3, 0x2, imm4_0, VRs1, 0x5, VRd, 0x2B)
#define vsra_vs(VPRs, VRd, VRs1, SRs2)          VECTOR_INSTR3(VPRs, 0x3, 0x1, SRs2,   VRs1, 0x5, VRd, 0x2B)
#define valh(VPRs, VRd, VRs1, SRs2)             VECTOR_INSTR3(VPRs, 0x0, 0x1, SRs2,   VRs1, 0x6, VRd, 0x2B)
#define valv(VPRs, VRd, VRs1, SRs2)             VECTOR_INSTR3(VPRs, 0x1, 0x1, SRs2,   VRs1, 0x6, VRd, 0x2B)
#define vlut(VPRs, VRd, VRs1, SRs2)             VECTOR_INSTR3(VPRs, 0x0, 0x0, SRs2,   VRs1, 0x7, VRd, 0x2B)
#define vpeq_vv(VPRd, VRs1, VRs2)               VECTOR_INSTR3(0x0,  0x0, 0x0, VRs2,   VRs1, 0x8,VPRd, 0x2B)
#define vpeq_vs(VPRd, VRs1, SRs2)               VECTOR_INSTR3(0x0,  0x0, 0x1, SRs2,   VRs1, 0x8,VPRd, 0x2B)
#define vpneq_vv(VPRd, VRs1, VRs2)              VECTOR_INSTR3(0x0,  0x1, 0x0, VRs2,   VRs1, 0x8,VPRd, 0x2B)
#define vpneq_vs(VPRd, VRs1, SRs2)              VECTOR_INSTR3(0x0,  0x1, 0x1, SRs2,   VRs1, 0x8,VPRd, 0x2B)
#define vplt_vv(VPRd, VRs1, VRs2)               VECTOR_INSTR3(0x0,  0x2, 0x0, VRs2,   VRs1, 0x8,VPRd, 0x2B)
#define vplt_vs(VPRd, VRs1, SRs2)               VECTOR_INSTR3(0x0,  0x2, 0x1, SRs2,   VRs1, 0x8,VPRd, 0x2B)
#define vpge_vv(VPRd, VRs1, VRs2)               VECTOR_INSTR3(0x0,  0x3, 0x0, VRs2,   VRs1, 0x8,VPRd, 0x2B)
#define vpge_vs(VPRd, VRs1, SRs2)               VECTOR_INSTR3(0x0,  0x3, 0x1, SRs2,   VRs1, 0x8,VPRd, 0x2B)
#define vpand(VPRd, VPRs1, VPRs2)               VECTOR_INSTR3(0x0,  0x0, 0x0, VPRs2, VPRs1, 0x9,VPRd, 0x2B)
#define vpor(VPRd, VPRs1, VPRs2)                VECTOR_INSTR3(0x0,  0x1, 0x0, VPRs2, VPRs1, 0x9,VPRd, 0x2B)
#define vpxor(VPRd, VPRs1, VPRs2)               VECTOR_INSTR3(0x0,  0x2, 0x0, VPRs2, VPRs1, 0x9,VPRd, 0x2B)
#define vpnot(VPRd, VPRs1, VPRs2)               VECTOR_INSTR3(0x0,  0x3, 0x0,   0x0, VPRs1, 0x9,VPRd, 0x2B)
#define vpswap(VPRs1, VPRs2)                    VECTOR_INSTR3(0x0,  0x0, 0x0, VPRs2, VPRs1, 0xA, 0x0, 0x2B)
#define vfadd_vv(VPRs, VRd, VRs1, VRs2)         VECTOR_INSTR3(VPRs, 0x0, 0x0, VRs2,  VRs1,  0xC, VRd, 0x2B)
#define vfadd_vs(VPRs, VRd, VRs1, SRs2)         VECTOR_INSTR3(VPRs, 0x0, 0x1, SRs2,  VRs1,  0xC, VRd, 0x2B)
#define vfsub_vv(VPRs, VRd, VRs1, VRs2)         VECTOR_INSTR3(VPRs, 0x1, 0x0, VRs2,  VRs1,  0xC, VRd, 0x2B)
#define vfsub_vs(VPRs, VRd, VRs1, SRs2)         VECTOR_INSTR3(VPRs, 0x1, 0x1, SRs2,  VRs1,  0xC, VRd, 0x2B)
#define vfmul_vv(VPRs, VRd, VRs1, VRs2)         VECTOR_INSTR3(VPRs, 0x2, 0x0, VRs2,  VRs1,  0xC, VRd, 0x2B)
#define vfmul_vs(VPRs, VRd, VRs1, SRs2)         VECTOR_INSTR3(VPRs, 0x2, 0x1, SRs2,  VRs1,  0xC, VRd, 0x2B)

//m: matrix operation([8, 8] * [8, 8])
// -- matrix multiply add
#define mmad(VPRs, VRd, SRs1, SRs2, VRs)        MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 0,  VRd, 11)
#define mmad_au(VPRs, VRd, SRs1, SRs2, VRs)     MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 8,  VRd, 11)
#define mmad_ad(VPRs, VRd, SRs1, SRs2, VRs)     MATRIX_INSTR(VPRs, 1, VRs, SRs2, SRs1, 8,  VRd, 11)
// -- matrix multiply accumulate
#define mmac(VPRs, VRd, SRs1, SRs2, VRs)        MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 1,  VRd, 11)
#define mmac_au(VPRs, VRd, SRs1, SRs2, VRs)     MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 9,  VRd, 11)
#define mmac_ad(VPRs, VRd, SRs1, SRs2, VRs)     MATRIX_INSTR(VPRs, 1, VRs, SRs2, SRs1, 9,  VRd, 11)
//md: matrix dot (对应位置乘法，乘完不累加，结果还是向量或矩阵，correspoding item multiply) operation([8,8] md [8,8]) or ([1,64] * [1, 64])
// -- [m]atrix [d]ot [m]ultiply [ad]d:
// -- VRd = VRs + [SRs1] md [SRs2]
#define mdmad(VPRs, VRd, SRs1, SRs2, VRs)       MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 2,  VRd, 11)
#define mdmad_au(VPRs, VRd, SRs1, SRs2, VRs)    MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 10, VRd, 11)
#define mdmad_ad(VPRs, VRd, SRs1, SRs2, VRs)    MATRIX_INSTR(VPRs, 1, VRs, SRs2, SRs1, 10, VRd, 11)
// -- [m]atrix [d]ot [m]ultiply [ac]cumulate
// -- VRd = VRs + \sigma_{i=0}^{N}{[SRs1_i] md [SRs2_i]}, accumulate: 根据blk_num, cluster_num 对于k_h, k_w 累加
#define mdmac(VPRs, VRd, SRs1, SRs2, VRs)       MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 3,  VRd, 11)
#define mdmac_au(VPRs, VRd, SRs1, SRs2, VRs)    MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 11, VRd, 11)
#define mdmac_ad(VPRs, VRd, SRs1, SRs2, VRs)    MATRIX_INSTR(VPRs, 1, VRs, SRs2, SRs1, 11, VRd, 11)
//mv: vector matrix operation([1, 64] * [64, 64])
// -- vector matrix multiply add
#define mvmad(VPRs, VRd, SRs1, SRs2, VRs)       MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 4,  VRd, 11)
#define mvmad_au(VPRs, VRd, SRs1, SRs2, VRs)    MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 12, VRd, 11)
#define mvmad_ad(VPRs, VRd, SRs1, SRs2, VRs)    MATRIX_INSTR(VPRs, 1, VRs, SRs2, SRs1, 12, VRd, 11)
// -- vector matrix multiply accumulate
#define mvmac(VPRs, VRd, SRs1, SRs2, VRs)       MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 5,  VRd, 11)
#define mvmac_au(VPRs, VRd, SRs1, SRs2, VRs)    MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 13, VRd, 11)
#define mvmac_ad(VPRs, VRd, SRs1, SRs2, VRs)    MATRIX_INSTR(VPRs, 1, VRs, SRs2, SRs1, 13, VRd, 11)
//mvb: vector matrix in batch mode operation(([1, 8] * [8, 8]) x 8)
// -- vector matrix multiply add in batch mode
#define mvbmad(VPRs, VRd, SRs1, SRs2, VRs)      MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 6,  VRd, 11)
#define mvbmad_au(VPRs, VRd, SRs1, SRs2, VRs)   MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 14, VRd, 11)
#define mvbmad_ad(VPRs, VRd, SRs1, SRs2, VRs)   MATRIX_INSTR(VPRs, 1, VRs, SRs2, SRs1, 14, VRd, 11)
// -- vector matrix multiply accumulate in batch mode
#define mvbmac(VPRs, VRd, SRs1, SRs2, VRs)      MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 7,  VRd, 11)
#define mvbmac_au(VPRs, VRd, SRs1, SRs2, VRs)   MATRIX_INSTR(VPRs, 0, VRs, SRs2, SRs1, 15, VRd, 11)
#define mvbmac_ad(VPRs, VRd, SRs1, SRs2, VRs)   MATRIX_INSTR(VPRs, 1, VRs, SRs2, SRs1, 15, VRd, 11)

// #define csrrw(csradd, rd, rs1)                  CSR_INSTR(csradd, rs1,    0x1, rd, 0x73)        
// #define csrrs(csradd, rd, rs1)                  CSR_INSTR(csradd, rs1,    0x2, rd, 0x73)
// #define csrrc(csradd, rd, rs1)                  CSR_INSTR(csradd, rs1,    0x3, rd, 0x73)
// #define csrrwi(csradd, rd, imm4_0)              CSR_INSTR(csradd, imm4_0, 0x5, rd, 0x73)
// #define csrrsi(csradd, rd, imm4_0)              CSR_INSTR(csradd, imm4_0, 0x6, rd, 0x73)
// #define csrrci(csradd, rd, imm4_0)              CSR_INSTR(csradd, imm4_0, 0x7, rd, 0x73)

#define csrwi(a, s)     asm("csrwi %0, %1"::"i"(a),"i"(s):)
#define csrw(a, s)      asm("csrw  %0, %1 "::"i"(a),"r"(s):)
#define csrr(s, a)      asm("csrr  %0, %1":"+r"(s):"i"(a):)

// Possible values of VPRs
#define VPR_NONE    0
#define VPR_0_EN    2
#define VPR_1_EN    3

#endif /*HIPU200CLIB_EXTEND_INSTR_GEN_H__*/
