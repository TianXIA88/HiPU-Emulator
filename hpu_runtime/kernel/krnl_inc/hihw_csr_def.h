/*
 * @Description: csr
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 14:33:45
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-09 20:46:10
 */
#ifndef __HIPU200CLIB_CSR_REG_DEF_H__
#define __HIPU200CLIB_CSR_REG_DEF_H__

// global definition
#define CSR_HPU_ID                 0xf15

// -- Scalar region
// -- NoC DMA

#define CSR_ADDR_NDMA_CTRL                      (0x7C0)
#define CSR_ADDR_NDMA_STATUS                    (0x7C1)
#define CSR_ADDR_NDMA_LCADDR                    (0x7C2)
#define CSR_ADDR_NDMA_RTADDR                    (0x7C3)
#define CSR_ADDR_NDMA_SIZE                      (0x7C4)
#define CSR_ADDR_NDMA_DESTXY                    (0x7C5)

//标量指令，向量（矩阵）指令，ndma 之间没有严格的保序关系，向量（矩阵）指令必定落后于标量指令的执行，
//如果标量指令的执行（ndma也是由标量指令发起的），依赖于向量（矩阵）指令的输出，则必须查询：CSR_ADDR_VMU_STATUS（0x7cf）== 0x 7,如果成立，说明向量（矩阵）指令执行完成
#define CSR_ADDR_VMU_STATUS                     (0x7Cf)   //readonly, == 0x7: now vector or matrix instructions have finished
#define vmu_poll()              {   uint32_t poll = 0;                      \
                                    do{                                     \
                                        csrr(poll, CSR_ADDR_VMU_STATUS);    \
                                    }while((poll) != (0x7));                \
                                }

// -- VMU
#define CSR_VMU_STATUS             0x7cf

// -- VCSR region
// -- Vec
#define CSR_ROUND_TYPE             0xbc0
#define CSR_FADD_SHIFT_NUM         0xbc1
#define CSR_FADD_PROT_HIGH         0xbc2
#define CSR_FADD_PROT_LOW          0xbc3
#define CSR_FSUB_SHIFT_NUM         0xbc4
#define CSR_FSUB_PROT_HIGH         0xbc5
#define CSR_FSUB_PROT_LOW          0xbc6
#define CSR_FMUL_SHIFT_NUM         0xbc7
#define CSR_FMUL_PROT_HIGH         0xbc8
#define CSR_FMUL_PROT_LOW          0xbc9

// -- Mtx
#define CSR_MTX_CLUSTER_START      0xbd0
#define CSR_MTX_CLUSTER_END        0xbd1
#define CSR_MTX_REGION_START       0xbd2
#define CSR_MTX_REGION_END         0xbd3
#define CSR_MTX_BLK_SIZE           0xbd4
#define CSR_MTX_BLK_NUM            0xbd5
#define CSR_MTX_CLS_NUM            0xbd6
#define CSR_MTXRW_BLK_STRIDE       0xbd7
#define CSR_MTXRW_CLS_STRIDE       0xbd8
#define CSR_MTXRO_BLK_STRIDE       0xbd9
#define CSR_MTXRO_CLS_STRIDE       0xbda
#define CSR_MTX_PAD_TYPE           0xbdb

#define CSR_WR_LUT_FIRST                        (0xBE0)
#define CSR_WR_LUT_INC                          (0xBE1)
#define CSR_FPRINT_ADDR                         (0xBE2)
#define CSR_FPRINT_LEN                          (0xBE3)

// -- LUT
#define CSR_VMU_WR_LUT_FIRST       CSR_WR_LUT_FIRST
#define CSR_VMU_WR_LUT_INC         CSR_WR_LUT_INC

#endif /*__HIPU200CLIB_CSR_REG_DEF_H__*/
