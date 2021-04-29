/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-30 17:53:20
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-11 10:37:16
 */
#ifndef __LIBHI_ADDR_DEF_H__
#define __LIBHI_ADDR_DEF_H__
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define HIPU200_USING_FPGA
#ifdef HIPU200_USING_FPGA
    #define HIPU200_SOC_CORE_NUM            (1)
#else
    #define HIPU200_SOC_CORE_NUM            (13)
#endif


//L1-I是8KB，L1-D是16KB， L1的cache line size都是64B， L2总大小是64KB，L2的cache line size是4KB
#define HIPU200_DMA_BUF_ALIGNSIZE       (1024)

#define HIPU200_CORE_0                  (0)
#define HIPU200_CORE_1                  (1)
#define HIPU200_CORE_2                  (2)
#define HIPU200_CORE_3                  (3)
#define HIPU200_CORE_4                  (4)
#define HIPU200_CORE_5                  (5)
#define HIPU200_CORE_6                  (6)
#define HIPU200_CORE_7                  (7)
#define HIPU200_CORE_8                  (8)
#define HIPU200_CORE_9                  (9)
#define HIPU200_CORE_10                 (10)
#define HIPU200_CORE_11                 (11)
#define HIPU200_CORE_12                 (12)

#define HIPU200_CORE_0_BIT              (0x01 << HIPU200_CORE_0 )
#define HIPU200_CORE_1_BIT              (0x01 << HIPU200_CORE_1 )
#define HIPU200_CORE_2_BIT              (0x01 << HIPU200_CORE_2 )
#define HIPU200_CORE_3_BIT              (0x01 << HIPU200_CORE_3 )
#define HIPU200_CORE_4_BIT              (0x01 << HIPU200_CORE_4 )
#define HIPU200_CORE_5_BIT              (0x01 << HIPU200_CORE_5 )
#define HIPU200_CORE_6_BIT              (0x01 << HIPU200_CORE_6 )
#define HIPU200_CORE_7_BIT              (0x01 << HIPU200_CORE_7 )
#define HIPU200_CORE_8_BIT              (0x01 << HIPU200_CORE_8 )
#define HIPU200_CORE_9_BIT              (0x01 << HIPU200_CORE_9 )
#define HIPU200_CORE_10_BIT             (0x01 << HIPU200_CORE_10)
#define HIPU200_CORE_11_BIT             (0x01 << HIPU200_CORE_11)
#define HIPU200_CORE_12_BIT             (0x01 << HIPU200_CORE_12)

#define HIPU200_CORE_BIT(x)             (0x01 << x)

#define HIPU200_CORE_ALL_BIT            (HIPU200_CORE_0_BIT  | \
                                         HIPU200_CORE_1_BIT  | \
                                         HIPU200_CORE_2_BIT  | \
                                         HIPU200_CORE_3_BIT  | \
                                         HIPU200_CORE_4_BIT  | \
                                         HIPU200_CORE_5_BIT  | \
                                         HIPU200_CORE_6_BIT  | \
                                         HIPU200_CORE_7_BIT  | \
                                         HIPU200_CORE_8_BIT  | \
                                         HIPU200_CORE_9_BIT  | \
                                         HIPU200_CORE_10_BIT | \
                                         HIPU200_CORE_11_BIT | \
                                         HIPU200_CORE_12_BIT   \
                                        )

#define HIPU200_NOC_SIZE_W              (4)
#define HIPU200_NOC_SIZE_H              (4)
#define HIPU200_NOC_NODE_NUM            (HIPU200_NOC_SIZE_W*HIPU200_NOC_SIZE_H)
#define HIPU200_NOC_ADDR_XSHIFT         (36)
#define HIPU200_NOC_ADDR_YSHIFT         (32)
//#define HIPU200_NOC_MAKEADDR(x,y,offs)  ( ((uint64_t)(x)<<HIPU200_NOC_ADDR_XSHIFT)+((uint64_t)(y)<<HIPU200_NOC_ADDR_YSHIFT)+offs )
#define HIPU200_NOC_MAKEADDR(node,offs) ( ((uint64_t)(node)<<32)+offs )
#define HIPU200_NOC_NMAP_XSHIFT         (4)
#define HIPU200_NOC_NMAP_YSHIFT         (0)
#define HIPU200_NOC_MAKENMAP(x,y)       ( ((uint8_t)(x)<<HIPU200_NOC_NMAP_XSHIFT)+((uint8_t)(y)<<HIPU200_NOC_NMAP_YSHIFT) )
#define HIPU200_NOC_MAKEMMAP_x64M(x)    ((uint8_t)(x))
#define HIPU200_NOC_MAKEMMAP(x)         ( (uint8_t)(x>>26) )
#define HIPU200_NOC_NODE_ADDR(addr)     ((addr & 0xFF00000000) >> 32)
#define HIPU200_NOC_NODE_ADDR_X_POS(addr)     ((addr & 0xF000000000) >> 36)
#define HIPU200_NOC_NODE_ADDR_Y_POS(addr)     ((addr & 0x0F00000000) >> 32)
#define HIPU200_NOC_LOCAL_ADDR(addr)     (addr & 0xFFFFFFFF) 

#define HIPU200_NOC_DDR0_X              (0)
#define HIPU200_NOC_DDR0_Y              (3)
#define HIPU200_NOC_DDR1_X              (3)
#define HIPU200_NOC_DDR1_Y              (3)
#define HIPU200_NOC_PCIE_X              (2)
#define HIPU200_NOC_PCIE_Y              (3)
#define HIPU200_NOC_COR0_X              (0)
#define HIPU200_NOC_COR0_Y              (0)
#define HIPU200_NOC_COR1_X              (0)
#define HIPU200_NOC_COR1_Y              (1)
#define HIPU200_NOC_COR2_X              (0)
#define HIPU200_NOC_COR2_Y              (2)
#define HIPU200_NOC_COR3_X              (1)
#define HIPU200_NOC_COR3_Y              (0)
#define HIPU200_NOC_COR4_X              (1)
#define HIPU200_NOC_COR4_Y              (1)
#define HIPU200_NOC_COR5_X              (1)
#define HIPU200_NOC_COR5_Y              (2)
#define HIPU200_NOC_COR6_X              (1)
#define HIPU200_NOC_COR6_Y              (3)
#define HIPU200_NOC_COR7_X              (2)
#define HIPU200_NOC_COR7_Y              (0)
#define HIPU200_NOC_COR8_X              (2)
#define HIPU200_NOC_COR8_Y              (1)
#define HIPU200_NOC_COR9_X              (2)
#define HIPU200_NOC_COR9_Y              (2)
#define HIPU200_NOC_COR10_X             (3)
#define HIPU200_NOC_COR10_Y             (0)
#define HIPU200_NOC_COR11_X             (3)
#define HIPU200_NOC_COR11_Y             (1)
#define HIPU200_NOC_COR12_X             (3)
#define HIPU200_NOC_COR12_Y             (2)

#define HIPU200_NOC_NODEADDR_DDR0       HIPU200_NOC_MAKENMAP(HIPU200_NOC_DDR0_X,  HIPU200_NOC_DDR0_Y)
#define HIPU200_NOC_NODEADDR_DDR1       HIPU200_NOC_MAKENMAP(HIPU200_NOC_DDR1_X,  HIPU200_NOC_DDR1_Y)
#define HIPU200_NOC_NODEADDR_PCIE       HIPU200_NOC_MAKENMAP(HIPU200_NOC_PCIE_X,  HIPU200_NOC_PCIE_Y)
#define HIPU200_NOC_NODEADDR_COR0       HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR0_X,  HIPU200_NOC_COR0_Y)
#define HIPU200_NOC_NODEADDR_COR1       HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR1_X,  HIPU200_NOC_COR1_Y)
#define HIPU200_NOC_NODEADDR_COR2       HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR2_X,  HIPU200_NOC_COR2_Y)
#define HIPU200_NOC_NODEADDR_COR3       HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR3_X,  HIPU200_NOC_COR3_Y)
#define HIPU200_NOC_NODEADDR_COR4       HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR4_X,  HIPU200_NOC_COR4_Y)
#define HIPU200_NOC_NODEADDR_COR5       HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR5_X,  HIPU200_NOC_COR5_Y)
#define HIPU200_NOC_NODEADDR_COR6       HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR6_X,  HIPU200_NOC_COR6_Y)
#define HIPU200_NOC_NODEADDR_COR7       HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR7_X,  HIPU200_NOC_COR7_Y)
#define HIPU200_NOC_NODEADDR_COR8       HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR8_X,  HIPU200_NOC_COR8_Y)
#define HIPU200_NOC_NODEADDR_COR9       HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR9_X,  HIPU200_NOC_COR9_Y)
#define HIPU200_NOC_NODEADDR_COR10      HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR10_X, HIPU200_NOC_COR10_Y)
#define HIPU200_NOC_NODEADDR_COR11      HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR11_X, HIPU200_NOC_COR11_Y)
#define HIPU200_NOC_NODEADDR_COR12      HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR12_X, HIPU200_NOC_COR12_Y)
#define HIPU200_NOC_NODEADDR_COR(x)     HIPU200_NOC_MAKENMAP(HIPU200_NOC_COR##x##_X, HIPU200_NOC_COR##x##_Y)

#define HIPU200_REG_MCU_CFG_0           (0x00)  /*default:0x0000_001F*/
#define HIPU200_REG_HPU_CFG_0           (0x04)  /*default:0x0000_0000*/
#define HIPU200_REG_HPU_RESTART_0       (0x0C)  /*default:0x0000_1FFF*/
#define HIPU200_REG_INITPC              (0x1C)  /*default:0x8000_0000*/
#define HIPU200_REG_PCIE_NOC_RXLEN      (0x80)  /*default:*/
#define HIPU200_REG_PCIE_DMA_RXNODE     (0x84)  /*default:*/
#define HIPU200_REG_PCIE_DMA_TXNODE     (0x88)  /*default:*/
#define HIPU200_REG_PCIE_DMA_DONE       (0x8C)  /*default:*/
#define HIPU200_REG_PCIE_NOC_TXLEN      (0x90)  /*default:*/
#define HIPU200_REG_MCU_STATUS          (0x100)  /*default:*/
#define HIPU200_REG_RAW_INT_STATUS_0    (0x200)  /*default:*/
#define HIPU200_REG_MSKED_INT_STATUS_0  (0x204)  /*default:*/
#define HIPU200_REG_INT_CLR_0           (0x208)  /*default:*/
#define HIPU200_REG_INT_MSK_0           (0x20C)  /*default:*/
#define HIPU200_REG_RAW_INT_STATUS_1    (0x210)  /*default:*/
#define HIPU200_REG_MSKED_INT_STATUS_1  (0x214)  /*default:*/
#define HIPU200_REG_INT_CLR_1           (0x218)  /*default:*/
#define HIPU200_REG_INT_MSK_1           (0x21C)  /*default:*/
#define HIPU200_REG_PLL_SEL             (0x280)  /*default:*/
#define HIPU200_REG_PLL0_CFG            (0x284)  /*default:*/
#define HIPU200_REG_PLL1_CFG            (0x288)  /*default:*/
#define HIPU200_REG_PLL2_CFG            (0x28C)  /*default:*/
#define HIPU200_REG_PLL3_CFG            (0x290)  /*default:*/
#define HIPU200_REG_PLL4_CFG            (0x294)  /*default:*/
#define HIPU200_REG_PLL5_CFG            (0x298)  /*default:*/

#define HIPU200_MEM_ATOMIC_START        (0x02010000)
#define HIPU200_MEM_ATOMIC_SIZE         (64)
#define HIPU200_MEM_LOCAL_MRA_START     (0x02100000)
#define HIPU200_MEM_LOCAL_MRA_SIZE      (256*1024)
#define HIPU200_MEM_LOCAL_MRB_START     (0x02140000)
#define HIPU200_MEM_LOCAL_MRB_SIZE      (256*1024)
#define HIPU200_MEM_LOCAL_START         (HIPU200_MEM_LOCAL_MRA_START)
#define HIPU200_MEM_LOCAL_SIZE          (HIPU200_MEM_LOCAL_MRA_SIZE+HIPU200_MEM_LOCAL_MRB_SIZE)
#define HIPU200_MEM_NMAP_START          (0x02012000)
#define HIPU200_MEM_NMAP_SIZE           (32)
#define HIPU200_MEM_MMAP_START          (0x02012020)
#define HIPU200_MEM_MMAP_SIZE           (32)
// #define HIPU200_KNL_PTABLE_ADDR         (HIPU200_MEM_ATOMIC_START+HIPU200_MEM_ATOMIC_SIZE-4)
// #define HIPU200_KNL_RTCODE_ADDR         (HIPU200_KNL_PTABLE_ADDR-4)

#define HIPU200_MEM_SSIG_ADDR           (0x02012040)            //risc-v 向此地址最低位bit写1 时， 向host 发起一个中断请求，与此同时，非阻塞1继续执行
#define HIPU200_MEM_SFMD_ADDR           (0x02012080)            //相应位写1时打开： bit 0: RISC-V issues 1 instr/cyc; bit 1: disable the recovery mode; bit 2: leave enough space for freelist.
#define HIPU200_MEM_APC_START           (0x02012060)
#define HIPU200_DDRMEM_CACHE_START      (0x80000000)
#define HIPU200_LOCAL_SPACE_MASK        (0xFFFFFFFF)

// #define HIPU200_MEM_MMAP_BLKSIZE            (64*1024*1024)
// #define HIPU200_MEM_MMAP_CODE_MONO_START    (0x80000000)
// #define HIPU200_MEM_MMAP_CODE_MONO_BLKNUM   (1)
// #define HIPU200_MEM_MMAP_CODE_SHAR_START    (HIPU200_MEM_MMAP_CODE_MONO_START+HIPU200_MEM_MMAP_CODE_MONO_BLKNUM*HIPU200_MEM_MMAP_BLKSIZE)
// #define HIPU200_MEM_MMAP_CODE_SHAR_BLKNUM   (3)

// #define HIPU200_MEM_MMAP_DATA_MONO_START    (0x90000000)
// #define HIPU200_MEM_MMAP_DATA_MONO_BLKNUM   (1)
// #define HIPU200_MEM_MMAP_DATA_SHAR_START    (HIPU200_MEM_MMAP_DATA_MONO_START+HIPU200_MEM_MMAP_DATA_MONO_BLKNUM*HIPU200_MEM_MMAP_BLKSIZE)
// #define HIPU200_MEM_MMAP_DATA_SHAR_BLKNUM   (3)

// //DDR0
// #define HIPU200_DDR_MMAP_CODE_MONO_START    (0)
// #define HIPU200_DDR_MMAP_CODE_MONO_OFFS(x)  (HIPU200_DDR_MMAP_CODE_MONO_START+(x)*HIPU200_MEM_MMAP_BLKSIZE)
// #define HIPU200_DDR_MMAP_CODE_MONO_BLKNUM   (HIPU200_SOC_CORE_NUM)
// #define HIPU200_DDR_MMAP_CODE_SHAR_START    (HIPU200_DDR_MMAP_CODE_MONO_START+HIPU200_DDR_MMAP_CODE_MONO_BLKNUM*HIPU200_MEM_MMAP_BLKSIZE)
// #define HIPU200_DDR_MMAP_CODE_SHAR_OFFS(x)  (HIPU200_DDR_MMAP_CODE_SHAR_START+(x)*HIPU200_MEM_MMAP_BLKSIZE)
// #define HIPU200_DDR_MMAP_CODE_SHAR_BLKNUM   (HIPU200_MEM_MMAP_CODE_SHAR_BLKNUM)

// //DDR1
// #define HIPU200_DDR_MMAP_DATA_MONO_START    (0)
// #define HIPU200_DDR_MMAP_DATA_MONO_OFFS(x)  (HIPU200_DDR_MMAP_DATA_MONO_START+(x)*HIPU200_MEM_MMAP_BLKSIZE)
// #define HIPU200_DDR_MMAP_DATA_MONO_BLKNUM   (HIPU200_SOC_CORE_NUM)
// #define HIPU200_DDR_MMAP_DATA_SHAR_START    (HIPU200_DDR_MMAP_DATA_MONO_START+HIPU200_DDR_MMAP_DATA_MONO_BLKNUM*HIPU200_MEM_MMAP_BLKSIZE)
// #define HIPU200_DDR_MMAP_DATA_SHAR_OFFS(x)  (HIPU200_DDR_MMAP_DATA_SHAR_START+(x)*HIPU200_MEM_MMAP_BLKSIZE)
// #define HIPU200_DDR_MMAP_DATA_SHAR_BLKNUM   (HIPU200_MEM_MMAP_DATA_SHAR_BLKNUM)


#ifdef __cplusplus
}
#endif
#endif /*__LIBHI_ADDR_DEF_H__*/
