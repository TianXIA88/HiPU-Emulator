/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-14 16:47:43
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-27 16:51:32
 */
#ifndef __HISDK_CONFIG_H__
#define __HISDK_CONFIG_H__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//是否打开dump数据的仿真开关
// #define HIRT_DUMP_DATA_FOR_SIMULATION

//是否打开dump数据的Qemu开关
// #define HIRT_DUMP_DATA_FOR_QEMU
// #define QEMU_DUMP_DATA_OUTPUT_PATH "/home/cag-driver/Code/Compiler/HPU-MNN/3rd_party/HPU-Runtime/kernel/qemu/qemu_bin/"

// #define HPU200_RSHIFT_WITH_DECREMENT1_THEN_VFADD_RSHIFT1   //rshift 前少shift 1 位，在vfadd时，先四舍五入，再rshift最后那一位，但此时如果原始的rshift == 0时，硬件将出问题，会rshift 很多很多位

#define HISDK_DEBUG
//undef HISDK_DEBUG
#define HISDK_INTERUPT_PATCH

#define LOGFILE_NAME                    "logfile"
#define KERNEL_FILE_NAME                "kernel_all.bin"
#define LOGFILE_SIZEMAX                 (1*1024*1024)
#define LOGFILE_MEDIA                   (HISDK_LOG_TO_FILE)
#define LOGFILE_DEBUG_EN                (HISDK_LOG_DEBUG_ENABLE)


#define MTX_SCALE       (8)
#define VEC_SCALE       (64)

#define HISDK_UTILS_ERROR_TAG           "hisdk"

#define HIRT_CMDQUEUE_SIZMAX            (1024)
#define HIRT_PARAMBUF_MAXSIZE           (4096)
// #define HIPU200_SOC_CORE_NUM         (13)
#define HIRT_HIPU200_MEM_CH_NUM         (2)
#define HIRT_HIPU200_MEM_CH_SIZ         (4096LL*1024*1024)
#define HIRT_HIPU200_MEM_BLK_SIZ        (64*1024*1024)
#define HIRT_HIPU200_MEM_CH_BLKNUM      (HIRT_HIPU200_MEM_CH_SIZ/HIRT_HIPU200_MEM_BLK_SIZ)
#define HIRT_HIPU200_MEM_CH0_NOCADDR    (0)
#define HIRT_HIPU200_MEM_CH1_NOCADDR    (1)
#define HIRT_HIPU200_DEBUG_PRINT_PING_PONG_SWITCH_BUF_LEFT_MINSIZE (255)    //when sizeof empty buf left in the debug_print_buf < 255, then ping pong switch 

//start from HIPU200_MEM_ATOMIC_START         
#define HIRT_HIPU200_CORE_ID_STORE_INDEX (0)        //one slot: u32_t
//#define HIRT_HIPU200_CORE_DEBUG_PRINT_PING_PONG_INDEX (1)                  //0(0~2M) or 1 (2~4M) buffer
#define HIRT_HIPU200_KNL_RTCODE_ADDR_INDEX (14)
#define HIRT_HIPU200_KNL_PTABLE_ADDR_INDEX (15)

typedef struct{
    uint32_t mstatus;               //HIPU200_KNL_MSTATUS_ADDR        
    uint32_t mtvalue;               //HIPU200_KNL_MTVAL_ADDR
    uint32_t mepc;                  //HIPU200_KNL_MEPC_ADDR           
    uint32_t mcause;                //HIPU200_KNL_MCAUSE_ADDR
}exception_result;
typedef union{
    uint32_t reserved[4];
    exception_result exception_result;
}exception_reserved;

typedef struct{
    uint8 _coreid:4;
    uint8 busy_ndma_channel_count:4;
    uint8 reserved2;
    uint16_t reserved3;
    uint32_t reserved4[9];
    exception_reserved exception_reserved;
    uint32_t return_code;               //HIPU200_KNL_RTCODE_ADDR
    uint32_t param_table;               //HIPU200_KNL_PTABLE_ADDR         
}amo_struct;

#define HIPU200_KNL_PTABLE_ADDR         (HIPU200_MEM_ATOMIC_START+HIPU200_MEM_ATOMIC_SIZE-4)
#define HIPU200_KNL_RTCODE_ADDR         (HIPU200_KNL_PTABLE_ADDR-4)
#define HIPU200_KNL_RTCODE_SUCCESS      (0x12345678)
#define HIPU200_KNL_RTCODE_BEGIN        (0xcccccccc)
#define HIPU200_KNL_MCAUSE_ADDR         (HIPU200_KNL_PTABLE_ADDR-8)
#define HIPU200_KNL_MEPC_ADDR           (HIPU200_KNL_PTABLE_ADDR-12)
#define HIPU200_KNL_MTVAL_ADDR          (HIPU200_KNL_PTABLE_ADDR-16)
#define HIPU200_KNL_MSTATUS_ADDR        (HIPU200_KNL_PTABLE_ADDR-20)

#define ALIGN(x,a)        __ALIGN_MASK((x),(typeof(x))(a)-1)
#define __ALIGN_MASK(x,mask)    (((x)+(mask))&~(mask))


//https://iaircag.yuque.com/staff-szibwc/ne5d38/amv6hy
//2G ~ 4G
//#define GMEM_BASE_ADDR      ()
#define GMEM_ALIGN_SIZE     (64)
//#define GMEM_ALIGN_CARRY    (GMEM_ALIGN_SIZE)
//#define GMEM_ALIGN_MASK     (GMEM_ALIGN_SIZE-1)
//#define GMEM_ALIGN(x)       (((x & GMEM_ALIGN_MASK) == 0) ? (x) : ((x & GMEM_ALIGN_MASK) + GMEM_ALIGN_CARRY))
#define GMEM_ALIGN(x)       ALIGN(x, GMEM_ALIGN_SIZE)

//0G ~ 2G
#define GLOCAL_MEM_ALIGN_SIZE     (32)
//#define GLOCAL_MEM_ALIGN_CARRY    (GLOCAL_MEM_ALIGN_SIZE)
//#define GLOCAL_MEM_ALIGN_MASK     (GLOCAL_MEM_ALIGN_SIZE-1)
//#define GLOCAL_MEM_ALIGN(x)       (((x & GLOCAL_MEM_ALIGN_MASK) == 0) ? (x) : ((x & GLOCAL_MEM_ALIGN_MASK) + GLOCAL_MEM_ALIGN_CARRY))
#define GLOCAL_MEM_ALIGN(x)       ALIGN(x, GLOCAL_MEM_ALIGN_SIZE)

//single hpu core 2G~4G, 32 blks (0~31) mapping configuration**************start*************

// .text section
#define HIRT_HIPU200_CORE_LOCAL_MEM_CODE_OFFSET    ((0x80000000) + HIRT_HIPU200_MEM_BLK_SIZ)
#define HIRT_HIPU200_CORE_LOCAL_MEM_CODE_BLKNUM    (1)         
#define HIRT_HIPU200_CORE_LOCAL_MEM_CODE_LENGTH    (HIRT_HIPU200_CORE_LOCAL_MEM_CODE_BLKNUM*HIRT_HIPU200_MEM_BLK_SIZ)
// .data section
#define HIRT_HIPU200_CORE_LOCAL_MEM_DATA_BLKNUM    (1)         
#define HIRT_HIPU200_CORE_LOCAL_MEM_DATA_OFFSET    (HIRT_HIPU200_CORE_LOCAL_MEM_CODE_OFFSET+HIRT_HIPU200_CORE_LOCAL_MEM_CODE_LENGTH)
#define HIRT_HIPU200_CORE_LOCAL_MEM_DATA_LENGTH    (HIRT_HIPU200_CORE_LOCAL_MEM_DATA_BLKNUM*HIRT_HIPU200_MEM_BLK_SIZ)

// share0 memory, 14 blks right after .data section
#define HIRT_HIPU200_CORE_LOCAL_MEM_SHA0_BLKNUM    (14)        
#define HIRT_HIPU200_CORE_LOCAL_MEM_SHA0_OFFSET    (HIRT_HIPU200_CORE_LOCAL_MEM_DATA_OFFSET+HIRT_HIPU200_CORE_LOCAL_MEM_DATA_LENGTH)
#define HIRT_HIPU200_CORE_LOCAL_MEM_SHA0_LENGTH    (HIRT_HIPU200_CORE_LOCAL_MEM_SHA0_BLKNUM*HIRT_HIPU200_MEM_BLK_SIZ)

// share1 memory, 14 blks right after share0 memory
#define HIRT_HIPU200_CORE_LOCAL_MEM_SHA1_BLKNUM    (14)        // .share1 memory
#define HIRT_HIPU200_CORE_LOCAL_MEM_SHA1_OFFSET    (HIRT_HIPU200_CORE_LOCAL_MEM_SHA0_OFFSET + HIRT_HIPU200_CORE_LOCAL_MEM_SHA0_LENGTH)
#define HIRT_HIPU200_CORE_LOCAL_MEM_SHA1_LENGTH    (HIRT_HIPU200_CORE_LOCAL_MEM_SHA1_BLKNUM*HIRT_HIPU200_MEM_BLK_SIZ)

// debug memory block, at the last 64M in 2G ~ 4G space
#define HIRT_HIPU200_CORE_LOCAL_MEM_DEBUG_BLKNUM   (1)
#define HIRT_HIPU200_CORE_LOCAL_MEM_DEBUG_OFFSET   (HIRT_HIPU200_CORE_LOCAL_MEM_SHA1_OFFSET + HIRT_HIPU200_CORE_LOCAL_MEM_SHA1_LENGTH)
#define HIRT_HIPU200_CORE_LOCAL_MEM_DEBUG_LENGTH   (4*1024*1024)
#define HIRT_HIPU200_CORE_LOCAL_MEM_DEBUG_STRING_BUF_LENGTH   (4*1024*1024 - 2 * GMEM_ALIGN_SIZE)

//below 2 points sit in the first 64 + 64 bytes, each 64 bytes align
#define HIPU200_KNL_DEBUG_FIFO_START_ADDR          (0x0)                   //debug print start offset
#define HIPU200_KNL_DEBUG_FIFO_PRINTED_ADDR        (HIPU200_KNL_DEBUG_FIFO_START_ADDR + GMEM_ALIGN_SIZE)       //debug print already printed offset

//single hpu core 2G~4G, 32blks (0~31) mapping configuration**************end***************

//physical memory mapping: NMAP MMAP **************start*************

//DDR0
// .text section in DDR0
#define HIRT_HIPU200_PHYS_MEM_CODE_SECTION_OFFSET (0x0)                                         
#define HIRT_HIPU200_PHYS_MEM_CODE_SECTION_LENGTH  (HIRT_HIPU200_MEM_BLK_SIZ)                   

// backup .text section in DDR0
#define HIRT_HIPU200_PHYS_MEM_BACKUP_CODE_SECTION_OFFSET    (HIRT_HIPU200_PHYS_MEM_CODE_SECTION_OFFSET + HIRT_HIPU200_MEM_BLK_SIZ)         
#define HIRT_HIPU200_PHYS_MEM_BACKUP_CODE_SECTION_LENGTH    (3 * HIRT_HIPU200_MEM_BLK_SIZ)         

#define HIRT_HIPU200_PHYS_MEM_SHA1_OFFSET    (HIRT_HIPU200_PHYS_MEM_BACKUP_CODE_SECTION_OFFSET + HIRT_HIPU200_PHYS_MEM_BACKUP_CODE_SECTION_LENGTH + 6 * HIRT_HIPU200_MEM_BLK_SIZ)         
#define HIRT_HIPU200_PHYS_MEM_SHA1_LENGTH    (14 * HIRT_HIPU200_MEM_BLK_SIZ)         

//DDR1
// debug region
#define HIRT_HIPU200_PHYS_MEM_DEBUG_OFFSET (0x0)                                         
#define HIRT_HIPU200_PHYS_MEM_DEBUG_LENGTH (HIRT_HIPU200_MEM_BLK_SIZ)

#define HIRT_HIPU200_PHYS_MEM_SHA0_OFFSET    (HIRT_HIPU200_PHYS_MEM_DEBUG_OFFSET + HIRT_HIPU200_PHYS_MEM_DEBUG_LENGTH + 7 * HIRT_HIPU200_MEM_BLK_SIZ)         
#define HIRT_HIPU200_PHYS_MEM_SHA0_LENGTH    (14 * HIRT_HIPU200_MEM_BLK_SIZ)         

//physical memory mapping: NMAP MMAP **************end*************



#ifdef __cplusplus
}
#endif
#endif /*__HISDK_CONFIG_H__*/

