/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-14 16:47:43
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-02-04 21:07:38
 */
#ifndef LIBHIRT_MM_H__
#define LIBHIRT_MM_H__

#include "hirt.h"
#include "hidv.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
    
typedef enum
{
    MB_FREE = 0,
    MB_USED = 1
} hirtMemBlockStat_t;

typedef struct
{
    hirtMemBlockStat_t m_mbstatus;
    hirtGMemAddress_t  m_mbbase;
} hirtMemBlockInfo_t;

typedef struct
{
    hirtGMemAddress_t m_mbbase;
    int m_blocknum;
    int m_ch;
    int m_beginBlk;
} hirtMemAllocRecord_t;

typedef struct
{
    hirtMemBlockInfo_t m_blocktbl[HIRT_HIPU200_MEM_CH_NUM][HIRT_HIPU200_MEM_CH_BLKNUM];
    hirtMemAllocRecord_t m_recordtbl[HIRT_HIPU200_MEM_CH_NUM*HIRT_HIPU200_MEM_CH_BLKNUM];
} hirtMemManager_t;

hisdkRet_t hirtMemManagerCreate(hirtMemManager_t **ppMemManager);
hisdkRet_t hirtMemManagerDestroy(hirtMemManager_t *pMemManager);
hisdkRet_t hirtMemManagerBlkGet(hirtMemManager_t *pMemManager, hirtGMemAddress_t *pAddr, int block_num);
hisdkRet_t hirtMemManagerBlkPut(hirtMemManager_t *pMemManager, hirtGMemAddress_t addr);

__R_HOST
hisdkRet_t hirtHostMalloc(void **pPtr, size_t nBytes/*, hirtMemType_t type*/);
__R_HOST
hisdkRet_t hirtHostFree(void *ptr);
__R_HOST
hisdkRet_t hirtGpuMalloc(hirtGMemAddress_t *pDevAddr, size_t nBytes, size_t *pActualSize, hirtGMemType_t memType = HIRT_GMEM_TYPE_SHARE1);
__R_HOST
hisdkRet_t hirtGpuFree(hirtGMemAddress_t devAddr);
__R_HOST
hisdkRet_t hirtMemcpy(void *dest, const void *src, size_t nBytes, hirtMemTransDir_t dir);
__R_HOST
u32_t hirtGetHPUCoreLocalMemorySpaceAddrForSHA0(hirtGMemAddress_t gPhysAddr);
__R_HOST
u32_t hirtGetHPUCoreLocalMemorySpaceAddrForSHA1(hirtGMemAddress_t gPhysAddr);

#ifdef __cplusplus
}
#endif
#endif /*LIBHIRT_FIFO_H__*/

