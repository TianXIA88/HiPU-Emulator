#ifndef _HIRT_CMDQUEUE_H__
#define _HIRT_CMDQUEUE_H__

#include <semaphore.h>
#include <pthread.h>
#include "hirt.h"
#include "hirt_fifo.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum
{
    CMDTYPE_KERNEL = 0,
    CMDTYPE_SYNC,
    CMDTYPE_NOTIFIER
} hirtCmdType_t;

typedef struct
{
    hirtCmdType_t type;
    hirtTaskDim_t dim;
    
    hirtKernelParamsBuffer_t *buf_param;
    hirtKernelBinBuffer_t    *buf_kernel;

    sem_t *sem_cmdsync;
} hirtCmdNode_t;

typedef struct
{
    hirtFifo_t *pfifo;
} hirtCmdQueue_t;

hisdkRet_t hirtCmdQueueCreate(hirtCmdQueue_t **ppQueue);
hisdkRet_t hirtCmdQueueDestroy(hirtCmdQueue_t *pQueue);
hisdkRet_t hirtCmdQueueSyncPut(hirtCmdQueue_t *pQueue, sem_t *pSem);
hisdkRet_t hirtCmdQueueKernelPut(hirtCmdQueue_t *pQueue, hirtKernelParamsBuffer_t *pParams, 
    hirtKernelBinBuffer_t *pKernelBin, hirtTaskDim_t taskDim);
hisdkRet_t hirtCmdQueueNodeGet(hirtCmdQueue_t *pQueue, hirtCmdNode_t *pNode);
hisdkRet_t hirtCmdQueueSync(hirtCmdQueue_t *pQueue);

#ifdef __cplusplus
}
#endif
#endif /*_HIRT_CMDQUEUE_H__*/

