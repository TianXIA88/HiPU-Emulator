/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2021-01-12 17:39:29
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-01-12 17:48:18
 */
#ifndef LIB_HIRT_DEBUG_PRINT_H__
#define LIB_HIRT_DEBUG_PRINT_H__

#include <pthread.h>
#include <semaphore.h>
#include "hirt.h"
#include "hirt_scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct
{
    pthread_t m_thread;
    hirtThreadFunction_t m_threadfunc;
    void* m_pArg;

    // hirtScoreboard_t *m_pScoreboard;
    // sem_t *m_pSemScheduler;
} hirtDebugHandler_t;

hisdkRet_t hirtDebugHandlerCreate(hirtDebugHandler_t **ppHandler);
hisdkRet_t hirtDebugHandlerDestroy(hirtDebugHandler_t *pHandler);
void*      hirtDebugHandlerThread(void* arg);

#ifdef __cplusplus
}
#endif
#endif /*LIB_HIRT_DEBUG_PRINT_H__*/

