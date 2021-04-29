/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2021-01-07 17:03:37
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-10 15:51:10
 */
#ifndef LIBHIRT_EVENT_H__
#define LIBHIRT_EVENT_H__

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

    hirtScoreboard_t *m_pScoreboard;
    sem_t *m_pSemScheduler;
} hirtEventHandler_t;

sem_t* hirtEventGetFinishSemaphore();
hisdkRet_t hirtEventHandlerCreate(hirtEventHandler_t **ppHandler, hirtScheduler_t *pScheduler);
hisdkRet_t hirtEventHandlerDestroy(hirtEventHandler_t *pHandler);
void*      hirtEventHandlerThread(void* arg);

#ifdef __cplusplus
}
#endif
#endif /*LIBHIRT_EVENT_H__*/

