#ifndef LIBHIRT_FIFO_H__
#define LIBHIRT_FIFO_H__

#include <pthread.h>
#include <semaphore.h>
#include "hirt.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct
{
    int   pWrite;
    int   pRead;
    int   item_num;
    int   item_siz;
    int   is_full;
    void  *buf;
    sem_t sem_read;
    //  sem_t sem_write;
} hirtFifo_t;

hisdkRet_t  hirtFifoCreate(hirtFifo_t **ppFifo, int item_num, int item_siz);
hisdkRet_t  hirtFifoDestroy(hirtFifo_t *fifo);
hisdkRet_t  hirtFifoPut(hirtFifo_t *fifo, void *pdata);
hisdkRet_t  hirtFifoGet(hirtFifo_t *fifo, void *pbuf);
int         hirtFifoIsFull(hirtFifo_t *fifo);

#ifdef __cplusplus
}
#endif
#endif /*LIBHIRT_FIFO_H__*/

