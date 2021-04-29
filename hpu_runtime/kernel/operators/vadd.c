/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2021-01-11 10:58:19
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-01 18:22:19
 */

#include <string.h>
// #include <stdio.h>

#include "hihw.h"
#include "libconv.h"
#include "dma.h"
#include "lock.h"
#include "int.h"
#include "operators/hi_krnl_param_vadd.h"
#include "hi_addr_def.h"
#include "hpu_util.h"
#include "hpu_api.h"

#define conv_xy(xy) (((xy & 0x30) >> 2) | (xy & 0x03))
#define SBUF_SIZE (64)

void vadd()
{
    // unsigned char buf_sram_a[SBUF_SIZE] __attribute__((aligned(64)));
    uint32_t buf_sram_a = 0;
    Krnl_hpu_malloc(&buf_sram_a, SBUF_SIZE, KERNEL_HPUMEM_TYPE_MMA);
    memset((void *)buf_sram_a, 0, SBUF_SIZE);
    printf("buf_sram_a 0x%x\n", buf_sram_a);

    uint32_t buf_sram_b = 0;
    Krnl_hpu_malloc(&buf_sram_b, SBUF_SIZE, KERNEL_HPUMEM_TYPE_MMA);
    memset((void *)buf_sram_b, 0, SBUF_SIZE);

    uint32_t buf_sram_c = 0;
    Krnl_hpu_malloc(&buf_sram_c, SBUF_SIZE, KERNEL_HPUMEM_TYPE_MMA);
    memset((void *)buf_sram_c, 0, SBUF_SIZE);

    unsigned int *_flags = (unsigned int *)HIPU200_MEM_ATOMIC_START;

    // int _coreid = get_hpuid(); /*get core id number*/
    unsigned int _coreid = _flags[HIRT_HIPU200_CORE_ID_STORE_INDEX];

    unsigned int _taskid;
    int i;
    int *_rtcode = (int *)HIPU200_KNL_RTCODE_ADDR; /*kernel return code to host runtime*/

    paramTableVAdd_t *_ptable = *((paramTableVAdd_t **)HIPU200_KNL_PTABLE_ADDR); /*get kernel param table from runtime*/
    int paramTableLen = _ptable->infoBase.table_size;
    int parallelCoreNum = _ptable->infoBase.task_dim;

    // int temp = (int)&parallelCoreNum;
    unsigned int temp = sizeof(unsigned long);
    // unsigned int temp = 2;

    // printf("parallelCoreNum's address %d\n", temp);
    
    // printf("parallelCoreNum's address %x\n", &parallelCoreNum);

    /*find the taskid of the current core*/
    for (i = 0; i < parallelCoreNum; ++i)
    {
        if (_ptable->infoBase.task_cores[i] == _coreid)
        {
            _taskid = i;
            break;
        }
    }

#if 1
    //test begin
    int ndma_len = SBUF_SIZE;
    int ndma_loc_addr, ndma_rmt_addr;
    // for (int i = 0; i < 1; ++i)
    {
        // dma load data from ddr input buffer to local sram buffer
        ndma_loc_addr = buf_sram_a;
        ndma_rmt_addr = _ptable->param.srcOffset_A;
        // _flags[8] = ndma_loc_addr;
        // _flags[9] = ndma_rmt_addr;
        ndma_load_data(ndma_loc_addr, ndma_rmt_addr, ndma_len, conv_xy(_ptable->param.srcNocnod_A));
        ndma_wait();

        // dma load data from ddr input buffer to local sram buffer
        ndma_loc_addr = buf_sram_b;
        ndma_rmt_addr = _ptable->param.srcOffset_B;
        // _flags[10] = ndma_loc_addr;
        // _flags[11] = ndma_rmt_addr;
        ndma_load_data(ndma_loc_addr, ndma_rmt_addr, ndma_len, conv_xy(_ptable->param.srcNocnod_B));
        ndma_wait();

        // do the actual calculation
        for (int j = 0; j < SBUF_SIZE; ++j)
        {
            ((unsigned char *)buf_sram_c)[j] = ((unsigned char *)buf_sram_a)[j] + ((unsigned char *)buf_sram_b)[j];
        }

        // dma save data from local sram buffer to ddr output buffer
        ndma_loc_addr = buf_sram_c;
        ndma_rmt_addr = _ptable->param.dstOffset;
        // _flags[12] = ndma_loc_addr;
        ndma_save_data(ndma_loc_addr, ndma_rmt_addr, ndma_len, conv_xy(_ptable->param.dstNocnod));
        ndma_wait();
        // _flags[14] = 0xaaaabbbb;
    }
//test end
#endif

#if 0
    int vlen_core = 32; //(vlen == 1) ? 1 : ((vlen f- 1) / cnum + 1);
    int iternum_core = (vlen_core - 1) / SBUF_SIZE + 1;
    int ndma_loc_addr, ndma_rmt_addr, ndma_len, itemnum;

    for (int i = 0; i < iternum_core; ++i)
    {
        if(((i+1) * SBUF_SIZE) <= vlen_core)
        {
            itemnum = SBUF_SIZE;
        }
        else
        {
            itemnum = vlen_core - i * SBUF_SIZE;
        }
        ndma_len = itemnum;

        // dma load data from ddr input buffer to local sram buffer
        ndma_loc_addr = buf_sram_a;
        ndma_rmt_addr = _ptable->param.srcOffset_A + _taskid * vlen_core + i * SBUF_SIZE;
        // _flags[8] = ndma_loc_addr;
        // _flags[9] = ndma_rmt_addr;
        ndma_load_data(ndma_loc_addr, ndma_rmt_addr, ndma_len, conv_xy(_ptable->param.srcNocnod_A));
        ndma_wait();

        // dma load data from ddr input buffer to local sram buffer 
        ndma_loc_addr = buf_sram_b;
        ndma_rmt_addr = _ptable->param.srcOffset_B + _taskid * vlen_core + i * SBUF_SIZE;
        // _flags[10] = ndma_loc_addr;
        // _flags[11] = ndma_rmt_addr;
        ndma_load_data(ndma_loc_addr, ndma_rmt_addr, ndma_len, conv_xy(_ptable->param.srcNocnod_B));
        ndma_wait();

        // do the actual calculation
        for (int j = 0; j < itemnum; ++j)
        {
            // buf_sram_c[j] = buf_sram_a[j] + buf_sram_b[j];
            ((unsigned char *)buf_sram_c)[j] = ((unsigned char *)buf_sram_a)[j] + ((unsigned char *)buf_sram_b)[j];
        }

        // dma save data from local sram buffer to ddr output buffer
        ndma_loc_addr = buf_sram_c;
        ndma_rmt_addr = _ptable->param.dstOffset + _taskid * vlen_core + i * SBUF_SIZE;
        // _flags[12] = ndma_loc_addr;
        // _flags[13] = ndma_rmt_addr;
        ndma_save_data(ndma_loc_addr, ndma_rmt_addr, ndma_len, conv_xy(_ptable->param.dstNocnod));
        ndma_wait();
        // _flags[7] = 0xaaaabbbb;
    }
#endif
}
