/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 14:35:14
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-22 20:01:17
 */
#include <string.h>
#include <assert.h>

#include "hpu_util.h"
#include "hisdk_config.h"
#include "krnl_log.h"
extern void flush_l2_cache();

// static unsigned int *_amo_flags = (unsigned int *)AMO_ADDR_S;

void read_from_ddr(void* addr, unsigned int size, void *data)
{
    //flush_l2_cache
    flush_l2_cache();
    //update local cache
    memcpy(data, addr, size);
}

int u_i_mod(unsigned long value, unsigned long base)
{
    assert(value >= 0);
    assert(base > 0);
    while(value >= 0 && value >= base)
    {
        // _amo_flags[8] = value;
        // _amo_flags[9] = base;
        value = value - base;
    }
    return value;
}

uint32_t bitwise_div_32 (uint32_t dividend, uint32_t divisor)
{
    if(divisor==0){
         printf("³ýÊý²»ÄÜÎªÁã\n");
    }
    uint32_t quot, rem, t;
    int bits_left = 8 * sizeof (uint32_t); //CHAR_BIT *
    quot = dividend;
    rem = 0;
    do {
            // (rem:quot) << 1
            t = quot;
            quot = quot + quot;
            rem = rem + rem + (quot < t);

            if (rem >= divisor) {
                rem = rem - divisor;
                quot = quot + 1;
            }
            bits_left--;
    } while (bits_left);
    return quot;
}

static uint32_t memUsedMMA = 0;
static uint32_t memUsedMMB = 0;

__R_HPU
int Krnl_hpu_malloc(uint32_t *pDevAddr, size_t nBytes, KrnlHPUMemType_t memType)
{
    int ret = 0;

    switch(memType)
    {
    case KERNEL_HPUMEM_TYPE_MMA:
        *pDevAddr = MMA_START_ADDR + memUsedMMA;
        memUsedMMA += GLOCAL_MEM_ALIGN(nBytes);
        break;
    case KERNEL_HPUMEM_TYPE_MMB:
        *pDevAddr = MMB_START_ADDR + memUsedMMB;
        memUsedMMB += GLOCAL_MEM_ALIGN(nBytes);
        break;
    default:
        ret = -1;
    }
    // HISDK_LOG_INFO(LOG_SYSTEM, "<GpuMalloc:size=%lu, addr=0x%lx", nBytes, *pDevAddr);
    return ret;
}

__R_HPU
void buf_print(uint32_t buf_addr, uint32_t buf_len)
{
#ifdef QEMU_ENV
    return;
#endif
#ifdef HIRT_DUMP_DATA_FOR_SIMULATION
    return;
#endif 
    char one_line_str[300] = {0};
    char temp[10] = {0};
    int line_num = buf_len / 64;
	KRNL_LOG_INFO(LOG_DEBUG, "buf_addr: 0x%x buf_len: %d line_num: %d", buf_addr, buf_len, line_num);
    for (int line = 0; line < line_num; line ++)
    // for (int line = 0; line < 3; line ++)
 	{
 	    for (int index = 63; index >= 0; index --)
 	    {
 	        if (index == 63)
 	        {
 	            snprintf(temp, 10, "line %d: ", line);
                memcpy(one_line_str, temp, strlen(temp));
                memset(temp, 0, 10);
 	        }
 	        snprintf(temp , 10, "%02x", ((unsigned char *)buf_addr)[index + line * 64]);
            memcpy(one_line_str + strlen(one_line_str), temp, strlen(temp));
 	    }
        memset(temp, 0, 10);
        memcpy(one_line_str + strlen(one_line_str), "\n", strlen("\n"));
	    KRNL_LOG_INFO(LOG_DEBUG, "buf: %s", one_line_str);
        memset(one_line_str, 0, 300);
 	}

}