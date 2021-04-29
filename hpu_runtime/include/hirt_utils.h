/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-14 16:47:43
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-03-25 10:41:18
 */
#ifndef _HIRT_UTILS_H__
#define _HIRT_UTILS_H__

#include "hirt.h"
#include "hidv.h"
#include "hirt_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

__R_HOST
const char *hirtGetErrorStr(hisdkRet_t errCode);
__R_HOST
hisdkRet_t  hirtGetLastErr(void);
__R_HOST
hisdkRet_t  hirtInit(unsigned int Flags);
__R_HOST
void        hirtDestroy(void);
__R_HOST
uint32_t    hirtGetCorePC(uint8_t node_xy);
__R_HOST
uint32_t    hirtGetFinishFlag();
__R_HOST
uint32_t    hirtUpdateFinishFlag(uint32_t finish_flag);
__R_HOST
hisdkRet_t hirtLoadFile(const char *filename, char **ppbuf, size_t *pSize);
// //为仿真而dump文件
// __R_HOST
// void hirtDumpData2FileForSimulation(uint64_t addr, uint64_t size, void *data);
// __R_HOST
// void hirtDumpData2FileForQemu(const char* src_file_path, uint64_t size, void *data, uint64_t dst_physical_gaddr, uint32_t dst_hpu_local_addr);


#ifdef __cplusplus
}
#endif
#endif /*_HIRT_UTILS_H__*/

