/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-30 10:47:13
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-01-29 19:36:50
 */
#ifndef __LIBHIDV_H__
#define __LIBHIDV_H__

#include "hisdk_type.h"
#include "hisdk_config.h"
#include "hisdk_port.h"
#include "hisdk_log.h"
#include "hisdk_error.h"
#include "hi_addr_def.h"


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define HIPU200_DEV_H2C_DEFAULT         "/dev/xdma0_h2c_0"
#define HIPU200_DEV_C2H_DEFAULT         "/dev/xdma0_c2h_0"
#define HIPU200_DEV_CTRL_DEFAULT        "/dev/xdma0_control"
#define HIPU200_DEV_USR_DEFAULT         "/dev/xdma0_user"
#define HIPU200_DEV_EVENT_DEFAULT       "/dev/xdma0_events_0"



uint8_t hidvGetNocNodeXY(uint32_t coreidx);
hisdkRet_t hidvInit();
void hidvDestroy();
void hidvWriteChipReg(uint32_t offset, uint32_t value);
uint32_t hidvReadChipReg(uint32_t offset);
void hidvWriteNocNodeMem(uint8_t node_xy, uint64_t offset, uint64_t size, void *data);
void hidvReadNocNodeMem(uint8_t node_xy, uint64_t offset, uint64_t size, void *data);
void hidvWriteFileNocNode(uint8_t node_xy, uint64_t offset, const char *filename);
void hidvWriteChipMem(uint64_t addr, uint64_t size, void *data);
void hidvReadChipMem(uint64_t addr, uint64_t size, void *data);
void hidvReadChipAMO(int coreidx, void *data);
void hidvCoreResetPC(uint32_t coreBitmap);
void hidvCoreActivate(uint32_t coreBitmap);
void hidvCoreDeactivate(uint32_t coreBitmap);
void hidvCoreWakeup(uint32_t coreBitmap);
// void hidvCoreSetMMAPCodeData(uint8_t node_xy, uint8_t mmap_code_mono, uint8_t mmap_code_data);
// void hidvCoreSetMMAPCode(uint8_t node_xy, uint8_t mmap_code_mono);
ssize_t read_to_buffer(const char *fname, int fd, char *buffer, uint64_t size, uint64_t base);
ssize_t write_from_buffer(const char *fname, int fd, char *buffer, uint64_t size, uint64_t base);

#ifdef __cplusplus
}
#endif
#endif /*__LIBHIDV_H__*/
