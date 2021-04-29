/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-14 16:47:43
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-22 19:55:15
 */
#ifndef __HISDK_UTILS_H__
#define __HISDK_UTILS_H__

#include "hisdk_type.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void loadFromCAGParamFileBuf(char* input_file_buf, size_t file_size, char *host_memBuf, int buf_size);
void loadFromCAGParamFileBufForBiasAndShift(char* input_file_buf, size_t file_size, char *out_host_memBS, char *out_host_memShift);
void OutputHexBufToCAGParamFileFormat(const unsigned char* input_hex_buf, size_t buf_size, const char *output_file_name);
void hirtPrintHpuBuf(const char* host_mem_buf, int buf_len);

//为仿真而dump文件
void hirtDumpData2FileForSimulation(uint64_t addr, uint64_t size, void *data);
//为QEMU而dump文件
void hirtDumpData2FileForQemu(const char* src_file_path, uint64_t size, void *data, uint64_t dst_physical_gaddr, uint32_t dst_hpu_local_addr, const char* extra_tag_name = "");

#ifdef __cplusplus
}
#endif
#endif /*__HISDK_UTILS_H__*/

