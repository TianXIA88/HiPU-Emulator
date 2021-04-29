#ifndef __LIBHIRT_H__
#define __LIBHIRT_H__

#include "hisdk_type.h"
#include "hisdk_config.h"
#include "hisdk_port.h"
#include "hisdk_log.h"
#include "hisdk_error.h"
#include "hi_addr_def.h"

//#include "hmodel.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// #define HIRT_DEVICE_NODE      "/dev/dri/renderD128"

/**< Code will be compiled for running on Host */
#ifndef __R_HOST
#define __R_HOST   // something like __attribute__ or #pragma
#endif /*__R_HOST*/

/**< Direction of data transmission */
typedef enum {
  HIRT_MEM_TRANS_DIR_HOST2GPU = 0,   /**< Host to Device*/
  HIRT_MEM_TRANS_DIR_GPU2GPU,        /**< Device to Device */
  HIRT_MEM_TRANS_DIR_GPU2HOST,       /**< Device to Host */
  HIRT_MEM_TRANS_DIR_HOST2HOST,      /**< Host to Host */
} hirtMemTransDir_t;

typedef enum {
  HIRT_GMEM_TYPE_CODE_BLOCK = 0,     /**< local block*/
  HIRT_GMEM_TYPE_SHARE0,         /**< share block */
  HIRT_GMEM_TYPE_SHARE1,              /**< share */
} hirtGMemType_t;

/**< Version and revision  */
typedef struct {
  u32_t   hirtPlatformVersion;     /**< platform version */
  u32_t   hirtPlatformRevision;    /**< platform revision */
  u32_t   hirtCoreVersion;         /**< core version */
  u32_t   hirtCoreRevision;        /**< core revision */
} hirtHardwareVersion_t;

/**< Device capabilities for compatibilities  */
typedef struct {
  u32_t   hirtCapsNumberOfCore;           /**< Number of cores */
  u32_t   hirtCapsMaxNumberOfStream;      /**< Maximum of stream that user can create */
} hirtCap_t;

/**< Data type and data order*/
typedef enum {
  HIRT_INVALID = 0x0,
  HIRT_FLOAT16 = 0x12,
  HIRT_FLOAT32 = 0x13,
  HIRT_FLOAT64 = 0x14,
  HIRT_INT8    = 0x21,
  HIRT_INT16   = 0x22,
  HIRT_INT32   = 0x23,
  HIRT_UINT8   = 0x31,
  HIRT_UINT32  = 0x33,
  HIRT_FIX8    = 0x41,
  HIRT_QUANT8  = 0x51,
  HIRT_BOOL    = 0x61,
} hirtDataType_t;

typedef enum {
  HIRT_NCHW = 0x0123,
  HIRT_NHWC = 0x0231,
} hirtDimOrder_t;

typedef int hirtTaskDim_t;
typedef unsigned long hirtDev_t;
typedef u64_t hirtGMemAddress_t;
typedef struct
{
    u8_t x_pos;
    u8_t y_pos;
    u32_t inner_addr;
} hirtNocAddress_t;

#define MAKE_DEVADDR(nodenoc, low32)    ((((u64_t)nodenoc) << 32) + low32)
#define MAKE_DEVADDR_BLK(nodenoc, blk)  ((((u64_t)nodenoc) << 32) + blk*HIRT_HIPU200_MEM_BLK_SIZ)

/**< Compiler */
/**< hirt.h */

/**< hirt_internal.h */
typedef struct {
  void              *pbuf_host;
  hirtGMemAddress_t pbuf_gpu;
  unsigned int max_param;
  unsigned int cur_param;
} hirtKernelParamsBuffer_t;

typedef struct {
  void *pbuf_host;
  hirtGMemAddress_t pbuf_gpu;
  unsigned int size;
} hirtKernelBinBuffer_t;

#if 0
typedef struct {
  u64_t function_start;
  u64_t function_end;
  u64_t function_size;
} hirtKernelFunction_t;
#endif

typedef struct {
  hirtGMemAddress_t base;
  size_t size;
} hirtKernelGMemAllocInfo_t;

typedef struct
{
  u32_t typeVersion;
  u32_t kernelparambuf_maxsize;
  u32_t task_parallelism;
  u32_t task_thread_coreid[HIPU200_SOC_CORE_NUM];
} hirtKernelParamParallel_t;

typedef struct
{
  u32_t mem_param_num;
  u32_t mem_input_num;
  u32_t mem_output_num;
  u32_t mem_weight_num;
  u32_t mem_bias_num;
  u32_t mem_feature_num;
  u32_t mem_lut_num;
}hirtKernelParamMemory_t;

typedef struct
{
    hirtKernelParamParallel_t parallel_info;
    hirtKernelParamMemory_t   memory_info;
}hirtKernelParamHdr_t;

//typedef struct hirtKernelTaskParam
//{
//  int task_parallelism;
//  int task_thread_coreid[HIPU200_SOC_CORE_NUM];
//  hirtKernelGMemAllocInfo_t mem_input;
//  hirtKernelGMemAllocInfo_t mem_output;
//  hirtKernelGMemAllocInfo_t mem_fm;
//  hirtKernelGMemAllocInfo_t mem_weight;
//  hirtKernelGMemAllocInfo_t mem_bias;
//  hirtKernelGMemAllocInfo_t mem_lut;
//} hirtKernelTaskParam_t;

typedef void (*hirtThreadFunction_t)(void *arg);

#ifdef __cplusplus
}
#endif
#endif /*__LIBHIRT_H__*/