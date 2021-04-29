#ifndef __HIML_H__
#define __HIML_H__

#include "hisdk_type.h"
#include "hisdk_config.h"
#include "hisdk_port.h"
#include "hisdk_log.h"
#include "hisdk_error.h"

#include "hirt.h"
#include "hirt_cqueue.h"
#include "hirt_mm.h"
#include "hirt_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    DATATYPE_INT8 = 0,
    DATATYPE_INT32 = 1,
    DATATYPE_FP32 = 2,
    DATATYPE_FP16 = 3
} himlDataType_t;

typedef enum
{
    TS_TYPE_TENSOR = 0,
    TS_TYPE_CONST = 1,
    TS_TYPE_WEIGHT = 2,
    TS_TYPE_BIAS = 3
} himlTensorType_t;

typedef enum
{
    TS_FORMAT_NCHW = 0,
    TS_FORMAT_NHWC = 1
} himlDataOrder_t;

typedef enum
{
    FUNCTYPE_C1 = 0,
    FUNCTYPE_C2 = 1,
    FUNCTYPE_C4 = 2,
    FUNCTYPE_C8 = 3,
    FUNCTYPE_CA = 4
} himlFuncType_t;

typedef struct
{
    u32_t dim;
    u32_t n;
    u32_t c;
    union
    {
        u32_t h;
        u32_t row;
    };
    union
    {
        u32_t w;
        u32_t col;
    };
} shape_t;

typedef struct
{
    himlTensorType_t tensor_type;
    himlDataType_t   data_type;
    himlDataOrder_t  data_order;
    shape_t          shape;
    u32_t            bufsize;
} himlCpuTensor_t;

typedef struct
{
    himlTensorType_t  tensor_type;
    himlDataType_t    data_type;
    shape_t           shape;
    u32_t             bufsize;
    hirtGMemAddress_t gmemaddr;
} himlGpuTensor_t;

typedef struct
{
    u32_t stride_w;
    u32_t stride_h;
    u32_t dilation_w;
    u32_t dilation_h;
    u32_t pad_w;
    u32_t pad_h;
} himlConvOpParam_t;

#define KERNEL_SYMBAL_NAME_MAX	(128)
typedef struct
{
    char kernel_symbal_name[KERNEL_SYMBAL_NAME_MAX];
    hirtKernelParamsBuffer_t *kernel_parambuf;
    hirtKernelBinBuffer_t    *kernel_binbuf;
} himlBaseOp_t;

hisdkRet_t himlCreateConvOp(
    himlBaseOp_t *op,
    himlConvOpParam_t *param,
    himlGpuTensor_t *inputTensor,
    himlGpuTensor_t *outputTensor,
    himlGpuTensor_t *filterTensor,
    himlGpuTensor_t *biasTensor);

hisdkRet_t himlComputeConvOpForward(
    himlBaseOp_t *op,
    hirtGMemAddress_t input,
    hirtGMemAddress_t output,
    himlFuncType_t type,
    hirtCmdQueue_t *pQueue);

hisdkRet_t himlCreateConvOpParam(himlConvOpParam_t **ppParam,
                int stride_h,
                int stride_w,
                int dilation_h,
                int dilation_w,
                int pad_h,
                int pad_w);

hisdkRet_t himlDestroyConvOpParam(himlConvOpParam_t *pParam);

hisdkRet_t himlCreateCpuTensor(himlCpuTensor_t** ppCpu_tensor,
                              himlTensorType_t  tensor_type,
                              himlDataType_t    data_type,
                              himlDataOrder_t   data_order,
                              int n,
                              int c,
                              int h,
                              int w);

hisdkRet_t himlDestroyCpuTensor(himlCpuTensor_t* pCpu_tensor);

hisdkRet_t himlCreateGpuTensor(himlGpuTensor_t** ppDev_tensor,
                              himlTensorType_t  tensor_type,
                              himlDataType_t    data_type,
                              int n,
                              int c,
                              int h,
                              int w);

hisdkRet_t himlDestroyGpuTensor(himlGpuTensor_t* pDev_tensor);

// hisdkAlloc himl tensor
hirtGMemAddress_t himlMallocBuffer(himlGpuTensor_t *tensor_himl);
        
// copy input to himl tensor
hisdkRet_t himlMemcpyTensorToDevice(
    himlCpuTensor_t  *tensor_cpu,
    void             *tensor_cpu_buf,
    himlGpuTensor_t  *tensor_dev,
    hirtGMemAddress_t tensor_dev_buf);

    
// copy input to himl tensor
hisdkRet_t himlMemcpyTensorToHost(
    himlGpuTensor_t  *tensor_dev,
    hirtGMemAddress_t tensor_dev_buf,
    himlCpuTensor_t  *tensor_cpu,
    void             *tensor_cpu_buf);

    
hisdkRet_t himlDumpTensor2File(const char *filename, himlCpuTensor_t *tensor_cpu);


#ifdef __cplusplus
}
#endif
#endif /*__LIBHIRT_H__*/

