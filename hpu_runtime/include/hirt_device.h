#ifndef _HIRT_DEVICE_H__
#define _HIRT_DEVICE_H__

#include "hirt.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

__R_HOST
hisdkRet_t hirtGetHardwareVersion(hirtHardwareVersion_t *ver);
__R_HOST
hisdkRet_t hirtGetDeviceHandle(hirtDev_t *dev, int ordinal);
__R_HOST
hisdkRet_t hirtSetCurrentDevice(hirtDev_t dev);
__R_HOST
hisdkRet_t hirtGetCurrentDevice(hirtDev_t *dev);
__R_HOST
hisdkRet_t hirtGetDeviceCapability(hirtCap_t *cap, hirtDev_t dev);
__R_HOST
hisdkRet_t hirtGetDeviceCount(unsigned *devnum);
__R_HOST
hisdkRet_t hirtGetDeviceMemorySizeMB(unsigned int *sz);

#ifdef __cplusplus
}
#endif
#endif /*_HIRT_DEVICE_H__*/

