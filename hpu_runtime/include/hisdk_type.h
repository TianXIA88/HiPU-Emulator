#ifndef __HISDK_H__
#define __HISDK_H__

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef uint64_t u64_t;
typedef int64_t  i64_t;
typedef uint64_t hirt_u64_t;
typedef int64_t  hirt_i64_t;
typedef uint32_t u32_t;
typedef uint16_t u16_t;
typedef uint8_t  u8_t;
typedef int32_t  i32_t;
typedef int16_t  i16_t;
typedef int8_t   i8_t;
typedef int      bool_t;
typedef u64_t    hirt_size_t;
typedef unsigned char   uint8;
typedef unsigned int    uint32;

#define HISDK_UNUSED(expr) do { (void)(expr); } while (0)

/**< Error codes */
typedef enum
{
    HISDK_RET_SUCCESS                 = 0,      /**< No error */
    HISDK_RET_WARNING_FAKE_DEVICE     = 1,      /**< Use fake device */
    HISDK_RET_ERR_INVALID             = 632007, /**< Invalid argument */
    HISDK_RET_ERR_NOMEM               = 632008, /**< Out of memory */
    HISDK_RET_ERR_NODEV               = 632009, /**< No such device */
    HISDK_RET_ERR_IO                  = 632010, /**< I/O error */
    HISDK_RET_ERR_SYS                 = 632011, /**< System error */
    HISDK_RET_ERR_ACCES               = 632012, /**< Permission denied */
    HISDK_RET_ERR_FAULT               = 632013, /**< Bad address */
    HISDK_RET_ERR_BUSY                = 632014, /**< Device or resource busy */
    HISDK_RET_ERR_TIMEOUT             = 632015, /**< Time expired */
    HISDK_RET_ERR_EXIST               = 632016, /**< Resource or file already exists */
    HISDK_RET_ERR_NOSYS               = 632017, /**< Function not implemenmted */
    HISDK_RET_ERR_AGAIN               = 632018, /**< try again later */
    HISDK_RET_ERR_NORES               = 632019, /**< Out of resource */
    HISDK_RET_ERR_UNSUPPORTED         = 632020, /**< Unsupported operation */
    HISDK_RET_ERR_INVALID_POINTER     = 632021, /**< Invalid pointer */
    HISDK_RET_ERR_NO_EXIST            = 632022, /**< Resource or file doesn't exist */
    HISDK_RET_ERR_BROKEN              = 632023, /**< Data transmission is broken */
    HISDK_RET_ERR_INIT                = 632024, /**< Uninitialized */
    HISDK_RET_ERR_STREAM              = 632025, /**< Failure on Stream */
    HISDK_RET_ERR_OUT_RANGE           = 632026, /**< Number out of range */
    HISDK_RET_ERR_MATH_OVERFLOW       = 632027, /**< Math result not representable */
    HISDK_RET_ERR_FUNC_CALL           = 632028, /**< Failure to call runtime functions */
    HISDK_RET_ERR_UNHANDLED           = 632029, /**< Unhandled error */
    HISDK_RET_ERR_INVALID_TYPE        = 632030, /**< Invalid type */
    HISDK_RET_ERR_INVALID_OP          = 632031, /**< Invalid operation */
    HISDK_RET_ERR_MLU                 = 632032, /**< MLU error */
    HISDK_RET_ERR_ONCHIP_CORE         = 632033, /**< Onchip core error */
    HISDK_RET_ERR_EVENT               = 632034, /**< Failure on event operation */
    HISDK_RET_ERR_RESHAPE             = 632035, /**< Failure on data reshape */
    HISDK_RET_ERR_MEMCPY              = 632036, /**< Failure on memory copy */
    HISDK_RET_ERR_ENCRYPT             = 632037, /**< Failure on encrypt */
    HISDK_RET_ERR_INVALID_DATADESC    = 632038, /**< Invalid data descriptor */
    HISDK_RET_ERR_FILEWRITEFAILED     = 632039, /**< FileWriteFailed      */
    HISDK_RET_ERR_FILEREADFAILED      = 632040, /**< FileReadFailed       */
    HISDK_RET_ERR_ENDOFFILE           = 632041, /**< EndOfFile            */
    HISDK_RET_ERR_FILEOPERATIONFAILED = 632042, /**< FileOperationFailed  */
    HISDK_RET_ERR_DIROPERATIONFAILED  = 632043, /**< DirOperationFailed   */
    HISDK_RET_ERR_ENDOFDIRLIST        = 632044, /**< EndOfDirList         */
    HISDK_RET_ERR_IOCTLFAILED         = 632045, /**< IoctlFailed          */
    HISDK_RET_ERR_PATHALREADYEXISTS   = 632046, /**< PathAlreadyExists    */
    HISDK_RET_ERR_SURFACENOTSUPPORTED = 632047, /**< SurfaceNotSupported  */
    HISDK_RET_ERR_NOTIMPLEMENTED      = 632048, /**< NotImplemented       */
    HISDK_RET_ERR_NOTINITIALIZED      = 632050, /**< NotInitialized       */
    HISDK_RET_ERR_BADPARAMETER        = 632051, /**< BadParameter         */
    HISDK_RET_ERR_INSUFFICIENTMEMORY  = 632053, /**< InsufficientMemory   */
    HISDK_RET_ERR_READONLYATTRIBUTE   = 632054, /**< ReadOnlyAttribute    */
    HISDK_RET_ERR_INVALIDSTATE        = 632055, /**< InvalidState         */
    HISDK_RET_ERR_INVALIDADDRESS      = 632056, /**< InvalidAddress       */
    HISDK_RET_ERR_INVALIDSIZE         = 632057, /**< InvalidSize          */
    HISDK_RET_ERR_BADVALUE            = 632058, /**< BadValue             */
    HISDK_RET_ERR_ALREADYALLOCATED    = 632059, /**< AlreadyAllocated     */
    HISDK_RET_ERR_MODULENOTPRESENT    = 632060, /**< ModuleNotPresent     */
    HISDK_RET_ERR_RESOURCEERROR       = 632061, /**< ResourceError        */
    HISDK_RET_ERR_COUNTMISMATCH       = 632062, /**< CountMismatch        */
    HISDK_RET_ERR_OVERFLOW            = 632063, /**< OverFlow             */
    HISDK_RET_ERR_DISCONNECTED        = 632064, /**< Disconnected         */
    HISDK_RET_ERR_FILENOTFOUND        = 632065, /**< FileNotFound         */
    HISDK_RET_ERR_TESTAPPFAILED       = 632066, /**< TestAppFailed        */
    HISDK_RET_ERR_DEVICENOTFOUND      = 632067, /**< DeviceNotFound       */
    HISDK_RET_ERR_FIFOEMPTY           = 632068, /**< fifoempty            */
    HISDK_RET_ERR_FIFOFULL            = 632069, /**< fifofull             */
    HISDK_RET_ERR_CREATETHREAD        = 632070, /**< createthread         */
    HISDK_RET_ERR_UNKNOWN             = 999991, /**< Unknown error */
    HISDK_RET_ERR_MAX,                          /**< The last one */
} hisdkRet_t; 


#ifdef __cplusplus
}
#endif
#endif /*__HISDK_H__*/

