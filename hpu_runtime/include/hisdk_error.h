#ifndef __HISDK_ERROR_H__
#define __HISDK_ERROR_H__

#include <stdint.h>
#include <stdbool.h>
#include "hisdk_type.h"
#include "hisdk_config.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//#ifdef HISDK_DEBUG
//#define HISDK_ASSERT(expr) do {} while (0);
//#else
#define HISDK_ASSERT(expr)                                  \
    do{                                                     \
        if (!(expr)) { \
            printf("\nASSERT failed at:\n  >File name: %s\n  >Function : %s\n  >Line No. : %d\n  >Condition: %s\n", \
                __FILE__, __FUNCTION__, __LINE__, #expr);   \
            exit(0);\
        } \
    }while(0);
    //exit(0);
//#endif

#define HISDK_ERRCHECK(ret) if (ret != HISDK_RET_SUCCESS) {\
    printf("error occur, func: %s, line: %d\n", __func__, __LINE__);\
    exit(0);\
    }


const char* hisdkUtilsGetErrorString(hisdkRet_t e);

/**
 * There are two ways for a client to customize the error logging macros, both of which
 * are performed by defining a macro value from the client before including this file:
 *  1) Define HISDK_UTILS_ERROR_TAG to be a string with the client name. This string will
 *     be appended to any error logs before being output by the default hisdkUtilsLogError.
 *     In addition, HISDK_UTILS_ERROR_PATH may be optionally defined so that this path will
 *     be stripped from any generated errors (if not set, the 'hisdk' path
 *     will be stripped by default).
 *  2) Define HISDK_UTILS_LOG_ERROR to point to a custom error logging function within the
 *     client. The signature for this macro should be the same as the default, below.
 */
#if defined (HISDK_UTILS_ERROR_TAG)
// Use the default hisdkUtilsLogError with the client tag appended.
void hisdkUtilsLogError(const char* tag, const char* path, hisdkRet_t e, const char* file, const char* func,
                        uint32_t line, bool propagating, const char* format, ...);
#if !defined (HISDK_UTILS_ERROR_PATH)
#define HISDK_UTILS_ERROR_PATH "hisdk"
#endif

#define HISDK_UTILS_LOG_ERROR(_err, _file, _func, _line, _propagating, _format, ...) \
    do { \
        hisdkUtilsLogError(HISDK_UTILS_ERROR_TAG, HISDK_UTILS_ERROR_PATH, \
                                (_err), (_file), (_func), (_line), \
                                (_propagating), (_format), ##__VA_ARGS__); \
    } \
    while (0)
#elif defined (HISDK_UTILS_LOG_ERROR)
// Use the client's custom error logging function.
#else
#error "One of HISDK_UTILS_ERROR_TAG or HISDK_UTILS_LOG_ERROR must be defined"
#endif

/**
 * Argument counting macros.
 */
#define HISDK_UTILS_CAT(A, B) A##B
#define HISDK_UTILS_SELECT(NAME, NUM) HISDK_UTILS_CAT(NAME##_, NUM)
#define HISDK_UTILS_GET_COUNT(_1, _2, _3, _4, _5, COUNT, ...) COUNT
#define HISDK_UTILS_VA_SIZE(...) HISDK_UTILS_GET_COUNT(__VA_ARGS__, 5, 4, 3, 2, 1)
#define HISDK_UTILS_VA_SELECT(NAME, ...) HISDK_UTILS_SELECT(NAME, HISDK_UTILS_VA_SIZE(__VA_ARGS__))(__VA_ARGS__)

/**
 * Simply report an error.
 * HISDK_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, 0);
 * HISDK_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format));
 */
#define REPORT_ERROR(...) HISDK_UTILS_VA_SELECT(REPORT_ERROR_IMPL, __VA_ARGS__)
#define REPORT_ERROR_IMPL_1(_err) \
    do { \
        e = (_err); \
        HISDK_LOG_INFO(LOG_ERROR, "err=%d ", e); \
        goto fail; \
    } while (0)
#define REPORT_ERROR_IMPL_2(_err, _format) \
    do { \
        e = (_err); \
        HISDK_LOG_INFO(LOG_ERROR, "err=%d "_format, e); \
        goto fail; \
    } while (0)
#define REPORT_ERROR_IMPL_3(_err, _format, ...) REPORT_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_IMPL_4(_err, _format, ...) REPORT_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_IMPL_5(_err, _format, ...) REPORT_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_IMPL_N(_err, _format, ...) \
    do { HISDK_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format), __VA_ARGS__); } while (0)

#if 0
/**
 * Report and return an error that was first detected in the current method.
 */
#define ORIGINATE_ERROR(...) HISDK_UTILS_VA_SELECT(REPORT_ERROR_IMPL, __VA_ARGS__)
#define REPORT_ERROR_IMPL_1(_err) \
    do { \
        HISDK_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, 0); \
        return (_err); \
    } while (0)
#define REPORT_ERROR_IMPL_2(_err, _format) \
    do { \
        HISDK_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format)); \
        return (_err); \
    } while (0)
#define REPORT_ERROR_IMPL_3(_err, _format, ...) REPORT_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_IMPL_4(_err, _format, ...) REPORT_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_IMPL_5(_err, _format, ...) REPORT_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_IMPL_N(_err, _format, ...) \
    do { \
        HISDK_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format), __VA_ARGS__); \
        return (_err); \
    } while (0)
#endif

/**
 * Report an error that was first detected in the current method, then jumps to the "fail:" label.
 * The variable "hisdkRet_t e" must have been previously declared.
 * HISDK_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, 0); \
 * HISDK_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format)); \
 */
#define HISDK_ERR_RPTFAIL(...) HISDK_UTILS_VA_SELECT(REPORT_ERROR_FAIL_IMPL, __VA_ARGS__)
#define REPORT_ERROR_FAIL_IMPL_1(_err) \
    do { \
        e = (_err); \
        HISDK_LOG_INFO(LOG_ERROR, "err=%d ", e); \
        goto fail; \
    } while (0)
#define REPORT_ERROR_FAIL_IMPL_2(_err, _format) \
    do { \
        e = (_err); \
        HISDK_LOG_INFO(LOG_ERROR, "err=%d " _format, e); \
        goto fail; \
    } while (0)
#define REPORT_ERROR_FAIL_IMPL_3(_err, _format, ...) REPORT_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_FAIL_IMPL_4(_err, _format, ...) REPORT_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_FAIL_IMPL_5(_err, _format, ...) REPORT_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_FAIL_IMPL_N(_err, _format, ...) \
    do { \
        e = (_err); \
        HISDK_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format), __VA_ARGS__); \
        goto fail; \
    } while (0)

#if 0
/**
 * Calls another function, and if an error was returned it is reported and returned.
 */
#define PROPAGATE_ERROR(...) HISDK_UTILS_VA_SELECT(FCALL_ERROR_IMPL, __VA_ARGS__)
#define FCALL_ERROR_IMPL_1(_err) \
    do { \
        hisdkRet_t peResult = (_err); \
        if (peResult != HISDK_RET_SUCCESS) \
        { \
            HISDK_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, 0); \
            return peResult; \
        } \
    } while (0)
#define FCALL_ERROR_IMPL_2(_err, _format) \
    do { \
        hisdkRet_t peResult = (_err); \
        if (peResult != HISDK_RET_SUCCESS) \
        { \
            HISDK_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, (_format)); \
            return peResult; \
        } \
    } while (0)
#define FCALL_ERROR_IMPL_3(_err, _format, ...) FCALL_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define FCALL_ERROR_IMPL_4(_err, _format, ...) FCALL_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define FCALL_ERROR_IMPL_5(_err, _format, ...) FCALL_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define FCALL_ERROR_IMPL_N(_err, _format, ...) \
    do { \
        hisdkRet_t peResult = (_err); \
        if (peResult != HISDK_RET_SUCCESS) \
        { \
            HISDK_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, (_format), __VA_ARGS__); \
            return peResult; \
        } \
    } while (0)
#endif

/**
 * Calls another function, and if an error was returned it is reported before jumping to the
 * "fail:" label. The variable "hisdkRet_t e" must have been previously declared.
 * HISDK_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, 0); \
 * HISDK_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, (_format)); \
 */
#define HISDK_ERR_FCALLFAIL(...) HISDK_UTILS_VA_SELECT(FCALL_ERROR_FAIL_IMPL, __VA_ARGS__)
#define FCALL_ERROR_FAIL_IMPL_1(_err) \
    do { \
        e = (_err); \
        if (e != HISDK_RET_SUCCESS) \
        { \
            HISDK_LOG_INFO(LOG_ERROR, "err=%d ", e); \
            goto fail; \
        } \
    } while (0)
#define FCALL_ERROR_FAIL_IMPL_2(_err, _format) \
    do { \
        e = (_err); \
        if (e != HISDK_RET_SUCCESS) \
        { \
            HISDK_LOG_INFO(LOG_ERROR, "err=%d " _format, e); \
            goto fail; \
        } \
    } while (0)
#define FCALL_ERROR_FAIL_IMPL_3(_err, _format, ...) FCALL_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define FCALL_ERROR_FAIL_IMPL_4(_err, _format, ...) FCALL_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define FCALL_ERROR_FAIL_IMPL_5(_err, _format, ...) FCALL_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define FCALL_ERROR_FAIL_IMPL_N(_err, _format, ...) \
    do { \
        e = (_err); \
        if (e != HISDK_RET_SUCCESS) \
        { \
            HISDK_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, (_format), __VA_ARGS__); \
            goto fail; \
        } \
    } while (0)

#if 0
/**
 * Calls another function, and if an error was returned it is reported and assigned to the
 * variable 'e' (which must have been previously declared). The caller does not return.
 */
#define FCALL_ERROR_CONTINUE(...) HISDK_UTILS_VA_SELECT(FCALL_ERROR_CONTINUE_IMPL, __VA_ARGS__)
#define FCALL_ERROR_CONTINUE_IMPL_1(_err) \
    do { \
        hisdkRet_t peResult = (_err); \
        if (peResult != HISDK_RET_SUCCESS) \
        { \
            HISDK_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, 0); \
            if (e == HISDK_RET_SUCCESS) \
                e = peResult; \
        } \
    } while (0)
#define FCALL_ERROR_CONTINUE_IMPL_2(_err, _format) \
    do { \
        hisdkRet_t peResult = (_err); \
        if (peResult != HISDK_RET_SUCCESS) \
        { \
            HISDK_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, (_format)); \
            if (e == HISDK_RET_SUCCESS) \
                e = peResult; \
        } \
    } while (0)
#define FCALL_ERROR_CONTINUE_IMPL_3(_err, _format, ...) FCALL_ERROR_CONTINUE_IMPL_N((_err), (_format), __VA_ARGS__)
#define FCALL_ERROR_CONTINUE_IMPL_4(_err, _format, ...) FCALL_ERROR_CONTINUE_IMPL_N((_err), (_format), __VA_ARGS__)
#define FCALL_ERROR_CONTINUE_IMPL_5(_err, _format, ...) FCALL_ERROR_CONTINUE_IMPL_N((_err), (_format), __VA_ARGS__)
#define FCALL_ERROR_CONTINUE_IMPL_N(_err, _format, ...) \
    do { \
        hisdkRet_t peResult = (_err); \
        if (peResult != HISDK_RET_SUCCESS) \
        { \
            HISDK_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, (_format), __VA_ARGS__); \
            if (e == HISDK_RET_SUCCESS) \
                e = peResult; \
        } \
    } while (0)
#endif

#ifdef __cplusplus
}
#endif // __cplusplus
#endif /*__HISDK_ERROR_H__*/
