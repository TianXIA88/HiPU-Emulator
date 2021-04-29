/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2021-01-14 18:21:52
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-07 19:40:08
 */
#ifndef _krnl_LOG_H__
#define _krnl_LOG_H__

#include <stdio.h>
#include "hisdk_type.h"
#include <string.h>

#ifdef QEMU_ENV
	#include "qemu.h"
#else
	#include "printf.h"
#endif
extern char *strrchr (const char *__s, int __c);

#ifdef __cplusplus
extern "C" {
#endif


#define STR_COMM_SIZE               128
#define STR_MAX_SIZE                1024

#define MAX_LOG_FILE_NUM		    (3)

#define NUMBER(type)                sizeof(type)/sizeof(type[0])

// #define __FILENAME__                ((strrchr(__FILE__, (int)'/') != NULL) ? (strrchr(__FILE__, '/') + 1) : __FILE__)
#define __FILENAME__                (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) : __FILE__)

enum
{
    LOG_DEBUG = 0,  
    LOG_ERROR,      
    LOG_WARNING,    
    LOG_ACTION,     
    LOG_SYSTEM,     
    LOG_HARDWARE_CMD,  
	LOG_NDMA,
    BOTTOM
};

extern unsigned long g_ulPrintDebugLogFlag;
extern unsigned long g_ulPrintHardwareCmdLogFlag;
extern unsigned long g_ulPrintSYSLogFlag;
extern unsigned long g_ulPrintNDMALogFlag;

unsigned long krnlLOGPrintLog(unsigned char ucType, unsigned char *pucLogInfo);

#define KRNL_LOG_INFO(type, fmt, ...) do{\
		if((0 == g_ulPrintDebugLogFlag) && (LOG_DEBUG == type)) \
		{\
			break;\
		}\
		if((0 == g_ulPrintHardwareCmdLogFlag) && (LOG_HARDWARE_CMD == type)) \
		{\
			break;\
		}\
		if((0 == g_ulPrintSYSLogFlag) && (LOG_SYSTEM == type)) \
		{\
			break;\
		}\
		if((0 == g_ulPrintNDMALogFlag) && (LOG_NDMA == type)) \
		{\
			break;\
		}\
    	unsigned char ucLogInfo[STR_MAX_SIZE] = {0}; \
		snprintf((char *)ucLogInfo, sizeof(ucLogInfo) - 1, fmt" [%s.%d][%s()]\n", ##__VA_ARGS__, __FILENAME__, __LINE__, __FUNCTION__); \
		krnlLOGPrintLog(type, ucLogInfo); \
	}while(0)

#ifdef __cplusplus
}
#endif
#endif //_krnl_LOG_H__

