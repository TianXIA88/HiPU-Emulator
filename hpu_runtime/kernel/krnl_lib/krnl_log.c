
#include "krnl_log.h"

unsigned long g_ulPrintDebugLogFlag = 0;
unsigned long g_ulPrintHardwareCmdLogFlag = 0;
unsigned long g_ulPrintSYSLogFlag = 0;
unsigned long g_ulPrintNDMALogFlag = 0;

// unsigned long krnlLOGPrintLogTime(unsigned char *ucTime, unsigned long ulBufLen)
// {
//     struct tm *pstTmSec;
//     struct timeval stTmMsec;

//     if (NULL == ucTime)
//     {
//         return -1;
//     }
//     gettimeofday(&stTmMsec, NULL);
//     pstTmSec = localtime(&stTmMsec.tv_sec);
// #if 0
//     snprintf((char *)ucTime, ulBufLen - 1, "%04d-%02d-%02d %02d:%02d:%02d %03ldms",
//              pstTmSec->tm_year + 1900, pstTmSec->tm_mon + 1, pstTmSec->tm_mday, pstTmSec->tm_hour,
//              pstTmSec->tm_min, pstTmSec->tm_sec, stTmMsec.tv_usec / 1000);
// #endif
//     snprintf((char *)ucTime, ulBufLen - 1, "%02d/%02d %02d:%02d:%02d.%03ld",
//              pstTmSec->tm_mon + 1, pstTmSec->tm_mday, pstTmSec->tm_hour,
//              pstTmSec->tm_min, pstTmSec->tm_sec, stTmMsec.tv_usec / 1000);

//     return 0;
// }

unsigned long krnlLOGLogTypeToStr(unsigned char ucType, unsigned char *pucTypeString, unsigned long ulBufLen)
{
    if (NULL == pucTypeString)
    {
        return -1;
    }
    ulBufLen -= 1;

    switch (ucType)
    {
    case LOG_DEBUG:
    {
        strncpy((char *)pucTypeString, "DBG", ulBufLen);
        break;
    }
    case LOG_ERROR:
    {
        strncpy((char *)pucTypeString, "ERR", ulBufLen);
        break;
    }
    case LOG_WARNING:
    {
        strncpy((char *)pucTypeString, "WAR", ulBufLen);
        break;
    }
    case LOG_ACTION:
    {
        strncpy((char *)pucTypeString, "ACT", ulBufLen);
        break;
    }
    case LOG_SYSTEM:
    {
        strncpy((char *)pucTypeString, "SYS", ulBufLen);
        break;
    }
    case LOG_HARDWARE_CMD:
    {
        strncpy((char *)pucTypeString, "HARDWARE_CMD", ulBufLen);
        break;
    }
    case LOG_NDMA:
    {
        strncpy((char *)pucTypeString, "NDMA", ulBufLen);
        break;
    }
    default:
    {
        strncpy((char *)pucTypeString, "UKN", ulBufLen);
        break;
    }
    }
    return 0;
}

unsigned long krnlLOGPrintLog(unsigned char ucType, unsigned char *pucLogInfo)
{
    unsigned long ulResult = 0;
    unsigned long ulFileLen = 0;
    unsigned char ucTime[STR_COMM_SIZE] = { 0 };
    unsigned char ucLogTypeStr[STR_COMM_SIZE] = { 0 };
    unsigned char ucLogInfo[STR_MAX_SIZE] = { 0 };

    if (NULL == pucLogInfo)
    {
        return -1;
    }

    ulResult = krnlLOGLogTypeToStr(ucType, ucLogTypeStr, sizeof(ucLogTypeStr));
    // ulResult += krnlLOGPrintLogTime(ucTime, sizeof(ucTime));
    if (0 != ulResult)
    {
        return -1;
    }
    snprintf((char *)ucLogInfo, sizeof(ucLogInfo) - 1, "[__%s__][%s] %s", __TIME__, ucLogTypeStr, pucLogInfo);
    printf("%s", ucLogInfo);
    return 0;
}


