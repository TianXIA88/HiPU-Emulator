/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2021-02-24 09:27:21
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-03-29 14:41:56
 */
#ifndef _HISDK_PORT_H__
#define _HISDK_PORT_H__

#include <stdlib.h>
#include <stddef.h>
#include <dirent.h>
#include "hisdk_type.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HISDK_OPEN_WRITE     (0x1)
#define HISDK_OPEN_READ      (0x2)
#define HISDK_OPEN_CREATE    (0x4)
#define HISDK_OPEN_APPEND    (0x8)

/*
 * Thread structures
 */
struct hisdkThreadRec 
{
    void *handle;
};
typedef struct hisdkThreadRec    hisdkThread_t;
typedef struct hisdkThreadRec*   hisdkThreadHandle_t;
typedef void (*hisdkThreadFunction_t)(void *args);

/*
 * Files and directory structures.
 */
typedef enum 
{
    HisdkFileType_Unknown = 0,
    HisdkFileType_File,
    HisdkFileType_Directory,
    HisdkFileType_Fifo,
    HisdkFileType_CharacterDevice,
    HisdkFileType_BlockDevice,

    HisdkFileType_Force32 = 0x7FFFFFFF
} hisdkFileType_t;

typedef struct
{
    u64_t size;
    hisdkFileType_t type;
    u64_t mtime;
} hisdkStatType_t;

typedef enum hisdkSeek 
{
    HisdkSeek_Set,
    HisdkSeek_Cur,
    HisdkSeek_End
} hisdkSeekEnum_t;

typedef struct 
{
    int fd;
} hisdkFile_t;

typedef struct 
{
    DIR *dir;
} hisdkDir_t;


void      *hisdkZAlloc(size_t size);
void      *hisdkAlloc(size_t size);
void       hisdkFree(void *ptr);
void      *hisdkMemset(void *str, int c, size_t n);
void      *hisdkMemcpy(void *destin, void *source, size_t n);
/*
 * File and directory operations
 */
hisdkRet_t hisdkPortOsStat(const char *filename, hisdkStatType_t *stat);
hisdkRet_t hisdkPortOsMkdir(char *dirname);
hisdkRet_t hisdkPortOsRm(const char *filename);
hisdkRet_t hisdkPortOsFopen(const char *path, u32_t flags, hisdkFile_t * *file);
void       hisdkPortOsFclose(hisdkFile_t *file);
hisdkRet_t hisdkPortOsFwrite(hisdkFile_t *file, const void *ptr, size_t size);
hisdkRet_t hisdkPortOsFread(hisdkFile_t *file, void *ptr, size_t size, size_t *bytes);
hisdkRet_t hisdkPortOsFseek(hisdkFile_t *file, i64_t offset, hisdkSeekEnum_t whence);
hisdkRet_t hisdkPortOsFstat(hisdkFile_t *file, hisdkStatType_t *stat);
u64_t      hisdkPortOsStatGetSize(hisdkStatType_t *stat);
hisdkRet_t hisdkPortOsFgetc(hisdkFile_t *file, u8_t *c);
void       hisdkPortOsMemset(void *s, u8_t c, size_t size);
hisdkRet_t hisdkPortOsOpenDir(const char *path, hisdkDir_t **dir);
hisdkRet_t hisdkPortOsReadDir(hisdkDir_t *dir, char *name, size_t size);
void       hisdkPortOsCloseDir(hisdkDir_t *dir);
void       hisdkPortOsDebugPrintf( const char *format, ... );
int        hisdkPortOsGetFileSize(const char *filename);

#ifdef __cplusplus
}
#endif
#endif /*_HISDK_PORT_H__*/