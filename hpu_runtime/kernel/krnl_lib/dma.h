/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 14:35:14
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-07 19:36:45
 */
#ifndef LIBDMA_H
#define LIBDMA_H

#include "hisdk_type.h"
#include "hihw.h"

#define DMA_TX_CMD                  (0)
#define DMA_RX_CMD                  (1)
#define DMA_SWAP                    (2)
#define DMA_RESERVED_CMD            (0xf)
#define DMA_STATUS_MASK             (0xff000000)            //bit[24:31] 8 bits, one bit for one channel bit[24] <--> channel index 0
#define DMA_STATUS_MASK_RIGHT_SHIFT_NUM        (24)            
#define NDMA_CHANNEL_INDEX_MASK         (0x7)           //bit[2:0] 8 ndma channel
#define NDMA_CHANNEL_INDEX_IDLE     (0)
#define NDMA_CHANNEL_INDEX_BUSY     (1)
#define NDMA_INTERRUPT_CLEAR_CMD    (3)

// 00: tx; 01: rx; 10:swap; 11:clear
#define ndma_xy(x, y)           (((x)&0x3)<<2)+((y)&0x3)
#define ndma_start(x)           csrw(CSR_ADDR_NDMA_CTRL, x)
#define ndma_over_poll(x)       do{                                     \
                                    csrr(x, CSR_ADDR_NDMA_STATUS);      \
                                }while(((x) & DMA_STATUS_MASK) == 0);   \
                                x = 0xffffffff;                         \
                                csrw(CSR_ADDR_NDMA_CTRL, x)

#define ndma_clr_signal_done(x) x = 0xff000003;                         \
                                csrw(CSR_ADDR_NDMA_CTRL, x)

#define ndma_cfg(xy, r, l, s)   csrw(CSR_ADDR_NDMA_RTADDR, (r));    \
                                csrw(CSR_ADDR_NDMA_DESTXY, (xy));   \
                                csrw(CSR_ADDR_NDMA_LCADDR, (l));    \
                                csrw(CSR_ADDR_NDMA_SIZE, (s));  

typedef void (*delegate_callback_for_each_ndma_channel)(uint32_t param);
typedef struct{
    uint8 ndma_xfer_cmd:4;
    //8 channel, each channel: 0 idle, 1 busy
    uint8 idle_status:4;
    uint32_t delegate_callback_param;
    delegate_callback_for_each_ndma_channel on_ndma_done_delegate;
}ndma_idle_status;

void ndma_callback(int ndma_status);
uint32 __ndma_read(uint32 cur_noc_node_localaddr,                                   \
                    uint32 remote_noc_node_x,                                       \
                    uint32 remote_noc_node_y,                                       \
                    uint32 remote_noc_node_localaddr,                               \
                    uint32 ndma_xfer_size,                                          \
                    delegate_callback_for_each_ndma_channel delegate_callback,      \
                    uint32_t delegate_callback_param);

uint32 __ndma_write(uint32 cur_noc_node_localaddr,                                      \
                    uint32 remote_noc_node_x,                                           \
                    uint32 remote_noc_node_y,                                           \
                    uint32 remote_noc_node_localaddr,                                   \
                    uint32 ndma_xfer_size,                                              \
                    delegate_callback_for_each_ndma_channel delegate_callback,          \
                   uint32_t delegate_callback_param);

void __ndma_poll();
void __rd_from_remote_var_non_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr);
void __wr_to_remote_var_non_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr);
uint32 __rd_from_remote_chunk_non_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr, uint32 size);
uint32 __wr_to_remote_chunk_non_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr, uint32 size);
void __rd_from_remote_var_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr);
void __wr_to_remote_var_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr);
uint32 __rd_from_remote_chunk_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr, uint32 size);
uint32 __wr_to_remote_chunk_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr, uint32 size);
void __swp_rmt_lock_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 lock_addr);



#endif /*LIBDMA_H*/