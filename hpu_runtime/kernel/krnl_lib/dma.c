/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 14:35:14
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-11 10:09:22
 */
// #include "hikl_param.h"
#include "hihw.h"
#include "dma.h"
#include "libconv.h"

extern void disable_ndma_intr();
extern void enable_ndma_intr();

//8 channel, each channel: 0 idle, 1 busy
static ndma_idle_status ndma_channel_idle_status[HPU200_NDMA_CHANNEL_NUM] = {0};

//get the ndma channel index
uint8_t __ndma_get_index(){
    asm("fence");
    uint32_t ndma_status = 0; 
    csrr(ndma_status, CSR_ADDR_NDMA_STATUS);

    KRNL_LOG_INFO(LOG_NDMA, "ndma_channel_index: %d", ndma_status & NDMA_CHANNEL_INDEX_MASK);
    return ndma_status & NDMA_CHANNEL_INDEX_MASK;
}

void __poll_until_ndma_channel_available(){
    amo_struct *amo_mem = (amo_struct *)AMO_ADDR_S;
    KRNL_LOG_INFO(LOG_NDMA, "busy_ndma_channel_count: %d", amo_mem->busy_ndma_channel_count);
    while(amo_mem->busy_ndma_channel_count >= HPU200_NDMA_CHANNEL_NUM)
    {
        //loop
    };
}

uint32 __ndma_xfer(uint32 cur_noc_node_localaddr,                                   \
                    uint32 remote_noc_node_x,                                       \
                    uint32 remote_noc_node_y,                                       \
                    uint32 remote_noc_node_localaddr,                               \
                    uint32 ndma_xfer_size,                                          \
                    uint8_t xfer_cmd,                                               \
                    delegate_callback_for_each_ndma_channel delegate_callback,      \
                    uint32_t delegate_callback_param){

    if(ndma_xfer_size & 0x3f) // must align with 64 Byte
        return -1;
    else{

        __poll_until_ndma_channel_available();
        
        disable_ndma_intr();

        ndma_cfg(ndma_xy(remote_noc_node_x, remote_noc_node_y), remote_noc_node_localaddr, cur_noc_node_localaddr, ndma_xfer_size);
        ndma_start(xfer_cmd);
        uint8_t ndma_channel_index = __ndma_get_index();
        ndma_channel_idle_status[ndma_channel_index].idle_status = NDMA_CHANNEL_INDEX_BUSY;
        ndma_channel_idle_status[ndma_channel_index].ndma_xfer_cmd = xfer_cmd;
        ndma_channel_idle_status[ndma_channel_index].on_ndma_done_delegate = delegate_callback;
        ndma_channel_idle_status[ndma_channel_index].delegate_callback_param = delegate_callback_param;
        ((amo_struct *)AMO_ADDR_S)->busy_ndma_channel_count ++;
        
        enable_ndma_intr();
        return 0;
    }
}

uint32 __ndma_read(uint32 cur_noc_node_localaddr,                                   \
                    uint32 remote_noc_node_x,                                       \
                    uint32 remote_noc_node_y,                                       \
                    uint32 remote_noc_node_localaddr,                               \
                    uint32 ndma_xfer_size,                                          \
                    delegate_callback_for_each_ndma_channel delegate_callback,      \
                    uint32_t delegate_callback_param){
    KRNL_LOG_INFO(LOG_NDMA, "ndma_read to 0x%x from [%x][%x]0x%x xfer size: %d, delegate_callback 0x%x, delegate_param 0x%x",       \
                    cur_noc_node_localaddr,                         \
                    remote_noc_node_x,                              \
                    remote_noc_node_y,                              \
                    remote_noc_node_localaddr,                      \
                    ndma_xfer_size,                                 \
                    delegate_callback,                              \
                    delegate_callback_param);

    return __ndma_xfer(cur_noc_node_localaddr, remote_noc_node_x, remote_noc_node_y, remote_noc_node_localaddr, ndma_xfer_size, DMA_RX_CMD, delegate_callback, delegate_callback_param);
}

uint32 __ndma_write(uint32 cur_noc_node_localaddr,                                      \
                    uint32 remote_noc_node_x,                                           \
                    uint32 remote_noc_node_y,                                           \
                    uint32 remote_noc_node_localaddr,                                   \
                    uint32 ndma_xfer_size,                                              \
                    delegate_callback_for_each_ndma_channel delegate_callback,          \
                    uint32_t delegate_callback_param){
    
	KRNL_LOG_INFO(LOG_NDMA, "ndma_write from 0x%x to [%x][%x]0x%x xfer size: %d, delegate_callback 0x%x, delegate_param 0x%x",       \
                    cur_noc_node_localaddr,                         \
                    remote_noc_node_x,                              \
                    remote_noc_node_y,                              \
                    remote_noc_node_localaddr,                      \
                    ndma_xfer_size,                                 \
                    delegate_callback,                              \
                    delegate_callback_param);
    return __ndma_xfer(cur_noc_node_localaddr, remote_noc_node_x, remote_noc_node_y, remote_noc_node_localaddr, ndma_xfer_size, DMA_TX_CMD, delegate_callback, delegate_callback_param);
}

void ndma_callback(int ndma_status) {
    uint8_t ndma_done_status = ((ndma_status) & DMA_STATUS_MASK) >> DMA_STATUS_MASK_RIGHT_SHIFT_NUM;
    csrw(CSR_ADDR_NDMA_CTRL, ndma_status + NDMA_INTERRUPT_CLEAR_CMD);
    for (int ndma_channel_index = 0; ndma_channel_index < HPU200_NDMA_CHANNEL_NUM; ndma_channel_index ++)
    {
        if (ndma_done_status & (1 << ndma_channel_index))
        {
            KRNL_LOG_INFO(LOG_NDMA, "ndma_callback for index: %d", ndma_channel_index);
            ndma_channel_idle_status[ndma_channel_index].idle_status = NDMA_CHANNEL_INDEX_IDLE;
            ndma_channel_idle_status[ndma_channel_index].ndma_xfer_cmd = DMA_RESERVED_CMD;
            if (ndma_channel_idle_status[ndma_channel_index].on_ndma_done_delegate != NULL)
            {
                ndma_channel_idle_status[ndma_channel_index].on_ndma_done_delegate(ndma_channel_idle_status[ndma_channel_index].delegate_callback_param);
            }
            ndma_channel_idle_status[ndma_channel_index].on_ndma_done_delegate = NULL;
            ndma_channel_idle_status[ndma_channel_index].delegate_callback_param = 0;
            ((amo_struct *)AMO_ADDR_S)->busy_ndma_channel_count --;
        }
    }
    return;
}









void __ndma_poll(){
    // uint32 x = 0;
    // ndma_clr_signal_done(x);
    uint32 poll = 0;
    ndma_over_poll(poll);
}

void __rd_from_remote_var_non_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr){
    ndma_cfg(ndma_xy(rm_x, rm_y), rm_addr, var, sizeof(slot_var_in_local_variable));
    ndma_start(DMA_RX_CMD);
}

void __wr_to_remote_var_non_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr){
    ndma_cfg(ndma_xy(rm_x, rm_y), rm_addr, var, sizeof(slot_var_in_local_variable));
    ndma_start(DMA_TX_CMD);
}

uint32 __rd_from_remote_chunk_non_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr, uint32 size){
    if(size & 0x3f) // must align with 64 Byte
        return -1;
    else{

        vmu_poll();
        ndma_cfg(ndma_xy(rm_x, rm_y), rm_addr, var, size);
        ndma_start(DMA_RX_CMD);
        return 0;
    }
}

uint32 __wr_to_remote_chunk_non_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr, uint32 size){
    if(size & 0x3f)
        return -1;
    else{

        vmu_poll();
        ndma_cfg(ndma_xy(rm_x, rm_y), rm_addr, var, size);
        ndma_start(DMA_TX_CMD);
        return 0;
    }
}

void __rd_from_remote_var_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr){
    __rd_from_remote_var_non_blocking(var, rm_x, rm_y, rm_addr);
    __ndma_poll();
}

void __wr_to_remote_var_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr){
    __wr_to_remote_var_non_blocking(var, rm_x, rm_y, rm_addr);
    __ndma_poll();
}

uint32 __rd_from_remote_chunk_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr, uint32 size){

    if(__rd_from_remote_chunk_non_blocking(var, rm_x, rm_y, rm_addr, size))
        return -1;
    __ndma_poll();
    return 0;
}

uint32 __wr_to_remote_chunk_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 rm_addr, uint32 size){

    if(__wr_to_remote_chunk_non_blocking(var, rm_x, rm_y, rm_addr, size))
        return -1;
    __ndma_poll();
    return 0;
}

void __swp_rmt_lock_blocking(uint32 *var, uint32 rm_x, uint32 rm_y, uint32 lock_addr){
    ndma_cfg(ndma_xy(rm_x, rm_y), lock_addr, var, 4);
    ndma_start(DMA_SWAP);
    __ndma_poll();
}
