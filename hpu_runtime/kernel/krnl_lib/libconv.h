/*
 * @Descripttion: 
 * @version: 
 * @Author: AlonzoChen
 * @Date: 2020-12-29 14:35:14
 * @LastEditors: AlonzoChen
 * @LastEditTime: 2021-04-22 19:37:44
 */
#ifndef LIBCONV_H
#define LIBCONV_H

#include "hisdk_type.h"
#include "operators/hi_krnl_param_conv2d.h"

typedef struct{
    uint32 value;
    uint32 reserved[7];
}slot_var_in_local_variable;

typedef struct{
    uint32 start_addr;
    uint32 end_addr;
    uint32 size;            // in 32-Bytes
    uint32 cur_cnt;
}local_variable;

typedef struct{
    uint32 start_addr;
    uint32 end_addr;
    uint32 row_slots_num;         // in rows
    uint32 row_size;        // in Bytes
    uint32 total_cnt;       // record the number of previous remote allocation 
} base_fm;

typedef struct {
    base_fm bfm;
    uint32 *row_slot_available_flgs;
    uint32 cur_cnt;         // how many valid rows are in current buffer
    uint32 cur_valid_idx;   // the index of current first valid row (range: 0 -> row_slots_num-1)
}local_fm;

typedef struct {
    base_fm bfm;
    hikl_addr_t var_in_remote_node_saving_flags;  // base address of flags of remote node
    uint32 *local_var_for_interact_with_remote_vars;     // a local var to interact with var of remote node
}remote_fm;

typedef struct {
    base_fm bfm;
    uint8   x_pos;
    uint8   y_pos;
}ddr_fm;

uint32 is_ifm_bottom(uint32 i, conv_shape_t *cshape, pad_shape_t *pshape);
uint32 is_ifm_top(uint32 i, conv_shape_t *cshape, pad_shape_t *pshape);
uint32 get_ifm_row_size(conv_shape_t* p_cshape, stride_shape_t *p_strd_shape);
uint32 get_ofm_row_size(conv_shape_t* p_cshape, stride_shape_t *p_strd_shape);

void init_local_var(local_variable *lvar, uint32 start_addr, uint32 end_addr, uint32 size);
uint32 alloc_var(local_variable *v, uint32 var_num);
void init_base_fm(base_fm* fm, uint32 row_slots_num, uint32 row_size, uint32 start);
uint32 alloc_fm(base_fm *fm, uint32 *idx);
void init_ddr_fm(ddr_fm *fm, uint32 row_slots_num, uint32 row_size, uint32 start, uint8 x, uint8 y);
uint32 alloc_ddr_fm(ddr_fm *fm);
void init_local_fm(local_fm* fm, local_variable* lvar, uint32 local_row_slots, uint32 row_size, uint32 start);
uint32 alloc_local_fm(local_fm *fm, uint32 num);
void dealloc_local_fm(local_fm *fm, uint32 num);
uint32 get_cur_valid_local_fm_addr(local_fm *fm);
void update_local_fm(local_fm *fm);
void init_remote_fm(remote_fm* fm, local_variable* lvar, uint32 row_slots_num, uint32 row_size, uint32 start, uint8 x, uint8 y, uint32 flag_addr);
uint32 alloc_remote_fm(remote_fm *fm, uint32 num);
void notify_remote_fm_ready(remote_fm *fm, uint32 fm_addr);

void init_local_fm_lock(local_fm* fm);
void acquire_local_fm_lock(uint32 row_slots_num);
void release_local_fm_lock(uint32 row_slots_num);
void acquire_remote_fm_lock(remote_fm* fm, uint32 row_slots_num);
void release_remote_fm_lock(remote_fm* fm, uint32 row_slots_num);
void acquire_flag_in_remote_node_blocking(remote_fm *fm, uint32 row_slots_num);
int check_flag_in_remote_node(remote_fm *fm, uint32 row_slots_num);
int _ndma_one_fm_row_from_localmem_to_remote_localmem_blocking(local_fm *ofm, remote_fm *rfm);
int _ndma_one_fm_row_from_localmem_to_ddr_blocking(local_fm *ofm, ddr_fm *dfm);
int _ndma_remain_ifm_rows_from_localmem_to_ddr_blocking(local_fm *ofm, ddr_fm *dfm);


// void set_conv_type(uint32_t input_conv_type);
// void __intrinsic_func__(uint32 ifm, uint32 wt, uint32 ofm, uint32 bias_start, uint32 shift_start, uint32 relu, int conv_type);
// void set_mmac_param_for_whole_conv(conv2d_params_t *conv2d_param);
// void set_mmac_cluster_params(conv2d_params_t *conv2d_param, uint32_t h_iter);
// uint32_t set_mmac_region_params_and_get_ifm_start(conv2d_params_t *conv2d_param, uint32_t w_iter);
// void set_wt_offset(conv2d_params_t *conv2d_param, uint32_t h_iter);

void one_row_conv(int h_iter, conv2d_params_t *conv2d_param,        	  	    \
						uint32_t wt_lcmem_start_addr,							\
						uint32_t ofm_row_lcmem_start_addr,						\
						uint32_t bias_lcmem_start_addr,							\
						uint32_t shift_lcmem_start_addr,						\
                    	uint32_t param_mmac_region_start,                     	\
                    	uint32_t param_mmac_region_end,                       	\
                    	uint32_t param_mmac_ifm_cluster_stride,               	\
                    	uint32_t param_mmac_cluster_start,                    	\
                    	uint32_t param_mmac_cluster_end,                      	\
                    	uint32_t param_mmac_cluster_num,                        \
						uint32_t input_conv_type,                               \
						uint32_t b_with_bias_shift);

void one_row_conv_add(
    int h_iter, conv2d_params_t *conv2d_par,             
    uint32_t wt_lcmem_start_addr,			             
    uint32_t ofm_row_lcmem_start_addr,		             
    uint32_t bias_lcmem_start_addr,			             
    uint32_t shift_lcmem_start_addr,		             
  	uint32_t param_mmac_region_start,      	             
  	uint32_t param_mmac_region_end,        	             
  	uint32_t param_mmac_ifm_cluster_stride,	             
  	uint32_t param_mmac_cluster_start,     	             
  	uint32_t param_mmac_cluster_end,       	             
  	uint32_t param_mmac_cluster_num,                     
    uint32_t input_conv_type,                            
    uint32_t b_with_bias_shift,uint32_t ifm_ptr_c_add, uint32_t add_shift, uint32_t add_clip);
#endif /*LIBCONV_H*/
