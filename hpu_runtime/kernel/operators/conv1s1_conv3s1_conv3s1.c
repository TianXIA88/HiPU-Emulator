#include "hihw.h"
#include "libconv.h"
#include "dma.h"
#include "lock.h"
#include "int.h"
#include "operators/hi_krnl_param_conv2d.h"
#include "operators/hi_krnl_param_conv1s1_conv3s1_conv3s1.h"
#include "hi_addr_def.h"
#include "krnl_log.h"
#include "hpu_util.h"
// #include "qemu.h"

#define LCMEM_CONV_A_IN0    MMA_BANK0_START_ADDR
#define LCMEM_CONV_A_IN1    MMA_BANK1_START_ADDR
#define LCMEM_CONV_BC_IN0   MMA_BANK2_START_ADDR
#define LCMEM_CONV_BC_IN1   MMA_BANK3_START_ADDR
#define LCMEM_CONV_BC_IN2   MMA_BANK4_START_ADDR
#define LCMEM_CONV_B_OUT    MMA_BANK5_START_ADDR
#define LCMEM_CONV_C_OUT    MMA_BANK6_START_ADDR
#define LCMEM_BIAS_SHIFT    MMA_BANK7_START_ADDR

static u32_t localmem_fm_table_conv_a[] = 
{
    LCMEM_CONV_A_IN0,
    LCMEM_CONV_A_IN1
};
static u32_t localmem_fm_index_conv_a = 0;
static const u32_t localmem_fm_num_conv_a = sizeof(localmem_fm_table_conv_a) / sizeof(u32_t);

static u32_t localmem_fm_table_conv_bc[] = 
{
    LCMEM_CONV_BC_IN0,
    LCMEM_CONV_BC_IN1,
    LCMEM_CONV_BC_IN2
};
static u32_t localmem_fm_index_conv_bc = 0;
static const u32_t localmem_fm_num_conv_bc = sizeof(localmem_fm_table_conv_bc) / sizeof(u32_t);

static u32_t localmem_fm_conv_b_out[] = 
{
    LCMEM_CONV_B_OUT
};

static u32_t localmem_fm_conv_c_out[] = 
{
    LCMEM_CONV_C_OUT
};

void _op_conv1s1_conv3s1_conv3s1 (   
    conv2d_params_t *conv1,
	conv2d_params_t *conv2,
	conv2d_params_t *conv3,

	hikl_addr_t *ifm_addr,
    hikl_addr_t *ofm_addr_b_cls1,
    hikl_addr_t *ofm_addr_c_reg1,

    hikl_addr_t *wt_addr_a, 
    hikl_addr_t *wt_addr_b, 
    hikl_addr_t *wt_addr_c, 

    hikl_addr_t *bs_addr_a, 
    hikl_addr_t *bs_addr_b, 
    hikl_addr_t *bs_addr_c, 

    hikl_addr_t *shift_addr_a,
    hikl_addr_t *shift_addr_b,
    hikl_addr_t *shift_addr_c
);

inline void set_debug_flag(int debug,int cmd,int sys,int dma ){
	g_ulPrintDebugLogFlag = debug;
	g_ulPrintHardwareCmdLogFlag = cmd;
	g_ulPrintSYSLogFlag = sys;
	g_ulPrintNDMALogFlag = dma;
}

void kernel_conv1s1_conv3s1_conv3s1( ){
	paramTableConv1s1_conv3s1_conv3s1_t *_pParamTable = *((paramTableConv1s1_conv3s1_conv3s1_t **)HIPU200_KNL_PTABLE_ADDR);/*get kernel param table from runtime*/
	paramTableConv1s1_conv3s1_conv3s1_Entry_t *p_op_entry = &_pParamTable->param;
   	KRNL_LOG_INFO(LOG_SYSTEM, "%s","=================== kernel =========================");
	
	_op_conv1s1_conv3s1_conv3s1(
		&p_op_entry->conv1,
		&p_op_entry->conv1a_c1,
		&p_op_entry->conv1b_r1,

		&p_op_entry->ifm_addr_conv1,
		&p_op_entry->ofm_addr_conv1a_c1,
		&p_op_entry->ofm_addr_conv1b_r1,

		&p_op_entry->wt_addr_conv1,
		&p_op_entry->wt_addr_conv1a_c1,
		&p_op_entry->wt_addr_conv1b_r1,

		&p_op_entry->bias_addr_conv1,
		&p_op_entry->bias_addr_conv1a_c1,
		&p_op_entry->bias_addr_conv1b_r1,

		&p_op_entry->shift_addr_conv1,
		&p_op_entry->shift_addr_conv1a_c1,
		&p_op_entry->shift_addr_conv1b_r1
	);
};

void _op_conv1s1_conv3s1_conv3s1 (   
    conv2d_params_t *conv1,
	conv2d_params_t *conv2,
	conv2d_params_t *conv3,

	hikl_addr_t *ifm_addr,
    hikl_addr_t *ofm_addr_b_cls1,
    hikl_addr_t *ofm_addr_c_reg1,

    hikl_addr_t *wt_addr_a, 
    hikl_addr_t *wt_addr_b, 
    hikl_addr_t *wt_addr_c, 

    hikl_addr_t *bs_addr_a, 
    hikl_addr_t *bs_addr_b, 
    hikl_addr_t *bs_addr_c, 

    hikl_addr_t *shift_addr_a,
    hikl_addr_t *shift_addr_b,
    hikl_addr_t *shift_addr_c
) {
	conv_shape_t *cshape_a = &(conv1->cshape); 
    conv_shape_t *cshape_b = &(conv2->cshape); 
    conv_shape_t *cshape_c = &(conv3->cshape); 
	bool relu_a = conv1->relu; 
    bool relu_b = conv2->relu;
    bool relu_c = conv3->relu;
	KRNL_LOG_INFO(LOG_SYSTEM, "%s","=========================into ============ kernel =========================");
	
    uint32 i, j, ndma_poll;
	uint32 h_iter_num_b;
	uint32 ifm_ptr;
	uint32 ofm_ptr;
	
	uint32 ifm_row_oneline_mrlen_a, ofm_row_oneline_mrlen_a = 0;
	uint32 ifm_row_oneline_mrlen_b, ofm_row_oneline_mrlen_b = 0;
	uint32 ifm_row_oneline_mrlen_c, ofm_row_oneline_mrlen_c = 0;
	
	uint32 ifm_c_group8_num_a, wt_cluster_size_a, cluster_num_a = 0;
	uint32 w_iter_num_a, kernel_group8_num_a = 0;
	uint32 wt_offset_a, cluster_start_a, cluster_end_a, wt_ptr_a, shift_ptr_a, bs_ptr_a, ifm_ptr_a, ofm_ptr_a = 0;

	uint32 ifm_c_group8_num_b, wt_cluster_size_b, cluster_num_b = 0;
	uint32 w_iter_num_b, kernel_group8_num_b = 0;
	uint32 wt_offset_b,  cluster_start_b, cluster_end_b,wt_ptr_b, shift_ptr_b, bs_ptr_b, ifm_ptr_b,ofm_ptr_b = 0;
	ofm_ptr_b = ofm_addr_b_cls1->lcaddr;

	uint32 ifm_c_group8_num_c, wt_cluster_size_c, cluster_num_c = 0;
	uint32 w_iter_num_c, kernel_group8_num_c = 0;
	uint32 wt_offset_c, cluster_start_c, cluster_end_c,wt_ptr_c, shift_ptr_c, bs_ptr_c, ifm_ptr_c, ofm_ptr_c = 0;
	ofm_ptr_c = ofm_addr_c_reg1->lcaddr;

	uint8  round_type, shift_num, prot_high, prot_low =0;
	
	uint32 wt_sz_a, bs_sz_a, shift_sz_a;
	uint32 wt_sz_b, bs_sz_b, shift_sz_b;
	uint32 wt_sz_c, bs_sz_c, shift_sz_c;
	uint32 wt_start_a , bs_start_a, shift_start_a;
	uint32 wt_start_b , bs_start_b, shift_start_b;
	uint32 wt_start_c , bs_start_c, shift_start_c;

    // Calculate total weight size
	wt_sz_a = cshape_a->k_w * cshape_a->k_h * cshape_a->ifm_c * cshape_a->ofm_c; // in Bytes
	wt_sz_b = cshape_b->k_w * cshape_b->k_h * cshape_b->ifm_c * cshape_b->ofm_c; // in Bytes
	wt_sz_c = cshape_c->k_w * cshape_c->k_h * cshape_c->ifm_c * cshape_c->ofm_c; // in Bytes
	//KRNL_LOG_INFO(LOG_DEBUG, "wt_sz %d \n", wt_sz);
	bs_sz_a = cshape_a->ofm_c * MTX_SCALE * 4;
	bs_sz_b = cshape_b->ofm_c * MTX_SCALE * 4;
	bs_sz_c = cshape_c->ofm_c * MTX_SCALE * 4;
	shift_sz_a = cshape_a->ofm_c * MTX_SCALE;
	shift_sz_b = cshape_b->ofm_c * MTX_SCALE;
	shift_sz_c = cshape_c->ofm_c * MTX_SCALE;
	wt_start_a = MMB_START_ADDR;
    wt_start_b = wt_start_a + wt_sz_a;
    wt_start_c = wt_start_b + wt_sz_b;
    bs_start_a = LCMEM_BIAS_SHIFT;
    bs_start_b = bs_start_a + bs_sz_a;
    bs_start_c = bs_start_b + bs_sz_b;
    shift_start_a = bs_start_c + bs_sz_c;
    shift_start_b = shift_start_a + shift_sz_a;
    shift_start_c = shift_start_b + shift_sz_b;

    // Load all weights
	//KRNL_LOG_INFO(LOG_DEBUG, "Load Weights: [%d%d%x]->[%x]\n\r", wt_addr->x_pos, wt_addr->y_pos, wt_addr->lcaddr, (uint32 *)MMB_ADDR);
	//MMB can't be access by HPU scalar ALU
	LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)wt_start_a, wt_addr_a->x_pos, wt_addr_a->y_pos, wt_addr_a->lcaddr,  GMEM_ALIGN(wt_sz_a)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)wt_start_b, wt_addr_b->x_pos, wt_addr_b->y_pos, wt_addr_b->lcaddr,  GMEM_ALIGN(wt_sz_b)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)wt_start_c, wt_addr_c->x_pos, wt_addr_c->y_pos, wt_addr_c->lcaddr,  GMEM_ALIGN(wt_sz_c)));

	// Load all bs
	//KRNL_LOG_INFO(LOG_DEBUG, "Load Bias: [%x]->[%x]\n\r", bias_addr->lcaddr, (uint32 *)(BIAS_SHIFT_BLK));
	LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)bs_start_a, bs_addr_a->x_pos, bs_addr_a->y_pos, bs_addr_a->lcaddr, GMEM_ALIGN(bs_sz_a)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)bs_start_b, bs_addr_b->x_pos, bs_addr_b->y_pos, bs_addr_b->lcaddr, GMEM_ALIGN(bs_sz_b)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)bs_start_c, bs_addr_c->x_pos, bs_addr_c->y_pos, bs_addr_c->lcaddr, GMEM_ALIGN(bs_sz_c)));
    
	// Load all shift_num
	//KRNL_LOG_INFO(LOG_DEBUG, "Load Shift_mtx: [%x]->[%x]\n\r", shift_addr->lcaddr, (uint32 *)(BIAS_SHIFT_BLK + bs_sz));
	LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)shift_start_a, shift_addr_a->x_pos, shift_addr_a->y_pos, shift_addr_a->lcaddr, GMEM_ALIGN(shift_sz_a)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)shift_start_b, shift_addr_b->x_pos, shift_addr_b->y_pos, shift_addr_b->lcaddr, GMEM_ALIGN(shift_sz_b)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)shift_start_c, shift_addr_c->x_pos, shift_addr_c->y_pos, shift_addr_c->lcaddr, GMEM_ALIGN(shift_sz_c)));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Calculate total iteration
	kernel_group8_num_a = cshape_a->ofm_c / MTX_SCALE;			
	kernel_group8_num_b = cshape_b->ofm_c / MTX_SCALE;			
	kernel_group8_num_c = cshape_c->ofm_c / MTX_SCALE;			
	w_iter_num_a = cshape_a->ifm_w / MTX_SCALE; 	        	
	w_iter_num_b = cshape_b->ifm_w / MTX_SCALE; 	        	
	w_iter_num_c = cshape_c->ifm_w / MTX_SCALE; 	        	
	ifm_c_group8_num_a = cshape_a->ifm_c / MTX_SCALE;		    
	ifm_c_group8_num_b = cshape_b->ifm_c / MTX_SCALE;		    
	ifm_c_group8_num_c = cshape_c->ifm_c / MTX_SCALE;		    
	ifm_row_oneline_mrlen_a = ByteToW64(cshape_a->ifm_w * cshape_a->ifm_c);
	ifm_row_oneline_mrlen_b = ByteToW64(cshape_b->ifm_w * cshape_b->ifm_c);
	ifm_row_oneline_mrlen_c = ByteToW64(cshape_b->ifm_w * cshape_c->ifm_c);
	ofm_row_oneline_mrlen_a = ByteToW64(cshape_a->ifm_w * cshape_a->ofm_c);
	ofm_row_oneline_mrlen_b = ByteToW64(cshape_b->ifm_w * cshape_b->ofm_c);
	ofm_row_oneline_mrlen_c = ByteToW64(cshape_a->ifm_w * cshape_c->ofm_c);
	wt_cluster_size_a = ifm_c_group8_num_a * cshape_a->k_w;
	wt_cluster_size_b = ifm_c_group8_num_b * cshape_b->k_w;
	wt_cluster_size_c = ifm_c_group8_num_c * cshape_b->k_w;
	h_iter_num_b = cshape_b->ifm_h /*+ pshape->top + pshape->bottom 2 - cshape_b->k_h + 1*/;
	// set_debug_flag(1,1,1,0);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	KRNL_LOG_INFO(LOG_SYSTEM, "%s","======01 pre compute start======");
    ifm_ptr = ifm_addr->lcaddr;
    for (int i = 0; i < 2; i++) {
		KRNL_LOG_INFO(LOG_SYSTEM, " pre_num : %d , ======01 pre compute start======", i);
		localmem_fm_index_conv_a  = (i + 1) % 2;     // 取到循环中对应的那个bank 0 or 1
		localmem_fm_index_conv_bc = (i + 1) % 3;
		LIBHIKL_NASSERT(__rd_from_remote_chunk_non_blocking((uint32 *)localmem_fm_table_conv_a[localmem_fm_index_conv_a], 
		ifm_addr->x_pos, ifm_addr->y_pos, ifm_ptr, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a))));
        __ndma_poll();
		ifm_ptr += W64ToByte(ifm_row_oneline_mrlen_a);

		cluster_start_a = ByteToW64(localmem_fm_table_conv_a[localmem_fm_index_conv_a] - MMA_START_ADDR);
		cluster_end_a = cluster_start_a + ifm_row_oneline_mrlen_a;       // always the end of the first ifm row
		cluster_num_a = 0; // always less than the full kernel height
		KRNL_LOG_INFO(LOG_DEBUG, " ==== cluster_start_a : %x", cluster_start_a);
		one_row_conv(i, conv1,        	  	    										      
					ByteToW64(wt_start_a-MMA_START_ADDR),											      
					ByteToW64(localmem_fm_table_conv_bc[localmem_fm_index_conv_bc] - MMA_START_ADDR),         
					ByteToW64(bs_start_a-MMA_START_ADDR),											      
					ByteToW64(shift_start_a-MMA_START_ADDR),										      
                    ByteToW64(localmem_fm_table_conv_a[0] - MEM_LCMEM_ADDR_S),	                     	  
                    ByteToW64(localmem_fm_table_conv_a[0] + localmem_fm_num_conv_a * MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),	  
                   	0,               								                                      
                    cluster_start_a,                    											      
                    cluster_end_a,                      											      
                    cluster_num_a,                                                                        
					CONV_TYPE_CLASSIC, 1);
	}
   	KRNL_LOG_INFO(LOG_SYSTEM, "%s","======00 pre compute over======");
	//test:
	// h_iter_num_b  = 10;
	
	KRNL_LOG_INFO(LOG_SYSTEM, "%s","======00 conv_bc loop start======");
	// set_debug_flag(0,0,0,0);
    for (int i = 0; i < h_iter_num_b; i++) {
		localmem_fm_index_conv_a  = (i + 1) % 2;
		KRNL_LOG_INFO(LOG_DEBUG, "=====ifm H iter: %d / %d=====\n\r", i, h_iter_num_b);
		if( i < h_iter_num_b - 1 ){ //只要不是最后一行
			//KRNL_LOG_INFO(LOG_DEBUG, "Load ifm row %d: [%x]->[%x]\n\r", (ifm->bfm.total_cnt - 1), ddr_mem_fm_addr, local_mem_fm_addr);
			LIBHIKL_NASSERT(__rd_from_remote_chunk_non_blocking((uint32 *)localmem_fm_table_conv_a[localmem_fm_index_conv_a], ifm_addr->x_pos, ifm_addr->y_pos, 
			ifm_ptr, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a))));
			ifm_ptr += W64ToByte(ifm_row_oneline_mrlen_a);
			__ndma_poll();
		}
		// else{
		// 	ifm_ptr += W64ToByte(ifm_row_oneline_mrlen_a);
		// }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		if(i < /*pshape->top*/1){   //如果当前是padding行 需要单独设置mmac参数 涉及到fm的跳转和wt的跳转
			// cluster_start = ByteToW64(MMA_BEGIN);
			// always start from the first ifm row, pad automatically increments inside mmac
			cluster_start_b = ByteToW64(localmem_fm_table_conv_bc[1] - MMA_START_ADDR); //指定bank3为起点。只在3 4 跳，硬件自动识别padding
			cluster_end_b = cluster_start_b + ifm_row_oneline_mrlen_b;        // always the end of the first ifm row
			cluster_num_b = i + cshape_b->k_h - 1 - /*pshape->top*/conv2->pshape.top;     // always less than the full kernel height
		}
		else{    //如果是最后一行，cluster需要特例设置；如果在中间，不需要
			localmem_fm_index_conv_bc = i % 3 ;// 0 -> 234 ; 1 -> 342
			cluster_start_b = ByteToW64(localmem_fm_table_conv_bc[localmem_fm_index_conv_bc] - MMA_START_ADDR);
			cluster_end_b = cluster_start_b + ifm_row_oneline_mrlen_b;
			if((i + cshape_b->k_h) > (cshape_b->ifm_h + /*pshape->top*/1))   // at bottom
				cluster_num_b = cshape_b->ifm_h - i;   // always less than the full kernel height
			else // at middle
				cluster_num_b = cshape_b->k_h - 1;   // always equal the full kernel height
		}
		if( i >= h_iter_num_b - 1 ) set_debug_flag(1,1,1,0);
		KRNL_LOG_INFO(LOG_DEBUG, "cluster_start: [%x] cluster_end: [%x]", cluster_start_b, cluster_end_b);

		KRNL_LOG_INFO(LOG_SYSTEM, "%s","======02 conv_b compute start======");
		one_row_conv(i, conv2,        	  	    								
					ByteToW64(wt_start_b - MMA_START_ADDR),									
					ByteToW64(localmem_fm_conv_b_out[0] - MMA_START_ADDR),							
					ByteToW64(bs_start_b - MMA_START_ADDR),									
					ByteToW64(shift_start_b - MMA_START_ADDR),									
                    ByteToW64(localmem_fm_table_conv_bc[0] - MEM_LCMEM_ADDR_S),	                     	
                    ByteToW64(localmem_fm_table_conv_bc[0] + localmem_fm_num_conv_bc * MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),	    
                   	ByteToW64(MMA_BANK_SIZE),         										
                    cluster_start_b,                    									
                    cluster_end_b,                      									
                    cluster_num_b,
					CONV_TYPE_CLASSIC, 1);
		// //st result
        LIBHIKL_NASSERT(__wr_to_remote_chunk_non_blocking((uint32 *)localmem_fm_conv_b_out[0], ofm_addr_b_cls1->x_pos, ofm_addr_b_cls1->x_pos, ofm_ptr_b, (W64ToByte(ofm_row_oneline_mrlen_b))));
        ofm_ptr_b += W64ToByte(ofm_row_oneline_mrlen_b);
		__ndma_poll();
		
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		if(i < /*pshape->top*/1){   //如果当前是padding行 需要单独设置mmac参数 涉及到fm的跳转和wt的跳转
            cluster_start_c = ByteToW64(localmem_fm_table_conv_bc[1] - MMA_START_ADDR); //指定bank3为起点。只在3 4 跳，硬件自动识别padding
			cluster_end_c = cluster_start_c + ifm_row_oneline_mrlen_c;        // always the end of the first ifm row
			cluster_num_c = i + cshape_c->k_h - 1/*pshape->top*/ - conv3->pshape.top;     // always less than the full kernel height
		}
		else{    //如果是最后一行，cluster需要特例设置；如果在中间，不需要
			cluster_start_c = ByteToW64(localmem_fm_table_conv_bc[localmem_fm_index_conv_bc] - MMA_START_ADDR);
			cluster_end_c = cluster_start_c + ifm_row_oneline_mrlen_c;
			if((i + cshape_c->k_h) > (cshape_c->ifm_h + /*pshape->top*/1))   // at bottom
				cluster_num_c = cshape_c->ifm_h - i;   // always less than the full kernel height
			else // at middle
				cluster_num_c = cshape_c->k_h - 1;   // always equal the full kernel height
		}
		KRNL_LOG_INFO(LOG_DEBUG, "cluster_start: [%x] cluster_end: [%x]", cluster_start_c, cluster_end_c);
		
		KRNL_LOG_INFO(LOG_SYSTEM, "%s","======02 conv_c compute start======");
		one_row_conv(i, conv3,        	  	    								
					ByteToW64(wt_start_c - MMA_START_ADDR),									
					ByteToW64(localmem_fm_conv_c_out[0] - MMA_START_ADDR),							
					ByteToW64(bs_start_c - MMA_START_ADDR),									
					ByteToW64(shift_start_c - MMA_START_ADDR),									
                    ByteToW64(localmem_fm_table_conv_bc[0] - MEM_LCMEM_ADDR_S),	                     	
                    ByteToW64(localmem_fm_table_conv_bc[0] + localmem_fm_num_conv_bc * MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),	    
                   	ByteToW64(MMA_BANK_SIZE),         										
                    cluster_start_c,                    									
                    cluster_end_c,                      									
                    cluster_num_c,
					CONV_TYPE_CLASSIC, 1);
		//st result
        LIBHIKL_NASSERT(__wr_to_remote_chunk_non_blocking((uint32 *)localmem_fm_conv_c_out[0], ofm_addr_c_reg1->x_pos, ofm_addr_c_reg1->x_pos, ofm_ptr_c, (W64ToByte(ofm_row_oneline_mrlen_c))));
        ofm_ptr_c += W64ToByte(ofm_row_oneline_mrlen_c);
        __ndma_poll();
		
		cluster_start_a = ByteToW64(localmem_fm_table_conv_a[localmem_fm_index_conv_a] - MMA_START_ADDR);  //每次都load 到 bank0
		cluster_end_a = cluster_start_a + ifm_row_oneline_mrlen_a;       // always the end of the first ifm row
		cluster_num_a = 0; // always less than the full kernel height
		localmem_fm_index_conv_bc = i % 3 ;
		// set_debug_flag(1,1,1,0);
		KRNL_LOG_INFO(LOG_SYSTEM, "%s","in loop : ======01 conv_a compute start======");
		one_row_conv(i + 2, conv1,        	  	    										      
					ByteToW64(wt_start_a-MMA_START_ADDR),											      
					ByteToW64(localmem_fm_table_conv_bc[localmem_fm_index_conv_bc] - MMA_START_ADDR),         
					ByteToW64(bs_start_a-MMA_START_ADDR),											      
					ByteToW64(shift_start_a-MMA_START_ADDR),										      
                    ByteToW64(localmem_fm_table_conv_a[0] - MEM_LCMEM_ADDR_S),	                     	  
                    ByteToW64(localmem_fm_table_conv_a[0] + localmem_fm_num_conv_a * MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),	  
                   	0,               								                                      
                    cluster_start_a,                    											      
                    cluster_end_a,                      											      
                    cluster_num_a,                                                                        
					CONV_TYPE_CLASSIC, 1);
		set_debug_flag(0,0,0,0);
	}
	KRNL_LOG_INFO(LOG_DEBUG, "calculation is end, ndma begins...\n\r");
	return;
}