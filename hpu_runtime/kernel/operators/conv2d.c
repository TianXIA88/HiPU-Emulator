

#include "hihw.h"
#include "libconv.h"
#include "dma.h"
#include "lock.h"
#include "int.h"
#include "operators/hi_krnl_param_conv2d.h"
#include "hi_addr_def.h"
#include "krnl_log.h"
#include "hpu_util.h"


#define LOCAL_IFM_BLK   MMA_BANK0_START_ADDR
#define LOCAL_OFM_BLK   MMA_BANK5_START_ADDR
#define LOCAL_VAR_BLK   MMA_BANK4_START_ADDR
#define BIAS_SHIFT_BLK  MMA_BANK7_START_ADDR
#define REMOT_IFM_BLK   MMA_BANK0_START_ADDR
#define VAR_BLK_IN_REMOTE_NODE   MMA_BANK6_START_ADDR

#define IFM_PREFETCH_ROW_NUM 1
#define OFM_OUTPUT_SLOTS_NUM 2

extern int get_core_id();

static uint32_t ifm_rsize, ofm_rsize, core_id, kernel_id;
static uint32_t wt_size = 0, bias_size = 0, shift_size = 0;

static paramTableConv2d_Entry_t *conv_table;
static hirtKernelParamTableBase_t *base_table;

static ddr_fm ddr_ifm, ddr_ofm;
static local_fm local_ifm, local_ofm;
static remote_fm remote_ifm;
static local_variable local_var;

static uint32_t conv_level_role = -1; // CONV_HEAD, CONV_TAIL, CONV_BODY, CONV_SINGLE 



void timer_handler(){
	KRNL_LOG_INFO(LOG_SYSTEM, "timer interrupt begins...");
	uint8_t ofm_node_addr = HIPU200_NOC_MAKENMAP(conv_table->ofm_addr.x_pos, conv_table->ofm_addr.y_pos);
	if(ofm_node_addr == HIPU200_NOC_NODEADDR_DDR0 || ofm_node_addr == HIPU200_NOC_NODEADDR_DDR1)
	{
		vmu_poll();
		_ndma_one_fm_row_from_localmem_to_ddr_blocking(&local_ofm, &ddr_ofm);
	}
	else
	{
		_ndma_one_fm_row_from_localmem_to_remote_localmem_blocking(&local_ofm, &remote_ifm);
	}
	KRNL_LOG_INFO(LOG_SYSTEM, "timer interrupt ends...");
	set_timer_tick(TIMER_TICK, 0);
	return;
}

void swi_handler(){
	_swi_print();
	clr_swi();
	return;
}

int get_core_id(){
	return 1;
}

uint32 trans_and_alloc_local_fm(local_fm *ofm, remote_fm *rfm, uint32 num){
	int ready = 1;
	uint32 local_mem_fm_addr = alloc_local_fm(ofm, num);
	if(local_mem_fm_addr == -1){
		KRNL_LOG_INFO(LOG_DEBUG, "the ofm is full, ndma begins...\n\r");
		do{
			ready = _ndma_one_fm_row_from_localmem_to_remote_localmem_blocking(ofm, rfm);
		}while(ready);
		KRNL_LOG_INFO(LOG_DEBUG, "ndma ends...\n\r");
		return alloc_local_fm(ofm, num);
	}
	return local_mem_fm_addr;
}

uint32 alloc_local_fm_with_drain_to_ddr_if_needed(local_fm *ofm, ddr_fm *dfm, uint32 num){
	int ready = 1;
	uint32 local_mem_fm_addr = alloc_local_fm(ofm, num);
	KRNL_LOG_INFO(LOG_DEBUG, "local_mem_fm_addr = %x\n", local_mem_fm_addr);
	if(local_mem_fm_addr == -1){
		KRNL_LOG_INFO(LOG_DEBUG, "the ofm is full, ndma begins...\n\r");
		vmu_poll();
		do{
			ready = _ndma_one_fm_row_from_localmem_to_ddr_blocking(ofm, dfm);
		}while(ready);
		KRNL_LOG_INFO(LOG_DEBUG, "ndma ends...\n\r");
		return alloc_local_fm(ofm, num);
	}
	return local_mem_fm_addr;
}

void check_conv_level_role(paramTableConv2d_Entry_t *p_conv2d_entry)
{
	uint8_t ifm_node_addr = HIPU200_NOC_MAKENMAP(p_conv2d_entry->ifm_addr.x_pos, p_conv2d_entry->ifm_addr.y_pos);
	uint8_t ofm_node_addr = HIPU200_NOC_MAKENMAP(p_conv2d_entry->ofm_addr.x_pos, p_conv2d_entry->ofm_addr.y_pos);
	if ( (ifm_node_addr == HIPU200_NOC_NODEADDR_DDR0 || ifm_node_addr == HIPU200_NOC_NODEADDR_DDR1) && \
	     (ofm_node_addr == HIPU200_NOC_NODEADDR_DDR0 || ofm_node_addr == HIPU200_NOC_NODEADDR_DDR1) )
 	{
		conv_level_role = CONV_SINGLE;
	}
	else if ( (ifm_node_addr == HIPU200_NOC_NODEADDR_DDR0 || ifm_node_addr == HIPU200_NOC_NODEADDR_DDR1) )
	{
		conv_level_role = CONV_HEAD;
	}
	else if ( (ofm_node_addr == HIPU200_NOC_NODEADDR_DDR0 || ofm_node_addr == HIPU200_NOC_NODEADDR_DDR1) )
	{
		conv_level_role = CONV_TAIL;
	}
	else 
	{
		conv_level_role = CONV_BODY;
	} 

	// unsigned int *_flags = ( unsigned int *)HIPU200_MEM_ATOMIC_START;
	// // Get the core ID
	// unsigned int _coreid = _flags[0];
	// char *conv_level_role_str = NULL;
	// switch (conv_level_role)
	// {
	// 	case CONV_SINGLE:
	// 		conv_level_role_str = "CONV_SINGLE";
	// 	break;
	// 	case CONV_HEAD:
	// 		conv_level_role_str = "CONV_HEAD";
	// 	break;
	// 	case CONV_TAIL:
	// 		conv_level_role_str = "CONV_TAIL";
	// 	break;
	// 	case CONV_BODY:
	// 		conv_level_role_str = "CONV_BODY";
	// 	break;
	// }
	// KRNL_LOG_INFO(LOG_SYSTEM, "core %d conv_level_role: %s", _coreid, conv_level_role_str);
}

void log_conv_param(paramTableConv2d_Entry_t *p_conv2d_entry)
{
	KRNL_LOG_INFO(LOG_SYSTEM, "ifm_h: %d", p_conv2d_entry->conv2d.cshape.ifm_h);
	KRNL_LOG_INFO(LOG_SYSTEM, "ifm_w: %d", p_conv2d_entry->conv2d.cshape.ifm_w);
	KRNL_LOG_INFO(LOG_SYSTEM, "ifm_c: %d", p_conv2d_entry->conv2d.cshape.ifm_c);
	KRNL_LOG_INFO(LOG_SYSTEM, "ofm_c: %d", p_conv2d_entry->conv2d.cshape.ofm_c);
	KRNL_LOG_INFO(LOG_SYSTEM, "k_h: %d", p_conv2d_entry->conv2d.cshape.k_h);
	KRNL_LOG_INFO(LOG_SYSTEM, "k_w: %d", p_conv2d_entry->conv2d.cshape.k_w);
	KRNL_LOG_INFO(LOG_SYSTEM, "pshape_top: %d",  p_conv2d_entry->conv2d.pshape.top);
	KRNL_LOG_INFO(LOG_SYSTEM, "pshape_bottom: %d", p_conv2d_entry->conv2d.pshape.bottom);
	KRNL_LOG_INFO(LOG_SYSTEM, "pshape_left: %d pshape_right: %d", p_conv2d_entry->conv2d.pshape.left, p_conv2d_entry->conv2d.pshape.right);
	KRNL_LOG_INFO(LOG_SYSTEM, "h_strd: %d", p_conv2d_entry->conv2d.strd_shape.h_strd);
	KRNL_LOG_INFO(LOG_SYSTEM, "w_strd: %d", p_conv2d_entry->conv2d.strd_shape.w_strd);
	KRNL_LOG_INFO(LOG_SYSTEM, "relu: %d", p_conv2d_entry->conv2d.relu);
}

void load_weight_bias_shift(paramTableConv2d_Entry_t *p_conv2d_entry)
{
	// Calculate total weight size
	wt_size = p_conv2d_entry->conv2d.cshape.k_w * p_conv2d_entry->conv2d.cshape.k_h * p_conv2d_entry->conv2d.cshape.ifm_c * p_conv2d_entry->conv2d.cshape.ofm_c; // in Bytes
	KRNL_LOG_INFO(LOG_DEBUG, "wt_size %d \n", wt_size);

	// Calculate total bias size
	bias_size = p_conv2d_entry->conv2d.cshape.ofm_c * MTX_SCALE * 4;	//bias 32bits per item

	// Calculate total shift size
	shift_size = p_conv2d_entry->conv2d.cshape.ofm_c * MTX_SCALE;
	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "bias_size %d  shift_size %d\n", bias_size, shift_size);

	// Load all weights
	KRNL_LOG_INFO(LOG_DEBUG, "Load Weights: [%d%d%x]->[%x]\n\r", p_conv2d_entry->wt_addr.x_pos, p_conv2d_entry->wt_addr.y_pos, p_conv2d_entry->wt_addr.lcaddr, (uint32 *)MMB_START_ADDR);
	//MMB can't be access by HPU scalar ALU
	LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)MMB_START_ADDR, p_conv2d_entry->wt_addr.x_pos, p_conv2d_entry->wt_addr.y_pos, p_conv2d_entry->wt_addr.lcaddr,  GMEM_ALIGN(wt_size)));

	// Load all bias
	KRNL_LOG_INFO(LOG_DEBUG, "Load Bias: [%x]->[%x]\n\r",  p_conv2d_entry->bias_addr.lcaddr, (uint32 *)(BIAS_SHIFT_BLK));
	LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)(BIAS_SHIFT_BLK), p_conv2d_entry->bias_addr.x_pos, p_conv2d_entry->bias_addr.y_pos, p_conv2d_entry->bias_addr.lcaddr, GMEM_ALIGN(bias_size)));
	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "Bias: [%x]:[%x]\n\r",  (uint32 *)(BIAS_SHIFT_BLK), *(uint32 *)(BIAS_SHIFT_BLK));

	// Load all shift_num
	KRNL_LOG_INFO(LOG_DEBUG, "Load Shift_mtx: [%x]->[%x]\n\r", p_conv2d_entry->shift_addr.lcaddr, (uint32 *)(BIAS_SHIFT_BLK + bias_size));
	LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)(BIAS_SHIFT_BLK + bias_size), p_conv2d_entry->shift_addr.x_pos, p_conv2d_entry->shift_addr.y_pos, p_conv2d_entry->shift_addr.lcaddr, GMEM_ALIGN(shift_size)));
}

void init_ifm_ofm_vars(paramTableConv2d_Entry_t *p_conv2d_entry)
{
	ifm_rsize = get_ifm_row_size(&p_conv2d_entry->conv2d.cshape, &p_conv2d_entry->conv2d.strd_shape);
	ofm_rsize = get_ofm_row_size(&p_conv2d_entry->conv2d.cshape, &p_conv2d_entry->conv2d.strd_shape);

	int ofm_h = p_conv2d_entry->conv2d.cshape.ifm_h / p_conv2d_entry->conv2d.strd_shape.h_strd;
	KRNL_LOG_INFO(LOG_DEBUG, "ifm_rsize: %d  ofm_rsize: %d \n", ifm_rsize, ofm_rsize);

    init_local_var(&local_var, LOCAL_VAR_BLK, LOCAL_VAR_BLK + MMA_BANK_SIZE, DIVIDE_BY_32BYTES(MMA_BANK_SIZE));
    init_local_fm(&local_ifm, &local_var, p_conv2d_entry->conv2d.cshape.k_h + IFM_PREFETCH_ROW_NUM, ifm_rsize, LOCAL_IFM_BLK);
    init_local_fm(&local_ofm, &local_var, OFM_OUTPUT_SLOTS_NUM, ofm_rsize, LOCAL_OFM_BLK);
	switch (conv_level_role)
	{
	case CONV_SINGLE:
		init_ddr_fm(&ddr_ifm, p_conv2d_entry->conv2d.cshape.ifm_h, ifm_rsize, p_conv2d_entry->ifm_addr.lcaddr, p_conv2d_entry->ifm_addr.x_pos, p_conv2d_entry->ifm_addr.y_pos);
        init_ddr_fm(&ddr_ofm, ofm_h, ofm_rsize, p_conv2d_entry->ofm_addr.lcaddr, p_conv2d_entry->ofm_addr.x_pos, p_conv2d_entry->ofm_addr.y_pos);
		break;
	case CONV_HEAD:
		init_remote_fm(&remote_ifm, &local_var, 4, ofm_rsize, REMOT_IFM_BLK, p_conv2d_entry->ofm_addr.x_pos, p_conv2d_entry->ofm_addr.y_pos, VAR_BLK_IN_REMOTE_NODE);
        init_ddr_fm(&ddr_ifm, p_conv2d_entry->conv2d.cshape.ifm_h, ifm_rsize, p_conv2d_entry->ifm_addr.lcaddr, p_conv2d_entry->ifm_addr.x_pos, p_conv2d_entry->ifm_addr.y_pos);
		break;
	case CONV_TAIL:
        init_ddr_fm(&ddr_ofm, ofm_h, ifm_rsize, p_conv2d_entry->ofm_addr.lcaddr, p_conv2d_entry->ofm_addr.x_pos, p_conv2d_entry->ofm_addr.y_pos);
		break;
	default:
		//CONV_BODY
        init_remote_fm(&remote_ifm, &local_var, 4, ofm_rsize, REMOT_IFM_BLK, p_conv2d_entry->ofm_addr.x_pos, p_conv2d_entry->ofm_addr.y_pos, VAR_BLK_IN_REMOTE_NODE);
		break;
	}
}

void initially_load_ifm_to_start_conv(paramTableConv2d_Entry_t *p_conv2d_entry)
{
	if (conv_level_role == CONV_BODY || conv_level_role == CONV_TAIL)
	{
		//  init local lock
		init_local_fm_lock(&local_ifm);
	}
	else
	{
		uint32 local_mem_fm_addr, ddr_mem_fm_addr;
		int total_retrieve_ifm_rows_num = p_conv2d_entry->conv2d.cshape.k_h - p_conv2d_entry->conv2d.pshape.top;
		// Initially Load Row-0 and Row-1 for kernel 3*3, padding: 1

		for(int i=0; i<total_retrieve_ifm_rows_num; i++){
			local_mem_fm_addr = alloc_local_fm(&local_ifm, 1);
			ddr_mem_fm_addr = alloc_ddr_fm(&ddr_ifm);
			LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)local_mem_fm_addr, ddr_ifm.x_pos, ddr_ifm.y_pos, ddr_mem_fm_addr, GMEM_ALIGN(local_ifm.bfm.row_size)));
			KRNL_LOG_INFO(LOG_HARDWARE_CMD, "Load local_ifm row %d: [%d%d%x]->[%x] size: %d Bytes\n\r", (local_ifm.bfm.total_cnt - 1), ddr_ifm.x_pos, ddr_ifm.y_pos, ddr_mem_fm_addr, \
						  local_mem_fm_addr, GMEM_ALIGN(local_ifm.bfm.row_size));
			//test log begin
			// buf_print(local_mem_fm_addr, GMEM_ALIGN(local_ifm.bfm.row_size));
			//test log end
		}
	}
}


void makesure_needed_ifm_rows_ready(int cur_ifm_row_index, paramTableConv2d_Entry_t *p_conv2d_entry)
{
	uint32 local_mem_fm_addr, ddr_mem_fm_addr;
	uint8 ready = 0;
	if(conv_level_role == CONV_BODY || conv_level_role == CONV_TAIL)
	{
		if(is_ifm_top(cur_ifm_row_index, &p_conv2d_entry->conv2d.cshape, &p_conv2d_entry->conv2d.pshape)){
			//  acquire the lock of core2 MMA
			do{
				for(int r=0; r<p_conv2d_entry->conv2d.cshape.k_h - p_conv2d_entry->conv2d.pshape.top; r++){
					acquire_local_fm_lock(r);
					ready = ready & local_ifm.row_slot_available_flgs[r];
				}
				if(!ready){
					for(int r=0; r<p_conv2d_entry->conv2d.cshape.k_h - p_conv2d_entry->conv2d.pshape.top ; r++)
						release_local_fm_lock(r);
					wait(15);
				}
				else{
					ready = 1;
				}
			}while(!ready);
		}

		else if(!is_ifm_bottom(cur_ifm_row_index, &p_conv2d_entry->conv2d.cshape, &p_conv2d_entry->conv2d.pshape)){
			//  acquire the lock of core2 MMA
			do{
				//cur_ifm_row_index + 1 should be refine for k_h > 3 ????
				acquire_local_fm_lock((cur_ifm_row_index + 1) % local_ifm.bfm.row_slots_num);
				if(!(local_ifm.row_slot_available_flgs[(cur_ifm_row_index + 1) %  local_ifm.bfm.row_slots_num])){
					release_local_fm_lock((cur_ifm_row_index + 1) % local_ifm.bfm.row_slots_num);
					wait(15);
				}
				else ready = 1;
			}while(!ready);
		}
	}
	else 
	{
		// Pre-load the next local_ifm row if we are not at bottom
		if(is_ifm_bottom(cur_ifm_row_index * p_conv2d_entry->conv2d.strd_shape.h_strd, &p_conv2d_entry->conv2d.cshape, &p_conv2d_entry->conv2d.pshape)){
			for(int i=0; i < p_conv2d_entry->conv2d.cshape.ifm_h - cur_ifm_row_index * p_conv2d_entry->conv2d.strd_shape.h_strd + p_conv2d_entry->conv2d.pshape.top; i++){
				local_mem_fm_addr = alloc_local_fm(&local_ifm, 1);
				ddr_mem_fm_addr = alloc_ddr_fm(&ddr_ifm);
				KRNL_LOG_INFO(LOG_DEBUG, "Load local_ifm row %d: [%x]->[%x]\n\r", (local_ifm.bfm.total_cnt - 1), ddr_mem_fm_addr, local_mem_fm_addr);
				#ifdef QEMU_ENV
					printf("Load local_ifm row %d: [%x]->[%x]\n\r", (local_ifm.bfm.total_cnt - 1), ddr_mem_fm_addr, local_mem_fm_addr); 
				#endif
				LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)local_mem_fm_addr, ddr_ifm.x_pos, ddr_ifm.y_pos, ddr_mem_fm_addr, GMEM_ALIGN(local_ifm.bfm.row_size)));
			}
		}
		else if(is_ifm_top(cur_ifm_row_index * p_conv2d_entry->conv2d.strd_shape.h_strd, &p_conv2d_entry->conv2d.cshape, &p_conv2d_entry->conv2d.pshape)){
			;
		}
		else{
			for(int i=0; i<p_conv2d_entry->conv2d.strd_shape.h_strd; i++){
				local_mem_fm_addr = alloc_local_fm(&local_ifm, 1);
				ddr_mem_fm_addr = alloc_ddr_fm(&ddr_ifm);
				KRNL_LOG_INFO(LOG_HARDWARE_CMD, "Load local_ifm row %d: [%x]->[%x]\n\r", (local_ifm.bfm.total_cnt - 1), ddr_mem_fm_addr, local_mem_fm_addr);
				#ifdef QEMU_ENV
					printf("Load local_ifm row %d: [%x]->[%x]\n\r", (local_ifm.bfm.total_cnt - 1), ddr_mem_fm_addr, local_mem_fm_addr); 
				#endif
				LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)local_mem_fm_addr, ddr_ifm.x_pos, ddr_ifm.y_pos, ddr_mem_fm_addr, GMEM_ALIGN(local_ifm.bfm.row_size)));
			}
		}
		// Make sure all required rows are ready
		if(is_ifm_top(cur_ifm_row_index, &p_conv2d_entry->conv2d.cshape, &p_conv2d_entry->conv2d.pshape)){
			LIBHIKL_ASSERT(local_ifm.cur_cnt >= p_conv2d_entry->conv2d.cshape.k_h - p_conv2d_entry->conv2d.pshape.top);
		}
		else if(is_ifm_bottom(cur_ifm_row_index, &p_conv2d_entry->conv2d.cshape, &p_conv2d_entry->conv2d.pshape)) {
			LIBHIKL_ASSERT(local_ifm.cur_cnt >= p_conv2d_entry->conv2d.cshape.k_h - p_conv2d_entry->conv2d.pshape.bottom);
		}
		else{
			LIBHIKL_ASSERT(local_ifm.cur_cnt >= p_conv2d_entry->conv2d.cshape.k_h);
		}
	}
}

void get_mmac_cluster_params(paramTableConv2d_Entry_t *p_conv2d_entry, uint32_t h_iter, 		\
							uint32_t *p_mmac_cluster_start,                    					\
                    		uint32_t *p_mmac_cluster_end,           		           			\
                    		uint32_t *p_mmac_cluster_num)
{

	uint32_t ifm_row_size = ByteToW64(p_conv2d_entry->conv2d.cshape.ifm_c * p_conv2d_entry->conv2d.cshape.ifm_w);
	if(is_ifm_top(h_iter * p_conv2d_entry->conv2d.strd_shape.h_strd, &p_conv2d_entry->conv2d.cshape, &p_conv2d_entry->conv2d.pshape)){
		// always start from the first ifm row, pad automatically increments inside mmac
		*p_mmac_cluster_start = ByteToHPU64BytesWord(local_ifm.bfm.start_addr - MMA_START_ADDR);
		if (conv_level_role == CONV_TAIL)
		{
			*p_mmac_cluster_end = *p_mmac_cluster_start + ByteToHPU64BytesWord(MMA_BANK_SIZE); 
		}
		else 
		{
			*p_mmac_cluster_end = *p_mmac_cluster_start + ifm_row_size;       // always the end of the first local_ifm row
		}
		*p_mmac_cluster_num = h_iter * p_conv2d_entry->conv2d.strd_shape.h_strd + p_conv2d_entry->conv2d.cshape.k_h - p_conv2d_entry->conv2d.pshape.top - 1; // always less than the full kernel height
		*p_mmac_cluster_num = *p_mmac_cluster_num > 0 ? *p_mmac_cluster_num : 0;
	}
	else{
		// from the middle of ifm
		*p_mmac_cluster_start = ByteToHPU64BytesWord(local_ifm.bfm.start_addr - MMA_START_ADDR) + \
						(h_iter * p_conv2d_entry->conv2d.strd_shape.h_strd - p_conv2d_entry->conv2d.pshape.top) % local_ifm.bfm.row_slots_num * ByteToHPU64BytesWord(MMA_BANK_SIZE);
		*p_mmac_cluster_end = *p_mmac_cluster_start + ifm_row_size;
		if(is_ifm_bottom(h_iter * p_conv2d_entry->conv2d.strd_shape.h_strd, &p_conv2d_entry->conv2d.cshape, &p_conv2d_entry->conv2d.pshape))
		{
			*p_mmac_cluster_num = p_conv2d_entry->conv2d.cshape.ifm_h + p_conv2d_entry->conv2d.pshape.top - h_iter * p_conv2d_entry->conv2d.strd_shape.h_strd - 1;   // always less than the full kernel height
			*p_mmac_cluster_num = *p_mmac_cluster_num > 0 ? *p_mmac_cluster_num : 0;
		}
		else // at middle
		{
			*p_mmac_cluster_num = p_conv2d_entry->conv2d.cshape.k_h - 1;   // always equal the full kernel height
		}
	}
}

void conv2d_core(paramTableConv2d_Entry_t *p_conv2d_entry, uint32_t input_conv_type)
{
#ifdef QEMU_ENV
	printf("func: conv2d_core\n");
#endif
	check_conv_level_role(p_conv2d_entry);
	log_conv_param(p_conv2d_entry);
 	load_weight_bias_shift(p_conv2d_entry);
	init_ifm_ofm_vars(p_conv2d_entry);
	initially_load_ifm_to_start_conv(p_conv2d_entry);

	// Calculate total bias size
	uint32 *local_ofm_output_row_addr = NULL;
	uint32 wt_start = 0, ifm_start = 0, ifm_inner_start = 0, ofm_start = 0;
	uint32 bias_start = 0, shift_start = 0;
	uint32_t param_mmac_cluster_start = 0, param_mmac_cluster_end = 0, param_mmac_cluster_num = 0; 

	// uint32 times_of_8_kernels = p_conv2d_entry->conv2d.cshape.ofm_c / MTX_SCALE;
	// uint32 w_iter_num = p_conv2d_entry->conv2d.cshape.ifm_w / MTX_SCALE; 		//local_ifm w 平均分成8份
	uint32 h_iter_num = (p_conv2d_entry->conv2d.cshape.ifm_h + p_conv2d_entry->conv2d.pshape.top + p_conv2d_entry->conv2d.pshape.bottom - p_conv2d_entry->conv2d.cshape.k_h) / p_conv2d_entry->conv2d.strd_shape.h_strd + 1;

	disable_ndma_intr();
	disable_timer_intr();
	// Start timer ticks
	set_timer_tick(TIMER_TICK, 0);
	enable_timer_intr();
	
	for(int h_iter=0; h_iter<h_iter_num; h_iter++){
		KRNL_LOG_INFO(LOG_SYSTEM, "***=====local_ifm H iter: %d / %d=====\n\r", h_iter, h_iter_num);
		makesure_needed_ifm_rows_ready(h_iter, p_conv2d_entry);
		// Allocate local space for 1 ofm row
		disable_timer_intr();
		local_ofm_output_row_addr = (uint32 *)alloc_local_fm_with_drain_to_ddr_if_needed(&local_ofm, &ddr_ofm, 1);
		enable_timer_intr();

		get_mmac_cluster_params(p_conv2d_entry, h_iter, 		\
								&param_mmac_cluster_start,      \
                   				&param_mmac_cluster_end, 		\
                   				&param_mmac_cluster_num);

		
		one_row_conv(h_iter, &p_conv2d_entry->conv2d,        	  	    										\
					ByteToHPU64BytesWord(MMB_BEGIN),															\
					ByteToHPU64BytesWord((int)local_ofm_output_row_addr - MMA_START_ADDR), 						\
					ByteToHPU64BytesWord(BIAS_SHIFT_BLK - MMA_START_ADDR),										\
					ByteToHPU64BytesWord(BIAS_SHIFT_BLK + bias_size - MMA_START_ADDR),							\
                    ByteToHPU64BytesWord(local_ifm.bfm.start_addr - MEM_LCMEM_ADDR_S),	                     	\
                    ByteToHPU64BytesWord(local_ifm.bfm.end_addr - MEM_LCMEM_ADDR_S),	                       	\
                   	ByteToHPU64BytesWord(MMA_BANK_SIZE),               											\
                    param_mmac_cluster_start,                    	\
                    param_mmac_cluster_end,                      	\
                    param_mmac_cluster_num,
					input_conv_type, 1);

		// Delete the row that is used up if we are not at the top
		if(!is_ifm_top(h_iter, &p_conv2d_entry->conv2d.cshape, &p_conv2d_entry->conv2d.pshape)){
			for(int delete_ifm_row = 0; delete_ifm_row < p_conv2d_entry->conv2d.strd_shape.h_strd; delete_ifm_row++){
				KRNL_LOG_INFO(LOG_DEBUG, "Release ifm row %d\n\r", local_ifm.cur_valid_idx);
			#ifdef QEMU_ENV
				printf("Release ifm row %d\n\r", local_ifm.cur_valid_idx);
			#endif
				dealloc_local_fm(&local_ifm, 1);
			}
		}
		else{
			for(int delete_ifm_row = 0; delete_ifm_row < p_conv2d_entry->conv2d.strd_shape.h_strd - p_conv2d_entry->conv2d.pshape.top; delete_ifm_row++){
				dealloc_local_fm(&local_ifm, p_conv2d_entry->conv2d.strd_shape.h_strd -  p_conv2d_entry->conv2d.pshape.top);
			}
		}
	}
	disable_timer_intr();
	KRNL_LOG_INFO(LOG_DEBUG, "calculation is end, ndma begins...\n\r");
	vmu_poll();
	_ndma_remain_ifm_rows_from_localmem_to_ddr_blocking(&local_ofm, &ddr_ofm);
	KRNL_LOG_INFO(LOG_DEBUG, "ndma ends...\n\r");
	return;
}


void conv2d_multi_layers()
{
#ifdef QEMU_ENV
    qemu_arch_setup();
	printf("=============================\n\r");
    printf("YOU ARE USING A QEMU INSTANCE\n\r");
    printf("=============================\n\r");
#endif
	
	// get paramTable for conv2d
	paramTableConv2d_t *_pParamTable = *((paramTableConv2d_t **)HIPU200_KNL_PTABLE_ADDR);/*get kernel param table from runtime*/
	base_table = &_pParamTable->infoBase;

	unsigned int *_flags = ( unsigned int *)HIPU200_MEM_ATOMIC_START;

	LIBHIKL_NASSERT(base_table->task_dim > HIPU200_SOC_CORE_NUM);
	LIBHIKL_NASSERT(base_table->task_dim == 0);

	// Get the core ID
	unsigned int _coreid = _flags[0];
	// unsigned int _coreid = get_core_id();
    
    int kernel_id = -1;

	// Get the position in the kernel and find the right param table
	unsigned int _taskid = 255;
	int *_rtcode = (int *)HIPU200_KNL_RTCODE_ADDR;

	//new
    for(int i=0; i < _pParamTable->count; i++){
        if(base_table->task_cores[i] == _coreid){
            kernel_id = i;
            conv_table = &_pParamTable->param[i];
            break;
        }
    }
    LIBHIKL_NASSERT(kernel_id == 255);
	uint32_t input_conv_type;
	switch(base_table->op_type)
	{
		case KERNEL_OP_DWCONV:
			input_conv_type = CONV_TYPE_DEPTH_WISE; 
		break;        
		case KERNEL_OP_CONV2D:
			input_conv_type = CONV_TYPE_CLASSIC; 
		break;        
	}
	
	conv2d_core(conv_table, input_conv_type);

#ifdef QEMU_ENV
	qemu_fprint(QEMU_LOG_MEM, conv_table->ofm_addr.lcaddr, 327680);
#endif
}

