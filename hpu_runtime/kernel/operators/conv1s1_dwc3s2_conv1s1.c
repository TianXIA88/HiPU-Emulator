#include "hihw.h"
#include "libconv.h"
#include "dma.h"
#include "lock.h"
#include "int.h"
#include "operators/hi_krnl_param_conv1s1_dwc3s2_conv1s1.h"
#include "hi_addr_def.h"
#include "krnl_log.h"
// #include "qemu.h"


#define LCMEM_CONV_A_IN0    MMA_BANK0_START_ADDR
#define LCMEM_CONV_A_IN1    MMA_BANK1_START_ADDR
#define LCMEM_DWC_B_IN0     MMA_BANK2_START_ADDR
#define LCMEM_DWC_B_IN1     MMA_BANK3_START_ADDR
#define LCMEM_DWC_B_IN2     MMA_BANK4_START_ADDR
#define LCMEM_CONV_C_IN0    MMA_BANK5_START_ADDR
#define LCMEM_CONV_C_OU0    MMA_BANK6_START_ADDR
#define LCMEM_BIAS_SHIFT    MMA_BANK7_START_ADDR

static const u32_t banktbl_conv_a[] = 
{
    LCMEM_CONV_A_IN0,
    LCMEM_CONV_A_IN1
};
static u32_t bankidx_conv_a = 0;
static const u32_t banknum_conv_a = sizeof(banktbl_conv_a) / sizeof(u32_t);

static const u32_t banktbl_dwc_b[] = 
{
    LCMEM_DWC_B_IN0,
    LCMEM_DWC_B_IN1,
    LCMEM_DWC_B_IN2
};
static u32_t bankidx_dwc_b = 0;
static const u32_t banknum_dwc_b = sizeof(banktbl_dwc_b) / sizeof(u32_t);

static const u32_t banktbl_conv_c[] = 
{
    LCMEM_CONV_C_IN0
};

static const u32_t banktbl_out[] = 
{
    LCMEM_CONV_C_OU0
};

void _op_conv1s1_dwc3s2_conv1s1
(
    conv2d_params_t *conv2d_params_a, 
    conv2d_params_t *conv2d_params_b, 
    conv2d_params_t *conv2d_params_c, 

    hikl_addr_t *ifm_addr,
    hikl_addr_t *ofm_addr,
    
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


void kernel_conv1s1_dwc3s2_conv1s1()
{
	paramTableConv1s1_dwc3s2_conv1s1_t *_pParamTable = *((paramTableConv1s1_dwc3s2_conv1s1_t **)HIPU200_KNL_PTABLE_ADDR);/*get kernel param table from runtime*/
	paramTableConv1s1_dwc3s2_conv1s1_Entry_t *p_op_entry = &_pParamTable->param;
	_op_conv1s1_dwc3s2_conv1s1(
		&p_op_entry->conv1,
		&p_op_entry->conv2,
		&p_op_entry->conv3,

		&p_op_entry->ifm_addr_conv1,
		&p_op_entry->ofm_addr_conv3,

		&p_op_entry->wt_addr_conv1,
		&p_op_entry->wt_addr_conv2,
		&p_op_entry->wt_addr_conv3,

		&p_op_entry->bias_addr_conv1,
		&p_op_entry->bias_addr_conv2,
		&p_op_entry->bias_addr_conv3,

		&p_op_entry->shift_addr_conv1,
		&p_op_entry->shift_addr_conv2,
		&p_op_entry->shift_addr_conv3);

}

void _op_conv1s1_dwc3s2_conv1s1
(
    conv2d_params_t *conv2d_params_a, 
    conv2d_params_t *conv2d_params_b, 
    conv2d_params_t *conv2d_params_c, 

    hikl_addr_t *ifm_addr,
    hikl_addr_t *ofm_addr,
    
    hikl_addr_t *wt_addr_a, 
    hikl_addr_t *wt_addr_b, 
    hikl_addr_t *wt_addr_c, 

    hikl_addr_t *bs_addr_a, 
    hikl_addr_t *bs_addr_b, 
    hikl_addr_t *bs_addr_c, 

    hikl_addr_t *shift_addr_a,
    hikl_addr_t *shift_addr_b,
    hikl_addr_t *shift_addr_c
)
{
	// conv2d_params_t conv2d_params_a;
	// memcpy(&conv2d_params_a->cshape, cshape_a, sizeof(conv_shape_t));

	// conv2d_params_a->pshape.top = 0;
	// conv2d_params_a->pshape.bottom = 0;
	// conv2d_params_a->pshape.left = 0;
	// conv2d_params_a->pshape.right = 0;

	// conv2d_params_a->strd_shape.w_strd = 1;
	// conv2d_params_a->strd_shape.h_strd = 1;

	// conv2d_params_a->dilat_shape.w_dilat = 1;
	// conv2d_params_a->dilat_shape.h_dilat = 1;

	// conv2d_params_a->cgroup_num.grp_num = 1;

	// conv2d_params_a->relu = relu_a;

	// conv2d_params_t conv2d_params_b;
	// memcpy(&conv2d_params_b->cshape, cshape_b, sizeof(conv_shape_t));

	// conv2d_params_b->pshape.top = 1;
	// conv2d_params_b->pshape.left = 1;
	// conv2d_params_b->pshape.bottom = 0;
	// conv2d_params_b->pshape.right = 0;

	// conv2d_params_b->strd_shape.w_strd = 2;
	// conv2d_params_b->strd_shape.h_strd = 2;

	// conv2d_params_b->dilat_shape.w_dilat = 1;
	// conv2d_params_b->dilat_shape.h_dilat = 1;

	// conv2d_params_b->cgroup_num.grp_num = 1;

	// conv2d_params_b->relu = relu_b;

	// conv2d_params_t conv2d_params_c;
	// memcpy(&conv2d_params_c->cshape, cshape_c, sizeof(conv_shape_t));

	// conv2d_params_c->pshape.top = 0;
	// conv2d_params_c->pshape.left = 0;
	// conv2d_params_c->pshape.bottom = 0;
	// conv2d_params_c->pshape.right = 0;

	// conv2d_params_c->strd_shape.w_strd = 1;
	// conv2d_params_c->strd_shape.h_strd = 1;

	// conv2d_params_c->dilat_shape.w_dilat = 1;
	// conv2d_params_c->dilat_shape.h_dilat = 1;

	// conv2d_params_c->cgroup_num.grp_num = 1;

	// conv2d_params_c->relu = relu_c;



	uint32 i, j;
	uint32 h_iter_num_b;
	uint32 ifm_ptr;
	uint32 ofm_ptr = ofm_addr->lcaddr;
	
	uint32 ifm_row_oneline_mrlen_a, ofm_row_oneline_mrlen_a = 0;
	uint32 ifm_row_oneline_mrlen_b = 0;
	uint32 ifm_row_oneline_mrlen_c, ofm_row_oneline_mrlen_c = 0;
	
	uint32 ifm_c_group8_num_a, cluster_num_a = 0;
	uint32 w_iter_num_a, kernel_group8_num_a = 0;
	uint32 wt_offset_a, cluster_start_a, cluster_end_a, wt_ptr_a, shift_ptr_a, bs_ptr_a, ifm_ptr_a, ofm_ptr_a = 0;

	uint32 ifm_c_group8_num_b, cluster_num_b = 0;
	uint32 w_iter_num_b, kernel_group8_num_b = 0;
	uint32 wt_offset_b, cluster_start_b, cluster_end_b, wt_ptr_b, shift_ptr_b, bs_ptr_b, ifm_ptr_b, ofm_ptr_b = 0;


	uint32 ifm_c_group8_num_c, cluster_num_c = 0;
	uint32 w_iter_num_c, kernel_group8_num_c = 0;
	uint32 wt_offset_c, cluster_start_c, cluster_end_c, wt_ptr_c, shift_ptr_c, bs_ptr_c, ifm_ptr_c, ofm_ptr_c = 0;

	uint8  round_type, shift_num, prot_high, prot_low =0;
	
	uint32 wt_sz_a, bs_sz_a, shift_sz_a;
	uint32 wt_sz_b, bs_sz_b, shift_sz_b;
	uint32 wt_sz_c, bs_sz_c, shift_sz_c;
	uint32 wt_start_a, bs_start_a, shift_start_a;
	uint32 wt_start_b, bs_start_b, shift_start_b;
	uint32 wt_start_c, bs_start_c, shift_start_c;

	// Calculate total weight size
	wt_sz_a = conv2d_params_a->cshape.k_w * conv2d_params_a->cshape.k_h * conv2d_params_a->cshape.ifm_c * conv2d_params_a->cshape.ofm_c; // in Bytes
	wt_sz_b = conv2d_params_b->cshape.k_w * conv2d_params_b->cshape.k_h * conv2d_params_b->cshape.ifm_c * MTX_SCALE; // in Bytes
	wt_sz_c = conv2d_params_c->cshape.k_w * conv2d_params_c->cshape.k_h * conv2d_params_c->cshape.ifm_c * conv2d_params_c->cshape.ofm_c; // in Bytes

	// // Calculate total bias size
	// bias_size = p_conv2d_entry->conv2d.cshape.ofm_c * MTX_SCALE * 4;	//bias 32bits per item
	// // Calculate total shift size
	// shift_size = p_conv2d_entry->conv2d.cshape.ofm_c * MTX_SCALE

	bs_sz_a = conv2d_params_a->cshape.ofm_c * MTX_SCALE * 4;
	bs_sz_b = conv2d_params_b->cshape.ofm_c * MTX_SCALE * 4;
	bs_sz_c = conv2d_params_c->cshape.ofm_c * MTX_SCALE * 4;
	shift_sz_a = conv2d_params_a->cshape.ofm_c * MTX_SCALE;
	shift_sz_b = conv2d_params_b->cshape.ofm_c * MTX_SCALE;
	shift_sz_c = conv2d_params_c->cshape.ofm_c * MTX_SCALE;

    wt_start_a = MMB_START_ADDR;
    wt_start_b = wt_start_a + GMEM_ALIGN(wt_sz_a);
    wt_start_c = wt_start_b + GMEM_ALIGN(wt_sz_b);
    bs_start_a = LCMEM_BIAS_SHIFT;
    bs_start_b = bs_start_a + GMEM_ALIGN(bs_sz_a);
    bs_start_c = bs_start_b + GMEM_ALIGN(bs_sz_b);
    shift_start_a = bs_start_c + GMEM_ALIGN(bs_sz_c);
    shift_start_b = shift_start_a + GMEM_ALIGN(shift_sz_a);
    shift_start_c = shift_start_b + GMEM_ALIGN(shift_sz_b);
	KRNL_LOG_INFO(LOG_DEBUG, "bs_start_a 0x%x bs_start_b 0x%x bs_start_c 0x%x", ByteToW64(bs_start_a - MMA_START_ADDR), ByteToW64(bs_start_b - MMA_START_ADDR), ByteToW64(bs_start_c - MMA_START_ADDR));
	KRNL_LOG_INFO(LOG_DEBUG, "shift_start_a 0x%x shift_start_b 0x%x shift_start_c 0x%x", ByteToW64(shift_start_a - MMA_START_ADDR), ByteToW64(shift_start_b - MMA_START_ADDR), ByteToW64(shift_start_c - MMA_START_ADDR));

	// Load all weights
	KRNL_LOG_INFO(LOG_DEBUG, "Load Weights");
	//MMB can't be access by HPU scalar ALU
	LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)wt_start_a, wt_addr_a->x_pos, wt_addr_a->y_pos, wt_addr_a->lcaddr,  GMEM_ALIGN(wt_sz_a)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)wt_start_b, wt_addr_b->x_pos, wt_addr_b->y_pos, wt_addr_b->lcaddr,  GMEM_ALIGN(wt_sz_b)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)wt_start_c, wt_addr_c->x_pos, wt_addr_c->y_pos, wt_addr_c->lcaddr,  GMEM_ALIGN(wt_sz_c)));
	// Load all bias
	KRNL_LOG_INFO(LOG_DEBUG, "Load Bias");
	LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)bs_start_a, bs_addr_a->x_pos, bs_addr_a->y_pos, bs_addr_a->lcaddr, GMEM_ALIGN(bs_sz_a)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)bs_start_b, bs_addr_b->x_pos, bs_addr_b->y_pos, bs_addr_b->lcaddr, GMEM_ALIGN(bs_sz_b)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)bs_start_c, bs_addr_c->x_pos, bs_addr_c->y_pos, bs_addr_c->lcaddr, GMEM_ALIGN(bs_sz_c)));
	// Load all shift_num
	KRNL_LOG_INFO(LOG_DEBUG, "Load Shift_mtx");
	LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)shift_start_a, shift_addr_a->x_pos, shift_addr_a->y_pos, shift_addr_a->lcaddr, GMEM_ALIGN(shift_sz_a)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)shift_start_b, shift_addr_b->x_pos, shift_addr_b->y_pos, shift_addr_b->lcaddr, GMEM_ALIGN(shift_sz_b)));
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)shift_start_c, shift_addr_c->x_pos, shift_addr_c->y_pos, shift_addr_c->lcaddr, GMEM_ALIGN(shift_sz_c)));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Calculate total iteration
	kernel_group8_num_a = conv2d_params_a->cshape.ofm_c / MTX_SCALE;			
	kernel_group8_num_b = conv2d_params_b->cshape.ofm_c / MTX_SCALE;			
	kernel_group8_num_c = conv2d_params_c->cshape.ofm_c / MTX_SCALE;			
	w_iter_num_a = conv2d_params_a->cshape.ifm_w / MTX_SCALE; 		
	w_iter_num_b = conv2d_params_b->cshape.ifm_w / MTX_SCALE; 		
	w_iter_num_c = conv2d_params_c->cshape.ifm_w / MTX_SCALE; 		
	//KRNL_LOG_INFO(LOG_DEBUG, "(ifm_c, pshape_top, pshap
	ifm_c_group8_num_a = conv2d_params_a->cshape.ifm_c / MTX_SCALE;		
	ifm_c_group8_num_b = conv2d_params_b->cshape.ifm_c / MTX_SCALE;		
	ifm_c_group8_num_c = conv2d_params_c->cshape.ifm_c / MTX_SCALE;		
	//KRNL_LOG_INFO(LOG_DEBUG, "ifm_c_blk_stride = ifm_c / 8 : %d\n", ifm_c_blk_num);
	ifm_row_oneline_mrlen_a = ByteToW64(get_ifm_row_size(&conv2d_params_a->cshape, &conv2d_params_a->strd_shape));
	ifm_row_oneline_mrlen_b = ByteToW64(get_ifm_row_size(&conv2d_params_b->cshape, &conv2d_params_b->strd_shape));
	ifm_row_oneline_mrlen_c = ByteToW64(get_ifm_row_size(&conv2d_params_c->cshape, &conv2d_params_c->strd_shape));
	ofm_row_oneline_mrlen_a = ByteToW64(get_ofm_row_size(&conv2d_params_a->cshape, &conv2d_params_a->strd_shape));
	ofm_row_oneline_mrlen_c = ByteToW64(get_ofm_row_size(&conv2d_params_c->cshape, &conv2d_params_c->strd_shape));

	// Calculate total iteration
	h_iter_num_b = conv2d_params_b->cshape.ifm_h /*+ pshape->top + pshape->bottom 2 - conv2d_params_b->cshape.k_h + 1*/;
	
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ifm_ptr = ifm_addr->lcaddr;
    //load row-0
	KRNL_LOG_INFO(LOG_DEBUG, "Load banktbl_conv_a");

    LIBHIKL_NASSERT(__rd_from_remote_chunk_non_blocking((uint32 *)banktbl_conv_a[0], ifm_addr->x_pos, ifm_addr->y_pos, ifm_ptr, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a))));
    ifm_ptr += W64ToByte(ifm_row_oneline_mrlen_a);

	// Initially Load Row-0 and Row-1 and Row-2 and conv0-1-2
	for(i=0; i<2; i++)
	{
		KRNL_LOG_INFO(LOG_SYSTEM, "=====ifm H iter for conv_a: %d / %d=====\n\r", i, 2);
	    __ndma_poll();
	    
		if(i<1)
		{
    		LIBHIKL_NASSERT(__rd_from_remote_chunk_non_blocking((uint32 *)banktbl_conv_a[1], ifm_addr->x_pos, ifm_addr->y_pos, ifm_ptr, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a))));
    		ifm_ptr += W64ToByte(ifm_row_oneline_mrlen_a);
        }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
		cluster_start_a = ByteToW64(banktbl_conv_a[i] - MMA_START_ADDR);
		cluster_end_a = cluster_start_a + ifm_row_oneline_mrlen_a;       // always the end of the first ifm row

		KRNL_LOG_INFO(LOG_SYSTEM, "bankidx_dwc_b: %d", bankidx_dwc_b);
		//mmac_cluster_num == k_h - 1
		cluster_num_a = 0; // always less than the full kernel height
		one_row_conv(i, conv2d_params_a,        	  	    										\
					ByteToW64(wt_start_a-MMA_START_ADDR),											\
					ByteToW64(banktbl_dwc_b[bankidx_dwc_b] - MMA_START_ADDR), 						\
					ByteToW64(bs_start_a-MMA_START_ADDR),											\
					ByteToW64(shift_start_a-MMA_START_ADDR),										\
                    ByteToW64(banktbl_conv_a[0] - MEM_LCMEM_ADDR_S),	                     		\
                    ByteToW64(banktbl_conv_a[0] + banknum_conv_a * MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),	      		\
                   	0,               								\
                    cluster_start_a,                    											\
                    cluster_end_a,                      											\
                    cluster_num_a,
					CONV_TYPE_CLASSIC, 1);


		// KRNL_LOG_INFO(LOG_DEBUG, "conv_a ofm line: %d", i);
		// buf_print(banktbl_dwc_b[bankidx_dwc_b], GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_b)));

		bankidx_dwc_b = (++bankidx_dwc_b >= banknum_dwc_b) ? 0 : bankidx_dwc_b;
        // if( (bankidx_dwc_b++) >= banknum_dwc_b)
        //        bankidx_dwc_b = 0;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}
	KRNL_LOG_INFO(LOG_SYSTEM, "=====ifm 2 rows finished for conv_2");

	// h_iter_num_b = 8;

	for(i=0; i<h_iter_num_b; i+=conv2d_params_b->strd_shape.h_strd)
	{
		KRNL_LOG_INFO(LOG_SYSTEM, "=====ifm H iter for conv_b: %d / %d=====\n\r", i, h_iter_num_b);

		// Pre-load line[0] if we are not at bottom
		// if(i<(h_iter_num_b-2))
		if(!is_ifm_bottom(i, &conv2d_params_b->cshape, &conv2d_params_b->pshape)){
			LIBHIKL_NASSERT(__rd_from_remote_chunk_non_blocking((uint32 *)banktbl_conv_a[0], ifm_addr->x_pos, ifm_addr->y_pos, ifm_ptr, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a))));
			ifm_ptr += W64ToByte(ifm_row_oneline_mrlen_a);
		}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

		if(is_ifm_top(i, &conv2d_params_b->cshape, &conv2d_params_b->pshape)){
			// cluster_start = ByteToW64(MMA_BEGIN);
			// always start from the first ifm row, pad automatically increments inside mmac
			cluster_start_b = ByteToW64(banktbl_dwc_b[0] - MMA_START_ADDR);
			cluster_end_b = cluster_start_b + ifm_row_oneline_mrlen_b;       // always the end of the first ifm row
			cluster_num_b = i + conv2d_params_b->cshape.k_h - 1 - /*pshape->top*/conv2d_params_b->pshape.top; // always less than the full kernel height
		}
		else{
			// cluster_start = ByteToW64(MMA_BEGIN) + (i - pshape->top) * ifm_row_stride;
			// from the middle of ifm
			cluster_start_b = ByteToW64(banktbl_dwc_b[( i - ((conv2d_params_b->cshape.k_h - 1) / 2) ) % banknum_dwc_b] - MMA_START_ADDR);
			cluster_end_b = cluster_start_b + ifm_row_oneline_mrlen_b;

		    // if(is_ifm_bottom(i, &conv2d_params_b->cshape, &conv2d_params_b->pshape)){
			// 	cluster_num_b = conv2d_params_b->cshape.ifm_h + /*pshape->top*/1 - i;   // always less than the full kernel height
			// }
			// else{ // at middle
			// 	cluster_num_b = conv2d_params_b->cshape.k_h - 1;   // always equal the full kernel height
			// }
			cluster_num_b = conv2d_params_b->cshape.k_h - 1;   // always equal the full kernel height
		}

		one_row_conv(i, conv2d_params_b,        	  	    									\
					ByteToW64(wt_start_b-MMA_START_ADDR),										\
					ByteToW64(banktbl_conv_c[0] - MMA_START_ADDR),								\
					ByteToW64(bs_start_b-MMA_START_ADDR),										\
					ByteToW64(shift_start_b-MMA_START_ADDR),										\
                    ByteToW64(banktbl_dwc_b[0] - MEM_LCMEM_ADDR_S),	                     		\
                    ByteToW64(banktbl_dwc_b[0] + banknum_dwc_b * MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),	       		\
                   	ByteToW64(MMA_BANK_SIZE),         											\
                    cluster_start_b,                    										\
                    cluster_end_b,                      										\
                    cluster_num_b,
					CONV_TYPE_DEPTH_WISE, 1);
		
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Pre-load line[1] if we are not at bottom
		// if(i<(h_iter_num_b-2))
		if(!is_ifm_bottom(i, &conv2d_params_b->cshape, &conv2d_params_b->pshape)){
            __ndma_poll();
            LIBHIKL_NASSERT(__rd_from_remote_chunk_non_blocking((uint32 *)banktbl_conv_a[1], ifm_addr->x_pos, ifm_addr->y_pos, ifm_ptr, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a))));
            ifm_ptr += W64ToByte(ifm_row_oneline_mrlen_a);
        }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //calc conv_c
		cluster_start_c = ByteToW64(banktbl_conv_c[0] - MMA_START_ADDR);
		cluster_end_c = cluster_start_c + ifm_row_oneline_mrlen_c;       // always the end of the first ifm row
		cluster_num_c = 0; // always less than the full kernel height

		KRNL_LOG_INFO(LOG_SYSTEM, "=====ifm H iter for conv_c: %d / %d=====\n\r", i, h_iter_num_b);
		one_row_conv(i, conv2d_params_c,        	  	    										\
					ByteToW64(wt_start_c-MMA_START_ADDR),											\
					ByteToW64(banktbl_out[0] - MMA_START_ADDR),										\
					ByteToW64(bs_start_c-MMA_START_ADDR),											\
					ByteToW64(shift_start_c-MMA_START_ADDR),										\
                    ByteToW64(banktbl_conv_c[0] - MEM_LCMEM_ADDR_S),	                     		\
                    ByteToW64(banktbl_conv_c[0] + MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),	       	\
                   	0,         												\
                    cluster_start_c,                    											\
                    cluster_end_c,                      											\
                    cluster_num_c,
					CONV_TYPE_CLASSIC, 1);

					
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// if(i<(h_iter_num_b-2))
		if(!is_ifm_bottom(i, &conv2d_params_b->cshape, &conv2d_params_b->pshape)){
			__ndma_poll();
		}
		//st result
        LIBHIKL_NASSERT(__wr_to_remote_chunk_non_blocking((uint32 *)banktbl_out[0], ofm_addr->x_pos, ofm_addr->y_pos, ofm_ptr, (W64ToByte(ofm_row_oneline_mrlen_c))));
		KRNL_LOG_INFO(LOG_SYSTEM, "write to ddr: 0x[%d%d]%x with length: %d", ofm_addr->x_pos, ofm_addr->y_pos, ofm_ptr, W64ToByte(ofm_row_oneline_mrlen_c));
        ofm_ptr += W64ToByte(ofm_row_oneline_mrlen_c);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //precalc conv_a[0] and conv_a[1]
		// if(i<(h_iter_num_b-2))
		if(!is_ifm_bottom(i, &conv2d_params_b->cshape, &conv2d_params_b->pshape)){
		    //precalc conv_a[0] and conv_a[1]
			int banktbl_conv_a_index = 0;
			for (int iter = 0; iter < conv2d_params_b->strd_shape.h_strd; iter ++)
			{
				int line_a_index = (i + 1) * conv2d_params_b->strd_shape.h_strd + iter;
				KRNL_LOG_INFO(LOG_SYSTEM, "=====ifm H iter for conv_a: %d=====\n\r", line_a_index);
				KRNL_LOG_INFO(LOG_SYSTEM, "bankidx_dwc_b: %d", bankidx_dwc_b);
				cluster_start_a = ByteToW64(banktbl_conv_a[banktbl_conv_a_index++] - MMA_START_ADDR);
   	 			cluster_end_a = cluster_start_a + ifm_row_oneline_mrlen_a;       // always the end of the first ifm row
   	 			cluster_num_a = 0; // always less than the full kernel height

				one_row_conv(line_a_index, conv2d_params_a,        	  	    										\
							ByteToW64(wt_start_a-MMA_START_ADDR),											\
							ByteToW64(banktbl_dwc_b[bankidx_dwc_b] - MMA_START_ADDR), 						\
							ByteToW64(bs_start_a-MMA_START_ADDR),											\
							ByteToW64(shift_start_a-MMA_START_ADDR),										\
     		                ByteToW64(banktbl_conv_a[0] - MEM_LCMEM_ADDR_S),	                     		\
     		                ByteToW64(banktbl_conv_a[0] + banknum_conv_a * MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),	      		\
     		              	0,               								\
     		                cluster_start_a,                    											\
     		                cluster_end_a,                      											\
     		                cluster_num_a,
							CONV_TYPE_CLASSIC, 1);

				bankidx_dwc_b = (++bankidx_dwc_b >= banknum_dwc_b) ? 0 : bankidx_dwc_b;
			}
    	}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        __ndma_poll();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		KRNL_LOG_INFO(LOG_SYSTEM, "=====ifm H iter finished : %d / %d=====\n\r", i, h_iter_num_b);
	}

	KRNL_LOG_INFO(LOG_SYSTEM, "=====conv1s1_dwc3s2_conv1s1 finished=====");
	return;
}
