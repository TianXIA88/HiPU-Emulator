#include "hihw.h"
#include "libconv.h"
#include "dma.h"
#include "lock.h"
#include "int.h"
#include "operators/hi_krnl_param_conv2d.h"
#include "hi_addr_def.h"
#include "krnl_log.h"
// #include "qemu.h"

#define LOCAL_IFM_BLK   MMA_B0_ADDR
#define LOCAL_OFM_BLK   MMA_B1_ADDR
#define LOCAL_BAS_BLK   MMA_B2_ADDR
#define LOCAL_VAR_BLK   MMA_B3_ADDR
#define REMOT_IFM_BLK   MMA_B0_ADDR
#define REMOT_VAR_BLK   MMA_B3_ADDR
#define BIAS_SHIFT_BLK  MMA_B7_ADDR

extern int get_core_id();
extern void conv2d_head(conv_shape_t *cshape, pad_shape_t *pshape, bool relu, local_fm *ifm, local_fm *ofm, remote_fm *rfm, ddr_fm *dfm, hikl_addr_t *wt_addr, hikl_addr_t *bias_addr, hikl_addr_t *shift_addr);
extern void conv2d_body(conv_shape_t *cshape, pad_shape_t *pshape, bool relu, local_fm *ifm, local_fm *ofm, remote_fm *rfm, hikl_addr_t *wt_addr, hikl_addr_t *bias_addr, hikl_addr_t *shift_addr);
extern void conv2d_tail(conv_shape_t *cshape, pad_shape_t *pshape, bool relu, local_fm *ifm, local_fm *ofm, ddr_fm *dfm, hikl_addr_t *wt_addr, hikl_addr_t *bias_addr, hikl_addr_t *shift_addr);
extern void conv2d_singlecore(conv_shape_t *cshape, stride_shape_t *stride_shape, pad_shape_t *pshape, bool relu, local_fm *ifm, local_fm *ofm, ddr_fm *ddr_ifm, ddr_fm *ddr_ofm, hikl_addr_t *wt_addr, hikl_addr_t *bias_addr, hikl_addr_t *shift_addr);

#define LCMEM_TENSOR_B0     MMA_B0_ADDR
#define LCMEM_TENSOR_B1     MMA_B1_ADDR

void upsample2x
(
    conv_shape_t *cshape_a,
    
    hikl_addr_t *ifm_addr,
    hikl_addr_t *ofm_addr
)
{
#define UPSAMPLE_SCALE      (2)

    uint32 i,j,k,ndma_pool;
    uint32 h_iter_num = cshape_a->ifm_h;
	uint32 ifm_row_size = CALC_FMWIDTH_GROUP8(cshape_a->ifm_w) * CALC_CHANNEL_GROUP8(cshape_a->ifm_c);
	uint32 ofm_row_size = ifm_row_size*2;

	uint32 ifm_c_group8_num_a;
	uint32 w_iter_num_a;
    uint32 ifm_line_ptr_global;
    uint32 ifm_line_ptr_local;
    uint32 ofm_line_ptr_global;
    uint32 ofm_line_ptr_local;
    uint32 ifm_ptr = 0;
    uint32 ofm_ptr = 0;

	// Calculate total iteration
	w_iter_num_a = cshape_a->ifm_w / MTX_SCALE; 		//ifm w 平均分成8�?
	ifm_c_group8_num_a = cshape_a->ifm_c / MTX_SCALE;		//ifm 输入fm channel 平均分成8�?

    ifm_line_ptr_global = ifm_addr->lcaddr;
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)LCMEM_TENSOR_B0, ifm_addr->x_pos, ifm_addr->y_pos, ifm_line_ptr_global,  ifm_row_size));

    for(i=0; i<h_iter_num; i++)
    {
        if( (i+1)<h_iter_num )
        {
            ifm_line_ptr_global = ifm_addr->lcaddr + (i+1) * ifm_row_size;
            LIBHIKL_NASSERT(__rd_from_remote_chunk_non_blocking((uint32 *)LCMEM_TENSOR_B0, ifm_addr->x_pos, ifm_addr->y_pos, ifm_line_ptr_global,  ifm_row_size));
        }
        
        for(j=0; j<w_iter_num_a; j++)
        {
            for(k=0; k<ifm_c_group8_num_a; k++)
            {
                ifm_ptr = LCMEM_TENSOR_B0+j*ifm_c_group8_num_a+k;
                asm volatile("mv t1, %0"::"r"(ifm_ptr):);
                vlw(vr1, t1, 0);

                ofm_ptr = LCMEM_TENSOR_B1+j*2*ifm_c_group8_num_a+k;
                asm volatile("mv t1, %0"::"r"(ofm_ptr):);
                vsw(vr1, t1, 0);
                ofm_ptr +=ifm_c_group8_num_a;
                asm volatile("mv t1, %0"::"r"(ofm_ptr):);
                vsw(vr1, t1, 0);
            }
        }

        __ndma_poll();
        ofm_line_ptr_global = ofm_addr->lcaddr + h_iter_num * UPSAMPLE_SCALE * ofm_row_size;
        LIBHIKL_NASSERT(__wr_to_remote_chunk_blocking((uint32 *)LCMEM_TENSOR_B1, ofm_addr->x_pos, ofm_addr->y_pos, ofm_line_ptr_global,  ofm_row_size));
        ofm_line_ptr_global += ofm_row_size;
        LIBHIKL_NASSERT(__wr_to_remote_chunk_blocking((uint32 *)LCMEM_TENSOR_B1, ofm_addr->x_pos, ofm_addr->y_pos, ofm_line_ptr_global,  ofm_row_size));
    }

	//KRNL_LOG_INFO(LOG_DEBUG, "ndma ends...\n\r");
	return;

}

void conv1s1_dwc3s2_conv1s1__
(
    conv_shape_t *cshape_a, 
    conv_shape_t *cshape_b, 
    conv_shape_t *cshape_c, 
//    stride_shape_t *stride_shape, 
//    pad_shape_t *pshape, 
    bool relu_a, 
    bool relu_b, 
    bool relu_c, 
    //local_fm *ifm, 
    //local_fm *ofm, 
    //ddr_fm *ddr_ifm, 
    //ddr_fm *ddr_ofm, 
    hikl_addr_t *ifm_addr,
    hikl_addr_t *ofm_addr,
    hikl_addr_t *wt_addr, 
    hikl_addr_t *bias_addr, 
    hikl_addr_t *shift_addr
){
	uint32 i, j, ndma_poll;
	uint32 h_iter_num_b, local_mem_fm_addr, ddr_mem_fm_addr;
	uint32 *ofm_ptr;
	
	uint32 ifm_row_stride_a, ofm_row_stride_a = 0;
	uint32 ifm_row_stride_b, ofm_row_stride_b = 0;
	uint32 ifm_row_stride_c, ofm_row_stride_c = 0;
	
	uint32 ifm_c_group8_num_a, wt_cluster_size_a, cluster_num_a = 0;
	uint32 w_iter_num_a, kernel_group8_num_a = 0;
	uint32 wt_offset_a, wt_start_a, ifm_start_a, cluster_start_a, cluster_end_a, ofm_start_a = 0;

	uint32 ifm_c_group8_num_b, wt_cluster_size_b, cluster_num_b = 0;
	uint32 w_iter_num_b, kernel_group8_num_b = 0;
	uint32 wt_offset_b, wt_start_b, ifm_start_b, cluster_start_b, cluster_end_b, ofm_start_b = 0;

	uint32 ifm_c_group8_num_c, wt_cluster_size_c, cluster_num_c = 0;
	uint32 w_iter_num_c, kernel_group8_num_c = 0;
	uint32 wt_offset_c, wt_start_c, ifm_start_c, cluster_start_c, cluster_end_c, ofm_start_c = 0;

	uint32 ifm_reserve_size, ofm_reserve_start = 0;
	uint8  round_type, shift_num, prot_high, prot_low =0;
	
	uint32 wt_sz_a, bs_sz_a, shift_sz_a;
	uint32 wt_sz_b, bs_sz_b, shift_sz_b;
	uint32 wt_sz_c, bs_sz_c, shift_sz_c;
	uint32 bias_start_a, shift_start_a;
	uint32 bias_start_b, shift_start_b;
	uint32 bias_start_c, shift_start_c;

	// Calculate total weight size
	wt_sz_a = cshape_a->k_w * cshape_a->k_h * cshape_a->ifm_c * cshape_a->ofm_c; // in Bytes
	wt_sz_b = cshape_b->k_w * cshape_b->k_h * cshape_b->ifm_c * cshape_b->ofm_c; // in Bytes
	wt_sz_c = cshape_c->k_w * cshape_c->k_h * cshape_c->ifm_c * cshape_c->ofm_c; // in Bytes
	//KRNL_LOG_INFO(LOG_DEBUG, "wt_sz %d \n", wt_sz);
	bs_sz_a = cshape_a->ofm_c / 8;
	bs_sz_b = cshape_b->ofm_c / 8;
	bs_sz_c = cshape_c->ofm_c / 8;
	shift_sz_a = bs_sz_a;
	shift_sz_b = bs_sz_b;
	shift_sz_c = bs_sz_c;

    int input_tensor_alloc_mrlen;
    int input_tensor_ddr_address;
    int input_tensor_aline_mrlen;
    int ra_output_tensor_aline_mrlen ;
    int rb_output_tensor_aline_mrlen ;
    int rc_output_tensor_aline_mrlen;
    int concat_output_tensor_aline_mrlen;//������Ra Lb Concat
    int output_tensor_alloc_mrlen;
    int output_tensor_ddr_address;
    int output_tensor_aline_mrlen;
    
    int Ra_wt_tensor_alloc_mrlen,  Ra_wt_tensor_ddr_address;
    int Ra_bs_tensor_alloc_mrlen,  Ra_bs_tensor_ddr_address;
    int Rb_wt_tensor_alloc_mrlen,  Rb_wt_tensor_ddr_address;
    int Rb_bs_tensor_alloc_mrlen,  Rb_bs_tensor_ddr_address;
    int Rc_wt_tensor_alloc_mrlen,  Rc_wt_tensor_ddr_address;
    int Rc_bs_tensor_alloc_mrlen,  Rc_bs_tensor_ddr_address;
    
    /*3.1.1 calc input/middle/output tensor mr_ddr_len*/
    input_tensor_aline_mrlen     = CALC_FMWIDTH_GROUP8(cshape_a->ifm_w) * CALC_CHANNEL_GROUP8(cshape_a->ifm_c);
    ra_output_tensor_aline_mrlen = CALC_FMWIDTH_GROUP8(cshape_b->ifm_w) * CALC_CHANNEL_GROUP8(cshape_b->ifm_c);
    rb_output_tensor_aline_mrlen = CALC_FMWIDTH_GROUP8(cshape_c->ifm_w) * CALC_CHANNEL_GROUP8(cshape_c->ifm_c);
    rc_output_tensor_aline_mrlen = CALC_FMWIDTH_GROUP8(cshape_c->ifm_w) * CALC_CHANNEL_GROUP8(cshape_c->ofm_c);
    //concat_output_tensor_aline_mrlen = CALC_FMWIDTH_GROUP8(output_concat_dims_w) * CALC_CHANNEL_GROUP8(output_concat_dims_c);

	// Load all weights
	//KRNL_LOG_INFO(LOG_DEBUG, "Load Weights: [%d%d%x]->[%x]\n\r", wt_addr->x_pos, wt_addr->y_pos, wt_addr->lcaddr, (uint32 *)MMB_ADDR);
	//MMB can't be access by HPU scalar ALU
	LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)MMB_ADDR, wt_addr->x_pos, wt_addr->y_pos, wt_addr->lcaddr,  GMEM_ALIGN(wt_sz_a)));
    LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)MMB_ADDR, wt_addr->x_pos, wt_addr->y_pos, wt_addr->lcaddr,  GMEM_ALIGN(wt_sz_b)));
    LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)MMB_ADDR, wt_addr->x_pos, wt_addr->y_pos, wt_addr->lcaddr,  GMEM_ALIGN(wt_sz_c)));
    
	// Load all bias
	//KRNL_LOG_INFO(LOG_DEBUG, "Load Bias: [%x]->[%x]\n\r", bias_addr->lcaddr, (uint32 *)(BIAS_SHIFT_BLK));
	LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)(BIAS_SHIFT_BLK), bias_addr->x_pos, bias_addr->y_pos, bias_addr->lcaddr, GMEM_ALIGN(bs_sz_a)));
    LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)(BIAS_SHIFT_BLK), bias_addr->x_pos, bias_addr->y_pos, bias_addr->lcaddr, GMEM_ALIGN(bs_sz_b)));
    LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)(BIAS_SHIFT_BLK), bias_addr->x_pos, bias_addr->y_pos, bias_addr->lcaddr, GMEM_ALIGN(bs_sz_c)));
    
	// Load all shift_num
	//KRNL_LOG_INFO(LOG_DEBUG, "Load Shift_mtx: [%x]->[%x]\n\r", shift_addr->lcaddr, (uint32 *)(BIAS_SHIFT_BLK + bs_sz));
	LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)(BIAS_SHIFT_BLK + bs_sz_a), shift_addr->x_pos, shift_addr->y_pos, shift_addr->lcaddr, GMEM_ALIGN(shift_sz_a)));
    LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)(BIAS_SHIFT_BLK + bs_sz_b), shift_addr->x_pos, shift_addr->y_pos, shift_addr->lcaddr, GMEM_ALIGN(shift_sz_b)));
    LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)(BIAS_SHIFT_BLK + bs_sz_c), shift_addr->x_pos, shift_addr->y_pos, shift_addr->lcaddr, GMEM_ALIGN(shift_sz_c)));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Calculate total iteration
	kernel_group8_num_a = cshape_a->ofm_c / MTX_SCALE;			//ofm channel 输出fm channel 平均分成8�? kernel num is devided by 8
	w_iter_num_a = cshape_a->ifm_w / MTX_SCALE; 		//ifm w 平均分成8�?
	//KRNL_LOG_INFO(LOG_DEBUG, "(ifm_c, pshape_top, pshape_bottom, k_h): %d %d %d %d\n",cshape->ifm_c , pshape->top , pshape->bottom, cshape->k_h);
	ifm_c_group8_num_a = cshape_a->ifm_c / MTX_SCALE;		//ifm 输入fm channel 平均分成8�?
	//KRNL_LOG_INFO(LOG_DEBUG, "ifm_c_blk_stride = ifm_c / 8 : %d\n", ifm_c_blk_num);
	ifm_row_stride_a = ByteToHPU64BytesWord(cshape_a->ifm_w * cshape_a->ifm_c);
	ofm_row_stride_a = ByteToHPU64BytesWord(cshape_a->ifm_w * cshape_a->ofm_c);
	wt_cluster_size_a = ifm_c_group8_num_a * cshape_a->k_w;

	_set_mmac_fm_blk_stride(ifm_c_group8_num_a * 1);
	_set_mmac_fm_cluster_stride(ifm_row_stride_a);
	_set_mmac_fm_blk_size(ifm_c_group8_num_a - 1);	
	_set_mmac_wt_blk_stride(ifm_c_group8_num_a);
	_set_mmac_wt_cluster_stride(ByteToHPU64BytesWord(cshape_a->ifm_c * cshape_a->k_w * MTX_SCALE));			
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Initially Load Row-0 and Row-1
	for(i=0; i<2; i++)
	{
		//local_mem_fm_addr = alloc_local_fm(ifm, 1);
		//ddr_mem_fm_addr = alloc_ddr_fm(ddr_ifm);
		LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)localmem_fm_table_conv_a[i], ddr_ifm->x_pos, ddr_ifm->y_pos, ddr_mem_fm_addr, GMEM_ALIGN(ifm->bfm.row_size)));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
    	_set_mmac_region_start(ByteToHPU64BytesWord(localmem_fm_table_conv_a[i] - MEM_LCMEM_ADDR_S));
    	_set_mmac_region_end(ByteToHPU64BytesWord(localmem_fm_table_conv_a[i] + MMA_BLOCK_SIZE - MEM_LCMEM_ADDR_S) + 1);
	
		cluster_start_a = ByteToHPU64BytesWord(localmem_fm_table_conv_a[i] - MMA_ADDR);
		cluster_end_a = cluster_start_a + ifm_row_stride_a;       // always the end of the first ifm row
		cluster_num_a = 1; // always less than the full kernel height
		wt_offset_a = 0;    // skip top rows of kernel
			
		_set_mmac_fm_cluster_start(cluster_start_a);						//64Bytes 为单�?
		_set_mmac_fm_cluster_end(cluster_end_a + 1);						//64Bytes 为单�?
		_set_mmac_fm_cluster_num(cluster_num_a - 1);
		_set_mmac_fm_blk_num(cshape_a->k_w - 1);		

		round_type = 1; //mid
		shift_num = 0;
		prot_high = 127;
		prot_low = -128;
		_set_mmac_round_type(round_type);    //mid
		_set_mmac_fadd_shift_num(shift_num);
		_set_mmac_fadd_prot_high(prot_high);
		_set_mmac_fadd_prot_low(prot_low);

		for(int j=0; j<w_iter_num_a; j++)
        {
			ifm_start_a = cluster_start_a;

			for(int k=0; k<kernel_group8_num_a; k++)
			{
				ofm_start_a   = ByteToHPU64BytesWord(localmem_fm_table_dwc_b[i] - MMA_ADDR) + j*kernel_group8_num_a + k;
				wt_start_a    = ByteToHPU64BytesWord(MMB_BEGIN) + k * cshape_a->k_h * wt_cluster_size_a + wt_offset_a;
				bias_start_a  = ByteToHPU64BytesWord(BIAS_SHIFT_BLK) + kernel_group8_num_a;
				shift_start_a = ByteToHPU64BytesWord(BIAS_SHIFT_BLK + bs_sz_a) + kernel_group8_num_a;
				
				__intrinsic_func_conv__(ifm_start_a, wt_start_a, ofm_start_a, bias_start_a, shift_start_a, relu_a);
			}
		}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}

	// Calculate total iteration
	//ifm_reserve_size = ifm->bfm.row_size*ifm->bfm.row_num;
	//ofm_reserve_start = ofm->bfm.start_addr - MEM_LCMEM_ADDR_S;
	kernel_group8_num_b = cshape_b->ofm_c / MTX_SCALE;			//ofm channel 输出fm channel 平均分成8�? kernel num is devided by 8
	w_iter_num_b = cshape_b->ifm_w / MTX_SCALE; 		//ifm w 平均分成8�?
	h_iter_num_b = cshape_b->ifm_h + /*pshape->top + pshape->bottom*/ 2 - cshape_b->k_h + 1;
	//KRNL_LOG_INFO(LOG_DEBUG, "(ifm_c, pshape_top, pshape_bottom, k_h): %d %d %d %d\n",cshape->ifm_c , pshape->top , pshape->bottom, cshape->k_h);
	ifm_c_group8_num_b = cshape_b->ifm_c / MTX_SCALE;		//ifm 输入fm channel 平均分成8�?
	//KRNL_LOG_INFO(LOG_DEBUG, "ifm_c_blk_stride = ifm_c / 8 : %d\n", ifm_c_blk_num);
	ifm_row_stride_b = ByteToHPU64BytesWord(cshape_b->ifm_w * cshape_b->ifm_c);
	ofm_row_stride_b = ByteToHPU64BytesWord(cshape_b->ifm_w / 2 * cshape_b->ofm_c);
	wt_cluster_size_b = ifm_c_group8_num_b * cshape_b->k_w;

	//weight, ifm 都有cluster_stride 和blk_stride 都需要設�?
	//所有的stride 都不需�?减去1)-1�?目的：stride * 64Bytes == 内存中的真实偏移�?
	//所有的size/num 都要(减去1)-1
	//region_start, region_end, cluster_start�?cluster_end, 所有的end均为上一个最后元素的地址 + 1
	//目的：end - start == size�?计算size时不需要额�?+ 1�?
	//如内存中存储�?,1,2,3,4,5,6,7. start == 0, end == 7 + 1 = 8, size = end - start = 8 - 0 = 8
	// csrwi(CSR_MTX_REGION_START, 0);
	// csrwi(CSR_MTX_REGION_END, 4);
	// csrwi(CSR_MTX_BLK_SIZE, 0);
	// csrwi(CSR_MTX_BLK_NUM, 0);
	// csrwi(CSR_MTX_CLUSTER_START, 0);
	// csrwi(CSR_MTX_CLUSTER_END, 1);
	// csrwi(CSR_MTX_CLS_NUM, 0);
	// csrwi(CSR_MTXRW_BLK_STRIDE, 0);
	// csrwi(CSR_MTXRW_CLS_STRIDE, 0);
	// csrwi(CSR_MTXRO_BLK_STRIDE, 0);
	// csrwi(CSR_MTXRO_CLS_STRIDE, 0);

	//region_start region_end 和HPU计算没关系，只是为了软件编程方便。用region 卡住范围后，凡是超出region_end的地址，都会循环buf，从region_start重新开始计算偏�?
	//此处一次load 4 行ifm，所以region_end - region_start = row_size * 4
	_set_mmac_region_start(ByteToHPU64BytesWord(localmem_fm_table_dwc_b[0] - MEM_LCMEM_ADDR_S));			//64Bytes 为单�?
	_set_mmac_region_end(ByteToHPU64BytesWord(localmem_fm_table_dwc_b[2] + MMA_BLOCK_SIZE - MEM_LCMEM_ADDR_S) + 1);			//64Bytes 为单�?
	//与卷积运算的stride有关，stride == 1 时，blk_stride = blk_size; stride == 2 时，blk_stride = 2 * blk_size
	//即本次kernel的filter计算左上角起始地址�?与下一次kernel的filter计算左上角起始地址的偏移量
	//以下代码默认stride == 1, 即stride_shape->w_strd == 1
	_set_mmac_fm_blk_stride(ifm_c_group8_num_b * 2);							//8的倍数
	//cluster_stride: ifm 一整片�?包括所有的ifm_c, 所有的w，一�?所在的内存空间大小, 即本行其实地址与下一行起始地址的偏移量
	_set_mmac_fm_cluster_stride(ifm_row_stride_b);												//64Bytes 为单�?
	//ifm_c �?的倍数	
	_set_mmac_fm_blk_size(ifm_c_group8_num_b - 1);	

	//同一个kernel weight 左上角的点到 同一个kernelweight 左上角向右一个点(w方向)的偏移量 === ifm_c / 8 
	_set_mmac_wt_blk_stride(ifm_c_group8_num_b);									//8的倍数
	//8 个kernel 内部，第一个kernel左上角的点，到第一个kernel左上角向下一个点(h方向)的偏移量:一整片,类似：ifm_row_stride
	_set_mmac_wt_cluster_stride(ByteToHPU64BytesWord(cshape_b->ifm_c * cshape_b->k_w * MTX_SCALE));												//64Bytes 为单�?

	//KRNL_LOG_INFO(LOG_DEBUG, "region start: 0x%x", ByteToHPU64BytesWord(ifm->bfm.start_addr - MEM_LCMEM_ADDR_S));
	//KRNL_LOG_INFO(LOG_DEBUG, "region end: 0x%x", ByteToHPU64BytesWord(ifm->bfm.end_addr - MEM_LCMEM_ADDR_S) + 1);
	//KRNL_LOG_INFO(LOG_DEBUG, "fm_blk_stride: 0x%x", ifm_c_blk_num * stride_shape->w_strd);
	//KRNL_LOG_INFO(LOG_DEBUG, "fm_cls_stride: 0x%x", ifm_row_stride);
	//KRNL_LOG_INFO(LOG_DEBUG, "fm_blk_size: 0x%x", ifm_c_blk_num - 1);
	//KRNL_LOG_INFO(LOG_DEBUG, "wt_blk_stide: 0x%x", ifm_c_blk_num);
	//KRNL_LOG_INFO(LOG_DEBUG, "wt_cluster_stride: 0x%x", ByteToHPU64BytesWord(cshape->ifm_c * cshape->k_w * MTX_SCALE));

	// Start timer ticks
	//set_timer_tick(0x60, 0);
	//enable_timer_intr();
	//KRNL_LOG_INFO(LOG_DEBUG, "h_iter = %d\n", h_iter_num);
	
	for(i=0; i<h_iter_num_b; i++)
	{
		KRNL_LOG_INFO(LOG_DEBUG, "=====ifm H iter: %d / %d=====\n\r", i, h_iter_num_b);
        pad_shape_t pad = {1,1,1};
		// Pre-load the next ifm row if we are not at bottom
		if(!is_conv_bot(i, cshape_b, pad)){
			//local_mem_fm_addr = alloc_local_fm(ifm, 1);
			//ddr_mem_fm_addr = alloc_ddr_fm(ddr_ifm);
			//KRNL_LOG_INFO(LOG_DEBUG, "Load ifm row %d: [%x]->[%x]\n\r", (ifm->bfm.total_cnt - 1), ddr_mem_fm_addr, local_mem_fm_addr);
			LIBHIKL_NASSERT(__rd_rmt_chunk_non_blocking((uint32 *)localmem_fm_table_conv_a[localmem_fm_index_conv_a], ddr_ifm->x_pos, ddr_ifm->y_pos, ddr_mem_fm_addr, GMEM_ALIGN(ifm->bfm.row_size)));
			if( (++localmem_fm_index_conv_a) >= localmem_fm_num_conv_a)
			    localmem_fm_index_conv_a = 0;
		}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Make sure all required rows are ready
		//if(is_conv_bot(i, cshape, pshape) || is_conv_top(i, cshape, pshape)){
		//	LIBHIKL_NASSERT(ifm->cur_cnt < 2);
		//}
		//else{
		//	LIBHIKL_NASSERT(ifm->cur_cnt < 3);
		//}

		// Allocate local space for 1 ofm row
		//disable_timer_intr();
		//ofm_ptr = (uint32 *)trans_and_alloc_local_fm_tail(ofm, ddr_ofm, 1);
		//enable_timer_intr();

		if(i < /*pshape->top*/1){
			// cluster_start = ByteToHPU64BytesWord(MMA_BEGIN);
			// always start from the first ifm row, pad automatically increments inside mmac
			cluster_start_b = ByteToHPU64BytesWord(ifm->bfm.start_addr - MMA_ADDR);
			cluster_end_b = cluster_start_b + ifm_row_stride_b;       // always the end of the first ifm row
			cluster_num_b = i + cshape_b->k_h - /*pshape->top*/1; // always less than the full kernel height
			wt_offset_b = wt_cluster_size_b * (/*pshape->top*/1 - i);    // skip top rows of kernel
		}
		else{
			// cluster_start = ByteToHPU64BytesWord(MMA_BEGIN) + (i - pshape->top) * ifm_row_stride;
			// from the middle of ifm
			cluster_start_b = ByteToHPU64BytesWord(ifm->bfm.start_addr - MMA_ADDR) + (i - /*pshape->top*/1) * ifm_row_stride_b;
			cluster_end_b = cluster_start_b + ifm_row_stride_b;
			wt_offset_b = 0; // always start from kernel top

			if((i + cshape_b->k_h) > (cshape_b->ifm_h + /*pshape->top*/1))   // at bottom
				cluster_num_b = cshape_b->ifm_h + /*pshape->top*/1 - i;   // always less than the full kernel height
			else // at middle
				cluster_num_b = cshape_b->k_h;   // always equal the full kernel height
		}

		//KRNL_LOG_INFO(LOG_DEBUG, "cluster_start: [%x] cluster_end: [%x]", cluster_start_b, cluster_end_b);
		//cluster_stride: ifm 一整片�?包括所有的ifm_c, 所有的w，一�?所在的内存空间大小, 即本行其实地址与下一行起始地址的偏移量
		_set_mmac_fm_cluster_start(cluster_start_b);						//64Bytes 为单�?
		_set_mmac_fm_cluster_end(cluster_end_b + 1);						//64Bytes 为单�?
		//cluster_num == k_h
		_set_mmac_fm_cluster_num(cluster_num_b - 1);
		//blk_num == k_w
		_set_mmac_fm_blk_num(cshape_b->k_w - 1);
		//KRNL_LOG_INFO(LOG_DEBUG, "cluster_num: 0x%x, blk_num: 0x%x", cluster_num_b, cshape_b->k_w);
		

		round_type = 1; //mid
		shift_num = 0;
		prot_high = 127;
		prot_low = -128;
		_set_mmac_round_type(round_type);    //mid
		_set_mmac_fadd_shift_num(shift_num);
		// asm volatile("mv t6, %0"::"r"(prot_high):);
		_set_mmac_fadd_prot_high(prot_high);
		_set_mmac_fadd_prot_low(prot_low);

		for(int j=0; j<w_iter_num_b; j++)        //w_iter_num = cshape->ifm_w / MTX_SCALE; 		//ifm w 平均分成8�?
        {
			ifm_start_b = cluster_start_b + (j - /*pshape->side*/1) * ifm_c_group8_num_b;
    		//KRNL_LOG_INFO(LOG_DEBUG, "=====ifm W iter: %d / %d===== ifm_start: %x", j, w_iter_num_b, ifm_start_b);
			
			for(int k=0; k<kernel_group8_num_b; k++){    //times_of_8_kernels = cshape->ofm_c / MTX_SCALE;			//ofm channel 输出fm channel 平均分成8�? kernel num is devided by 8
				//each iteration, process 8 kernels
    			//KRNL_LOG_INFO(LOG_DEBUG, "=====8 kernels iter: %d / %d=====", k, kernel_group8_num_b);
				wt_start_b = ByteToHPU64BytesWord(MMB_BEGIN) + k * cshape_b->k_h * wt_cluster_size_b + wt_offset_b;
				
				//ifm_start = cluster_start + offset
				//KRNL_LOG_INFO(LOG_DEBUG, "cluster_start = %x, offset = %x\n", cluster_start_b, (j - /*pshape->side*/1) * ifm_c_group8_num_b);
				// ofm_start = ByteToHPU64BytesWord(ofm_reserve_start) + i*ofm_row_stride + j*times_of_8_kernels + k;
				ofm_start_b = ByteToHPU64BytesWord(localmem_fm_table_dwc_b[localmem_fm_index_dwc_b] - MMA_ADDR) + j*kernel_group8_num_b + k;
				
				//KRNL_LOG_INFO(LOG_DEBUG, "ofm_start = 0x%lx\n", ofm_start_b);
				bias_start_b = ByteToHPU64BytesWord(BIAS_SHIFT_BLK) + kernel_group8_num_b;
				shift_start_b = ByteToHPU64BytesWord(BIAS_SHIFT_BLK + bs_sz) + kernel_group8_num_b;
				//KRNL_LOG_INFO(LOG_DEBUG, "***************ifm = %x, wt= %x, ofm= %x \n", ifm_start_b, wt_start_b, ofm_start_b);
				
				__intrinsic_func_dwconv__(ifm_start_b, wt_start_b, ofm_start_b, bias_start_b, shift_start_b, relu);
			}
		}
		
		if( (++localmem_fm_index_dwc_b) >= localmem_fm_num_dwc_b)
			   localmem_fm_index_dwc_b = 0;
		//KRNL_LOG_INFO(LOG_DEBUG, "Workload: ifm=[%x] ofm=[%x]\n\r", local_mem_fm_addr, ofm_ptr);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cluster_start_c = ByteToHPU64BytesWord(ifm->bfm.start_addr - MMA_ADDR);
		cluster_end_c = cluster_start_c + ifm_row_stride_c;       // always the end of the first ifm row
		cluster_num_c = i + cshape_c->k_h - /*pshape->top*/0; // always less than the full kernel height
		wt_offset_c = 0;    // skip top rows of kernel
			
		_set_mmac_fm_cluster_start(cluster_start_c);						//64Bytes 为单�?
		_set_mmac_fm_cluster_end(cluster_end_c + 1);						//64Bytes 为单�?
		_set_mmac_fm_cluster_num(cluster_num_c - 1);
		_set_mmac_fm_blk_num(cshape_c->k_w - 1);		

		round_type = 1; //mid
		shift_num = 0;
		prot_high = 127;
		prot_low = -128;
		_set_mmac_round_type(round_type);    //mid
		_set_mmac_fadd_shift_num(shift_num);
		_set_mmac_fadd_prot_high(prot_high);
		_set_mmac_fadd_prot_low(prot_low);

		for(int j=0; j<w_iter_num_c; j++)
        {
			ifm_start_c = cluster_start_c + (j - /*pshape->side*/0) * ifm_c_group8_num_c;
    		//KRNL_LOG_INFO(LOG_DEBUG, "=====ifm W iter: %d / %d===== ifm_start: %x", j, w_iter_num_c, ifm_start_c);
			
			for(int k=0; k<kernel_group8_num_c; k++)
			{
				wt_start_c    = ByteToHPU64BytesWord(MMB_BEGIN) + k * cshape_c->k_h * wt_cluster_size_c + wt_offset_c;
				ofm_start_c   = ByteToHPU64BytesWord((int)ofm_ptr - MMA_ADDR) + j*kernel_group8_num_c + k;
				bias_start_c  = ByteToHPU64BytesWord(BIAS_SHIFT_BLK) + kernel_group8_num_c;
				shift_start_c = ByteToHPU64BytesWord(BIAS_SHIFT_BLK + bs_sz_c) + kernel_group8_num_c;
				
				__intrinsic_func_conv__(ifm_start_c, wt_start_c, ofm_start_c, bias_start_c, shift_start_c, relu_c);
			}
		}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Delete the row that is used up if we are not at the top
		//if(!is_conv_top(i, cshape_c, pshape_c)){
		//	KRNL_LOG_INFO(LOG_DEBUG, "Release ifm row %d\n\r", ifm->cur_valid_idx);
		//	dealloc_local_fm(ifm, 1);
		//}

		// Check for the next ifm row if we are not at bottom
		pad_shape_t pad = {1,1,1};
		if(!is_conv_bot(i, cshape_c, pad))
			__ndma_poll();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cluster_start_a = ByteToHPU64BytesWord(localmem_fm_table_conv_c[0] - MMA_ADDR);
		cluster_end_a = cluster_start_a + ifm_row_stride_a;       // always the end of the first ifm row
		cluster_num_a = 1; // always less than the full kernel height
		wt_offset_a = 0;    // skip top rows of kernel
			
		_set_mmac_fm_cluster_start(cluster_start_a);						//64Bytes 为单�?
		_set_mmac_fm_cluster_end(cluster_end_a + 1);						//64Bytes 为单�?
		_set_mmac_fm_cluster_num(cluster_num_a - 1);
		_set_mmac_fm_blk_num(cshape_a->k_w - 1);		

		round_type = 1; //mid
		shift_num = 0;
		prot_high = 127;
		prot_low = -128;
		_set_mmac_round_type(round_type);    //mid
		_set_mmac_fadd_shift_num(shift_num);
		_set_mmac_fadd_prot_high(prot_high);
		_set_mmac_fadd_prot_low(prot_low);

		for(int j=0; j<w_iter_num_a; j++)
        {
			ifm_start_a = cluster_start_a + (j - /*pshape->side*/0) * ifm_c_group8_num_a;
    		//KRNL_LOG_INFO(LOG_DEBUG, "=====ifm W iter: %d / %d===== ifm_start: %x", j, w_iter_num_a, ifm_start_a);
			
			for(int k=0; k<kernel_group8_num_a; k++)
			{
				wt_start_a = ByteToHPU64BytesWord(MMB_BEGIN) + k * cshape_a->k_h * wt_cluster_size_a + wt_offset_a;
				ofm_start_a = ByteToHPU64BytesWord(localmem_fm_table_out[0] - MMA_ADDR) + j*kernel_group8_num_a + k;
				bias_start_a = ByteToHPU64BytesWord(BIAS_SHIFT_BLK) + kernel_group8_num_a;
				shift_start_a = ByteToHPU64BytesWord(BIAS_SHIFT_BLK + bs_sz_a) + kernel_group8_num_a;
				
				__intrinsic_func_conv__(ifm_start_a, wt_start_a, ofm_start_a, bias_start_a, shift_start_a, relu_a);
			}
		}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
	}

	//disable_timer_intr();
	//KRNL_LOG_INFO(LOG_DEBUG, "calculation is end, ndma begins...\n\r");
	//_ndma_to_ddr_nowait(ofm, ddr_ofm);
	//_ndma_to_ddr_nowait(ofm, ddr_ofm);
	//_ndma_to_ddr_nowait(ofm, ddr_ofm);
	//KRNL_LOG_INFO(LOG_DEBUG, "ndma ends...\n\r");
	
	return;
}


