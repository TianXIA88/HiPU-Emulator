#include "hihw.h"
#include "libconv.h"
#include "dma.h"
#include "lock.h"
#include "int.h"
#include "krnl_log.h"
#include "hisdk_config.h"
#include "operators/hi_krnl_param_conv2d.h"

// #define check_32bits_output

extern void buf_print(uint32_t buf_addr, uint32_t buf_len);

static uint32_t mmac_region_start = 0;
static uint32_t mmac_region_end = 0;
static uint32_t mmac_ifm_blk_size = 0;
static uint32_t ifm_c_group8_num = 0;
// static uint32_t mmac_ifm_cluster_stride = ByteToW64(MMA_BANK_SIZE);
static uint32_t mmac_ifm_cluster_stride = 0;
static uint32_t mmac_ifm_blk_stride = 0;
static uint32_t mmac_cluster_start = 0, mmac_cluster_end = 0, mmac_cluster_num = 0;

static uint32_t ifm_row_size = 0;
static uint32_t wt_cluster_size = 0;
static uint32_t conv_type = CONV_TYPE_CLASSIC; 	// CONV_TYPE_CLASSIC, CONV_TYPE_DEPTH_WISE
static uint32_t wt_offset = 0;

// void set_conv_type(uint32_t input_conv_type)
// {
//     conv_type = input_conv_type;
// }

//所有矩阵指令，向量指令的寻址都是相对于于MMA的首地址的64Bytes的倍数
//比如ifm == ByteToHPU64BytesWord(addr_in_x_bank_in_MMA - MMA)
//比如bias == ByteToHPU64BytesWord(addr_in_x_bank_in_MMA - MMA)
//比如shift == ByteToHPU64BytesWord(addr_in_x_bank_in_MMA - MMA)
//wt == ByteToW64(addr_in_x_bank_in_MMB - MMB)
void __intrinsic_func__(uint32 ifm, uint32 wt, uint32 ofm, uint32 bias_start, uint32 shift_start, uint32 relu, int conv_type, uint32_t b_with_bias_shift) {

	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "kernel mac: ifm 0x%x wt 0x%x ofm 0x%x bias 0x%x shift 0x%x relu %d conv_type %d, check_origins_output %d",			\
					ifm, wt, ofm, bias_start, shift_start, relu, conv_type, b_with_bias_shift);
	uint32 vfadd_bias = 0;

	asm volatile("mv t1, %0"::"r"(ifm):);
	asm volatile("mv t2, %0"::"r"(wt):);
	asm volatile("mv t3, %0"::"r"(ofm):);
	asm volatile("mv t4, %0"::"r"(vfadd_bias):);
	asm volatile("mv t5, %0"::"r"(bias_start));
	asm volatile("mv t6, %0"::"r"(shift_start));
	_clr_vreg(vr1);
	switch(conv_type)
	{
		case CONV_TYPE_CLASSIC:
			mmac(VPR_NONE, vr1, t1, t2, vr1);
		break;
		case CONV_TYPE_DEPTH_WISE:
			mdmac(VPR_NONE, vr1, t1, t2, vr1);
		break;
	}
	vlw(vr2, t5, 0);
	vlb(vr3, t6, 0);

	// vsb(vr2, t3, 0);							//可以输出到ofm，供调试查看
	if(b_with_bias_shift > 0) 
	{
	 	vadd_vv(VPR_NONE, vr1, vr1, vr2);
	#ifdef HPU200_RSHIFT_WITH_DECREMENT1_THEN_VFADD_RSHIFT1   
		uint32 decrement = 1;
		asm volatile("mv t0, %0"::"r"(decrement));
		vsub_vs(VPR_NONE, vr3, vr3, t0);
	#endif
		vsra_vv(VPR_NONE, vr1, vr1, vr3);		//算术移位，保留符号位
		// vsrl_vv(VPR_NONE, vr1, vr1, vr3);			//逻辑移位，不保留符号位
		// vsrl_vi(VPR_NONE, vr1, vr1, 4);		/shift for debug
		//now still 32 bits per item
		if (relu)
		{
			// KRNL_LOG_INFO(LOG_DEBUG, "relu_not_zero : %d", relu);
			vmax_vs(VPR_NONE, vr1, vr1, 0);
		}
		vfadd_vs(VPR_NONE, vr1, vr1, t4);
		// can follow other vector operations ...
	}

	if(b_with_bias_shift == 0) 
	{
		// vsrl_vi(VPR_NONE, vr1, vr1, 16);			//向右偏移8bit， 可以将origins 的乘加结果的第二个8bits 输出，以供调试
	}
	vsb(vr1, t3, 0);

	// vsw(vr1, t3, 0);
	// qemu_fprint(QEMU_LOG_MEM, ofm, 64);
	return;
}

void _set_mmac_vfadd_param_()
{
	uint8 round_type = 1; //mid
#ifdef HPU200_RSHIFT_WITH_DECREMENT1_THEN_VFADD_RSHIFT1   
	uint8 shift_num = 1;			//先加 再移位， 这个移位带四舍五入
#else
	uint8 shift_num = 0;
#endif 
	uint8 prot_high = 127;
	uint8 prot_low = -128;
	_set_mmac_round_type(round_type);    //mid
	_set_mmac_fadd_shift_num(shift_num);
	_set_mmac_fadd_prot_high(prot_high);
	_set_mmac_fadd_prot_low(prot_low);
}

void _set_mmac_param_for_whole_conv_(conv2d_params_t *conv2d_param)
{
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

	ifm_c_group8_num = conv2d_param->cshape.ifm_c / MTX_SCALE;		//local_ifm 输入fm channel 平均分成8份
	ifm_row_size = ByteToW64(conv2d_param->cshape.ifm_c * conv2d_param->cshape.ifm_w);
	wt_cluster_size = ifm_c_group8_num * conv2d_param->cshape.k_w;

	//weight, local_ifm 都有cluster_stride 和blk_stride 都需要設置
	//所有的stride 都不需要(减去1)-1， 目的：stride * 64Bytes == 内存中的真实偏移量
	//所有的size/num 都要(减去1)-1
	//mmac_region_start, mmac_region_end, cluster_start， mmac_cluster_end, 所有的end均为上一个最后元素的地址 + 1
	//目的：end - start == size， 计算size时不需要额外 + 1了
	//如内存中存储了0,1,2,3,4,5,6,7. start == 0, end == 7 + 1 = 8, size = end - start = 8 - 0 = 8
	
	//与卷积运算的stride有关，stride.w == 1 时，blk_stride = blk_size; stride.w == 2 时，blk_stride = 2 * blk_size
	//即本次ofm 8 个输出点中第一个点，与下一次同一行（同一个大片，w方向下8个输出点）中8个输出点中第一个点的内存地址偏移量
	mmac_ifm_blk_stride = ifm_c_group8_num * conv2d_param->dilat_shape.w_dilat;
	_set_mmac_fm_blk_stride(mmac_ifm_blk_stride);							//8的倍数
	//cluster_stride: local_ifm 一整片行(包括所有的ifm_c, 所有的w，一行)所在的内存空间大小, 即本行起始地址与下一行起始地址的偏移量
	//对于1x1的卷积，_set_mmac_fm_cluster_stride 无效
	_set_mmac_fm_cluster_stride(mmac_ifm_cluster_stride);
	_set_mtx_pad_type(MTX_PAD_TYPE_1);

	//8 个kernel 内部，第一个kernel左上角的点，到第一个kernel左上角向下一个点(h方向)的偏移量:一整片,类似：ifm_row_stride
	uint32_t mmac_wt_cluster_stride = ByteToW64(conv2d_param->cshape.ifm_c * conv2d_param->cshape.k_w * MTX_SCALE);
	//对于1x1的卷积，_set_mmac_wt_cluster_stride 无效
	_set_mmac_wt_cluster_stride(mmac_wt_cluster_stride);					//64Bytes 为单位
	_set_mmac_fm_blk_num(conv2d_param->cshape.k_w - 1);

	//ifm_c 的8的倍数	
	if (conv_type == CONV_TYPE_DEPTH_WISE)
	{
		mmac_ifm_blk_size = 0;
	}
	else
	{
		mmac_ifm_blk_size = ifm_c_group8_num - 1;
	}
	_set_mmac_fm_blk_size(mmac_ifm_blk_size);	

	//同一个kernel weight 左上角的点到 同一个kernelweight 左上角向右一个点(w方向)的偏移量 === ifm_c / 8 
	//对于1x1的卷积，_set_mmac_wt_blk_stride 无效
	_set_mmac_wt_blk_stride(ifm_c_group8_num);									//8的倍数

	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "fm_blk_stride: 0x%x", mmac_ifm_blk_stride);
	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "fm_cluster_stride: 0x%x", mmac_ifm_cluster_stride);
	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "fm_blk_size: 0x%x", mmac_ifm_blk_size);
	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "fm_blk_num: 0x%x", conv2d_param->cshape.k_w - 1);
	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "wt_blk_stide: 0x%x", ifm_c_group8_num);
	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "mmac_wt_cluster_stride: 0x%x", mmac_wt_cluster_stride);
}

void _set_mmac_cluster_params_()
{
	//1. 每次mmac的cls start和cls end必须在reg start和reg end之间，在mmac执行过程中，如果地址超过region end ，会自动循环到region start。
	//2. 每次mmac开始前，cls start设置为此次卷积所需的第一行ifm在mma中的实际地址。
	//3. ifm start不要求在region_start ~ region_end内，基于cls start计算ifm start就可以，不需要考虑ifm start是不是在region_start ~ region_end内。
	//4. ifm_start = mmac_cluster_start + padding_offset
	//cluster_stride: ifm 一整片行(包括所有的ifm_c, 所有的w，一行)所在的内存空间大小, 即本行其实地址与下一行起始地址的偏移量
	//mmac_cluster_start mmac_cluster_end	必须是此次卷积运算的第一行ifm的实际的物理地址
	_set_mmac_fm_cluster_start(mmac_cluster_start);						//64Bytes 为单位	
	_set_mmac_fm_cluster_end(mmac_cluster_end);						//64Bytes 为单位
	//mmac_cluster_num == k_h - 1
	_set_mmac_fm_cluster_num(mmac_cluster_num); //mmac_cluster_num == k_h - 1
	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "mmac_cluster_start: 0x[%x] mmac_cluster_end: 0x[%x]", mmac_cluster_start, mmac_cluster_end);
	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "mmac_cluster_num: 0x%x", mmac_cluster_num);
}

uint32_t _set_mmac_region_params_and_get_ifm_start_(conv2d_params_t *conv2d_param, uint32_t w_iter)
{
	uint32_t ifm_start = 0;
	ifm_start = mmac_cluster_start + (w_iter - conv2d_param->pshape.left) * ifm_c_group8_num;
	if (ifm_start == mmac_cluster_start)
	{
		//no padding here
		//mmac_region_start mmac_region_end 和HPU计算没关系，只是为了软件编程方便。用region 卡住范围后，凡是超出region_end的地址，都会循环buf，从region_start重新开始计算偏移
		//此处一次load 4 行ifm，所以region_end - mmac_region_start = row_size * 4
		//mmac_region_start, mmac_region_end 必须减去相应的padding(有padding 运算的时候，没有padding的时候不考虑)
	}
	else
	{
		//has padding here
		//mmac_region_start mmac_region_end 和HPU计算没关系，只是为了软件编程方便。用region 卡住范围后，凡是超出region_end的地址，都会循环buf，从region_start重新开始计算偏移
		//此处一次load 4 行ifm，所以region_end - mmac_region_start = row_size * 4
		//mmac_region_start, mmac_region_end 必须减去相应的padding(有padding 运算的时候，没有padding的时候不考虑)
		mmac_region_start -= conv2d_param->pshape.left * ifm_c_group8_num;
		mmac_region_end -= conv2d_param->pshape.left * ifm_c_group8_num;
	}
	_set_mmac_region_start(mmac_region_start);		//64Bytes 为单位
	_set_mmac_region_end(mmac_region_end);			//64Bytes 为单位

	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "region start: 0x%x", mmac_region_start);
	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "region end: 0x%x", mmac_region_end);
	return ifm_start;
}

uint32_t _set_mmac_region_params_and_get_ifm_add_start_( uint32_t ifm_ptr_c_add ,conv2d_params_t *conv2d_param, uint32_t w_iter)
{
	uint32_t ifm_add_start = 0;
	ifm_add_start = ifm_ptr_c_add + (w_iter - conv2d_param->pshape.left) * ifm_c_group8_num;
	if (ifm_add_start == ifm_ptr_c_add)
	{
		//no padding here
		//mmac_region_start mmac_region_end 和HPU计算没关系，只是为了软件编程方便。用region 卡住范围后，凡是超出region_end的地址，都会循环buf，从region_start重新开始计算偏移
		//此处一次load 4 行ifm，所以region_end - mmac_region_start = row_size * 4
		//mmac_region_start, mmac_region_end 必须减去相应的padding(有padding 运算的时候，没有padding的时候不考虑)
	}
	else
	{
		//has padding here
		//mmac_region_start mmac_region_end 和HPU计算没关系，只是为了软件编程方便。用region 卡住范围后，凡是超出region_end的地址，都会循环buf，从region_start重新开始计算偏移
		//此处一次load 4 行ifm，所以region_end - mmac_region_start = row_size * 4
		//mmac_region_start, mmac_region_end 必须减去相应的padding(有padding 运算的时候，没有padding的时候不考虑)
		mmac_region_start -= conv2d_param->pshape.left * ifm_c_group8_num;
		mmac_region_end -= conv2d_param->pshape.left * ifm_c_group8_num;
	}
	_set_mmac_region_start(mmac_region_start);		//64Bytes 为单位
	_set_mmac_region_end(mmac_region_end);			//64Bytes 为单位

	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "region start: 0x%x", mmac_region_start);
	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "region end: 0x%x", mmac_region_end);
	return ifm_add_start;
}


void _set_wt_offset_(conv2d_params_t *conv2d_param, uint32_t h_iter)
{
	if(is_ifm_top(h_iter * conv2d_param->strd_shape.h_strd, &conv2d_param->cshape, &conv2d_param->pshape)){
		wt_offset = wt_cluster_size * (conv2d_param->pshape.top - h_iter*conv2d_param->strd_shape.h_strd);    // skip top rows of kernel
	}
	else{
		wt_offset = 0; // always start from kernel top
	}
}

//
void one_row_conv(int h_iter, conv2d_params_t *conv2d_param,        	  	\
						uint32_t wt_lcmem_start_addr,							\
						uint32_t ofm_row_lcmem_start_addr,						\
						uint32_t bias_lcmem_start_addr,							\
						uint32_t shift_lcmem_start_addr,						\
                    	uint32_t param_mmac_region_start,                     	\
                    	uint32_t param_mmac_region_end,                       	\
                    	uint32_t param_mmac_ifm_cluster_stride,               	\
                    	uint32_t param_mmac_cluster_start,                    	\
                    	uint32_t param_mmac_cluster_end,                      	\
                    	uint32_t param_mmac_cluster_num,						\
						uint32_t input_conv_type,								\
						uint32_t b_with_bias_shift)
{
	KRNL_LOG_INFO(LOG_SYSTEM, "=====ifm row index(h_iter): %d", h_iter);

	// KRNL_LOG_INFO(LOG_SYSTEM, "ifm_h: %d", conv2d_param->cshape.ifm_h);
	// KRNL_LOG_INFO(LOG_SYSTEM, "ifm_w: %d", conv2d_param->cshape.ifm_w);
	// KRNL_LOG_INFO(LOG_SYSTEM, "ifm_c: %d", conv2d_param->cshape.ifm_c);
	// KRNL_LOG_INFO(LOG_SYSTEM, "ofm_c: %d", conv2d_param->cshape.ofm_c);
	// KRNL_LOG_INFO(LOG_SYSTEM, "k_h: %d", conv2d_param->cshape.k_h);
	// KRNL_LOG_INFO(LOG_SYSTEM, "k_w: %d", conv2d_param->cshape.k_w);
	// KRNL_LOG_INFO(LOG_SYSTEM, "pshape_top: %d",  conv2d_param->pshape.top);
	// KRNL_LOG_INFO(LOG_SYSTEM, "pshape_bottom: %d", conv2d_param->pshape.bottom);
	// KRNL_LOG_INFO(LOG_SYSTEM, "pshape_left: %d pshape_right: %d", conv2d_param->pshape.left, conv2d_param->pshape.right);
	// KRNL_LOG_INFO(LOG_SYSTEM, "h_strd: %d", conv2d_param->strd_shape.h_strd);
	// KRNL_LOG_INFO(LOG_SYSTEM, "w_strd: %d", conv2d_param->strd_shape.w_strd);
	// KRNL_LOG_INFO(LOG_SYSTEM, "relu: %d", conv2d_param->relu);
	// KRNL_LOG_INFO(LOG_DEBUG, "ofm_row_lcmem_start_addr 0x%x", ofm_row_lcmem_start_addr);
    conv_type = input_conv_type;
	mmac_region_start = param_mmac_region_start;
	mmac_region_end = param_mmac_region_end;
	mmac_ifm_cluster_stride = param_mmac_ifm_cluster_stride;
	mmac_cluster_start = param_mmac_cluster_start; 
	mmac_cluster_end = param_mmac_cluster_end; 
	mmac_cluster_num = param_mmac_cluster_num;

	// makesure_needed_ifm_rows_ready(h_iter, conv2d_param);
	// Allocate local space for 1 ofm row
	// local_ofm_output_row_addr = (uint32 *)alloc_local_fm_with_drain_to_ddr_if_needed(&local_ofm, &ddr_ofm, 1);

	_set_mmac_vfadd_param_();
	_set_mmac_param_for_whole_conv_(conv2d_param);
	_set_mmac_cluster_params_(conv2d_param, h_iter);
	_set_wt_offset_(conv2d_param, h_iter);
	
	uint32 wt_start = 0, ifm_start = 0, ifm_inner_start = 0, ofm_start = 0, bias_start = 0, shift_start = 0;

	uint32 w_iter_num = conv2d_param->cshape.ifm_w / MTX_SCALE; 		//local_ifm w 平均分成8份
	uint32 kernel_group8_num = conv2d_param->cshape.ofm_c / MTX_SCALE;

	for(int w_iter=0; w_iter<w_iter_num; w_iter+=conv2d_param->strd_shape.w_strd){    //w_iter_num = p_conv2d_entry->conv2d.cshape.ifm_w / MTX_SCALE; 		//ifm w 平均分成8份

		ifm_start = _set_mmac_region_params_and_get_ifm_start_(conv2d_param, w_iter);

   		// KRNL_LOG_INFO(LOG_SYSTEM, "=====ifm W iter: %d / %d===== ifm_start: 0x%x", w_iter, w_iter_num, ifm_start);

		//ofm channel 输出fm channel 平均分成8份, kernel num is devided by 8
		for(int k=0; k<kernel_group8_num; k++){    //kernel_group8_num = conv2d_param->cshape.ofm_c / MTX_SCALE;			
   			// KRNL_LOG_INFO(LOG_SYSTEM, "=====ifm kernel_group8_num iter: %d / %d", k, kernel_group8_num);
			if (conv_type == CONV_TYPE_DEPTH_WISE)
			{
				//each kernel has only 1 channel
				ifm_inner_start = ifm_start + k;
				// ifm_start += k;
				wt_start = wt_lcmem_start_addr + k + wt_offset;
				// ofm_start = ofm_row_lcmem_start_addr + w_iter*kernel_group8_num + k;
				// bias_start = bias_lcmem_start_addr +  k * 4;		//bias 32bits per item
				// shift_start = shift_lcmem_start_addr + k;
				// __intrinsic_func__(ifm_inner_start, wt_start, ofm_start, bias_start, shift_start, conv2d_param->relu, conv_type);
			}
			else
			{
				ifm_inner_start = ifm_start;
				//each iteration, process 8 kernels
				wt_start = wt_lcmem_start_addr + k * conv2d_param->cshape.k_h * wt_cluster_size + wt_offset;
				// __intrinsic_func__(ifm_start, wt_start, ofm_start, bias_start, shift_start, conv2d_param->relu, conv_type);
			}

			ofm_start = ofm_row_lcmem_start_addr + (w_iter/conv2d_param->strd_shape.w_strd) * kernel_group8_num + k;
			bias_start = bias_lcmem_start_addr +  k * 4;		//bias 32bits per item
			shift_start = shift_lcmem_start_addr + k;
			__intrinsic_func__(ifm_inner_start, wt_start, ofm_start, bias_start, shift_start, conv2d_param->relu, conv_type, b_with_bias_shift);
			if (g_ulPrintDebugLogFlag)
			{
				vmu_poll();
				KRNL_LOG_INFO(LOG_DEBUG, "ifm_inner_start");
				buf_print(MMA_START_ADDR + W64ToByte(ifm_inner_start), GMEM_ALIGN(64));
				// KRNL_LOG_INFO(LOG_HARDWARE_CMD, "mmac_cluster_start");
				// buf_print(MMA_START_ADDR + W64ToByte(mmac_cluster_start), GMEM_ALIGN(64));
				KRNL_LOG_INFO(LOG_DEBUG, "ofm_start");
				buf_print(MMA_START_ADDR + W64ToByte(ofm_start), GMEM_ALIGN(64));
				// KRNL_LOG_INFO(LOG_DEBUG, "bias_start");
				// buf_print(MMA_START_ADDR + W64ToByte(bias_start), GMEM_ALIGN(64));
				// KRNL_LOG_INFO(LOG_DEBUG, "shift_start");
				// buf_print(MMA_START_ADDR + W64ToByte(shift_start), GMEM_ALIGN(64));
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void __intrinsic_conv_add__(uint32 ifm, uint32 wt, uint32 ofm, uint32 bias_start, uint32 shift_start, uint32 relu, int conv_type, uint32_t b_with_bias_shift, uint32_t ifm_add, uint32_t add_shift, uint32_t add_clip) {

	KRNL_LOG_INFO(LOG_HARDWARE_CMD, "conv_add kernel mac: ifm 0x%x ifm_add 0x%x add_shift %d add_clip %d wt 0x%x ofm 0x%x bias 0x%x shift 0x%x relu %d conv_type %d, check_origins_output %d",			\
					ifm, ifm_add , add_shift, add_clip ,wt, ofm, bias_start, shift_start, relu, conv_type, b_with_bias_shift );
	uint32 vfadd_bias = 0;

	asm volatile("mv t1, %0"::"r"(ifm):);
	asm volatile("mv t2, %0"::"r"(wt):);
	asm volatile("mv t3, %0"::"r"(ofm):);
	asm volatile("mv t4, %0"::"r"(vfadd_bias):);
	asm volatile("mv t5, %0"::"r"(bias_start));
	asm volatile("mv t6, %0"::"r"(shift_start));
	_clr_vreg(vr1);
	switch(conv_type)
	{
		case CONV_TYPE_CLASSIC:
			mmac(VPR_NONE, vr1, t1, t2, vr1);
		break;
		case CONV_TYPE_DEPTH_WISE:
			mdmac(VPR_NONE, vr1, t1, t2, vr1);
		break;
	}
	vlw(vr2, t5, 0);
	vlb(vr3, t6, 0);

	// vsb(vr2, t3, 0);							//可以输出到ofm，供调试查看
	if(b_with_bias_shift > 0) 
	{
	 	vadd_vv(VPR_NONE, vr1, vr1, vr2);
	#ifdef HPU200_RSHIFT_WITH_DECREMENT1_THEN_VFADD_RSHIFT1   
		uint32 decrement = 1;
		asm volatile("mv t0, %0"::"r"(decrement));
		vsub_vs(VPR_NONE, vr3, vr3, t0);
	#endif
		vsra_vv(VPR_NONE, vr1, vr1, vr3);		//算术移位，保留符号位
		// vsrl_vv(VPR_NONE, vr1, vr1, vr3);			//逻辑移位，不保留符号位
		// vsrl_vi(VPR_NONE, vr1, vr1, 4);		/shift for debug
		//now still 32 bits per item
		if (relu)
		{
			// KRNL_LOG_INFO(LOG_DEBUG, "relu_not_zero : %d", relu);
			vmax_vs(VPR_NONE, vr1, vr1, 0);
		}
		vfadd_vs(VPR_NONE, vr1, vr1, t4);
		// can follow other vector operations ...
	}

	if(b_with_bias_shift == 0) 
	{
		// vsrl_vi(VPR_NONE, vr1, vr1, 16);			//向右偏移8bit， 可以将origins 的乘加结果的第二个8bits 输出，以供调试
	}
	//调试中间tensor:
	vsb(vr1, t3, 0);
	vmu_poll();
	KRNL_LOG_INFO(LOG_DEBUG, "conv_c_out");
	buf_print(MMA_START_ADDR + W64ToByte(ofm), GMEM_ALIGN(64));

	//ifm_add -> add_shift右移 -> add_clip右移 . 注意是vsra_vs
	asm volatile( "mv t4, %0" :: "r"(ifm_add): );
	asm volatile( "mv t1, %0" :: "r"(add_shift): );
	vlb(vr2, t4, 0);
	vlb(vr3, t1, 0);
	vsra_vs(VPR_NONE, vr2, vr2, t1);
	vadd_vv(VPR_NONE, vr1, vr1, vr2);
	asm volatile( "mv t1, %0" :: "r"(add_clip): );
	vsra_vs(VPR_NONE, vr1, vr1, t1);
	
	asm volatile("mv t3, %0"::"r"(ofm):);
	vsb(vr1, t3, 0);
	vmu_poll();
	KRNL_LOG_INFO(LOG_DEBUG, "conv_c add ofm");
	buf_print(MMA_START_ADDR + W64ToByte(ofm), GMEM_ALIGN(64));
	return;
}


void one_row_conv_add(int h_iter, conv2d_params_t *conv2d_param,        	  	\
						uint32_t wt_lcmem_start_addr,							\
						uint32_t ofm_row_lcmem_start_addr,						\
						uint32_t bias_lcmem_start_addr,							\
						uint32_t shift_lcmem_start_addr,						\
                    	uint32_t param_mmac_region_start,                     	\
                    	uint32_t param_mmac_region_end,                       	\
                    	uint32_t param_mmac_ifm_cluster_stride,               	\
                    	uint32_t param_mmac_cluster_start,                    	\
                    	uint32_t param_mmac_cluster_end,                      	\
                    	uint32_t param_mmac_cluster_num,						\
						uint32_t input_conv_type,								\
						uint32_t b_with_bias_shift,
						uint32_t ifm_ptr_c_add, uint32_t add_shift, uint32_t add_clip)
{
	KRNL_LOG_INFO(LOG_SYSTEM, "=====convadd ifm row index(h_iter): %d", h_iter);

	// KRNL_LOG_INFO(LOG_SYSTEM, "ifm_h: %d", conv2d_param->cshape.ifm_h);
	// KRNL_LOG_INFO(LOG_SYSTEM, "ifm_w: %d", conv2d_param->cshape.ifm_w);
	// KRNL_LOG_INFO(LOG_SYSTEM, "ifm_c: %d", conv2d_param->cshape.ifm_c);
	// KRNL_LOG_INFO(LOG_SYSTEM, "ofm_c: %d", conv2d_param->cshape.ofm_c);
	// KRNL_LOG_INFO(LOG_SYSTEM, "k_h: %d", conv2d_param->cshape.k_h);
	// KRNL_LOG_INFO(LOG_SYSTEM, "k_w: %d", conv2d_param->cshape.k_w);
	// KRNL_LOG_INFO(LOG_SYSTEM, "pshape_top: %d",  conv2d_param->pshape.top);
	// KRNL_LOG_INFO(LOG_SYSTEM, "pshape_bottom: %d", conv2d_param->pshape.bottom);
	// KRNL_LOG_INFO(LOG_SYSTEM, "pshape_left: %d pshape_right: %d", conv2d_param->pshape.left, conv2d_param->pshape.right);
	// KRNL_LOG_INFO(LOG_SYSTEM, "h_strd: %d", conv2d_param->strd_shape.h_strd);
	// KRNL_LOG_INFO(LOG_SYSTEM, "w_strd: %d", conv2d_param->strd_shape.w_strd);
	// KRNL_LOG_INFO(LOG_SYSTEM, "relu: %d", conv2d_param->relu);
	// KRNL_LOG_INFO(LOG_DEBUG, "ofm_row_lcmem_start_addr 0x%x", ofm_row_lcmem_start_addr);
    conv_type = input_conv_type;
	mmac_region_start = param_mmac_region_start;
	mmac_region_end = param_mmac_region_end;
	mmac_ifm_cluster_stride = param_mmac_ifm_cluster_stride;
	mmac_cluster_start = param_mmac_cluster_start; 
	mmac_cluster_end = param_mmac_cluster_end; 
	mmac_cluster_num = param_mmac_cluster_num;

	// makesure_needed_ifm_rows_ready(h_iter, conv2d_param);
	// Allocate local space for 1 ofm row
	// local_ofm_output_row_addr = (uint32 *)alloc_local_fm_with_drain_to_ddr_if_needed(&local_ofm, &ddr_ofm, 1);

	_set_mmac_vfadd_param_();
	_set_mmac_param_for_whole_conv_(conv2d_param);
	_set_mmac_cluster_params_(conv2d_param, h_iter);
	_set_wt_offset_(conv2d_param, h_iter);
	
	uint32 wt_start = 0, ifm_start = 0, ifm_inner_start = 0, ofm_start = 0, bias_start = 0, shift_start = 0, ifm_add_inner_start = 0;
	uint32 w_iter_num = conv2d_param->cshape.ifm_w / MTX_SCALE; 		//local_ifm w 平均分成8份
	uint32 kernel_group8_num = conv2d_param->cshape.ofm_c / MTX_SCALE;

	for(int w_iter=0; w_iter<w_iter_num; w_iter+=conv2d_param->strd_shape.w_strd){    //w_iter_num = p_conv2d_entry->conv2d.cshape.ifm_w / MTX_SCALE; 		//ifm w 平均分成8份

		ifm_start = _set_mmac_region_params_and_get_ifm_start_(conv2d_param, w_iter);

   		// KRNL_LOG_INFO(LOG_SYSTEM, "=====ifm W iter: %d / %d===== ifm_start: 0x%x", w_iter, w_iter_num, ifm_start);

		//ofm channel 输出fm channel 平均分成8份, kernel num is devided by 8
		for(int k=0; k<kernel_group8_num; k++){    //kernel_group8_num = conv2d_param->cshape.ofm_c / MTX_SCALE;			
   			// KRNL_LOG_INFO(LOG_SYSTEM, "=====ifm kernel_group8_num iter: %d / %d", k, kernel_group8_num);
			if (conv_type == CONV_TYPE_DEPTH_WISE)
			{
				//each kernel has only 1 channel
				ifm_inner_start = ifm_start + k;
				// ifm_start += k;
				wt_start = wt_lcmem_start_addr + k + wt_offset;
				
				// ofm_start = ofm_row_lcmem_start_addr + w_iter*kernel_group8_num + k;
				// bias_start = bias_lcmem_start_addr +  k * 4;		//bias 32bits per item
				// shift_start = shift_lcmem_start_addr + k;
				// __intrinsic_func__(ifm_inner_start, wt_start, ofm_start, bias_start, shift_start, conv2d_param->relu, conv_type);
			}
			else
			{
				ifm_inner_start = ifm_start;
				//each iteration, process 8 kernels
				wt_start = wt_lcmem_start_addr + k * conv2d_param->cshape.k_h * wt_cluster_size + wt_offset;
				// __intrinsic_func__(ifm_start, wt_start, ofm_start, bias_start, shift_start, conv2d_param->relu, conv_type);
			}
			ifm_add_inner_start = ifm_ptr_c_add + k + (w_iter / conv2d_param->strd_shape.w_strd) * kernel_group8_num;
			ofm_start = ofm_row_lcmem_start_addr  + k + (w_iter / conv2d_param->strd_shape.w_strd)* kernel_group8_num;
			bias_start = bias_lcmem_start_addr +  k * 4;		//bias 32bits per item
			shift_start = shift_lcmem_start_addr + k;
			// __intrinsic_func__(ifm_inner_start, wt_start, ofm_start, bias_start, shift_start, conv2d_param->relu, conv_type, b_with_bias_shift);
			__intrinsic_conv_add__(ifm_inner_start, wt_start, ofm_start, bias_start, shift_start, conv2d_param->relu,  conv_type, b_with_bias_shift , ifm_add_inner_start, add_shift ,add_clip);
			
			if (g_ulPrintDebugLogFlag)
			{
				vmu_poll();
				KRNL_LOG_INFO(LOG_DEBUG, "ifm_inner_start");
				buf_print(MMA_START_ADDR + W64ToByte(ifm_inner_start), GMEM_ALIGN(64));
				KRNL_LOG_INFO(LOG_DEBUG, "ifm_add_inner_start");
				buf_print(MMA_START_ADDR + W64ToByte(ifm_ptr_c_add), GMEM_ALIGN(64));
				// KRNL_LOG_INFO(LOG_HARDWARE_CMD, "mmac_cluster_start");
				// buf_print(MMA_START_ADDR + W64ToByte(mmac_cluster_start), GMEM_ALIGN(64));
				// KRNL_LOG_INFO(LOG_DEBUG, "ofm_start");
				// buf_print(MMA_START_ADDR + W64ToByte(ofm_start), GMEM_ALIGN(64));
				// KRNL_LOG_INFO(LOG_DEBUG, "bias_start");
				// buf_print(MMA_START_ADDR + W64ToByte(bias_start), GMEM_ALIGN(64));
				// KRNL_LOG_INFO(LOG_DEBUG, "shift_start");
				// buf_print(MMA_START_ADDR + W64ToByte(shift_start), GMEM_ALIGN(64));
			}
		}
	}
}

uint32 is_ifm_bottom(uint32 cur_row_index, conv_shape_t *cshape, pad_shape_t *pshape){
    return (cur_row_index + ( ( cshape->k_h - 1 ) >> 1 )) >= cshape->ifm_h;
}

uint32 is_ifm_top(uint32 cur_row_index, conv_shape_t *cshape, pad_shape_t *pshape){
    return cur_row_index < pshape->top;
}

uint32 get_ifm_row_size(conv_shape_t* p_cshape, stride_shape_t *p_strd_shape){
    return p_cshape->ifm_w * p_cshape->ifm_c;
}

uint32 get_ofm_row_size(conv_shape_t* p_cshape, stride_shape_t *p_strd_shape){
    return (p_cshape->ifm_w / p_strd_shape->w_strd) * p_cshape->ofm_c;
}


//每一条都占用32Bytes 大小，因为local mem ndma 的最小粒度为32Bytes
void init_local_var(local_variable *local_var, uint32 start_addr, uint32 end_addr, uint32 size){
    local_var->start_addr = start_addr;
    local_var->end_addr = end_addr;
    local_var->size = size;
    local_var->cur_cnt = 0;
}

uint32 alloc_var(local_variable *v, uint32 var_num){
    uint32 ptr;
    LIBHIKL_NASSERT(var_num > (v->size - v->cur_cnt));
    ptr = v->start_addr + v->cur_cnt * sizeof(slot_var_in_local_variable);   //每个var 占用 32Bytes 大小, 第0个Byte 保存有效值
    v->cur_cnt += var_num;
    return ptr;
}

void init_base_fm(base_fm* fm, uint32 row_slots_num, uint32 row_size, uint32 start){
    fm->row_slots_num = row_slots_num;
    fm->row_size = row_size;    // size of one complete row in ifm
    fm->start_addr = start;
    // fm->end_addr = fm->start_addr + fm->row_size * fm->row_slots_num;
    fm->end_addr = fm->start_addr + MMA_BANK_SIZE * fm->row_slots_num;
    fm->total_cnt = 0;
}

uint32 alloc_fm(base_fm *fm, uint32 *idx){
    uint32 ret;
    *idx = fm->total_cnt % fm->row_slots_num;
    // ret = (*idx) * fm->row_size + fm->start_addr;
    ret = (*idx) * MMA_BANK_SIZE + fm->start_addr;
    fm->total_cnt += 1;
    return ret;
}

uint32 alloc_ddr_ofm(base_fm *fm, uint32 *idx){
    uint32 ret;
    *idx = fm->total_cnt % fm->row_slots_num;
    ret = fm->total_cnt * fm->row_size + fm->start_addr;
    fm->total_cnt += 1;
    return ret;
}

void init_ddr_fm(ddr_fm* fm, uint32 row_slots_num, uint32 row_size, uint32 start, uint8 x, uint8 y){
    init_base_fm(&fm->bfm, row_slots_num, row_size, start);
    fm->bfm.end_addr = fm->bfm.start_addr + fm->bfm.row_size * fm->bfm.row_slots_num;
    fm->x_pos = x;
    fm->y_pos = y;
}

uint32 alloc_ddr_fm(ddr_fm *fm){
    int tmp;
    return alloc_ddr_ofm(&fm->bfm, &tmp);
}

void __attribute__((optimize("O1"))) init_local_fm(local_fm* fm, local_variable* local_var, uint32 row_slots_num, uint32 row_size, uint32 start){
    init_base_fm(&fm->bfm, row_slots_num, row_size, start);
    fm->row_slot_available_flgs = (uint32 *)alloc_var(local_var, row_slots_num);
    fm->cur_cnt = 0;
    fm->cur_valid_idx = 0; 

    // init all row flags to 0
    for(int i=0; i<row_slots_num; i++){
        fm->row_slot_available_flgs[i] = 0;
    }
}

// for now, only support allocation of one row for each time
uint32 alloc_local_fm(local_fm *fm, uint32 num){
    uint32 idx, fm_addr;
    LIBHIKL_NASSERT(num > 1);
    if(num > fm->bfm.row_slots_num - fm->cur_cnt)
        return -1;
    fm_addr = alloc_fm(&fm->bfm, &idx);
    fm->row_slot_available_flgs[idx] = 1;
    fm->cur_cnt += 1;
    return fm_addr;
}

// uint32 alloc_local_ddr_fm(local_fm *fm, uint32 num){
//     uint32 idx, fm_addr;
//     LIBHIKL_NASSERT(num > 1);
//     if(num > fm->bfm.row_slots_num - fm->cur_cnt)
//         return -1;
//     fm_addr = alloc_ddr_ofm(&fm->bfm, &idx);
//     fm->row_slot_available_flgs[idx] = 1;
//     fm->cur_cnt += 1;
//     return fm_addr;
// }

void dealloc_local_fm(local_fm *fm, uint32 num){
    LIBHIKL_NASSERT(num > 1);
    LIBHIKL_NASSERT(num > fm->cur_cnt);
    fm->row_slot_available_flgs[fm->cur_valid_idx] = 0;
    fm->cur_valid_idx = (fm->cur_valid_idx + 1) % fm->bfm.row_slots_num;
    fm->cur_cnt -= 1;
}

// Return address of current valid local fm address
uint32 get_cur_valid_local_fm_addr(local_fm *fm){
    // return fm->bfm.start_addr + fm->cur_valid_idx * fm->bfm.row_size; 
    return fm->bfm.start_addr + fm->cur_valid_idx * MMA_BANK_SIZE; 
}

// update the status of local fm struct 
void update_local_fm(local_fm *fm){
    fm->cur_cnt = 0;
    for(int i=0; i<fm->bfm.row_slots_num; i++){
        fm->cur_cnt += fm->row_slot_available_flgs[i];
    }
}

void init_remote_fm(remote_fm* fm, local_variable* local_var, uint32 row_slots_num, uint32 row_size, uint32 start, uint8 remote_node_x, uint8 remote_node_y, uint32 remote_node_var_saving_flags){
    init_base_fm(&fm->bfm, row_slots_num, row_size, start);
    fm->var_in_remote_node_saving_flags.x_pos = remote_node_x;
    fm->var_in_remote_node_saving_flags.y_pos = remote_node_y;
    fm->var_in_remote_node_saving_flags.lcaddr = remote_node_var_saving_flags;
    fm->local_var_for_interact_with_remote_vars = (uint32 *)alloc_var(local_var, 1);
}

// for now, only support allocation of one row for each time
uint32 alloc_remote_fm(remote_fm *fm, uint32 num){
    uint32 *val;
    uint32 idx, fm_addr, flag_addr;

    LIBHIKL_NASSERT(num > 1);
    fm_addr = alloc_fm(&fm->bfm, &idx);
    flag_addr = fm->var_in_remote_node_saving_flags.lcaddr + sizeof(slot_var_in_local_variable) * idx;
    
    // Polling until the remote fm slot flag is marked empty
    val = fm->local_var_for_interact_with_remote_vars;
    *val = 1;
    while(*val){
        __rd_from_remote_var_blocking(val, fm->var_in_remote_node_saving_flags.x_pos, fm->var_in_remote_node_saving_flags.y_pos, flag_addr);
    }

    return fm_addr;
}

// try to acquire the remote_lock until the flag is ready. For flag, if 1:release and retry; else: break
void acquire_flag_in_remote_node_blocking(remote_fm *fm, uint32 row_slots_num){
    uint32 flag_addr;
    uint32 *val;
    flag_addr = fm->var_in_remote_node_saving_flags.lcaddr + sizeof(slot_var_in_local_variable) * row_slots_num;
    val = fm->local_var_for_interact_with_remote_vars;
    *val = 1;
    do{
        acquire_remote_fm_lock(fm, row_slots_num);
        __rd_from_remote_var_blocking(val, fm->var_in_remote_node_saving_flags.x_pos, fm->var_in_remote_node_saving_flags.y_pos, flag_addr);
        if(*val){
            release_remote_fm_lock(fm, row_slots_num);
            wait(15);
        }
    }while(*val);

}

// try to acquire the remote_lock until the flag is ready. For flag, if 1:release and break; else: break
int check_flag_in_remote_node(remote_fm *fm, uint32 row_slots_num){
    uint32 flag_addr;
    uint32 *val;
    flag_addr = fm->var_in_remote_node_saving_flags.lcaddr + sizeof(slot_var_in_local_variable) * row_slots_num;
    val = fm->local_var_for_interact_with_remote_vars;
    *val = 1;
    acquire_remote_fm_lock(fm, row_slots_num);
    __rd_from_remote_var_blocking(val, fm->var_in_remote_node_saving_flags.x_pos, fm->var_in_remote_node_saving_flags.y_pos, flag_addr);
    if(*val){
        release_remote_fm_lock(fm, row_slots_num);
        }
    return *val;

}


// After use NDMA to write data to next node fm, notify it the data is ready by setting remote flag
void notify_remote_fm_ready(remote_fm *fm, uint32 fm_addr){
    uint32 idx, flag_addr;
    // idx = (fm_addr - fm->bfm.start_addr) / fm->bfm.row_size;
    idx = (fm_addr - fm->bfm.start_addr) / MMA_BANK_SIZE;
    flag_addr = fm->var_in_remote_node_saving_flags.lcaddr + idx * sizeof(slot_var_in_local_variable);

    uint32 *val = fm->local_var_for_interact_with_remote_vars;
    *val = 1;
    __wr_to_remote_var_blocking(val, fm->var_in_remote_node_saving_flags.x_pos, fm->var_in_remote_node_saving_flags.y_pos, flag_addr);
}

// initialize local fm locks
void init_local_fm_lock(local_fm* fm){
    uint32* lock = (uint32*)AMO_ADDR_S;
    for (int idx = 0; idx < fm->bfm.row_slots_num; idx++){
        lock[idx] = 0;
    }
}

// aquire lock for one local fm row
void acquire_local_fm_lock(uint32 row_slots_num){
    uint32 lock_addr = AMO_ADDR_S + row_slots_num * 4;
    acquire_local_lock(lock_addr);
}

// release lock for one local fm row
void release_local_fm_lock(uint32 row_slots_num){
    uint32 lock_addr = AMO_ADDR_S + row_slots_num * 4;
    release_local_lock(lock_addr);
}

// aquire lock for one remote fm row
void acquire_remote_fm_lock(remote_fm* fm, uint32 row_slots_num){
    uint32 *val = fm->local_var_for_interact_with_remote_vars;
    uint32 lock_addr = AMO_ADDR_S + row_slots_num * 4;
    acquire_remote_lock(val, lock_addr, fm->var_in_remote_node_saving_flags.x_pos, fm->var_in_remote_node_saving_flags.y_pos);
}

// Release lock for one remote row
void release_remote_fm_lock(remote_fm* fm, uint32 row_slots_num){
    uint32 *val = fm->local_var_for_interact_with_remote_vars;
    uint32 lock_addr = AMO_ADDR_S + row_slots_num * 4;
    *val = 0;
    release_remote_lock(val, lock_addr, fm->var_in_remote_node_saving_flags.x_pos, fm->var_in_remote_node_saving_flags.y_pos);
}

int _ndma_one_fm_row_from_localmem_to_remote_localmem_blocking(local_fm *ofm, remote_fm *rfm) {
    int row_idx;
    int fm_addr, local_fm_addr;
    int ready = 0;
    if(ofm->cur_cnt){
        row_idx = rfm->bfm.total_cnt % rfm->bfm.row_slots_num;
    #ifndef QEMU_ENV
        // Acquire the lock of core2 MMA, the oldest row
        ready = check_flag_in_remote_node(rfm, row_idx);
        if(ready == 1)
            return ready;
    #endif
        // Acquire the addr of oldest row in rfm
        // fm_addr = row_idx * rfm->bfm.row_size + rfm->bfm.start_addr;
        fm_addr = row_idx * MMA_BANK_SIZE + rfm->bfm.start_addr;

        // Acquire the addr of oldest row in ofm
        local_fm_addr = get_cur_valid_local_fm_addr(ofm);

        // Write to core2 MMA
        LIBHIKL_NASSERT(__wr_to_remote_chunk_blocking((uint32 *)local_fm_addr, rfm->var_in_remote_node_saving_flags.x_pos, rfm->var_in_remote_node_saving_flags.y_pos, fm_addr, rfm->bfm.row_size));
        
        notify_remote_fm_ready(rfm, fm_addr);

    #ifndef QEMU_ENV
        // Release the lock of next core MMA
        release_remote_fm_lock(rfm, row_idx);
    #endif

        KRNL_LOG_INFO(LOG_DEBUG, "Transfer ofm: [%x]->[%x]\n\r", local_fm_addr, fm_addr);

        // Release local ofm after transaction is over
        KRNL_LOG_INFO(LOG_DEBUG, "Release ofm row %d\n\r", ofm->cur_valid_idx);
        rfm->bfm.total_cnt += 1;
        dealloc_local_fm(ofm, 1);

    }
    return ready;
}

int _ndma_remain_ifm_rows_from_localmem_to_ddr_blocking(local_fm *ofm, ddr_fm *dfm) {
    do{
        _ndma_one_fm_row_from_localmem_to_ddr_blocking(ofm, dfm);
    }while(ofm->cur_cnt > 0);
}

int _ndma_one_fm_row_from_localmem_to_ddr_blocking(local_fm *ofm, ddr_fm *dfm) {
    int ddr_fm_addr, local_fm_addr;
    if(ofm->cur_cnt){
        // Acquire the addr of dfm
        ddr_fm_addr = alloc_ddr_fm(dfm);//dfm->bfm.total_cnt * dfm->bfm.row_size + dfm->bfm.start_addr;//

        // Acquire the addr of oldest row in ofm
        local_fm_addr = get_cur_valid_local_fm_addr(ofm);

        // KRNL_LOG_INFO(LOG_DEBUG, "local ofm: ");
        // for (int i = 0; i < 64; i++)
        // {
        //     if (i % VEC_SCALE == 0)
        //     {
        //         KRNL_LOG_INFO(LOG_DEBUG, "\n");
        //     }
        //     #ifdef  check_32bits_output
        //         KRNL_LOG_INFO(LOG_DEBUG, "ofm local addr %x : 0x%08x", ((int32_t *)local_fm_addr + i), *((int32_t *)local_fm_addr + i));
        //     #else
        //         KRNL_LOG_INFO(LOG_DEBUG, "addr %x : 0x%02x", ((char *)local_fm_addr + i), *((char *)local_fm_addr + i));
        //     #endif
        // }

        // Write to ddr
        #ifdef  check_32bits_output
            LIBHIKL_NASSERT(__wr_to_remote_chunk_blocking((uint32 *)local_fm_addr, dfm->x_pos, dfm->y_pos, ddr_fm_addr, dfm->bfm.row_size * 4));  // to output 32 bits per item
            KRNL_LOG_INFO(LOG_DEBUG, "Transfer ofm[%d]Bytes: [%x]->[%x], total_cnt = %d ddr_dofm start_addr: %d%d%x\n\r", \
                          dfm->bfm.row_size * 4, local_fm_addr, ddr_fm_addr, dfm->bfm.total_cnt, dfm->x_pos, dfm->y_pos, dfm->bfm.start_addr); // to output 32 bits per item
        #else
            LIBHIKL_NASSERT(__wr_to_remote_chunk_blocking((uint32 *)local_fm_addr, dfm->x_pos, dfm->y_pos, ddr_fm_addr, dfm->bfm.row_size));
            KRNL_LOG_INFO(LOG_DEBUG, "Transfer ofm[%d]Bytes: [%x]->[%x], total_cnt = %d ddr_dofm start_addr: %d%d%x\n\r", \
                          dfm->bfm.row_size, local_fm_addr, ddr_fm_addr, dfm->bfm.total_cnt, dfm->x_pos, dfm->y_pos, dfm->bfm.start_addr);
			// buf_print(local_fm_addr, dfm->bfm.row_size);

        #endif

        // Release local ofm after transaction is over
        KRNL_LOG_INFO(LOG_DEBUG, "Release ofm row %d\n\r", ofm->cur_valid_idx);
        // dfm->bfm.total_cnt += 1;
        dealloc_local_fm(ofm, 1);
    }
    return 0;
}
