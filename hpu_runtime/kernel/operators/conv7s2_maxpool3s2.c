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

static uint32 ifm_rsize, ofm_rsize, core_id, kernel_id;
static u32_t *kernel_id_table;
static paramTableConv2d_Entry_t *conv_table;
static hirtKernelParamTableBase_t *base_table;
static ddr_fm ddr_ifm, ddr_ofm;
static local_fm local_ifm, local_ofm;
static remote_fm remote_ifm;
static local_variable local_var;
static paramTableConv2d_t param_table;

/*extern int _ifm;*/
/*extern int _wt1;*/
/*extern int _data_out;*/

// typedef struct{
//     int typeVersion;
//     int tableSiz;
//     int parallelism;        // the number of parallel cores
//     int parallelTable[13];  // the ID of each parallel core, Table[0] is head, Table[1] is tail, Table[2,3,...] are body
// } paramTableBase_t;

// typedef struct{
//     conv2d_params_t conv2d;
//     hikl_addr_t wt_addr;
//     hikl_addr_t ifm_addr;
//     hikl_addr_t ofm_addr;
// }paramTableConv2d_Entry_t;

// This is supposed to be populated by runtime
// struct{
//     paramTableBase_t infoBase;
//     paramTableConv2d_Entry_t head;
//     paramTableConv2d_Entry_t tail;
//     paramTableConv2d_Entry_t body2;
//     paramTableConv2d_Entry_t body1;
// } paramTableConv2d_t;
// struct{
//     paramTableBase_t infoBase;
//     paramTableConv2d_Entry_t param;
// } param_table;


void timer_handler(){
	KRNL_LOG_INFO(LOG_DEBUG, "timer interrupt begins...\n\r");
	if(kernel_id == 1 || base_table->task_dim == 1) // if the CPU is Conv tail
		_ndma_to_ddr_nowait(&local_ofm, &ddr_ofm);
	else
		_ndma_to_next_nowait(&local_ofm, &remote_ifm);
	KRNL_LOG_INFO(LOG_DEBUG, "timer interrupt ends...\n\r");
	set_timer_tick(0x60, 0);
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
			ready = _ndma_to_next_nowait(ofm, rfm);
		}while(ready);
		KRNL_LOG_INFO(LOG_DEBUG, "ndma ends...\n\r");
		return alloc_local_fm(ofm, num);
	}
	return local_mem_fm_addr;
}

uint32 trans_and_alloc_local_fm_tail(local_fm *ofm, ddr_fm *dfm, uint32 num){
	int ready = 1;
	uint32 local_mem_fm_addr = alloc_local_fm(ofm, num);
	KRNL_LOG_INFO(LOG_DEBUG, "local_mem_fm_addr = %x\n", local_mem_fm_addr);
	if(local_mem_fm_addr == -1){
		KRNL_LOG_INFO(LOG_DEBUG, "the ofm is full, ndma begins...\n\r");
		do{
			ready = _ndma_to_ddr_nowait(ofm, dfm);
		}while(ready);
		KRNL_LOG_INFO(LOG_DEBUG, "ndma ends...\n\r");
		return alloc_local_fm(ofm, num);
	}
	return local_mem_fm_addr;
}

static void __intrinsic_func_conv__(uint32 ifm, uint32 wt, uint32 ofm, uint32 bias_start, uint32 shift_start, bool relu) 
{
	uint32 vfadd_bias = 0;
	asm volatile("mv t1, %0"::"r"(ifm):);
	asm volatile("mv t2, %0"::"r"(wt):);
	asm volatile("mv t3, %0"::"r"(ofm):);
	asm volatile("mv t4, %0"::"r"(vfadd_bias):);
	asm volatile("mv t5, %0"::"r"(bias_start));
	asm volatile("mv t6, %0"::"r"(shift_start));
	_clr_vreg(vr1);
	mmac(VPR_NONE, vr1, t1, t2, vr1);
	// mdmac(VPR_NONE, vr1, t1, t2, vr1);
	// mmad(VPR_NONE, vr1, t1, t2, vr1);
	// vlw(vr2, t5, 0);
	// vlw(vr3, t6, 0);
	// vadd_vv(VPR_NONE, vr1, vr1, vr2);
	// vsrl_vv(VPR_NONE, vr1, vr1, vr3);
	//now still 32 bits per item
	if (relu)
	{
		vmax_vs(VPR_NONE, vr1, vr1, 0);
	}
	// vfadd_vs(VPR_NONE, vr1, vr1, t4);
	// can follow other vector operations ...
	// vsb(vr1, t3, 0);
	vsw(vr1, t3, 0);
	// qemu_fprint(QEMU_LOG_MEM, ofm, 64);
	return;
}

static void __intrinsic_func_dwconv__(uint32 ifm, uint32 wt, uint32 ofm, uint32 bias_start, uint32 shift_start, bool relu) 
{
	uint32 vfadd_bias = 0;
	asm volatile("mv t1, %0"::"r"(ifm):);
	asm volatile("mv t2, %0"::"r"(wt):);
	asm volatile("mv t3, %0"::"r"(ofm):);
	asm volatile("mv t4, %0"::"r"(vfadd_bias):);
	asm volatile("mv t5, %0"::"r"(bias_start));
	asm volatile("mv t6, %0"::"r"(shift_start));
	_clr_vreg(vr1);
	mmac(VPR_NONE, vr1, t1, t2, vr1);
	// mdmac(VPR_NONE, vr1, t1, t2, vr1);
	// mmad(VPR_NONE, vr1, t1, t2, vr1);
	// vlw(vr2, t5, 0);
	// vlw(vr3, t6, 0);
	// vadd_vv(VPR_NONE, vr1, vr1, vr2);
	// vsrl_vv(VPR_NONE, vr1, vr1, vr3);
	//now still 32 bits per item
	if (relu)
	{
		vmax_vs(VPR_NONE, vr1, vr1, 0);
	}
	// vfadd_vs(VPR_NONE, vr1, vr1, t4);
	// can follow other vector operations ...
	// vsb(vr1, t3, 0);
	vsw(vr1, t3, 0);
	// qemu_fprint(QEMU_LOG_MEM, ofm, 64);
	return;
}


#define LCMEM_TENSOR_B0     MMA_B0_ADDR
#define LCMEM_TENSOR_B1     MMA_B1_ADDR
#define LCMEM_TENSOR_S0     MMA_B2_ADDR
#define LCMEM_TENSOR_S1     (MMA_B2_ADDR+MMA_BLOCK_SIZE/4)
#define LCMEM_TENSOR_S2     (MMA_B2_ADDR+MMA_BLOCK_SIZE/2)
#define LCMEM_TENSOR_S3     MMA_B3_ADDR

#define LCMEM_BIAS_SHIFT    MMA_B7_ADDR

static u32_t localmem_usagetbl_tensor_b01[]
{
    LCMEM_TENSOR_B0,
    LCMEM_TENSOR_B1
};
static u32_t localmem_index_tensor_b01 = 0;
static const u32_t localmem_num_tensor_b01 = sizeof(localmem_usagetbl_tensor_b01) / sizeof(u32_t);

static u32_t localmem_usagetbl_tensor_s012[]
{
    LCMEM_TENSOR_S0,
    LCMEM_TENSOR_S1,
    LCMEM_TENSOR_S2
};
static u32_t localmem_index_tensor_s012 = 0;
static const u32_t localmem_num_tensor_s012 = sizeof(localmem_usagetbl_tensor_s012) / sizeof(u32_t);

static u32_t localmem_usagetbl_tensor_out[]
{
    LCMEM_TENSOR_S3
};

#define GET_TENSOR_BUFFER_B01()     (localmem_usagetbl_tensor_b01[localmem_index_tensor_b01])
#define GET_TENSOR_BUFFER_S012()    (localmem_usagetbl_tensor_s012[localmem_index_tensor_s012])
#define INC_TENSOR_BUFFER_B01()     {localmem_index_tensor_b01++;if(localmem_index_tensor_b01>=localmem_num_tensor_b01)localmem_index_tensor_b01=0}
#define INC_TENSOR_BUFFER_S012()    {localmem_index_tensor_s012++;if(localmem_index_tensor_s012>=localmem_num_tensor_s012)localmem_index_tensor_s012=0}
#define GET_TENSOR_BUFFER_OUT()     {localmem_usagetbl_tensor_out[0]}

void conv7s2_maxpool3s2
(
    conv_shape_t *cshape_a, 
    conv_shape_t *cshape_b, 
    stride_shape_t *stride_a,
    stride_shape_t *stride_b,
    pad_shape_t *pad_a,
    pad_shape_t *pad_b,
    bool relu_a, 
    
    hikl_addr_t *ifm_addr,
    hikl_addr_t *ofm_addr,
    hikl_addr_t *wt_addr, 
    hikl_addr_t *bias_addr, 
    hikl_addr_t *shift_addr
)
{
    uint32 i,j,ndma_pool;
    uint32 h_iter_num;
	uint8  round_type, shift_num, prot_high, prot_low =0;
	
	uint32 ifm_row_stride_a, ofm_row_stride_a = 0;
	uint32 wt_sz_a, bs_sz_a, shift_sz_a;
	uint32 bias_start_a, shift_start_a;


	uint32 ifm_c_group8_num_a, wt_cluster_size_a, cluster_num_a = 0;
	uint32 w_iter_num_a, kernel_group8_num_a = 0;
	uint32 wt_offset_a, wt_start_a, ifm_start_a, cluster_start_a, cluster_end_a, ofm_start_a = 0;

    uint32 ifm_c_group8_num_b;
    uint32 cluster_num_b = 0;
    uint32 cluster_stride_b;
    uint32 w_iter_num_b;
    uint32 ifm_row_stride_b;
    uint32 ofm_row_stride_b;
	uint32 ifm_start_b, cluster_start_b, cluster_end_b, ofm_start_b = 0;

	wt_sz_a = cshape_a->k_w * cshape_a->k_h * cshape_a->ifm_c * cshape_a->ofm_c; // in Bytes
	bs_sz_a = cshape_a->ofm_c / 8;
	shift_sz_a = bs_sz_a;

    uint32 conv_a_lcmem_alloc_aline_size = MMA_BLOCK_SIZE / 8;
    uint32 pool_b_lcmem_alloc_aline_size = MMA_BLOCK_SIZE / 4;
	uint32 input_tensor_aline_word64len = CALC_FMWIDTH_GROUP8(cshape_a->ifm_w) * CALC_CHANNEL_GROUP8(cshape_a->ifm_c);
	uint32 mpool_tensor_aline_word64len = input_tensor_aline_word64len / stride_a->w_strd;
	uint32 output_tensor_aline_word64len = mpool_tensor_aline_word64len / stride_b->w_strd;

	// Load all weights
	LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)MMB_ADDR, wt_addr->x_pos, wt_addr->y_pos, wt_addr->lcaddr,  GMEM_ALIGN(wt_sz_a)));
	// Load all bias
	LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)(BIAS_SHIFT_BLK), bias_addr->x_pos, bias_addr->y_pos, bias_addr->lcaddr, GMEM_ALIGN(bs_sz_a)));
	// Load all shift_num
	LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)(BIAS_SHIFT_BLK + bs_sz_a), shift_addr->x_pos, shift_addr->y_pos, shift_addr->lcaddr, GMEM_ALIGN(shift_sz_a)));

	// Calculate total iteration
	kernel_group8_num_a = cshape_a->ofm_c / MTX_SCALE;			//ofm channel 鏉堟挸鍤璮m channel 楠炲啿娼庨崚鍡樺灇8娴�? kernel num is devided by 8
	w_iter_num_a = cshape_a->ifm_w / MTX_SCALE; 		//ifm w 楠炲啿娼庨崚鍡樺灇8娴�?
	ifm_c_group8_num_a = cshape_a->ifm_c / MTX_SCALE;		//ifm 鏉堟挸鍙唂m channel 楠炲啿娼庨崚鍡樺灇8娴�?
	ifm_row_stride_a = ByteToHPU64BytesWord(cshape_a->ifm_w * cshape_a->ifm_c);
	ofm_row_stride_a = ByteToHPU64BytesWord(cshape_a->ifm_w * cshape_a->ofm_c);
	wt_cluster_size_a = ifm_c_group8_num_a * cshape_a->k_w;

    ifm_c_group8_num_b = cshape_a->ofm_c / MTX_SCALE;
    w_iter_num_b = cshape_a->ifm_w / stride_a->w_strd;
    ifm_row_stride_b = ;
    ofm_row_stride_b = ;

    //convpad0//////////////////////////////////////////////////////////////////////////////////
	_set_mmac_fm_blk_stride(ifm_c_group8_num_a * 1);
	_set_mmac_fm_cluster_stride(ifm_row_stride_a);
	_set_mmac_fm_blk_size(ifm_c_group8_num_a - 1);	
	_set_mmac_wt_blk_stride(ifm_c_group8_num_a);
	_set_mmac_wt_cluster_stride(ByteToHPU64BytesWord(cshape_a->ifm_c * cshape_a->k_w * MTX_SCALE));	
    _set_mmac_region_start(ByteToHPU64BytesWord(localmem_usagetbl_tensor_b01[0] - MEM_LCMEM_ADDR_S));
    _set_mmac_region_end(ByteToHPU64BytesWord(localmem_usagetbl_tensor_b01[1] + MMA_BLOCK_SIZE - MEM_LCMEM_ADDR_S) + 1);
	round_type = 1;shift_num = 0;prot_high = 127;prot_low = -128;
	_set_mmac_round_type(round_type);    //mid
	_set_mmac_fadd_shift_num(shift_num);
	_set_mmac_fadd_prot_high(prot_high);
	_set_mmac_fadd_prot_low(prot_low);

	//convpad0,convpad1,maxpool0
	{
	    //convpad0
		//LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)localmem_fm_table_conv_a[i], ddr_ifm->x_pos, ddr_ifm->y_pos, ddr_mem_fm_addr, GMEM_ALIGN(ifm->bfm.row_size)));
		cluster_start_a = ByteToHPU64BytesWord(localmem_usagetbl_tensor_b01[0] - MMA_ADDR);
		cluster_end_a = cluster_start_a + ifm_row_stride_a;       // always the end of the first ifm row
		cluster_num_a = cshape_a->k_h-pad_a->top; // always less than the full kernel height
		wt_offset_a = wt_cluster_size_a * pad_a->top;    // skip top rows of kernel
		_set_mmac_fm_cluster_start(cluster_start_a);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_end(cluster_end_a + 1);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_num(cluster_num_a - 1);
		_set_mmac_fm_blk_num(cshape_a->k_w - 1);		

		for(int j=0; j<w_iter_num_a; j+=stride_a->w_strd)
        {
			ifm_start_a = cluster_start_a + (j - pad_a->side) * ifm_c_group8_num_a;

			for(int k=0; k<kernel_group8_num_a; k++)
			{
				ofm_start_a   = ByteToHPU64BytesWord(GET_TENSOR_BUFFER_S012() - MMA_ADDR) + (j/stride_a->w_strd)*kernel_group8_num_a + k;
				wt_start_a    = ByteToHPU64BytesWord(MMB_BEGIN) + k * cshape_a->k_h * wt_cluster_size_a + wt_offset_a;
				bias_start_a  = ByteToHPU64BytesWord(BIAS_SHIFT_BLK) + kernel_group8_num_a;
				shift_start_a = ByteToHPU64BytesWord(BIAS_SHIFT_BLK + bs_sz_a) + kernel_group8_num_a;
				__intrinsic_func_conv__(ifm_start_a, wt_start_a, ofm_start_a, bias_start_a, shift_start_a, relu_a);
			}
		}
		INC_TENSOR_BUFFER_S012();

        //convpad1
		//LIBHIKL_NASSERT(__rd_rmt_chunk_blocking((uint32 *)localmem_fm_table_conv_a[i], ddr_ifm->x_pos, ddr_ifm->y_pos, ddr_mem_fm_addr, GMEM_ALIGN(ifm->bfm.row_size)));
		cluster_start_a = ByteToHPU64BytesWord(localmem_usagetbl_tensor_b01[0] - MMA_ADDR);
		cluster_end_a = cluster_start_a + ifm_row_stride_a;       // always the end of the first ifm row
		cluster_num_a = cshape_a->k_h-pad_a->top+stride_a->h_strd; // always less than the full kernel height
		wt_offset_a = wt_cluster_size_a * (pad_a->top-stride_a->h_strd);    // skip top rows of kernel
		_set_mmac_fm_cluster_start(cluster_start_a);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_end(cluster_end_a + 1);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_num(cluster_num_a - 1);
		_set_mmac_fm_blk_num(cshape_a->k_w - 1);		

		for(int j=0; j<w_iter_num_a; j+=stride_a->w_strd)
        {
			ifm_start_a = cluster_start_a + (j - pad_a->side) * ifm_c_group8_num_a;

			for(int k=0; k<kernel_group8_num_a; k++)
			{
				ofm_start_a   = ByteToHPU64BytesWord(GET_TENSOR_BUFFER_S012() - MMA_ADDR) + (j/stride_a->w_strd)*kernel_group8_num_a + k;
				wt_start_a    = ByteToHPU64BytesWord(MMB_BEGIN) + k * cshape_a->k_h * wt_cluster_size_a + wt_offset_a;
				bias_start_a  = ByteToHPU64BytesWord(BIAS_SHIFT_BLK) + kernel_group8_num_a;
				shift_start_a = ByteToHPU64BytesWord(BIAS_SHIFT_BLK + bs_sz_a) + kernel_group8_num_a;
				
				__intrinsic_func_conv__(ifm_start_a, wt_start_a, ofm_start_a, bias_start_a, shift_start_a, relu_a);
			}
		}
		INC_TENSOR_BUFFER_S012();

        //maxpool0
        cluster_stride_b = pool_b_lcmem_alloc_aline_size;
		for(int j=0; j<w_iter_num_b; j+=stride_b->w_strd)
        {
            int32_t ifm_offset = (j - pad_b->side) * ifm_c_group8_num_b;
            ofm_start_b = ;//ByteToHPU64BytesWord((int)local_ofm_output_row_addr - MEM_LCMEM_ADDR_S) + j*times_of_8_kernels + k;

            for(int cc=0; cc<ifm_c_group8_num_b; cc++)
            {
                int32_t ifm_ptr;
                int32_t ofm_ptr=ofm_start_b+(j/stride_b->w_strd)*ifm_c_group8_num_b+cc;
                _clr_vreg(vr1);
                
                for(int hh=0; hh<(cshape_b->k_h-pad_b->top); hh++)
                {
                    cluster_start_b = GET_TENSOR_BUFFER_S012();
                    cluster_end_b = cluster_start_b+mpool_tensor_aline_word64len;
                    ifm_start_b = cluster_start_b+ifm_offset;
                    
                    for(int ww=0; ww<cshape_b->k_w; ww++)
                    {
                        #define ALIGN_NONE  (0)
                        #define ALIGN_UP    (1)
                        #define ALIGN_DN    (2)
                        int alignflag = ALIGN_NONE;
                        ifm_ptr = ifm_start_b + ww * ifm_c_group8_num_b + cc;
                        if(ifm_ptr < cluster_start_b)
                        {
                            //alignup()
                            alignflag = ALIGN_UP;
                            ifm_ptr = cluster_end_b - (cluster_start_b-ifm_ptr);
                        }
                        else if(ifm_ptr >= cluster_end_b)
                        {
                            //aligndn()
                            alignflag = ALIGN_DN;
                            ifm_ptr = cluster_start_b + (ifm_ptr-cluster_end_b);
                        }
                     	asm volatile("mv t1, %0"::"r"(ifm_ptr):);
                     	vlw(vr2, t1, 0);
                     	if(alignflag==ALIGN_UP)
                     	    //
                     	else if(alignflag==ALIGN_DN)
                     	    //
                     	vmax_vv(VPR_NONE, vr1, vr1, vr2);
                    }
                }
                
                asm volatile("mv t1, %0"::"r"(ofm_ptr):);
                vsw(vr1, t1, 0);
            }
            
		}
	}

    uint32 h_iter_group = cshape_a->ifm_h / (stride_a->h_strd*stride_b->h_strd*4);
	for(i=0; i<h_iter_group; i++)
	{
		if(i != (h_iter_group-1)){
			LIBHIKL_NASSERT(__rd_rmt_chunk_non_blocking((uint32 *)GET_TENSOR_BUFFER_B01(), ddr_ifm->x_pos, ddr_ifm->y_pos, ddr_mem_fm_addr, GMEM_ALIGN(ifm->bfm.row_size)));
			INC_TENSOR_BUFFER_B01();
		}

        //conv1//////////////////////////////////////////////////////////////////////////////////
		cluster_start_a = ByteToHPU64BytesWord(localmem_fm_table_conv_a[i] - MMA_ADDR);
		cluster_end_a = cluster_start_a + ifm_row_stride_a;       // always the end of the first ifm row
		cluster_num_a = 1; // always less than the full kernel height
		wt_offset_a = 0;    // skip top rows of kernel
		_set_mmac_fm_cluster_start(cluster_start_a);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_end(cluster_end_a + 1);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_num(cluster_num_a - 1);
		_set_mmac_fm_blk_num(cshape_a->k_w - 1);		

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
        //conv2//////////////////////////////////////////////////////////////////////////////////
		cluster_start_a = ByteToHPU64BytesWord(localmem_fm_table_conv_a[i] - MMA_ADDR);
		cluster_end_a = cluster_start_a + ifm_row_stride_a;       // always the end of the first ifm row
		cluster_num_a = 1; // always less than the full kernel height
		wt_offset_a = 0;    // skip top rows of kernel
		_set_mmac_fm_cluster_start(cluster_start_a);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_end(cluster_end_a + 1);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_num(cluster_num_a - 1);
		_set_mmac_fm_blk_num(cshape_a->k_w - 1);		

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
		//pool1//////////////////////////////////////////////////////////////////////////////////
        
        //conv3//////////////////////////////////////////////////////////////////////////////////
        //conv4//////////////////////////////////////////////////////////////////////////////////
        //pool2+ld0//////////////////////////////////////////////////////////////////////////////
        
        //conv5//////////////////////////////////////////////////////////////////////////////////
        //conv6//////////////////////////////////////////////////////////////////////////////////
        //pool3//////////////////////////////////////////////////////////////////////////////////
        
        //conv7//////////////////////////////////////////////////////////////////////////////////
        //conv8//////////////////////////////////////////////////////////////////////////////////
        //pool4+ld1//////////////////////////////////////////////////////////////////////////////
        
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
    int concat_output_tensor_aline_mrlen;//闅愯棌浜哛a Lb Concat
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
	kernel_group8_num_a = cshape_a->ofm_c / MTX_SCALE;			//ofm channel 鏉堟挸鍤璮m channel 楠炲啿娼庨崚鍡樺灇8娴�? kernel num is devided by 8
	w_iter_num_a = cshape_a->ifm_w / MTX_SCALE; 		//ifm w 楠炲啿娼庨崚鍡樺灇8娴�?
	//KRNL_LOG_INFO(LOG_DEBUG, "(ifm_c, pshape_top, pshape_bottom, k_h): %d %d %d %d\n",cshape->ifm_c , pshape->top , pshape->bottom, cshape->k_h);
	ifm_c_group8_num_a = cshape_a->ifm_c / MTX_SCALE;		//ifm 鏉堟挸鍙唂m channel 楠炲啿娼庨崚鍡樺灇8娴�?
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
			
		_set_mmac_fm_cluster_start(cluster_start_a);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_end(cluster_end_a + 1);						//64Bytes 娑撳搫宕熸担?
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
	kernel_group8_num_b = cshape_b->ofm_c / MTX_SCALE;			//ofm channel 鏉堟挸鍤璮m channel 楠炲啿娼庨崚鍡樺灇8娴�? kernel num is devided by 8
	w_iter_num_b = cshape_b->ifm_w / MTX_SCALE; 		//ifm w 楠炲啿娼庨崚鍡樺灇8娴�?
	h_iter_num_b = cshape_b->ifm_h + /*pshape->top + pshape->bottom*/ 2 - cshape_b->k_h + 1;
	//KRNL_LOG_INFO(LOG_DEBUG, "(ifm_c, pshape_top, pshape_bottom, k_h): %d %d %d %d\n",cshape->ifm_c , pshape->top , pshape->bottom, cshape->k_h);
	ifm_c_group8_num_b = cshape_b->ifm_c / MTX_SCALE;		//ifm 鏉堟挸鍙唂m channel 楠炲啿娼庨崚鍡樺灇8娴�?
	//KRNL_LOG_INFO(LOG_DEBUG, "ifm_c_blk_stride = ifm_c / 8 : %d\n", ifm_c_blk_num);
	ifm_row_stride_b = ByteToHPU64BytesWord(cshape_b->ifm_w * cshape_b->ifm_c);
	ofm_row_stride_b = ByteToHPU64BytesWord(cshape_b->ifm_w / 2 * cshape_b->ofm_c);
	wt_cluster_size_b = ifm_c_group8_num_b * cshape_b->k_w;

	//weight, ifm 闁姤婀乧luster_stride 閸滃異lk_stride 闁粙娓剁憰浣脚嶇純?
	//閹碘偓閺堝娈憇tride 闁垝绗夐棁鈧憰?閸戝繐骞�1)-1閿�?閻╊喚娈戦敍姝磘ride * 64Bytes == 閸愬懎鐡ㄦ稉顓犳畱閻喎鐤勯崑蹇曅╅柌?
	//閹碘偓閺堝娈憇ize/num 闁€燁洣(閸戝繐骞�1)-1
	//region_start, region_end, cluster_start閿�?cluster_end, 閹碘偓閺堝娈慹nd閸у洣璐熸稉濠佺娑擃亝娓堕崥搴″帗缁辩姷娈戦崷鏉挎絻 + 1
	//閻╊喚娈戦敍姝焠d - start == size閿�?鐠侊紕鐣籹ize閺冩湹绗夐棁鈧憰渚€顤傛径?+ 1娴�?
	//婵″倸鍞寸€涙ü鑵戠€涙ê鍋嶆禍?,1,2,3,4,5,6,7. start == 0, end == 7 + 1 = 8, size = end - start = 8 - 0 = 8
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

	//region_start region_end 閸滃瓛PU鐠侊紕鐣诲▽鈥冲彠缁紮绱濋崣顏呮Ц娑撹桨绨℃潪顖欐缂傛牜鈻奸弬閫涚┒閵嗗倻鏁egion 閸椻€茬秶閼煎啫娲块崥搴礉閸戔剝妲哥搾鍛毉region_end閻ㄥ嫬婀撮崸鈧敍宀勫厴娴兼艾鎯婇悳鐥搖f閿涘奔绮爎egion_start闁插秵鏌婂鈧慨瀣吀缁犳浜哥粔?
	//濮濄倕顦╂稉鈧▎顡瞣ad 4 鐞涘fm閿涘本澧嶆禒顧竐gion_end - region_start = row_size * 4
	_set_mmac_region_start(ByteToHPU64BytesWord(localmem_fm_table_dwc_b[0] - MEM_LCMEM_ADDR_S));			//64Bytes 娑撳搫宕熸担?
	_set_mmac_region_end(ByteToHPU64BytesWord(localmem_fm_table_dwc_b[2] + MMA_BLOCK_SIZE - MEM_LCMEM_ADDR_S) + 1);			//64Bytes 娑撳搫宕熸担?
	//娑撳骸宓庣粔顖濈箥缁犳娈憇tride閺堝鍙ч敍瀹籺ride == 1 閺冭绱漛lk_stride = blk_size; stride == 2 閺冭绱漛lk_stride = 2 * blk_size
	//閸楄櫕婀板▎顡眅rnel閻ㄥ垿ilter鐠侊紕鐣诲锔跨瑐鐟欐帟鎹ｆ慨瀣勾閸р偓閿�?娑撳簼绗呮稉鈧▎顡眅rnel閻ㄥ垿ilter鐠侊紕鐣诲锔跨瑐鐟欐帟鎹ｆ慨瀣勾閸р偓閻ㄥ嫬浜哥粔濠氬櫤
	//娴犮儰绗呮禒锝囩垳姒涙ǹ顓籹tride == 1, 閸楃tride_shape->w_strd == 1
	_set_mmac_fm_blk_stride(ifm_c_group8_num_b * 2);							//8閻ㄥ嫬鈧秵鏆�
	//cluster_stride: ifm 娑撯偓閺佸澧栫悰?閸栧懏瀚幍鈧張澶屾畱ifm_c, 閹碘偓閺堝娈憌閿涘奔绔寸悰?閹碘偓閸︺劎娈戦崘鍛摠缁屾椽妫挎径褍鐨�, 閸楄櫕婀扮悰灞藉従鐎圭偛婀撮崸鈧稉搴濈瑓娑撯偓鐞涘矁鎹ｆ慨瀣勾閸р偓閻ㄥ嫬浜哥粔濠氬櫤
	_set_mmac_fm_cluster_stride(ifm_row_stride_b);												//64Bytes 娑撳搫宕熸担?
	//ifm_c 閻�?閻ㄥ嫬鈧秵鏆�	
	_set_mmac_fm_blk_size(ifm_c_group8_num_b - 1);	

	//閸氬奔绔存稉鐚璭rnel weight 瀹革缚绗傜憴鎺旀畱閻愮懓鍩� 閸氬奔绔存稉鐚璭rnelweight 瀹革缚绗傜憴鎺戞倻閸欏厖绔存稉顏嗗仯(w閺傜懓鎮�)閻ㄥ嫬浜哥粔濠氬櫤 === ifm_c / 8 
	_set_mmac_wt_blk_stride(ifm_c_group8_num_b);									//8閻ㄥ嫬鈧秵鏆�
	//8 娑撶尛ernel 閸愬懘鍎撮敍宀€顑囨稉鈧稉鐚璭rnel瀹革缚绗傜憴鎺旀畱閻愮櫢绱濋崚鎵儑娑撯偓娑撶尛ernel瀹革缚绗傜憴鎺戞倻娑撳绔存稉顏嗗仯(h閺傜懓鎮�)閻ㄥ嫬浜哥粔濠氬櫤:娑撯偓閺佸澧�,缁鎶€閿涙fm_row_stride
	_set_mmac_wt_cluster_stride(ByteToHPU64BytesWord(cshape_b->ifm_c * cshape_b->k_w * MTX_SCALE));												//64Bytes 娑撳搫宕熸担?

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
		//cluster_stride: ifm 娑撯偓閺佸澧栫悰?閸栧懏瀚幍鈧張澶屾畱ifm_c, 閹碘偓閺堝娈憌閿涘奔绔寸悰?閹碘偓閸︺劎娈戦崘鍛摠缁屾椽妫挎径褍鐨�, 閸楄櫕婀扮悰灞藉従鐎圭偛婀撮崸鈧稉搴濈瑓娑撯偓鐞涘矁鎹ｆ慨瀣勾閸р偓閻ㄥ嫬浜哥粔濠氬櫤
		_set_mmac_fm_cluster_start(cluster_start_b);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_end(cluster_end_b + 1);						//64Bytes 娑撳搫宕熸担?
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

		for(int j=0; j<w_iter_num_b; j++)        //w_iter_num = cshape->ifm_w / MTX_SCALE; 		//ifm w 楠炲啿娼庨崚鍡樺灇8娴�?
        {
			ifm_start_b = cluster_start_b + (j - /*pshape->side*/1) * ifm_c_group8_num_b;
    		//KRNL_LOG_INFO(LOG_DEBUG, "=====ifm W iter: %d / %d===== ifm_start: %x", j, w_iter_num_b, ifm_start_b);
			
			for(int k=0; k<kernel_group8_num_b; k++){    //times_of_8_kernels = cshape->ofm_c / MTX_SCALE;			//ofm channel 鏉堟挸鍤璮m channel 楠炲啿娼庨崚鍡樺灇8娴�? kernel num is devided by 8
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
			
		_set_mmac_fm_cluster_start(cluster_start_c);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_end(cluster_end_c + 1);						//64Bytes 娑撳搫宕熸担?
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
			
		_set_mmac_fm_cluster_start(cluster_start_a);						//64Bytes 娑撳搫宕熸担?
		_set_mmac_fm_cluster_end(cluster_end_a + 1);						//64Bytes 娑撳搫宕熸担?
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


