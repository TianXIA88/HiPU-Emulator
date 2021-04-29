/* This is the fully-connected layer code  */
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

uint32 alloc_local_fm_with_drain_to_ddr_if_needed_1(local_fm *ofm, ddr_fm *dfm, uint32 num)
{
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

void _kernel_type_4(uint32 ifm,uint32 wt,uint32 ofm)
{
    asm volatile("mv t1, %0"::"r"(ifm):);
    asm volatile("mv t2, %0"::"r"(wt):);
    asm volatile("mv t3, %0"::"r"(ofm):);
    _clr_vreg(vr1);
    mvmac(VPR_NONE, vr1, t1, t2, vr1);
    vsb(vr1, t3, 0);
    // qemu_fprint(QEMU_LOG_MMAB, ofm, 64);
    return;
}

void fc_core(paramTableConv2d_Entry_t *p_conv2d_entry)
{
    int iterations = p_conv2d_entry->conv2d.cshape.ofm_c / VEC_SCALE;
    uint32 wt_size = p_conv2d_entry->conv2d.cshape.ifm_c * VEC_SCALE;
    uint32 wt_start,ofm_start,ifm_start;
    uint32 wt_addr = p_conv2d_entry->wt_addr.lcaddr;
    local_fm local_ofm,local_ifm;
    ddr_fm ddr_ofm,ddr_ifm;
    local_variable local_var;
    // init ifm ofm vars
    uint32 ifm_rsize = p_conv2d_entry->conv2d.cshape.ifm_c;
    uint32 ofm_rsize = p_conv2d_entry->conv2d.cshape.ofm_c;

    init_local_var(&local_var, LOCAL_VAR_BLK, LOCAL_VAR_BLK + MMA_BANK_SIZE, DIVIDE_BY_32BYTES(MMA_BANK_SIZE));
    init_local_fm(&local_ifm, &local_var, 1, ifm_rsize, LOCAL_IFM_BLK);
    init_local_fm(&local_ofm, &local_var, 1, ofm_rsize, LOCAL_OFM_BLK);
    init_ddr_fm(&ddr_ifm, 1, ifm_rsize, p_conv2d_entry->ifm_addr.lcaddr, p_conv2d_entry->ifm_addr.x_pos, p_conv2d_entry->ifm_addr.y_pos);
    init_ddr_fm(&ddr_ofm, 1, ofm_rsize, p_conv2d_entry->ofm_addr.lcaddr, p_conv2d_entry->ofm_addr.x_pos, p_conv2d_entry->ofm_addr.y_pos);
    // initially load ifm to start conv
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32 *)LOCAL_IFM_BLK, p_conv2d_entry->ifm_addr.x_pos, p_conv2d_entry->ifm_addr.y_pos, p_conv2d_entry->ifm_addr.lcaddr, GMEM_ALIGN(ifm_rsize)));
    // initially load wt to start
    LIBHIKL_NASSERT(__rd_from_remote_chunk_non_blocking((uint32 *)(MMB_START_ADDR), p_conv2d_entry->wt_addr.x_pos, p_conv2d_entry->wt_addr.y_pos, p_conv2d_entry->wt_addr.lcaddr,  GMEM_ALIGN(wt_size)));
    // set mmac param
    
    _set_mmac_fm_blk_stride(ByteToHPU64BytesWord(p_conv2d_entry->conv2d.cshape.ifm_c));
    _set_mmac_wt_cluster_stride(ByteToHPU64BytesWord(p_conv2d_entry->conv2d.cshape.ifm_c));
    _set_mmac_wt_blk_stride(ByteToHPU64BytesWord(p_conv2d_entry->conv2d.cshape.ifm_c));
    _set_mmac_region_start(ByteToHPU64BytesWord(local_ifm.bfm.start_addr - MEM_LCMEM_ADDR_S));
    _set_mmac_region_end(ByteToHPU64BytesWord(local_ifm.bfm.start_addr - MEM_LCMEM_ADDR_S + p_conv2d_entry->conv2d.cshape.ifm_c));

    _set_mmac_fm_cluster_stride(ByteToHPU64BytesWord(p_conv2d_entry->conv2d.cshape.ifm_c));
		
    _set_mmac_fm_blk_size(p_conv2d_entry->conv2d.cshape.ifm_c/VEC_SCALE - 1);
    _set_mmac_fm_cluster_num(0);
    _set_mmac_fm_blk_num(0);
    _set_mmac_fm_cluster_start(ByteToHPU64BytesWord(local_ifm.bfm.start_addr - MEM_LCMEM_ADDR_S));
    _set_mmac_fm_cluster_end(ByteToHPU64BytesWord(local_ifm.bfm.start_addr - MEM_LCMEM_ADDR_S + p_conv2d_entry->conv2d.cshape.ifm_c));

    ifm_start = ByteToHPU64BytesWord(MMA_BEGIN);
    alloc_local_fm_with_drain_to_ddr_if_needed_1(&local_ofm, &ddr_ofm, 1);
    for(int i=0;i<iterations;i++)
    {
        __ndma_poll();
        if(i!=(iterations-1))
        {
           wt_addr = wt_addr + p_conv2d_entry->conv2d.cshape.ifm_c * VEC_SCALE;
            LIBHIKL_NASSERT(__rd_from_remote_chunk_non_blocking((uint32 *)(MMB_START_ADDR+((i+1)%2)*4*MMA_BANK_SIZE), p_conv2d_entry->wt_addr.x_pos, p_conv2d_entry->wt_addr.y_pos, wt_addr,  GMEM_ALIGN(wt_size)));
        }
        wt_start = ByteToHPU64BytesWord(MMB_BEGIN + (i%2)*4*MMA_BANK_SIZE);
        ofm_start = ByteToHPU64BytesWord((int)LOCAL_OFM_BLK - MEM_LCMEM_ADDR_S + i * VEC_SCALE);
        _kernel_type_4(ifm_start, wt_start, ofm_start);
    }
    _ndma_remain_ifm_rows_from_localmem_to_ddr_blocking(&local_ofm, &ddr_ofm);
}

void fc_multi_layers()
{
#ifdef QEMU_ENV
    qemu_arch_setup();
    printf("=============================\n\r");
    printf("YOU ARE USING A QEMU INSTANCE\n\r");
    printf("=============================\n\r");
#endif
	
    // get paramTable for conv2d
    paramTableConv2d_t *_pParamTable = *((paramTableConv2d_t **)HIPU200_KNL_PTABLE_ADDR);/*get kernel param table from runtime*/
    hirtKernelParamTableBase_t *base_table = &_pParamTable->infoBase;
    //paramTableConv2d_Entry_t *_pConv2dTable = &_pParamTable->param;
    //conv2d_params_t *_pConv2dParams = &_pConv2dTable->conv2d;

    unsigned int *_flags = ( unsigned int *)HIPU200_MEM_ATOMIC_START;
    paramTableConv2d_Entry_t *conv_table;

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
            //conv_table = (paramTableConv2d_Entry_t *)((int)table + sizeof(hirtKernelParamTableBase_t) + i * sizeof(paramTableConv2d_Entry_t));
            conv_table = &_pParamTable->param[i];
            break;
        }
    }
    LIBHIKL_NASSERT(kernel_id == 255);

    fc_core(conv_table);

#ifdef QEMU_ENV
    qemu_fprint(QEMU_LOG_MEM, conv_table->ofm_addr.lcaddr, 1024);
#endif
}
