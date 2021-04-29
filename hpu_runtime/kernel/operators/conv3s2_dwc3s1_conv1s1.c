#include "dma.h"
#include "hi_addr_def.h"
#include "hihw.h"
#include "int.h"
#include "krnl_log.h"
#include "libconv.h"
#include "lock.h"
#include "operators/hi_krnl_param_conv2d.h"
#include "operators/hi_krnl_param_conv3s2_dwc3s1_conv1s1.h"
// #include "qemu.h"
//

#define LOCAL_IFM_BLK MMA_BANK0_START_ADDR
#define LOCAL_OFM_BLK MMA_BANK1_START_ADDR
#define LOCAL_BAS_BLK MMA_BANK2_START_ADDR
#define LOCAL_VAR_BLK MMA_BANK3_START_ADDR
#define REMOT_IFM_BLK MMA_BANK7_START_ADDR
#define REMOT_VAR_BLK MMA_BANK3_START_ADDR
#define BIAS_SHIFT_BLK MMA_B

#define LCMEM_CONV_A_IN0 MMA_BANK0_START_ADDR
#define LCMEM_CONV_A_IN1 MMA_BANK1_START_ADDR
#define LCMEM_DWC_B_IN0 MMA_BANK2_START_ADDR
#define LCMEM_DWC_B_IN1 MMA_BANK3_START_ADDR
#define LCMEM_CONV_C_IN0 MMA_BANK4_START_ADDR
#define LCMEM_CONV_C_OU0 MMA_BANK5_START_ADDR
#define LCMEM_BIAS_SHIFT MMA_BANK7_START_ADDR

#define INPUT_FM_BASE_ADDR

#define PRE_LOAD_LINE 2
#define IFM_LINE_IN_BANK_CONV_A 4
#define IFM_LINE_IN_BANK_DWC_B 2
#define IFM_PADDING_LINE_NUM 1
// const int ifm_padding_line_num = (cshape_a->k_h - 1) >> 1;

static u32_t ddr_mem_fm_addr = 0;

static const u32_t banktbl_conv_a[] = {LCMEM_CONV_A_IN0, LCMEM_CONV_A_IN1};
static u32_t       bankidx_conv_a   = 0;
static const u32_t banknum_conv_a   = sizeof(banktbl_conv_a) / sizeof(u32_t);

static u32_t       banktbl_dwc_b[]         = {LCMEM_DWC_B_IN0, LCMEM_DWC_B_IN1};
static u32_t       localmem_fm_index_dwc_b = 0;
static const u32_t banknum_dwc_b           = sizeof(banktbl_dwc_b) / sizeof(u32_t);

static u32_t banktbl_conv_c[] = {LCMEM_CONV_C_IN0};

static u32_t banktbl_out[] = {LCMEM_CONV_C_OU0};

//      Bank 0
//         |
//         |
//    +----\----+
//    |  conv1 |
//    |         |
//    +----/----+
//         |
//         |
//         \
//  Bank 1/2 (4 lines)
//         /
//         |
//   +------------+
//   |            |
//   |   Conv2    |
//   |            |
//   +-----/------+
//         |
//         \
//  Bank 3/4 (2 lines)
//         /
//         |
//         |
//         \
//   +------------+
//   |            |
//   |   Conv3    |
//   |            |
//   +-----/------+
//         |
//         |
//         \
//      Bank 5

extern void buf_print(uint32_t buf_addr, uint32_t buf_len);
extern int  get_core_id();
extern void conv2d_head(conv_shape_t* cshape, pad_shape_t* pshape, bool relu, local_fm* ifm, local_fm* ofm, remote_fm* rfm, ddr_fm* dfm, hikl_addr_t* wt_addr, hikl_addr_t* bias_addr, hikl_addr_t* shift_addr);
extern void conv2d_body(conv_shape_t* cshape, pad_shape_t* pshape, bool relu, local_fm* ifm, local_fm* ofm, remote_fm* rfm, hikl_addr_t* wt_addr, hikl_addr_t* bias_addr, hikl_addr_t* shift_addr);
extern void conv2d_tail(conv_shape_t* cshape, pad_shape_t* pshape, bool relu, local_fm* ifm, local_fm* ofm, ddr_fm* dfm, hikl_addr_t* wt_addr, hikl_addr_t* bias_addr, hikl_addr_t* shift_addr);
extern void conv2d_singlecore(conv_shape_t*   cshape,
                              stride_shape_t* stride_shape,
                              pad_shape_t*    pshape,
                              bool            relu,
                              local_fm*       ifm,
                              local_fm*       ofm,
                              ddr_fm*         ifm_addr,
                              ddr_fm*         ddr_ofm,
                              hikl_addr_t*    wt_addr,
                              hikl_addr_t*    bias_addr,
                              hikl_addr_t*    shift_addr);

static uint32                      ifm_rsize, ofm_rsize, core_id, kernel_id;
static u32_t*                      kernel_id_table;
static paramTableConv2d_Entry_t*   conv_table;
static hirtKernelParamTableBase_t* base_table;
static ddr_fm                      ifm_addr, ddr_ofm;
static local_fm                    local_ifm, local_ofm;
static remote_fm                   remote_ifm;
static local_variable              local_var;
static paramTableConv2d_t          param_table;

/*extern int _ifm;*/
/*extern int _wt1;*/
/*extern int _data_out;*/

// typedef struct{
//     int typeVersion;
//     int tableSiz;
//     int parallelism;        // the number of parallel cores
//     int parallelTable[13];  // the ID of each parallel core, Table[0] is
//     head, Table[1] is tail, Table[2,3,...] are body
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

static void __set_fadd_prot(const uint8 round_type, const uint8 shift_num, const uint8 prot_high, const uint8 prot_low)
{
    _set_mmac_round_type(round_type);  // mid
    _set_mmac_fadd_shift_num(shift_num);
    _set_mmac_fadd_prot_high(prot_high);
    _set_mmac_fadd_prot_low(prot_low);
}

static uint32_t __set_bank_lines_addr(const uint32_t base_addr,
                                      const int      lines_num,
                                      const int      line_idx)
{
    // lines_num : num of lines in a bank
    // line_idx : index of lines in a bank (0, 1, 2. ...)
    const uint32_t bank_size = 32 * 1024;

    return base_addr + bank_size / lines_num * line_idx;
}

// static uint32_t __cal_input_ddr_addr(const uint32_t base_addr,
//                                      const uint32_t offset) {
//   ddr_mem_fm_addr = base_addr + d return base_addr + offset;
// }
static void __rd_chunks(const int lines_num, const uint32_t local_mr_addr, const uint32_t row_size, uint32* p_ifm_ptr, hikl_addr_t* ifm_addr)
{
    // local_mr_addr: 使用bank的基地址
    // ddr_offset:
    // 加载一行的ifm大小，这里ddr地址为全局(ddr_mem_fm_addr)，进行修改，每次是修改后的基地址+offset
    const int tmp = lines_num / 2;
    const int slice_line_num = (lines_num % 2 == 0)? lines_num : tmp * 2 + 2; 
    const int bank_offset = MMA_BANK_SIZE / slice_line_num;
    //循环加载 n 行到一个bank， 每次修改全局变量 ddr基地址
    for (int i = 0; i < lines_num; ++i) {
        
        uint32_t localmem_fm_mr_addr = local_mr_addr + i * bank_offset;
        LIBHIKL_NASSERT(__rd_from_remote_chunk_non_blocking(
            (uint32*)localmem_fm_mr_addr, ifm_addr->x_pos, ifm_addr->y_pos, *p_ifm_ptr,
            GMEM_ALIGN(W64ToByte(row_size))));
        *p_ifm_ptr = *p_ifm_ptr + W64ToByte(row_size);
            
        KRNL_LOG_INFO(LOG_DEBUG, " ==== Load ifm ==== ifm addr %d/%d : %x", i, lines_num, localmem_fm_mr_addr);            

        __ndma_poll();
    }
}

static uint32_t _calc_cluster_start_addr(const uint32_t bank_base_addr,
                                         const int      line_num_in_bank,
                                         const int      line_idx)
{
    return ByteToW64(bank_base_addr - MMA_START_ADDR + MMA_BANK_SIZE / line_num_in_bank * line_idx);
}

static void _set_log_flag(const int flag){
    // g_ulPrintDebugLogFlag = debug;
    // g_ulPrintHardwareCmdLogFlag = hardware;
    // g_ulPrintSYSLogFlag = sys;
    // g_ulPrintNDMALogFlag = dma;
    g_ulPrintDebugLogFlag = flag;
    g_ulPrintHardwareCmdLogFlag = flag;
    g_ulPrintSYSLogFlag = flag;
    g_ulPrintNDMALogFlag = 0;
}

void _op_conv3s2_dwc3s1_conv1s1(
    conv2d_params_t* conv2d_param_a,
    conv2d_params_t* conv2d_param_b,
    conv2d_params_t* conv2d_param_c,

    hikl_addr_t* ifm_addr,
    hikl_addr_t* ofm_addr,
    hikl_addr_t* wt_addr_a,
    hikl_addr_t* wt_addr_b,
    hikl_addr_t* wt_addr_c,
    hikl_addr_t* bs_addr_a,
    hikl_addr_t* bs_addr_b,
    hikl_addr_t* bs_addr_c,
    hikl_addr_t* shift_addr_a,
    hikl_addr_t* shift_addr_b,
    hikl_addr_t* shift_addr_c);

void kernel_conv3s2_dwc3s1_conv1s1()
{
    paramTableConv3s2_dwc3s1_conv1s1_t*       _pParamTable = *((paramTableConv3s2_dwc3s1_conv1s1_t**)HIPU200_KNL_PTABLE_ADDR); /*get kernel param table from runtime*/
    paramTableConv3s2_dwc3s1_conv1s1_Entry_t* p_op_entry   = &_pParamTable->param;
    _op_conv3s2_dwc3s1_conv1s1(
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
void _op_conv3s2_dwc3s1_conv1s1(
    conv2d_params_t* conv2d_params_a,
    conv2d_params_t* conv2d_params_b,
    conv2d_params_t* conv2d_params_c,

    hikl_addr_t* ifm_addr,
    hikl_addr_t* ofm_addr,
    hikl_addr_t* wt_addr_a,
    hikl_addr_t* wt_addr_b,
    hikl_addr_t* wt_addr_c,
    hikl_addr_t* bs_addr_a,
    hikl_addr_t* bs_addr_b,
    hikl_addr_t* bs_addr_c,
    hikl_addr_t* shift_addr_a,
    hikl_addr_t* shift_addr_b,
    hikl_addr_t* shift_addr_c)
{
    conv_shape_t* cshape_a = &( conv2d_params_a->cshape );
    conv_shape_t* cshape_b = &( conv2d_params_b->cshape );
    conv_shape_t* cshape_c = &( conv2d_params_c->cshape );


    uint32 i, j, ndma_poll;
    uint32 h_iter_num_b, local_mem_fm_addr, ddr_mem_fm_addr;
    uint32 ifm_ptr;
    uint32 ofm_ptr = ofm_addr->lcaddr;

    uint32 ifm_row_oneline_mrlen_a, ofm_row_oneline_mrlen_a = 0;
    uint32 ifm_row_oneline_mrlen_b, ofm_row_oneline_mrlen_b = 0;
    uint32 ifm_row_oneline_mrlen_c, ofm_row_oneline_mrlen_c = 0;

    uint32 ifm_c_group8_num_a, wt_cluster_size_a, cluster_num_a = 0;
    uint32 w_iter_num_a, kernel_group8_num_a = 0;
    uint32 wt_offset_a, cluster_start_a, cluster_end_a, wt_ptr_a, shift_ptr_a, bs_ptr_a, ifm_ptr_a, ofm_ptr_a = 0;

    uint32 ifm_c_group8_num_b, wt_cluster_size_b, cluster_num_b = 0;
    uint32 w_iter_num_b, kernel_group8_num_b = 0;
    uint32 wt_offset_b, cluster_start_b, cluster_end_b, wt_ptr_b, shift_ptr_b, bs_ptr_b, ifm_ptr_b, ofm_ptr_b = 0;

    uint32 ifm_c_group8_num_c, wt_cluster_size_c, cluster_num_c = 0;
    uint32 w_iter_num_c, kernel_group8_num_c = 0;
    uint32 wt_offset_c, cluster_start_c, cluster_end_c, wt_ptr_c, shift_ptr_c, bs_ptr_c, ifm_ptr_c, ofm_ptr_c = 0;

    uint8 round_type, shift_num, prot_high, prot_low = 0;

    uint32 wt_sz_a, bs_sz_a, shift_sz_a;
    uint32 wt_sz_b, bs_sz_b, shift_sz_b;
    uint32 wt_sz_c, bs_sz_c, shift_sz_c;
    uint32 wt_start_a, bs_start_a, shift_start_a;
    uint32 wt_start_b, bs_start_b, shift_start_b;
    uint32 wt_start_c, bs_start_c, shift_start_c;

    uint32 ifm_row_cluster_stride_a, ifm_row_cluster_stride_b, ifm_row_cluster_stride_c;

    uint32 bankidx_ld = 1;
    uint32 bankidx_conv_a_in, lineidx_conv_a_in;
    uint32 bankidx_conv_a_ou, lineidx_conv_a_ou;
    uint32 bankidx_dwc_b_in, lineidx_dwc_b_in;

    // Calculate total weight size
    wt_sz_a = cshape_a->k_w * cshape_a->k_h * cshape_a->ifm_c * cshape_a->ofm_c;  // in Bytes
    wt_sz_b = cshape_b->k_w * cshape_b->k_h * cshape_b->ifm_c * cshape_b->ofm_c * MTX_SCALE;  // in Bytes
    wt_sz_c = cshape_c->k_w * cshape_c->k_h * cshape_c->ifm_c * cshape_c->ofm_c;  // in Bytes
	bs_sz_a = cshape_a->ofm_c * MTX_SCALE * 4;
	bs_sz_b = cshape_b->ofm_c * MTX_SCALE * 4;
	bs_sz_c = cshape_c->ofm_c * MTX_SCALE * 4;
	shift_sz_a = cshape_a->ofm_c * MTX_SCALE;
	shift_sz_b = cshape_b->ofm_c * MTX_SCALE;
	shift_sz_c = cshape_c->ofm_c * MTX_SCALE;

    wt_start_a = MMB_START_ADDR;
    wt_start_b = wt_start_a + GMEM_ALIGN(wt_sz_a);
    wt_start_c = wt_start_b + GMEM_ALIGN(wt_sz_b);
    bs_start_a = LCMEM_BIAS_SHIFT;
    bs_start_b = bs_start_a + GMEM_ALIGN(bs_sz_a);
    bs_start_c = bs_start_b + GMEM_ALIGN(bs_sz_b);
    shift_start_a = bs_start_c + GMEM_ALIGN(bs_sz_c);
    shift_start_b = shift_start_a + GMEM_ALIGN(shift_sz_a);
    shift_start_c = shift_start_b + GMEM_ALIGN(shift_sz_b);

    ifm_row_cluster_stride_a = ByteToW64(MMA_BANK_SIZE / IFM_LINE_IN_BANK_CONV_A);
    ifm_row_cluster_stride_b = ByteToW64(MMA_BANK_SIZE / IFM_LINE_IN_BANK_DWC_B);
    ifm_row_cluster_stride_c = ByteToW64(MMA_BANK_SIZE);

    // Load all weights
    // KRNL_LOG_INFO(LOG_DEBUG, "Load Weights: [%d%d%x]->[%x]\n\r",
    // wt_addr->x_pos, wt_addr->y_pos, wt_addr->lcaddr, (uint32 *)MMB_START_ADDR);
    // MMB can't be access by HPU scalar ALU
    LIBHIKL_NASSERT( __rd_from_remote_chunk_blocking((uint32*)wt_start_a, wt_addr_a->x_pos, wt_addr_a->y_pos, wt_addr_a->lcaddr, GMEM_ALIGN(wt_sz_a)));
    LIBHIKL_NASSERT( __rd_from_remote_chunk_blocking((uint32*)wt_start_b, wt_addr_b->x_pos, wt_addr_b->y_pos, wt_addr_b->lcaddr, GMEM_ALIGN(wt_sz_b)));
    LIBHIKL_NASSERT( __rd_from_remote_chunk_blocking((uint32*)wt_start_c, wt_addr_c->x_pos, wt_addr_c->y_pos, wt_addr_c->lcaddr, GMEM_ALIGN(wt_sz_c)));
    // Load all bias
    // KRNL_LOG_INFO(LOG_DEBUG, "Load Bias: [%x]->[%x]\n\r", bs_addr->lcaddr,
    // (uint32 *)(BIAS_SHIFT_BLK));
    LIBHIKL_NASSERT( __rd_from_remote_chunk_blocking((uint32*)bs_start_a, bs_addr_a->x_pos, bs_addr_a->y_pos, bs_addr_a->lcaddr, GMEM_ALIGN(bs_sz_a)));
    LIBHIKL_NASSERT( __rd_from_remote_chunk_blocking((uint32*)bs_start_b, bs_addr_b->x_pos, bs_addr_b->y_pos, bs_addr_b->lcaddr, GMEM_ALIGN(bs_sz_b)));
    LIBHIKL_NASSERT( __rd_from_remote_chunk_blocking((uint32*)bs_start_c, bs_addr_c->x_pos, bs_addr_c->y_pos, bs_addr_c->lcaddr, GMEM_ALIGN(bs_sz_c)));
    // Load all shift_num
    // KRNL_LOG_INFO(LOG_DEBUG, "Load bs shift finished");
    // KRNL_LOG_INFO(LOG_DEBUG, "Load Shift_mtx: [%x]->[%x]\n\r",
    // shift_addr->lcaddr, (uint32 *)(BIAS_SHIFT_BLK + bs_sz));

    LIBHIKL_NASSERT( __rd_from_remote_chunk_blocking((uint32*)shift_start_a, shift_addr_a->x_pos, shift_addr_a->y_pos, shift_addr_a->lcaddr, GMEM_ALIGN(shift_sz_a)));
    LIBHIKL_NASSERT( __rd_from_remote_chunk_blocking((uint32*)shift_start_b, shift_addr_b->x_pos, shift_addr_b->y_pos, shift_addr_b->lcaddr, GMEM_ALIGN(shift_sz_b)));
    LIBHIKL_NASSERT( __rd_from_remote_chunk_blocking((uint32*)shift_start_c, shift_addr_c->x_pos, shift_addr_c->y_pos, shift_addr_c->lcaddr, GMEM_ALIGN(shift_sz_c)));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Calculate total iteration
    kernel_group8_num_a = cshape_a->ofm_c / MTX_SCALE;  // ofm channel čžĺşfm channel
                                                        // ĺšłĺĺć8ďż? kernel num is devided by 8
    kernel_group8_num_b = cshape_b->ofm_c / MTX_SCALE;  // ofm channel čžĺşfm channel
                                                        // ĺšłĺĺć8ďż? kernel num is devided by 8
    kernel_group8_num_c = cshape_c->ofm_c / MTX_SCALE;  // ofm channel čžĺşfm channel
                                                        // ĺšłĺĺć8ďż? kernel num is devided by 8
    w_iter_num_a = cshape_a->ifm_w / MTX_SCALE;         // ifm w ĺšłĺĺć8ďż?
    w_iter_num_b = cshape_b->ifm_w / MTX_SCALE;         // ifm w ĺšłĺĺć8ďż?
    w_iter_num_c = cshape_c->ifm_w / MTX_SCALE;         // ifm w ĺšłĺĺć8ďż?
    // KRNL_LOG_INFO(LOG_DEBUG, "(ifm_c, pshape_top, pshape_bottom, k_h): %d %d %d
    // %d\n",cshape->ifm_c , pshape->top , pshape->bottom, cshape->k_h);
    ifm_c_group8_num_a = cshape_a->ifm_c / MTX_SCALE;  // ifm čžĺĽfm channel ĺšłĺĺć8ďż?
    ifm_c_group8_num_b = cshape_b->ifm_c / MTX_SCALE;  // ifm čžĺĽfm channel ĺšłĺĺć8ďż?
    ifm_c_group8_num_c = cshape_c->ifm_c / MTX_SCALE;  // ifm čžĺĽfm channel ĺšłĺĺć8ďż?
    // KRNL_LOG_INFO(LOG_DEBUG, "ifm_c_blk_stride = ifm_c / 8 : %d\n",
    // ifm_c_blk_num);
    // TODO:这里ifm ofm 中间变量是一样大的？？？ 注意检查大小
    ifm_row_oneline_mrlen_a = ByteToW64(cshape_a->ifm_w * cshape_a->ifm_c);
    ifm_row_oneline_mrlen_b = ByteToW64(cshape_b->ifm_w * cshape_b->ifm_c);
    ifm_row_oneline_mrlen_c = ByteToW64(cshape_c->ifm_w * cshape_c->ifm_c);
    ofm_row_oneline_mrlen_a = ByteToW64(cshape_a->ifm_w / conv2d_params_a->strd_shape.h_strd  * cshape_a->ofm_c);
    ofm_row_oneline_mrlen_b = ByteToW64(cshape_b->ifm_w * cshape_b->ofm_c);
    ofm_row_oneline_mrlen_c = ByteToW64(cshape_c->ifm_w * cshape_c->ofm_c);
    wt_cluster_size_a       = ifm_c_group8_num_a * cshape_a->k_w;
    wt_cluster_size_b       = ifm_c_group8_num_b * cshape_b->k_w;
    wt_cluster_size_c       = ifm_c_group8_num_c * cshape_c->k_w;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Calculate total iteration
    h_iter_num_b = cshape_b->ifm_h;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // =========================== conv_a 3s2 ===================================
    ifm_ptr           = ifm_addr->lcaddr;
    uint32 *p_ifm_ptr = &ifm_ptr;
    uint32 ofm_addr_conv_a = 0;

    //* conv_a ofm line : 1 
    __rd_chunks(IFM_LINE_IN_BANK_CONV_A - 1, banktbl_conv_a[0] + MMA_BANK_SIZE / IFM_LINE_IN_BANK_CONV_A,
                    ifm_row_oneline_mrlen_a, p_ifm_ptr, ifm_addr);
    __rd_chunks(IFM_LINE_IN_BANK_CONV_A, banktbl_conv_a[1], ifm_row_oneline_mrlen_a, p_ifm_ptr, ifm_addr);

    cluster_start_a = _calc_cluster_start_addr(banktbl_conv_a[0], IFM_LINE_IN_BANK_CONV_A, 1);
    cluster_end_a = cluster_start_a + ifm_row_oneline_mrlen_a;              // always the end of the first ifm row
    cluster_num_a = cshape_a->k_h - 1 - conv2d_params_a->pshape.top;  // cluster_num = k_h
    ofm_addr_conv_a = banktbl_dwc_b[0] + 0 * (MMA_BANK_SIZE / IFM_LINE_IN_BANK_DWC_B) - MMA_START_ADDR;  

    KRNL_LOG_INFO(LOG_DEBUG, "==== cluster_start_a ： %x ==== ", cluster_start_a);

    KRNL_LOG_INFO(LOG_DEBUG, " === ifm [conv_a][0] === ");
    buf_print(W64ToByte(cluster_start_a) + MMA_START_ADDR, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a)));
    

    


    one_row_conv(
        0, conv2d_params_a,
        ByteToW64(wt_start_a - MMA_START_ADDR),
        ByteToW64(ofm_addr_conv_a),
        ByteToW64(bs_start_a - MMA_START_ADDR),
        ByteToW64(shift_start_a - MMA_START_ADDR),
        ByteToW64(banktbl_conv_a[0] - MEM_LCMEM_ADDR_S),
        ByteToW64(banktbl_conv_a[0] + banknum_conv_a * MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),
        ifm_row_cluster_stride_a,
        cluster_start_a,
        cluster_end_a,
        cluster_num_a,
        CONV_TYPE_CLASSIC, 1);

        KRNL_LOG_INFO(LOG_DEBUG, " === ofm [conv_a][0] === ");
        buf_print(ofm_addr_conv_a + MMA_START_ADDR, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_b)));


    //* conv_a ofm line : 2
    // NOTE: 这里 使用 i+2 体现 stride_h
    cluster_start_a = _calc_cluster_start_addr(banktbl_conv_a[0], IFM_LINE_IN_BANK_CONV_A, 2);
    cluster_end_a = cluster_start_a + ifm_row_oneline_mrlen_a;  // always the end of the first ifm row
    cluster_num_a = cshape_a->k_h - 1;             // cluster_num = k_h
    ofm_addr_conv_a = banktbl_dwc_b[0] + 1 * (MMA_BANK_SIZE / IFM_LINE_IN_BANK_DWC_B) - MMA_START_ADDR;  
    
    _set_log_flag(0);
    KRNL_LOG_INFO(LOG_DEBUG, " === ifm [conv_a][1-0] === ");
    buf_print(W64ToByte(_calc_cluster_start_addr(banktbl_conv_a[0], IFM_LINE_IN_BANK_CONV_A, 2)) + MMA_START_ADDR, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a)));
    KRNL_LOG_INFO(LOG_DEBUG, " === ifm [conv_a][1-1] === ");
    buf_print(W64ToByte(_calc_cluster_start_addr(banktbl_conv_a[0], IFM_LINE_IN_BANK_CONV_A, 3)) + MMA_START_ADDR, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a)));
    KRNL_LOG_INFO(LOG_DEBUG, " === ifm [conv_a][1-2] === ");
    buf_print(W64ToByte(_calc_cluster_start_addr(banktbl_conv_a[1], IFM_LINE_IN_BANK_CONV_A, 0)) + MMA_START_ADDR, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a)));
    _set_log_flag(0);

    one_row_conv(
        0 + 1 * conv2d_params_a->strd_shape.h_strd, conv2d_params_a,
        ByteToW64(wt_start_a - MMA_START_ADDR),
        ByteToW64(ofm_addr_conv_a),
        ByteToW64(bs_start_a - MMA_START_ADDR), 
        ByteToW64(shift_start_a - MMA_START_ADDR), 
        ByteToW64(banktbl_conv_a[0] - MEM_LCMEM_ADDR_S),
        ByteToW64(banktbl_conv_a[0] + banknum_conv_a * MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),
        ifm_row_cluster_stride_a,
        cluster_start_a,
        cluster_end_a,
        cluster_num_a, 
        CONV_TYPE_CLASSIC, 1);

        KRNL_LOG_INFO(LOG_DEBUG, " === ofm [conv_a][1] === ");
        _set_log_flag(0);
        buf_print(ofm_addr_conv_a + MMA_START_ADDR, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_b)));
        _set_log_flag(0);

    // =================== dwconv_b =========================

    // dwc_b uses 2 bank * 2 lines
    // h_iter_num_b = 4;
    for (i = 0; i < h_iter_num_b; i++) {
        KRNL_LOG_INFO(LOG_DEBUG, "=====ifm H iter: %d / %d=====\n\r", i,
                      h_iter_num_b);

        // REVIEW: 这里判断bottom， 需要分情况，因为这里隔行加载，
        bankidx_conv_a_in = ((i + 2) % 4) / 2, lineidx_conv_a_in = (i % 2) * 2;
        bankidx_conv_a_ou = ((i + 2) % 4) / 2, lineidx_conv_a_ou = i % 2;
        bankidx_dwc_b_in = (( i - 1 ) % 4) / 2, lineidx_dwc_b_in = ( i - 1 ) % 2;
        // int bankidx_dwc_b_ou = 4, lineidx_dwc_b_ou = 0;
        // int banktbl_conv_c_in = 4, lineidx_conv_c_in = 0;
        // int bankidx_conv_c_ou = 5, lineidx_conv_c_ou = 0;
        if ( i % 2 == 1){  // 每两行加载一个bank,4 lines
            bankidx_ld = (bankidx_ld+1) > 1? 0 : 1;
            if (i < h_iter_num_b - 2) {
            {  
                __rd_chunks(IFM_LINE_IN_BANK_CONV_A, banktbl_conv_a[bankidx_ld], ifm_row_oneline_mrlen_a, p_ifm_ptr, ifm_addr);
            }
            }else if (i == h_iter_num_b - 2) {  // bottom 的情况只需要加载一行
                __rd_chunks(IFM_LINE_IN_BANK_CONV_A, banktbl_conv_a[bankidx_ld], ifm_row_oneline_mrlen_a, p_ifm_ptr, ifm_addr);
            }
        }

        
        // ===
        if (is_ifm_top(i, &conv2d_params_b->cshape, &conv2d_params_b->pshape)) {
            bankidx_dwc_b_in = 0, lineidx_dwc_b_in = 0;
            cluster_num_b = cshape_b->k_h - 1 - conv2d_params_b->pshape.top;  // always less than the full kernel height
        }
        else {
            // REVIEW: stride == 2 的时候不会进入 > 的情况，刚好卷积完
            if (is_ifm_bottom(i, &conv2d_params_b->cshape, &conv2d_params_b->pshape))  // at bottom
                cluster_num_b = cshape_b->k_h - 1 - conv2d_params_b->pshape.bottom;
            else                                // at middle
                cluster_num_b = cshape_b->k_h - 1;  // always equal the full kernel height
        }
        _set_log_flag(1);
        cluster_start_b = _calc_cluster_start_addr(banktbl_dwc_b[bankidx_dwc_b_in], IFM_LINE_IN_BANK_DWC_B, lineidx_dwc_b_in);
        KRNL_LOG_INFO(LOG_DEBUG, " === ld params: (bank): (%d)(in)", bankidx_ld); 
        KRNL_LOG_INFO(LOG_DEBUG, " === conv_a params: (bank,line): (%d, %d)(in), (%d, %d)(ou)", 
        bankidx_conv_a_in, lineidx_conv_a_in, bankidx_conv_a_ou, lineidx_conv_a_ou );
        KRNL_LOG_INFO(LOG_DEBUG, " === dwc_b params: (bank,line): (%d, %d)(in), cluster_start_b: %d", bankidx_dwc_b_in, lineidx_dwc_b_in, cluster_start_b);
        cluster_end_b = cluster_start_b + ifm_row_oneline_mrlen_b;  // always the end of the first ifm row
        _set_log_flag(0);

        if ( i >= 77)
        _set_log_flag(1);
        KRNL_LOG_INFO(LOG_DEBUG, " === [Dwc_b][%d] Start === ", i);
        one_row_conv(
            i, conv2d_params_b,
            ByteToW64(wt_start_b - MMA_START_ADDR),
            ByteToW64(banktbl_conv_c[0]  - MMA_START_ADDR),
            ByteToW64(bs_start_b - MMA_START_ADDR), 
            ByteToW64(shift_start_b - MMA_START_ADDR), 
            ByteToW64(banktbl_dwc_b[0] - MEM_LCMEM_ADDR_S),
            ByteToW64(banktbl_dwc_b[0] + banknum_dwc_b * MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),
            ifm_row_cluster_stride_b,
            cluster_start_b,
            cluster_end_b,
            cluster_num_b, 
            CONV_TYPE_DEPTH_WISE, 1);

        //* [LOG] ofm
        _set_log_flag(1);
        KRNL_LOG_INFO(LOG_DEBUG, " === ofm [dwc_b][%d] === ", i);
        buf_print(banktbl_conv_c[0], GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_c)));
        _set_log_flag(0);
       
        KRNL_LOG_INFO(LOG_DEBUG, "===== Dwc_b Finished : %d / %d=====\n\r", i, h_iter_num_b);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        cluster_start_c = ByteToW64(banktbl_conv_c[0] - MMA_START_ADDR);
        cluster_end_c = cluster_start_c + ifm_row_oneline_mrlen_c;  // always the end of the first ifm row
        cluster_num_c = cshape_c->k_h - 1;

        if ( i >= 77)
            _set_log_flag(1);
        KRNL_LOG_INFO(LOG_DEBUG, " === [Conv_c][%d] Start === ", i);
        one_row_conv(i, conv2d_params_c,        	  	    									
                ByteToW64(wt_start_c-MMA_START_ADDR),											
                ByteToW64(banktbl_out[0] - MMA_START_ADDR),									
                ByteToW64(bs_start_c-MMA_START_ADDR),											
                ByteToW64(shift_start_c-MMA_START_ADDR),										
                ByteToW64(banktbl_conv_c[0] - MEM_LCMEM_ADDR_S),	                     		
                ByteToW64(banktbl_conv_c[0] + MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),	       	
                0,        												
                cluster_start_c,                    											
                cluster_end_c,                      											
                cluster_num_c,
                CONV_TYPE_CLASSIC, 1);
        _set_log_flag(0);

        KRNL_LOG_INFO(LOG_DEBUG, "=====Conv_c Finished : %d / %d=====\n\r", i,
                      h_iter_num_b);

        _set_log_flag(0);
        KRNL_LOG_INFO(LOG_DEBUG, " === ld params: (bank): (%d)(in)", bankidx_ld); 
        KRNL_LOG_INFO(LOG_DEBUG, " === conv_a params: (bank,line): (%d, %d)(in), (%d, %d)(ou)", 
        bankidx_conv_a_in, lineidx_conv_a_in, bankidx_conv_a_ou, lineidx_conv_a_ou );
        KRNL_LOG_INFO(LOG_DEBUG, " === dwc_b params: (bank,line): (%d, %d)(in), cluster_start_b: %d", bankidx_dwc_b_in, lineidx_dwc_b_in, cluster_start_b);
        KRNL_LOG_INFO(LOG_DEBUG, " === ofm [conv_c][%d] === ", i);
        buf_print(banktbl_out[0], GMEM_ALIGN(W64ToByte(ofm_row_oneline_mrlen_c)));
        _set_log_flag(0);

        // Check for the next ifm row if we are not at bottom

        // __ndma_poll();

        // st results
        LIBHIKL_NASSERT(__wr_to_remote_chunk_non_blocking((uint32*)banktbl_out[0], ofm_addr->x_pos, ofm_addr->x_pos, ofm_ptr,
            (W64ToByte(ofm_row_oneline_mrlen_c))));
        __ndma_poll();
        ofm_ptr += W64ToByte(ofm_row_oneline_mrlen_c);

        if (i < h_iter_num_b - 2) {
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // TODO: conv_a计算每次load 4 lines, 计算使用 3 lines, 需要分开指定bank
            cluster_start_a = _calc_cluster_start_addr(banktbl_conv_a[bankidx_conv_a_in],
                                         IFM_LINE_IN_BANK_CONV_A, lineidx_conv_a_in);
            cluster_end_a = cluster_start_a + ifm_row_oneline_mrlen_a;  // always the end of the first ifm row
            cluster_num_a = cshape_a->k_h - 1;             // always less than the full kernel height
            ofm_addr_conv_a = banktbl_dwc_b[bankidx_conv_a_ou] + lineidx_conv_a_ou * (MMA_BANK_SIZE / IFM_LINE_IN_BANK_DWC_B) - MMA_START_ADDR;  
    
            if (i >= 75)
            _set_log_flag(1);
            KRNL_LOG_INFO(LOG_DEBUG, " === [Conv_a][%d] Start === ", ( i+2 )*2);
            one_row_conv(0 + (i+2) * conv2d_params_a->strd_shape.h_strd, 
                            conv2d_params_a,        	  	    										
							ByteToW64(wt_start_a-MMA_START_ADDR),											
							ByteToW64(ofm_addr_conv_a), 						
							ByteToW64(bs_start_a-MMA_START_ADDR),											
							ByteToW64(shift_start_a-MMA_START_ADDR),										
     		                ByteToW64(banktbl_conv_a[0] - MEM_LCMEM_ADDR_S),	                     		
     		                ByteToW64(banktbl_conv_a[0] + banknum_conv_a * MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),	      	
     		              	ifm_row_cluster_stride_a,               								
     		                cluster_start_a,                    										
     		                cluster_end_a,                      										
     		                cluster_num_a,
							CONV_TYPE_CLASSIC, 1);

            
            if (i >= 75)
            _set_log_flag(1);
            KRNL_LOG_INFO(LOG_DEBUG, " === ofm [conv_a][%d] === ", ( i+2 )*2);
            buf_print(ofm_addr_conv_a + MMA_START_ADDR, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_b)));
            _set_log_flag(0);


            KRNL_LOG_INFO(LOG_DEBUG, "=====Conv_a Finished : %d / %d=====\n\r", i,
                          h_iter_num_b);
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }

    
    

    KRNL_LOG_INFO(LOG_DEBUG, "=====Compute Finished =====\n\r");

    return;
}
