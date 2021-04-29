#include "dma.h"
#include "hi_addr_def.h"
#include "hihw.h"
#include "int.h"
#include "krnl_log.h"
#include "libconv.h"
#include "lock.h"
#include "operators/hi_krnl_param_conv2d.h"
#include "operators/hi_krnl_param_conv1s1_upsmp2x_add.h"

// #include "qemu.h"

extern void buf_print(uint32_t buf_addr, uint32_t buf_len);
extern int  get_core_id();
static void __intrinsic_func_upsample_add__(uint32 ifm, uint32 wt, uint32 ofm, uint32 bias_start, uint32 shift_start, bool relu, uint32 ifm_add)
{
    uint32 vfadd_bias = 0;
    asm volatile("mv t1, %0" ::"r"(ifm)
                 :);
    asm volatile("mv t2, %0" ::"r"(wt)
                 :);
    asm volatile("mv t3, %0" ::"r"(ofm)
                 :);
    asm volatile("mv t4, %0" ::"r"(vfadd_bias)
                 :);
    asm volatile("mv t5, %0" ::"r"(shift_start));
    asm volatile("mv t6, %0" ::"r"(ifm_add));

    _clr_vreg(vr1);
    mmac(VPR_NONE, vr1, t1, t2, vr1);
    // now still 32 bits per item
    vlw(vr2, t5, 0);
    vlb(vr3, t6, 0);
    vadd_vv(VPR_NONE, vr1, vr1, vr2);

#ifdef HPU200_RSHIFT_WITH_DECREMENT1_THEN_VFADD_RSHIFT1
    uint32 decrement = 1;
    asm volatile("mv t0, %0" ::"r"(decrement));
    vsub_vs(VPR_NONE, vr3, vr3, t0);
#endif

    vsra_vv(VPR_NONE, vr1, vr1, vr3);
    if (relu) {
        KRNL_LOG_INFO(LOG_DEBUG, "relu_not_zero : %d", relu);
        vmax_vs(VPR_NONE, vr1, vr1, 0);
    }
    vfadd_vs(VPR_NONE, vr1, vr1, t4);
    // TODO
    asm volatile("mv t4, %0" ::"r"(ifm_add)
                 :);
    // can follow other vector operations ...
    vlb(vr4, t6, 0);  // ifm_add --> vr4
    vfadd_vv(VPR_NONE, vr1, vr1, vr4);
    vsb(vr1, t3, 0);
    // qemu_fprint(QEMU_LOG_MEM, ofm, 64);
    return;
}

static void _set_log_flag(const int flag)
{
    // g_ulPrintDebugLogFlag = debug;
    // g_ulPrintHardwareCmdLogFlag = hardware;
    // g_ulPrintSYSLogFlag = sys;
    // g_ulPrintNDMALogFlag = dma;
    g_ulPrintDebugLogFlag       = flag;
    g_ulPrintHardwareCmdLogFlag = flag;
    g_ulPrintSYSLogFlag         = flag;
    g_ulPrintNDMALogFlag        = 0;
}

#define UPSAMPLE_SCALE (2)
#define LCMEM_TENSOR_CONV_IN MMA_BANK0_START_ADDR
#define LCMEM_TENSOR_VADD_IN MMA_BANK1_START_ADDR
#define LCMEM_TENSOR_CONV_OU MMA_BANK2_START_ADDR
#define LCMEM_TENSOR_SMPL_OU MMA_BANK3_START_ADDR
#define LCMEM_TENSOR_VADD_OU MMA_BANK4_START_ADDR
#define LCMEM_BIAS_SHIFT MMA_BANK7_START_ADDR

static u32_t       ddr_mem_fm_addr  = 0;
static const u32_t banktbl_conv_a[] = {LCMEM_TENSOR_CONV_IN};
static const u32_t banknum_conv_a   = sizeof(banktbl_conv_a) / sizeof(u32_t);

static const u32_t banktbl_smpl[] = {LCMEM_TENSOR_CONV_OU};
static const u32_t banknum_smpl   = sizeof(banktbl_smpl) / sizeof(u32_t);

static const u32_t banktbl_add_b[] = {LCMEM_TENSOR_VADD_IN, LCMEM_TENSOR_SMPL_OU};
static const u32_t banknum_add_b   = sizeof(banktbl_add_b) / sizeof(u32_t);

static const u32_t banktbl_ou[] = {LCMEM_TENSOR_VADD_OU};
static const u32_t banknum_ou   = sizeof(banktbl_ou) / sizeof(u32_t);

//     bank0          bank1
//       |              |
//       |              |
// /-----\-----\        |
// |           |        |
// |   conv_a  |        |
// |           |        |
// \-----/-----/        |
//       |              |
//     bank2            |
//       |              |
// /-----------\        |
// |           |        |
// |   smp_2x  |        |
// |           |        |
// \-----------/        |
//        `,            |
//          .           |
//           \          |
//           bank3      |
//              .       |
//          /----'------\---\
//          |               |
//          |     add       |
//          |               |
//          \-------/-------/
//                  |
//                  |
//                bank4

void _op_conv1s1_upsmp2x_add(
    conv2d_params_t* conv_a,
    add_params_t*    add_b,

    hikl_addr_t* ifm_addr_conv_a,
    hikl_addr_t* ofm_addr_conv_a,

    hikl_addr_t* ifm_addr_add_b,
    hikl_addr_t* ofm_addr_add_b,

    hikl_addr_t* wt_addr_conv_a,
    hikl_addr_t* bias_addr_conv_a,
    hikl_addr_t* shift_addr_conv_a);

void kernel_conv1s1_upsmp2x_add()
{
    paramTableConv1s1_upsmp2x_add_t*       _pParamTable = *((paramTableConv1s1_upsmp2x_add_t**)HIPU200_KNL_PTABLE_ADDR); /*get kernel param table from runtime*/
    paramTableConv1s1_upsmp2x_add_Entry_t* p_op_entry   = &_pParamTable->param;
    _op_conv1s1_upsmp2x_add(
        &p_op_entry->conv1,
        &p_op_entry->add1,

        &p_op_entry->ifm_addr_conv1,
        &p_op_entry->ofm_addr_conv1,

        &p_op_entry->ifm_addr_add1,
        &p_op_entry->ofm_addr_add1,

        &p_op_entry->wt_addr_conv1,
        &p_op_entry->bias_addr_conv1,
        &p_op_entry->shift_addr_conv1);
}
void _op_conv1s1_upsmp2x_add(
    conv2d_params_t* conv_a,
    add_params_t*    add_b,

    hikl_addr_t* ifm_addr_conv_a,
    hikl_addr_t* ofm_addr_conv_a,

    hikl_addr_t* ifm_addr_add_b,
    hikl_addr_t* ofm_addr_add_b,

    hikl_addr_t* wt_addr_conv_a,
    hikl_addr_t* bias_addr_conv_a,
    hikl_addr_t* shift_addr_conv_a)
{
    conv_shape_t* cshape_a = &(conv_a->cshape);

    uint32 i, j, k, ndma_pool;
    uint32 h_iter_num = cshape_a->ifm_h * UPSAMPLE_SCALE;
    uint32 ifm_row_size;

    uint32 ifm_c_group8_num_a, ifm_c_group8_num_b;
    uint32 w_iter_num_a, w_iter_num_b;
    uint32 ifm_line_ptr_global;
    uint32 ifm_line_ptr_local;
    uint32 ifm_line_ptr_global_b;
    uint32 ifm_line_ptr_local_b;
    uint32 ofm_line_ptr_global;
    uint32 ofm_line_ptr_local;
    uint32 ifm_ptr = 0;
    uint32 ofm_ptr = 0;

    uint32 ifm_row_size_b;
    uint32 ofm_row_size;
    uint32 kernel_group8_num_a, wt_cluster_size_a;
    uint32 ifm_row_oneline_mrlen_a, ofm_row_oneline_mrlen_a;
    uint32 ifm_row_oneline_mrlen_b, ofm_row_oneline_mrlen_b;

    uint32 wt_sz_a, bs_sz_a, shift_sz_a;
    uint32 wt_offset_a, wt_start_a, bs_start_a, bs_end_a, ifm_start_a,
        cluster_start_a, cluster_end_a, cluster_num_a, ofm_start_a = 0;
    uint32 bias_start_a, shift_start_a;

    wt_sz_a       = cshape_a->k_w * cshape_a->k_h * cshape_a->ifm_c * cshape_a->ofm_c;  // in Bytes
    bs_sz_a       = cshape_a->ofm_c * MTX_SCALE * 4;
    shift_sz_a    = cshape_a->ofm_c * MTX_SCALE;
    wt_start_a    = MMB_START_ADDR;
    bs_start_a    = LCMEM_BIAS_SHIFT;
    shift_start_a = bs_start_a + GMEM_ALIGN(bs_sz_a);

    // Load all weights
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32*)wt_start_a, wt_addr_conv_a->x_pos, wt_addr_conv_a->y_pos, wt_addr_conv_a->lcaddr, GMEM_ALIGN(wt_sz_a)));
    // Load all bias
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32*)bs_start_a, bias_addr_conv_a->x_pos, bias_addr_conv_a->y_pos, bias_addr_conv_a->lcaddr, GMEM_ALIGN(bs_sz_a)));
    // Load all shift_num
    LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32*)shift_start_a, shift_addr_conv_a->x_pos, shift_addr_conv_a->y_pos, shift_addr_conv_a->lcaddr, GMEM_ALIGN(shift_sz_a)));

    // Calculate total iteration
    w_iter_num_a            = cshape_a->ifm_w / MTX_SCALE;
    w_iter_num_b            = cshape_a->ifm_w / MTX_SCALE;
    ifm_c_group8_num_a      = cshape_a->ifm_c / MTX_SCALE;
    ifm_c_group8_num_b      = cshape_a->ofm_c / MTX_SCALE;
    kernel_group8_num_a     = cshape_a->ofm_c / MTX_SCALE;  // ofm channel ??fm channel ????8??
    wt_cluster_size_a       = ifm_c_group8_num_a * cshape_a->k_w;
    ifm_row_size            = (cshape_a->ifm_w / MTX_SCALE) * (cshape_a->ifm_c / MTX_SCALE);
    ifm_row_size_b          = ifm_row_size * 2;
    ofm_row_size            = ifm_row_size * 2;
    ifm_row_oneline_mrlen_a = ByteToW64(cshape_a->ifm_w * cshape_a->ifm_c);
    ifm_row_oneline_mrlen_b = ByteToW64(cshape_a->ifm_w * cshape_a->ofm_c);
    ofm_row_oneline_mrlen_a = ByteToW64(cshape_a->ifm_w * cshape_a->ofm_c);
    ofm_row_oneline_mrlen_b = ByteToW64(cshape_a->ifm_w * 2 * cshape_a->ofm_c);

    _set_log_flag(1);
    KRNL_LOG_INFO(LOG_DEBUG, " === Kernel Start === ");
    _set_log_flag(0);

    ifm_line_ptr_global = ifm_addr_conv_a->lcaddr;
    ifm_line_ptr_global_b = ifm_addr_add_b->lcaddr;

    for (i = 0; i < h_iter_num; i++) {
        // load & conv & smp2x
        // if ((h_iter_num & 0x00000001) == 0)
		_set_log_flag(1);
        KRNL_LOG_INFO(LOG_DEBUG, " === ifm [ifm_buf_addr : 0x%x] === ", ifm_line_ptr_global);
		_set_log_flag(0);

        LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32*)banktbl_conv_a[0], ifm_addr_conv_a->x_pos, ifm_addr_conv_a->y_pos,
                                                        ifm_line_ptr_global, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a))));
        ifm_line_ptr_global += W64ToByte(ifm_row_oneline_mrlen_a);

		_set_log_flag(1);
        KRNL_LOG_INFO(LOG_DEBUG, " === ifm [conv_a][%d][buf_addr : 0x%x] === ", i, banktbl_conv_a[0]);
        buf_print((banktbl_conv_a[0]), GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a)));
		_set_log_flag(0);

        // calc conv1s1KRNL_LOG_INFO(LOG_DEBUG, " === [Conv_c] Start === ");
        cluster_start_a = ByteToW64(banktbl_conv_a[0] - MMA_START_ADDR);
        cluster_end_a   = cluster_start_a + ifm_row_oneline_mrlen_a;
        cluster_num_a   = cshape_a->k_h - 1;

        _set_log_flag(1);
        one_row_conv(i, conv_a,
                     ByteToW64(wt_start_a - MMA_START_ADDR),
                     ByteToW64(banktbl_smpl[0] - MMA_START_ADDR),
                     ByteToW64(bs_start_a - MMA_START_ADDR),
                     ByteToW64(shift_start_a - MMA_START_ADDR),
                     ByteToW64(banktbl_conv_a[0] - MEM_LCMEM_ADDR_S),
                     ByteToW64(banktbl_conv_a[0] + MMA_BANK_SIZE - MEM_LCMEM_ADDR_S),
                     0,
                     cluster_start_a,
                     cluster_end_a,
                     cluster_num_a,
                     CONV_TYPE_CLASSIC, 1);
        _set_log_flag(0);

        // _set_log_flag(0);
        // KRNL_LOG_INFO(LOG_DEBUG, " === ifm [conv_a][1-0] === ");
        // buf_print(W64ToByte(_calc_cluster_start_addr(banktbl_conv_a[0], IFM_LINE_IN_BANK_CONV_A, 2)) + MMA_START_ADDR, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a)));
        // KRNL_LOG_INFO(LOG_DEBUG, " === ifm [conv_a][1-1] === ");
        // buf_print(W64ToByte(_calc_cluster_start_addr(banktbl_conv_a[0], IFM_LINE_IN_BANK_CONV_A, 3)) + MMA_START_ADDR, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a)));
        // KRNL_LOG_INFO(LOG_DEBUG, " === ifm [conv_a][1-2] === ");
        // buf_print(W64ToByte(_calc_cluster_start_addr(banktbl_conv_a[1], IFM_LINE_IN_BANK_CONV_A, 0)) + MMA_START_ADDR, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_a)));
        // _set_log_flag(0);

        // smp2x
        for (j = 0; j < w_iter_num_a; j++) {
            for (k = 0; k < ifm_c_group8_num_a; k++) {
                ifm_ptr = banktbl_smpl[0] + j * ifm_c_group8_num_a + k;
                asm volatile("mv t1, %0" ::"r"(ifm_ptr)
                             :);
                vlw(vr1, t1, 0);

                ofm_ptr = banktbl_add_b[1] + j * 2 * ifm_c_group8_num_a + k;
                asm volatile("mv t1, %0" ::"r"(ofm_ptr)
                             :);
                vsw(vr1, t1, 0);
                ofm_ptr += ifm_c_group8_num_a;
                asm volatile("mv t1, %0" ::"r"(ofm_ptr)
                             :);
                vsw(vr1, t1, 0);
            }
        }

        LIBHIKL_NASSERT(__rd_from_remote_chunk_blocking((uint32*)banktbl_add_b[0], ifm_addr_add_b->x_pos, ifm_addr_add_b->y_pos,
                                                        ifm_line_ptr_global_b, GMEM_ALIGN(W64ToByte(ifm_row_oneline_mrlen_b))));
        ifm_line_ptr_global_b += W64ToByte(ifm_row_oneline_mrlen_b);

        // vadd
        for (j = 0; j < w_iter_num_a; j++) {
            for (k = 0; k < ifm_c_group8_num_a; k++) {
                ifm_ptr = banktbl_add_b[0] + j * ifm_c_group8_num_a + k;
                asm volatile("mv t1, %0" ::"r"(ifm_ptr)
                             :);
                vlw(vr1, t1, 0);

                ifm_ptr = banktbl_add_b[1] + j * ifm_c_group8_num_a + k;
                asm volatile("mv t2, %0" ::"r"(ifm_ptr)
                             :);
                vlw(vr2, t2, 0);

                ofm_ptr = banktbl_ou[0] + j * 2 * ifm_c_group8_num_a + k;
                asm volatile("mv t1, %0" ::"r"(ofm_ptr)
                             :);
                vadd_vv(VPR_NONE, vr1, vr1, vr2);
                vsw(vr1, t1, 0);
            }
        }

        //__ndma_poll();
        ofm_line_ptr_global = ofm_addr_add_b->lcaddr + h_iter_num * UPSAMPLE_SCALE * ofm_row_size;
        LIBHIKL_NASSERT(__wr_to_remote_chunk_blocking((uint32*)banktbl_ou[0], ofm_addr_add_b->x_pos, ofm_addr_add_b->y_pos, ofm_line_ptr_global, ofm_row_size));
    }

    // KRNL_LOG_INFO(LOG_DEBUG, "ndma ends...\n\r");
    return;
}
