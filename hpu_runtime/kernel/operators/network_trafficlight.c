#include "hihw.h"
#include "libconv.h"
#include "dma.h"
#include "lock.h"
#include "int.h"
#include "hi_addr_def.h"
#include "krnl_log.h"
#include "operators/hi_krnl_param_net_trafficlight.h"
#include "operators/hi_krnl_param_conv1s1_dwc3s2_conv1s1.h"
#include "operators/hi_krnl_param_conv1s1_dwc3s1_conv1s1_add.h"
#include "operators/hi_krnl_param_conv1s1_dwc3s2_conv1s1.h"
#include "operators/hi_krnl_param_conv1s1_upsmp2x_add.h"
#include "operators/hi_krnl_param_conv1s1_conv3s1_conv3s1.h"

extern void _op_conv1s1_dwc3s1_conv1s1_add
(
	conv2d_params_t *conv1,
	conv2d_params_t *conv2,
	conv2d_params_t *conv3,
    
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
    hikl_addr_t *shift_addr_c,

	uint32 add_shift,
	uint32 add_clip
);

extern void _op_conv1s1_conv3s1_conv3s1 (   
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

extern void _op_conv1s1_dwc3s2_conv1s1(
	conv_shape_t *cshape_a,
	conv_shape_t *cshape_b,
	conv_shape_t *cshape_c,

	bool relu_a,
	bool relu_b,
	bool relu_c,

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
	hikl_addr_t *shift_addr_c);

extern void _op_conv1s1_upsmp2x_add(
    conv2d_params_t* conv_a,
    add_params_t*    add_b,

    hikl_addr_t* ifm_addr_conv_a,
    hikl_addr_t* ofm_addr_conv_a,

    hikl_addr_t* ifm_addr_add_b,
    hikl_addr_t* ofm_addr_add_b,

    hikl_addr_t* wt_addr_conv_a,
    hikl_addr_t* bias_addr_conv_a,
    hikl_addr_t* shift_addr_conv_a);


extern void _op_conv3s2_dwc3s1_conv1s1(
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

void kernel_network_trafficlight()
{
  	KRNL_LOG_INFO(LOG_SYSTEM, "enter into kernel_network_trafficlight");
	paramTable_net_trafficlight_t *_pParamTable = *((paramTable_net_trafficlight_t **)HIPU200_KNL_PTABLE_ADDR);/*get kernel param table from runtime*/

    paramTableConv3s2_dwc3s1_conv1s1_Entry_t        *param_blk1  = &_pParamTable->param.param_blk1;
    paramTableConv1s1_dwc3s2_conv1s1_Entry_t        *param_blk2  = &_pParamTable->param.param_blk2;
    paramTableConv1s1_dwc3s1_conv1s1_add_Entry_t    *param_blk3  = &_pParamTable->param.param_blk3;
    paramTableConv1s1_dwc3s2_conv1s1_Entry_t        *param_blk4  = &_pParamTable->param.param_blk4;
    paramTableConv1s1_dwc3s1_conv1s1_add_Entry_t    *param_blk5  = &_pParamTable->param.param_blk5;
    paramTableConv1s1_dwc3s2_conv1s1_Entry_t        *param_blk6  = &_pParamTable->param.param_blk6;
    paramTableConv1s1_dwc3s1_conv1s1_add_Entry_t    *param_blk7  = &_pParamTable->param.param_blk7;
    paramTableConv1s1_upsmp2x_add_Entry_t           *param_blk8  = &_pParamTable->param.param_blk8;
    paramTableConv1s1_upsmp2x_add_Entry_t           *param_blk9  = &_pParamTable->param.param_blk9;
    paramTableConv1s1_conv3s1_conv3s1_Entry_t       *param_blk10 = &_pParamTable->param.param_blk10;
    paramTableConv1s1_conv3s1_conv3s1_Entry_t       *param_blk11 = &_pParamTable->param.param_blk11;
    paramTableConv1s1_conv3s1_conv3s1_Entry_t       *param_blk12 = &_pParamTable->param.param_blk12;

	/*blk1*/
	_op_conv3s2_dwc3s1_conv1s1(
		&param_blk1->conv1,
		&param_blk1->conv2,
		&param_blk1->conv3,

		&param_blk1->ifm_addr_conv1,
		&param_blk1->ofm_addr_conv3,

		&param_blk1->wt_addr_conv1,
		&param_blk1->wt_addr_conv2,
		&param_blk1->wt_addr_conv3,

		&param_blk1->bias_addr_conv1,
		&param_blk1->bias_addr_conv2,
		&param_blk1->bias_addr_conv3,

		&param_blk1->shift_addr_conv1,
		&param_blk1->shift_addr_conv2,
		&param_blk1->shift_addr_conv3);

	/*blk2*/
	_op_conv1s1_dwc3s2_conv1s1(
		&param_blk2->conv1.cshape,
		&param_blk2->conv2.cshape,
		&param_blk2->conv3.cshape,

		param_blk2->conv1.relu,
		param_blk2->conv2.relu,
		param_blk2->conv3.relu,

		&param_blk2->ifm_addr_conv1,
		&param_blk2->ofm_addr_conv3,

		&param_blk2->wt_addr_conv1,
		&param_blk2->wt_addr_conv2,
		&param_blk2->wt_addr_conv3,

		&param_blk2->bias_addr_conv1,
		&param_blk2->bias_addr_conv2,
		&param_blk2->bias_addr_conv3,

		&param_blk2->shift_addr_conv1,
		&param_blk2->shift_addr_conv2,
		&param_blk2->shift_addr_conv3);

	/*blk3*/
	_op_conv1s1_dwc3s1_conv1s1_add(
		&param_blk3->conv1,
		&param_blk3->conv2,
		&param_blk3->conv3,

		&param_blk3->ifm_addr_conv1,
		&param_blk3->ofm_addr_conv3,

		&param_blk3->wt_addr_conv1,
		&param_blk3->wt_addr_conv2,
		&param_blk3->wt_addr_conv3,

		&param_blk3->bias_addr_conv1,
		&param_blk3->bias_addr_conv2,
		&param_blk3->bias_addr_conv3,

		&param_blk3->shift_addr_conv1,
		&param_blk3->shift_addr_conv2,
		&param_blk3->shift_addr_conv3,
		param_blk3->add1.shift,
		param_blk3->add1.clip
		);

	/*blk4*/
	_op_conv1s1_dwc3s2_conv1s1(
		&param_blk4->conv1.cshape,
		&param_blk4->conv2.cshape,
		&param_blk4->conv3.cshape,

		param_blk4->conv1.relu,
		param_blk4->conv2.relu,
		param_blk4->conv3.relu,

		&param_blk4->ifm_addr_conv1,
		&param_blk4->ofm_addr_conv3,

		&param_blk4->wt_addr_conv1,
		&param_blk4->wt_addr_conv2,
		&param_blk4->wt_addr_conv3,

		&param_blk4->bias_addr_conv1,
		&param_blk4->bias_addr_conv2,
		&param_blk4->bias_addr_conv3,

		&param_blk4->shift_addr_conv1,
		&param_blk4->shift_addr_conv2,
		&param_blk4->shift_addr_conv3
		);

	/*blk5*/
	_op_conv1s1_dwc3s1_conv1s1_add(
		&param_blk5->conv1,
		&param_blk5->conv2,
		&param_blk5->conv3,

		&param_blk5->ifm_addr_conv1,
		&param_blk5->ofm_addr_conv3,

		&param_blk5->wt_addr_conv1,
		&param_blk5->wt_addr_conv2,
		&param_blk5->wt_addr_conv3,

		&param_blk5->bias_addr_conv1,
		&param_blk5->bias_addr_conv2,
		&param_blk5->bias_addr_conv3,

		&param_blk5->shift_addr_conv1,
		&param_blk5->shift_addr_conv2,
		&param_blk5->shift_addr_conv3,
		param_blk5->add1.shift,
		param_blk5->add1.clip
		);

	/*blk6*/
	_op_conv1s1_dwc3s2_conv1s1(
		&param_blk6->conv1.cshape,
		&param_blk6->conv2.cshape,
		&param_blk6->conv3.cshape,

		param_blk6->conv1.relu,
		param_blk6->conv2.relu,
		param_blk6->conv3.relu,

		&param_blk6->ifm_addr_conv1,
		&param_blk6->ofm_addr_conv3,

		&param_blk6->wt_addr_conv1,
		&param_blk6->wt_addr_conv2,
		&param_blk6->wt_addr_conv3,

		&param_blk6->bias_addr_conv1,
		&param_blk6->bias_addr_conv2,
		&param_blk6->bias_addr_conv3,

		&param_blk6->shift_addr_conv1,
		&param_blk6->shift_addr_conv2,
		&param_blk6->shift_addr_conv3);

	/*blk7*/
	_op_conv1s1_dwc3s1_conv1s1_add(
		&param_blk7->conv1,
		&param_blk7->conv2,
		&param_blk7->conv3,

		&param_blk7->ifm_addr_conv1,
		&param_blk7->ofm_addr_conv3,

		&param_blk7->wt_addr_conv1,
		&param_blk7->wt_addr_conv2,
		&param_blk7->wt_addr_conv3,

		&param_blk7->bias_addr_conv1,
		&param_blk7->bias_addr_conv2,
		&param_blk7->bias_addr_conv3,

		&param_blk7->shift_addr_conv1,
		&param_blk7->shift_addr_conv2,
		&param_blk7->shift_addr_conv3,
		param_blk7->add1.shift,
		param_blk7->add1.clip
		);

	/*blk8*/
	_op_conv1s1_upsmp2x_add(
		&param_blk8->conv1,
		&param_blk8->add1,

		&param_blk8->ifm_addr_conv1,
		&param_blk8->ofm_addr_conv1,

		&param_blk8->ifm_addr_add1,
		&param_blk8->ofm_addr_add1,

		&param_blk8->wt_addr_conv1,
		&param_blk8->bias_addr_conv1,
		&param_blk8->shift_addr_conv1
		);
	/*blk9*/

	_op_conv1s1_upsmp2x_add(
		&param_blk9->conv1,
		&param_blk9->add1,

		&param_blk9->ifm_addr_conv1,
		&param_blk9->ofm_addr_conv1,

		&param_blk9->ifm_addr_add1,
		&param_blk9->ofm_addr_add1,

		&param_blk9->wt_addr_conv1,
		&param_blk9->bias_addr_conv1,
		&param_blk9->shift_addr_conv1
		);

	/*blk10*/
	_op_conv1s1_conv3s1_conv3s1(
		&param_blk10->conv1,
		&param_blk10->conv1a_c1,
		&param_blk10->conv1b_r1,
		
		&param_blk10->ifm_addr_conv1,
		&param_blk10->ofm_addr_conv1a_c1,
		&param_blk10->ofm_addr_conv1b_r1,

		&param_blk10->wt_addr_conv1,
		&param_blk10->wt_addr_conv1a_c1,
		&param_blk10->wt_addr_conv1b_r1,

		&param_blk10->bias_addr_conv1,
		&param_blk10->bias_addr_conv1a_c1,
		&param_blk10->bias_addr_conv1b_r1,

		&param_blk10->shift_addr_conv1,
		&param_blk10->shift_addr_conv1a_c1,
		&param_blk10->shift_addr_conv1b_r1);

	/*blk11*/
	_op_conv1s1_conv3s1_conv3s1(
		&param_blk11->conv1,
		&param_blk11->conv1a_c1,
		&param_blk11->conv1b_r1,

		&param_blk11->ifm_addr_conv1,
		&param_blk11->ofm_addr_conv1a_c1,
		&param_blk11->ofm_addr_conv1b_r1,

		&param_blk11->wt_addr_conv1,
		&param_blk11->wt_addr_conv1a_c1,
		&param_blk11->wt_addr_conv1b_r1,

		&param_blk11->bias_addr_conv1,
		&param_blk11->bias_addr_conv1a_c1,
		&param_blk11->bias_addr_conv1b_r1,

		&param_blk11->shift_addr_conv1,
		&param_blk11->shift_addr_conv1a_c1,
		&param_blk11->shift_addr_conv1b_r1);

	/*blk12*/
	_op_conv1s1_conv3s1_conv3s1(
		&param_blk12->conv1,
		&param_blk12->conv1a_c1,
		&param_blk12->conv1b_r1,

		&param_blk12->ifm_addr_conv1,
		&param_blk12->ofm_addr_conv1a_c1,
		&param_blk12->ofm_addr_conv1b_r1,

		&param_blk12->wt_addr_conv1,
		&param_blk12->wt_addr_conv1a_c1,
		&param_blk12->wt_addr_conv1b_r1,

		&param_blk12->bias_addr_conv1,
		&param_blk12->bias_addr_conv1a_c1,
		&param_blk12->bias_addr_conv1b_r1,

		&param_blk12->shift_addr_conv1,
		&param_blk12->shift_addr_conv1a_c1,
		&param_blk12->shift_addr_conv1b_r1);
}

