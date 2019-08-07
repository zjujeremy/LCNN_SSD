#ifndef PERMUTE_LAYER_H
#define PERMUTE_LAYER_H

#include "lcnn_param.h"

class PermuteLayer
{
public:
	PermuteLayer();
	~PermuteLayer();

	void layer_set_up(vector<Blob>& bottom_data_, vector<Blob>& top_data_);
	void forward_cpu(vector<Blob>& bottom_data_, vector<Blob>& top_data_);

private:
	int num_axes_; //4 
	bool need_permute_; //True

	Blob permute_order_; //[0,2,3,1]
	Blob old_steps_; // Blob以一维数组存放，计算原张量每一维数据在一维上的间隔距离 [CHW, HW, W, 1]
	Blob new_steps_; // [HWC, WC, C, 1]
};

#endif // !PERMUTE_LAYER_H

