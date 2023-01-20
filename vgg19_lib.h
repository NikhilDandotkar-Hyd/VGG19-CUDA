#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <string>

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}




struct pData{
    /* This structure contains the pointer 
     * to the output of the respective later
     */
    float* input;
    float* conv1_1;
    float* conv1_2;
    float* maxpool1_1;
    float* conv2_1;
    float* conv2_2;
    float* maxpool2_1;
    float* conv3_1;
    float* conv3_2;
    float* conv3_3;
    float* conv3_4;
    float* maxpool3_1;
    float* conv4_1;
    float* conv4_2;
    float* conv4_3;
    float* conv4_4;
    float* maxpool4_1;
    float* conv5_1;
    float* conv5_2;
    float* conv5_3;
    float* conv5_4;
    float* maxpool5_1;
    float* fc1;
    float* fc2;
    float* fc3;
    float* out_data;
	int* y;//actual output
	
	float* conv1_1_actv;
    float* conv1_2_actv;
    
    float* conv2_1_actv;
    float* conv2_2_actv;
    
    float* conv3_1_actv;
    float* conv3_2_actv;
    float* conv3_3_actv;
    float* conv3_4_actv;
    
    float* conv4_1_actv;
    float* conv4_2_actv;
    float* conv4_3_actv;
    float* conv4_4_actv;
    
    float* conv5_1_actv;
    float* conv5_2_actv;
    float* conv5_3_actv;
    float* conv5_4_actv;
    
    float* fc1_actv;
    float* fc2_actv;
    float* fc3_actv;
	
};

struct pW_n_B{
    /* This struct contains pointers to the 
     * weights and biases of the each layer
     */ 
    float* conv1_1_w;
    float* conv1_2_w;
    float* conv2_1_w;
    float* conv2_2_w;
    float* conv3_1_w;
    float* conv3_2_w;
    float* conv3_3_w;
    float* conv3_4_w;
    float* conv4_1_w;
    float* conv4_2_w;
    float* conv4_3_w;
    float* conv4_4_w;
    float* conv5_1_w;
    float* conv5_2_w;
    float* conv5_3_w;
    float* conv5_4_w;
    float* fc1_w;
    float* fc2_w;
    float* fc3_w;
    float* conv1_1_b;
    float* conv1_2_b;
    float* conv2_1_b;
    float* conv2_2_b;
    float* conv3_1_b;
    float* conv3_2_b;
    float* conv3_3_b;
    float* conv3_4_b;
    float* conv4_1_b;
    float* conv4_2_b;
    float* conv4_3_b;
    float* conv4_4_b;
    float* conv5_1_b;
    float* conv5_2_b;
    float* conv5_3_b;
    float* conv5_4_b;
    float* fc1_b;
    float* fc2_b;
    float* fc3_b;
};


struct layerDim{
  int n,c,h,w;  
};

struct layerInfo{
    cudnnTensorDescriptor_t ten_des_conv1_1;
    cudnnTensorDescriptor_t ten_des_conv1_2;
    cudnnTensorDescriptor_t ten_des_maxpool1_1;
    cudnnTensorDescriptor_t ten_des_conv2_1;
    cudnnTensorDescriptor_t ten_des_conv2_2;
    cudnnTensorDescriptor_t ten_des_maxpool2_1;
    cudnnTensorDescriptor_t ten_des_conv3_1;
    cudnnTensorDescriptor_t ten_des_conv3_2;
    cudnnTensorDescriptor_t ten_des_conv3_3;
    cudnnTensorDescriptor_t ten_des_conv3_4;
    cudnnTensorDescriptor_t ten_des_maxpool3_1;
    cudnnTensorDescriptor_t ten_des_conv4_1;
    cudnnTensorDescriptor_t ten_des_conv4_2;
    cudnnTensorDescriptor_t ten_des_conv4_3;
    cudnnTensorDescriptor_t ten_des_conv4_4;
    cudnnTensorDescriptor_t ten_des_maxpool4_1;
    cudnnTensorDescriptor_t ten_des_conv5_1;
    cudnnTensorDescriptor_t ten_des_conv5_2;
    cudnnTensorDescriptor_t ten_des_conv5_3;
    cudnnTensorDescriptor_t ten_des_conv5_4;
    cudnnTensorDescriptor_t ten_des_maxpool5_1;
    cudnnTensorDescriptor_t ten_des_FC_1;
    cudnnTensorDescriptor_t ten_des_FC_2;
    cudnnTensorDescriptor_t ten_des_FC_3;
    cudnnTensorDescriptor_t ten_des_output;
    
    cudnnFilterDescriptor_t fil_des_W1_1;
    cudnnFilterDescriptor_t fil_des_W1_2;
    cudnnFilterDescriptor_t fil_des_W2_1;
    cudnnFilterDescriptor_t fil_des_W2_2;
    cudnnFilterDescriptor_t fil_des_W3_1;
    cudnnFilterDescriptor_t fil_des_W3_2;
    cudnnFilterDescriptor_t fil_des_W3_3;
    cudnnFilterDescriptor_t fil_des_W3_4;
    cudnnFilterDescriptor_t fil_des_W4_1;
    cudnnFilterDescriptor_t fil_des_W4_2;
    cudnnFilterDescriptor_t fil_des_W4_3;
    cudnnFilterDescriptor_t fil_des_W4_4;
    cudnnFilterDescriptor_t fil_des_W5_1;
    cudnnFilterDescriptor_t fil_des_W5_2;
    cudnnFilterDescriptor_t fil_des_W5_3;
    cudnnFilterDescriptor_t fil_des_W5_4;
    cudnnFilterDescriptor_t fil_des_FC_1;
    cudnnFilterDescriptor_t fil_des_FC_2;
    cudnnFilterDescriptor_t fil_des_FC_3;
	
	cudnnTensorDescriptor_t ten_des_W1_1;
    cudnnTensorDescriptor_t ten_des_W1_2;
    cudnnTensorDescriptor_t ten_des_W2_1;
    cudnnTensorDescriptor_t ten_des_W2_2;
    cudnnTensorDescriptor_t ten_des_W3_1;
    cudnnTensorDescriptor_t ten_des_W3_2;
    cudnnTensorDescriptor_t ten_des_W3_3;
    cudnnTensorDescriptor_t ten_des_W3_4;
    cudnnTensorDescriptor_t ten_des_W4_1;
    cudnnTensorDescriptor_t ten_des_W4_2;
    cudnnTensorDescriptor_t ten_des_W4_3;
    cudnnTensorDescriptor_t ten_des_W4_4;
    cudnnTensorDescriptor_t ten_des_W5_1;
    cudnnTensorDescriptor_t ten_des_W5_2;
    cudnnTensorDescriptor_t ten_des_W5_3;
    cudnnTensorDescriptor_t ten_des_W5_4;
    cudnnTensorDescriptor_t ten_des_FC1_W;
    cudnnTensorDescriptor_t ten_des_FC2_W;
    cudnnTensorDescriptor_t ten_des_FC3_W;
	
    cudnnTensorDescriptor_t ten_des_B1_1;
    cudnnTensorDescriptor_t ten_des_B1_2;
    cudnnTensorDescriptor_t ten_des_B2_1;
    cudnnTensorDescriptor_t ten_des_B2_2;
    cudnnTensorDescriptor_t ten_des_B3_1;
    cudnnTensorDescriptor_t ten_des_B3_2;
    cudnnTensorDescriptor_t ten_des_B3_3;
    cudnnTensorDescriptor_t ten_des_B3_4;
    cudnnTensorDescriptor_t ten_des_B4_1;
    cudnnTensorDescriptor_t ten_des_B4_2;
    cudnnTensorDescriptor_t ten_des_B4_3;
    cudnnTensorDescriptor_t ten_des_B4_4;
    cudnnTensorDescriptor_t ten_des_B5_1;
    cudnnTensorDescriptor_t ten_des_B5_2;
    cudnnTensorDescriptor_t ten_des_B5_3;
    cudnnTensorDescriptor_t ten_des_B5_4;
    cudnnTensorDescriptor_t ten_des_B_FC_1;
    cudnnTensorDescriptor_t ten_des_B_FC_2;
    cudnnTensorDescriptor_t ten_des_B_FC_3;
    
    cudnnConvolutionDescriptor_t des_conv1_1;
    cudnnConvolutionDescriptor_t des_conv1_2;
    cudnnConvolutionDescriptor_t des_conv2_1;
    cudnnConvolutionDescriptor_t des_conv2_2;
    cudnnConvolutionDescriptor_t des_conv3_1;
    cudnnConvolutionDescriptor_t des_conv3_2;
    cudnnConvolutionDescriptor_t des_conv3_3;
    cudnnConvolutionDescriptor_t des_conv3_4;
    cudnnConvolutionDescriptor_t des_conv4_1;
    cudnnConvolutionDescriptor_t des_conv4_2;
    cudnnConvolutionDescriptor_t des_conv4_3;
    cudnnConvolutionDescriptor_t des_conv4_4;
    cudnnConvolutionDescriptor_t des_conv5_1;
    cudnnConvolutionDescriptor_t des_conv5_2;
    cudnnConvolutionDescriptor_t des_conv5_3;
    cudnnConvolutionDescriptor_t des_conv5_4;
	cudnnConvolutionDescriptor_t des_fc_1;
    cudnnConvolutionDescriptor_t des_fc_2;
    cudnnConvolutionDescriptor_t des_fc_3;
    
    cudnnPoolingDescriptor_t des_maxpool1_1;
    cudnnPoolingDescriptor_t des_maxpool2_1;
    cudnnPoolingDescriptor_t des_maxpool3_1;
    cudnnPoolingDescriptor_t des_maxpool4_1;
    cudnnPoolingDescriptor_t des_maxpool5_1;
    
    cudnnConvolutionFwdAlgo_t conv1_1algo;
    cudnnConvolutionFwdAlgo_t conv1_2algo;
    cudnnConvolutionFwdAlgo_t conv2_1algo;
    cudnnConvolutionFwdAlgo_t conv2_2algo;
    cudnnConvolutionFwdAlgo_t conv3_1algo;
    cudnnConvolutionFwdAlgo_t conv3_2algo;
    cudnnConvolutionFwdAlgo_t conv3_3algo;
    cudnnConvolutionFwdAlgo_t conv3_4algo;
    cudnnConvolutionFwdAlgo_t conv4_1algo;
    cudnnConvolutionFwdAlgo_t conv4_2algo;
    cudnnConvolutionFwdAlgo_t conv4_3algo;
    cudnnConvolutionFwdAlgo_t conv4_4algo;
    cudnnConvolutionFwdAlgo_t conv5_1algo;
    cudnnConvolutionFwdAlgo_t conv5_2algo;
    cudnnConvolutionFwdAlgo_t conv5_3algo;
    cudnnConvolutionFwdAlgo_t conv5_4algo;
	cudnnConvolutionFwdAlgo_t fc_1_algo;
    cudnnConvolutionFwdAlgo_t fc_2_algo;
    cudnnConvolutionFwdAlgo_t fc_3_algo;
    
    cudnnActivationDescriptor_t act_conv1_1;
    cudnnActivationDescriptor_t act_conv1_2;
    cudnnActivationDescriptor_t act_conv2_1;
    cudnnActivationDescriptor_t act_conv2_2;
    cudnnActivationDescriptor_t act_conv3_1;
    cudnnActivationDescriptor_t act_conv3_2;
    cudnnActivationDescriptor_t act_conv3_3;
    cudnnActivationDescriptor_t act_conv3_4;
    cudnnActivationDescriptor_t act_conv4_1;
    cudnnActivationDescriptor_t act_conv4_2;
    cudnnActivationDescriptor_t act_conv4_3;
    cudnnActivationDescriptor_t act_conv4_4;
    cudnnActivationDescriptor_t act_conv5_1;
    cudnnActivationDescriptor_t act_conv5_2;
    cudnnActivationDescriptor_t act_conv5_3;
    cudnnActivationDescriptor_t act_conv5_4;
    cudnnActivationDescriptor_t act_FC_1;
    cudnnActivationDescriptor_t act_FC_2;
    cudnnActivationDescriptor_t act_FC_3;
    
    float* conv1_1ws_data;
    float* conv1_2ws_data;
    float* conv2_1ws_data;
    float* conv2_2ws_data;
    float* conv3_1ws_data;
    float* conv3_2ws_data;
    float* conv3_3ws_data;
    float* conv3_4ws_data;
    float* conv4_1ws_data;
    float* conv4_2ws_data;
    float* conv4_3ws_data;
    float* conv4_4ws_data;
    float* conv5_1ws_data;
    float* conv5_2ws_data;
    float* conv5_3ws_data;
    float* conv5_4ws_data;
	float* fc1_ws_data;
	float* fc2_ws_data;
	float* fc3_ws_data;
	float* ws_data;
    
    size_t conv1_1ws_size;
    size_t conv1_2ws_size;
    size_t conv2_1ws_size;
    size_t conv2_2ws_size;
    size_t conv3_1ws_size;
    size_t conv3_2ws_size;
    size_t conv3_3ws_size;
    size_t conv3_4ws_size;
    size_t conv4_1ws_size;
    size_t conv4_2ws_size;
    size_t conv4_3ws_size;
    size_t conv4_4ws_size;
    size_t conv5_1ws_size;
    size_t conv5_2ws_size;
    size_t conv5_3ws_size;
    size_t conv5_4ws_size;
	size_t fc1_ws_size;
	size_t fc2_ws_size;
	size_t fc3_ws_size;
	size_t ws_size;
    
    layerDim dim_conv1_1;
    layerDim dim_conv1_2;
    layerDim dim_maxpool1_1;
    layerDim dim_conv2_1;
    layerDim dim_conv2_2;
    layerDim dim_maxpool2_1;
    layerDim dim_conv3_1;
    layerDim dim_conv3_2;
    layerDim dim_conv3_3;
    layerDim dim_conv3_4;
    layerDim dim_maxpool3_1;
    layerDim dim_conv4_1;
    layerDim dim_conv4_2;
    layerDim dim_conv4_3;
    layerDim dim_conv4_4;
    layerDim dim_maxpool4_1;
    layerDim dim_conv5_1;
    layerDim dim_conv5_2;
    layerDim dim_conv5_3;
    layerDim dim_conv5_4;
    layerDim dim_maxpool5_1;
	layerDim dim_fc1;
	layerDim dim_fc2;
	layerDim dim_fc3;
};

struct pgradData{
	
	/* This struct contains the pointers to 
	 * the gradients of weights and biases
	 */
	float* ones_batch_size;
	float* grad_loss;
	float* grad_softmax;
	float* grad_conv1_1_w;
    float* grad_conv1_2_w;
    float* grad_conv2_1_w;
    float* grad_conv2_2_w;
    float* grad_conv3_1_w;
    float* grad_conv3_2_w;
    float* grad_conv3_3_w;
    float* grad_conv3_4_w;
    float* grad_conv4_1_w;
    float* grad_conv4_2_w;
    float* grad_conv4_3_w;
    float* grad_conv4_4_w;
    float* grad_conv5_1_w;
    float* grad_conv5_2_w;
    float* grad_conv5_3_w;
    float* grad_conv5_4_w;
    float* grad_fc1_w;
    float* grad_fc2_w;
    float* grad_fc3_w;
    float* grad_conv1_1_b;
    float* grad_conv1_2_b;
    float* grad_conv2_1_b;
    float* grad_conv2_2_b;
    float* grad_conv3_1_b;
    float* grad_conv3_2_b;
    float* grad_conv3_3_b;
    float* grad_conv3_4_b;
    float* grad_conv4_1_b;
    float* grad_conv4_2_b;
    float* grad_conv4_3_b;
    float* grad_conv4_4_b;
    float* grad_conv5_1_b;
    float* grad_conv5_2_b;
    float* grad_conv5_3_b;
    float* grad_conv5_4_b;
    float* grad_fc1_b;
    float* grad_fc2_b;
    float* grad_fc3_b;
	
	float* grad_data;
	float* grad_fc3_data;
	float* grad_fc2_data;
	float* grad_fc1_data;
	float* grad_maxpool5_1_data;
	float* grad_conv5_4_data;
	float* grad_conv5_3_data;
	float* grad_conv5_2_data;
	float* grad_conv5_1_data;
	float* grad_maxpool4_1_data;
	float* grad_conv4_4_data;
	float* grad_conv4_3_data;
	float* grad_conv4_2_data;
	float* grad_conv4_1_data;
	float* grad_maxpool3_1_data;
	float* grad_conv3_4_data;
	float* grad_conv3_3_data;
	float* grad_conv3_2_data;
	float* grad_conv3_1_data;
	float* grad_maxpool2_1_data;
	float* grad_conv2_2_data;
	float* grad_conv2_1_data;
	float* grad_maxpool1_1_data;
	float* grad_conv1_2_data;
	float* grad_conv1_1_data;
	
	float* grad_fc3_actv;
	float* grad_fc2_actv;
	float* grad_fc1_actv;
	float* grad_maxpool5_1_actv;
	float* grad_conv5_4_actv;
	float* grad_conv5_3_actv;
	float* grad_conv5_2_actv;
	float* grad_conv5_1_actv;
	float* grad_maxpool4_1_actv;
	float* grad_conv4_4_actv;
	float* grad_conv4_3_actv;
	float* grad_conv4_2_actv;
	float* grad_conv4_1_actv;
	float* grad_maxpool3_1_actv;
	float* grad_conv3_4_actv;
	float* grad_conv3_3_actv;
	float* grad_conv3_2_actv;
	float* grad_conv3_1_actv;
	float* grad_maxpool2_1_actv;
	float* grad_conv2_2_actv;
	float* grad_conv2_1_actv;
	float* grad_maxpool1_1_actv;
	float* grad_conv1_2_actv;
	float* grad_conv1_1_actv;
	
	
	float* vel_conv1_1_w;
    float* vel_conv1_2_w;
    float* vel_conv2_1_w;
    float* vel_conv2_2_w;
    float* vel_conv3_1_w;
    float* vel_conv3_2_w;
    float* vel_conv3_3_w;
    float* vel_conv3_4_w;
    float* vel_conv4_1_w;
    float* vel_conv4_2_w;
    float* vel_conv4_3_w;
    float* vel_conv4_4_w;
    float* vel_conv5_1_w;
    float* vel_conv5_2_w;
    float* vel_conv5_3_w;
    float* vel_conv5_4_w;
    float* vel_fc1_w;
    float* vel_fc2_w;
    float* vel_fc3_w;
    float* vel_conv1_1_b;
    float* vel_conv1_2_b;
    float* vel_conv2_1_b;
    float* vel_conv2_2_b;
    float* vel_conv3_1_b;
    float* vel_conv3_2_b;
    float* vel_conv3_3_b;
    float* vel_conv3_4_b;
    float* vel_conv4_1_b;
    float* vel_conv4_2_b;
    float* vel_conv4_3_b;
    float* vel_conv4_4_b;
    float* vel_conv5_1_b;
    float* vel_conv5_2_b;
    float* vel_conv5_3_b;
    float* vel_conv5_4_b;
    float* vel_fc1_b;
    float* vel_fc2_b;
    float* vel_fc3_b;
};

struct backLayerInfo{
	
	
	float* loss_i;
	cudnnTensorDescriptor_t des_grad_fc3;
	cudnnFilterDescriptor_t fil_ones;
	cudnnConvolutionDescriptor_t des_ones_conv;
	layerDim grad_dim_fc3;
	cudnnConvolutionFwdAlgo_t grad_fc_3_algo;
	
	cudnnTensorDescriptor_t des_diffSoftmax;
	cudnnTensorDescriptor_t grad_ten_des_conv1_1;
    cudnnTensorDescriptor_t grad_ten_des_conv1_2;
    cudnnTensorDescriptor_t grad_ten_des_maxpool1_1;
    cudnnTensorDescriptor_t grad_ten_des_conv2_1;
    cudnnTensorDescriptor_t grad_ten_des_conv2_2;
    cudnnTensorDescriptor_t grad_ten_des_maxpool2_1;
    cudnnTensorDescriptor_t grad_ten_des_conv3_1;
    cudnnTensorDescriptor_t grad_ten_des_conv3_2;
    cudnnTensorDescriptor_t grad_ten_des_conv3_3;
    cudnnTensorDescriptor_t grad_ten_des_conv3_4;
    cudnnTensorDescriptor_t grad_ten_des_maxpool3_1;
    cudnnTensorDescriptor_t grad_ten_des_conv4_1;
    cudnnTensorDescriptor_t grad_ten_des_conv4_2;
    cudnnTensorDescriptor_t grad_ten_des_conv4_3;
    cudnnTensorDescriptor_t grad_ten_des_conv4_4;
    cudnnTensorDescriptor_t grad_ten_des_maxpool4_1;
    cudnnTensorDescriptor_t grad_ten_des_conv5_1;
    cudnnTensorDescriptor_t grad_ten_des_conv5_2;
    cudnnTensorDescriptor_t grad_ten_des_conv5_3;
    cudnnTensorDescriptor_t grad_ten_des_conv5_4;
    cudnnTensorDescriptor_t grad_ten_des_maxpool5_1;
    cudnnTensorDescriptor_t grad_ten_des_FC_1;
    cudnnTensorDescriptor_t grad_ten_des_FC_2;
    cudnnTensorDescriptor_t grad_ten_des_FC_3;
    cudnnTensorDescriptor_t grad_ten_des_output;
	
	cudnnConvolutionBwdFilterAlgo_t conv1_1bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv1_2bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv2_1bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv2_2bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv3_1bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv3_2bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv3_3bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv3_4bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv4_1bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv4_2bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv4_3bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv4_4bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv5_1bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv5_2bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv5_3bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t conv5_4bw_fil_algo;
	cudnnConvolutionBwdFilterAlgo_t fc_1_bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t fc_2_bw_fil_algo;
    cudnnConvolutionBwdFilterAlgo_t fc_3_bw_fil_algo;
	
	cudnnConvolutionBwdDataAlgo_t conv1_1bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv1_2bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv2_1bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv2_2bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv3_1bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv3_2bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv3_3bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv3_4bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv4_1bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv4_2bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv4_3bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv4_4bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv5_1bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv5_2bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv5_3bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t conv5_4bw_dt_algo;
	cudnnConvolutionBwdDataAlgo_t fc_1_bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t fc_2_bw_dt_algo;
    cudnnConvolutionBwdDataAlgo_t fc_3_bw_dt_algo;
	
	cudnnTensorDescriptor_t ten_des_VW1_1;
    cudnnTensorDescriptor_t ten_des_VW1_2;
    cudnnTensorDescriptor_t ten_des_VW2_1;
    cudnnTensorDescriptor_t ten_des_VW2_2;
    cudnnTensorDescriptor_t ten_des_VW3_1;
    cudnnTensorDescriptor_t ten_des_VW3_2;
    cudnnTensorDescriptor_t ten_des_VW3_3;
    cudnnTensorDescriptor_t ten_des_VW3_4;
    cudnnTensorDescriptor_t ten_des_VW4_1;
    cudnnTensorDescriptor_t ten_des_VW4_2;
    cudnnTensorDescriptor_t ten_des_VW4_3;
    cudnnTensorDescriptor_t ten_des_VW4_4;
    cudnnTensorDescriptor_t ten_des_VW5_1;
    cudnnTensorDescriptor_t ten_des_VW5_2;
    cudnnTensorDescriptor_t ten_des_VW5_3;
    cudnnTensorDescriptor_t ten_des_VW5_4;
    cudnnTensorDescriptor_t ten_des_VFC1_W;
    cudnnTensorDescriptor_t ten_des_VFC2_W;
    cudnnTensorDescriptor_t ten_des_VFC3_W;
	
    cudnnTensorDescriptor_t ten_des_VB1_1;
    cudnnTensorDescriptor_t ten_des_VB1_2;
    cudnnTensorDescriptor_t ten_des_VB2_1;
    cudnnTensorDescriptor_t ten_des_VB2_2;
    cudnnTensorDescriptor_t ten_des_VB3_1;
    cudnnTensorDescriptor_t ten_des_VB3_2;
    cudnnTensorDescriptor_t ten_des_VB3_3;
    cudnnTensorDescriptor_t ten_des_VB3_4;
    cudnnTensorDescriptor_t ten_des_VB4_1;
    cudnnTensorDescriptor_t ten_des_VB4_2;
    cudnnTensorDescriptor_t ten_des_VB4_3;
    cudnnTensorDescriptor_t ten_des_VB4_4;
    cudnnTensorDescriptor_t ten_des_VB5_1;
    cudnnTensorDescriptor_t ten_des_VB5_2;
    cudnnTensorDescriptor_t ten_des_VB5_3;
    cudnnTensorDescriptor_t ten_des_VB5_4;
    cudnnTensorDescriptor_t ten_des_VB_FC_1;
    cudnnTensorDescriptor_t ten_des_VB_FC_2;
    cudnnTensorDescriptor_t ten_des_VB_FC_3;
	
	size_t conv1_1_bw_ws_size;
    size_t conv1_2_bw_ws_size;
    size_t conv2_1_bw_ws_size;
    size_t conv2_2_bw_ws_size;
    size_t conv3_1_bw_ws_size;
    size_t conv3_2_bw_ws_size;
    size_t conv3_3_bw_ws_size;
    size_t conv3_4_bw_ws_size;
    size_t conv4_1_bw_ws_size;
    size_t conv4_2_bw_ws_size;
    size_t conv4_3_bw_ws_size;
    size_t conv4_4_bw_ws_size;
    size_t conv5_1_bw_ws_size;
    size_t conv5_2_bw_ws_size;
    size_t conv5_3_bw_ws_size;
    size_t conv5_4_bw_ws_size;
	size_t fc1_bw_ws_size;
	size_t fc2_bw_ws_size;
	size_t fc3_bw_ws_size;
	
	float* conv1_1_bw_ws_data;
    float* conv1_2_bw_ws_data;
    float* conv2_1_bw_ws_data;
    float* conv2_2_bw_ws_data;
    float* conv3_1_bw_ws_data;
    float* conv3_2_bw_ws_data;
    float* conv3_3_bw_ws_data;
    float* conv3_4_bw_ws_data;
    float* conv4_1_bw_ws_data;
    float* conv4_2_bw_ws_data;
    float* conv4_3_bw_ws_data;
    float* conv4_4_bw_ws_data;
    float* conv5_1_bw_ws_data;
    float* conv5_2_bw_ws_data;
    float* conv5_3_bw_ws_data;
    float* conv5_4_bw_ws_data;
	float* fc1_bw_ws_data;
	float* fc2_bw_ws_data;
	float* fc3_bw_ws_data;
};

struct inputinfo{
    layerDim in_dim;
    cudnnTensorDescriptor_t in_desc;
};


void singleconvWeightDescriptor(cudnnFilterDescriptor_t& weight_desc ,int weight_k,int weight_c,int weight_h,int weight_w);

void singleconvBiasDescriptor(cudnnTensorDescriptor_t& bias_desc,int weight_c);

void singlefcBiasDescriptor(cudnnTensorDescriptor_t& bias_desc);

void setWeightnBiasDescriptor(layerInfo* p_layer);

void singleconvDescriptor(cudnnConvolutionDescriptor_t& conv_desc);

void convDescriptor(layerInfo* p_layer);

void singlemaxpoolDescriptor(cudnnPoolingDescriptor_t& poolDesc);

void maxpoolDescriptor(layerInfo* stpoolDesc);

void inputDimDesc(inputinfo* input);

void getsingleLayerDimensions(cudnnConvolutionDescriptor_t& conv_desc,cudnnFilterDescriptor_t& weight_desc,
                        cudnnTensorDescriptor_t& in_desc,cudnnTensorDescriptor_t& out_desc,layerDim& out_dim,pData* data);

void getsingleMaxpoolDimensions(cudnnPoolingDescriptor_t& pool_desc,cudnnTensorDescriptor_t& in_desc,
                                cudnnTensorDescriptor_t& out_desc,layerDim& out_dim,pData* data);

void getAllLayerDimensions(layerInfo* p_layer,inputinfo* input,pData* output);

void setsingleAlgorithmWorkspace(cudnnHandle_t& cudnn,cudnnTensorDescriptor_t& in_desc,cudnnFilterDescriptor_t& filt_desc,
                                 cudnnConvolutionDescriptor_t& conv_desc,cudnnTensorDescriptor_t& out_desc,cudnnConvolutionFwdAlgo_t* algo,float* ws_data);

void setallAlgorithmWorkspace(layerInfo* layerDimDesc,inputinfo* input,cudnnHandle_t& cudnn);

void setAllLayers(layerInfo* p_layer,inputinfo* input,pData* output, cudnnHandle_t& cudnn);

void computeConvol(cudnnHandle_t& cudnn,cudnnTensorDescriptor_t& in_desc,float* input_data,cudnnFilterDescriptor_t& filt_desc,float* filt_data,
                   cudnnConvolutionDescriptor_t& conv_desc,cudnnConvolutionFwdAlgo_t& algo,cudnnTensorDescriptor_t& out_desc,cudnnTensorDescriptor_t& bias_desc,
                   cudnnActivationDescriptor_t& activate,float* ws_data,size_t& ws_size,float* out_data,float* bias_data);
				   
void computeFCActivation(cudnnHandle_t& cudnn,cudnnTensorDescriptor_t& out_desc,float* out_data,cudnnActivationDescriptor_t& activate);
                           
void computeForwardpass(cudnnHandle_t& cudnn, layerInfo* p_layer, inputinfo* input, pData* p_layerdata, pW_n_B* p_weightdata);

void backLayerDesc(cudnnHandle_t& cudnn,inputinfo* input,layerInfo* p_layer, 
                   backLayerInfo* bak_layer, pgradData* p_graddata,int num_class, int batch_size,std::string optimizer);

void cal_gradient(cudnnHandle_t& cudnn, layerInfo* p_layer, 
                  pData* p_layerdata, pW_n_B* p_weightdata,inputinfo* input,
				  backLayerInfo* bak_layer,pgradData* p_graddata, int num_class, 
				  int batch_size);
void update_weight(cudnnHandle_t& cudnn,layerInfo* p_layer,pW_n_B* WeightData,pgradData* gradData,float rho,float learning_rate,std::string optimizer);

void printoutputdimensions(layerInfo* d_layer,inputinfo* d_input);


void destroy(layerInfo* layer_info,pData* output);

