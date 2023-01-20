#include "vgg19_lib.h"
#include <math.h>
#include <iostream>
#include <random>     // mt19937 and uniform_int_distribution
#include <algorithm>  // generate
#include <functional> // bind

using namespace std;



void singleconvWeightDescriptor(cudnnFilterDescriptor_t& weight_desc ,cudnnTensorDescriptor_t& ten_desc, 
                                int weight_k, int weight_c, int weight_h, int weight_w){
    
    /* INPUT:
     *    weight_k = number of filters
     *    weight_c = number of channnels
     *    weight_h = height of each kernel
     *    weight_w = width of each kernel
     * OUTPUT:
     *    weight_desc = cudnnFilterDescriptor_t with parameters
     * Description:
     *    This function fills the parameters of the single kernel in cudnnFilterDescriptor_t variable
     */ 
    
    CUDNN_CALL(cudnnCreateFilterDescriptor(&weight_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(weight_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, weight_k,
                                          weight_c, weight_h, weight_w));
	
	CUDNN_CALL(cudnnCreateTensorDescriptor(&ten_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(ten_desc, CUDNN_TENSOR_NCHW, 
	                                      CUDNN_DATA_FLOAT,weight_k,
                                          weight_c, weight_h, weight_w));

    
    
}

void singleconvBiasDescriptor(cudnnTensorDescriptor_t& bias_desc, int weight_c){
	
    /* INPUT:
     *    weight_c = number of channnels
     * OUTPUT:
     *    bias_desc = cudnnTensorDescriptor_t with parameters
     * Description:
     *    This function fills the cudnnTensorDescriptor_t which is used to add the bias after convolution
     */ 
    
	CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, 
	                                      CUDNN_DATA_FLOAT,1,
                                          weight_c, 1, 1));
   
}





void setWeightnBiasDescriptor(layerInfo* p_layer ){

    /* INPUT:
     *    p_layer = struct of all the filter descriptor, convolution descriptors, pool descriptors and output descriptors and activation descriptors
     * Description:
     *    This function fills the parameters of all the filter descriptors of   VGG19
     */
     
    //conv1_1
    int k=64, c=3, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W1_1, p_layer->ten_des_W1_1, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B1_1, k);
    
    //conv1_2
    k=64, c=64, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W1_2, p_layer->ten_des_W1_2, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B1_2, k);
    
    //conv2_1
    k=128, c=64, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W2_1, p_layer->ten_des_W2_1, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B2_1, k);
    
    
    //conv2_2
    k=128, c=128, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W2_2, p_layer->ten_des_W2_2, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B2_2, k);
     
    
    //conv3_1
    k=256, c=128, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W3_1, p_layer->ten_des_W3_1, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B3_1, k);
    
    
    //conv3_2
    k=256, c=256, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W3_2, p_layer->ten_des_W3_2, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B3_2, k);
    
    
    //conv3_3
    k=256, c=256, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W3_3, p_layer->ten_des_W3_3, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B3_3, k);
     
    
    //conv3_4
    k=256, c=256, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W3_4, p_layer->ten_des_W3_4, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B3_4, k);
    
    
    
    //conv4_1
    k=512, c=256, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W4_1, p_layer->ten_des_W4_1, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B4_1, k);
    
    
    //conv4_2
    k=512, c=512, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W4_2, p_layer->ten_des_W4_2, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B4_2, k);
   
    
    //conv4_3
    k=512, c=512, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W4_3, p_layer->ten_des_W4_3, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B4_3, k);
    
    
    //conv4_4
    k=512, c=512, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W4_4, p_layer->ten_des_W4_4, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B4_4, k);
   
    
    //conv5_1
    k=512, c=512, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W5_1, p_layer->ten_des_W5_1, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B5_1, k);
   
    
    //conv5_2
    k=512, c=512, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W5_2, p_layer->ten_des_W5_2, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B5_2, k);
    
    
    //conv5_3
    k=512, c=512, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W5_3, p_layer->ten_des_W5_3, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B5_3, k);
    
    
    //conv5_4
    k=512, c=512, w=3, h=3;
    singleconvWeightDescriptor(p_layer->fil_des_W5_4, p_layer->ten_des_W5_4, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B5_4, k);
    
    
	/*NOTE: Dense layer operation is performed using convolution API*/
	
	//Fully Connected Layer 1
	c=512;
	k=4096, w=7, h=7;
	singleconvWeightDescriptor(p_layer->fil_des_FC_1, p_layer->ten_des_FC1_W, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B_FC_1, k);
   
	
	//Fully Connected Layer 2
	k=4096, c=4096, w=1, h=1;
	singleconvWeightDescriptor(p_layer->fil_des_FC_2, p_layer->ten_des_FC2_W, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B_FC_2, k);
   
	
	//Fully Connected Layer 3
	k=1000, c=4096, w=1, h=1;
	singleconvWeightDescriptor(p_layer->fil_des_FC_3, p_layer->ten_des_FC3_W, k, c, h, w);
    singleconvBiasDescriptor(p_layer->ten_des_B_FC_3, k);
   
}

void singleconvDescriptor(cudnnConvolutionDescriptor_t& conv_desc,string s){
    
    /* INPUT:
     *    conv_desc = need to will the parameters of cudnnConvolutionDescriptor_t
     * Description:
     *    This function fills the parameters cudnnConvolutionDescriptor_t variable with padding = 1
     *    stride = 1, dillation = 1 and in VGG19 every convolution has the same operation.
     */ 
     
    int pad_h ;
    int pad_w ;
    int str_h = 1;
    int str_w = 1;
    int dil_h = 1;
    int dil_w = 1;
	if(s.compare("conv") == 0){
		pad_h = 1;
        pad_w = 1;
	}
	else{
		pad_h = 0;
        pad_w = 0;
	}
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, 
	                                           pad_w, str_h, 
											   str_w, dil_h, 
											   dil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
}

void convDescriptor(layerInfo* stconv_desc){
    
    /* INPUT:
     *    stconv_desc = struct of all the filter descriptor, convolution descriptors, pool descriptors and output descriptors and activation descriptors
     * Description:
     *    This function fills the parameters of all the convolution descriptors of VGG19
     */
    
    singleconvDescriptor(stconv_desc->des_conv1_1,"conv");
    singleconvDescriptor(stconv_desc->des_conv1_2,"conv");
    singleconvDescriptor(stconv_desc->des_conv2_1,"conv");
    singleconvDescriptor(stconv_desc->des_conv2_2,"conv");
    singleconvDescriptor(stconv_desc->des_conv3_1,"conv");
    singleconvDescriptor(stconv_desc->des_conv3_2,"conv");
    singleconvDescriptor(stconv_desc->des_conv3_3,"conv");
    singleconvDescriptor(stconv_desc->des_conv3_4,"conv");
    singleconvDescriptor(stconv_desc->des_conv4_1,"conv");
    singleconvDescriptor(stconv_desc->des_conv4_2,"conv");
    singleconvDescriptor(stconv_desc->des_conv4_3,"conv");
    singleconvDescriptor(stconv_desc->des_conv4_4,"conv");
    singleconvDescriptor(stconv_desc->des_conv5_1,"conv");
    singleconvDescriptor(stconv_desc->des_conv5_2,"conv");
    singleconvDescriptor(stconv_desc->des_conv5_3,"conv");
    singleconvDescriptor(stconv_desc->des_conv5_4,"conv");
	singleconvDescriptor(stconv_desc->des_fc_1,"fc");
    singleconvDescriptor(stconv_desc->des_fc_2,"fc");
    singleconvDescriptor(stconv_desc->des_fc_3,"fc");
	
}

void singlemaxpoolDescriptor(cudnnPoolingDescriptor_t& poolDesc){
    
    /* INPUT:
     *    poolDesc = need to will the parameters of cudnnPoolingDescriptor_t
     * Description:
     *    This function fills the parameters of cudnnPoolingDescriptor_t variable with padding = 0
     *    stride = 2x2, window_size = 2x2 and in VGG19 every maxpool has the same operation.
     */
     
    int win_h=2, win_w=2, ver_padd=0, hor_padd=0, ver_str=2, hor_str=2;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolDesc));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, 
	                                       CUDNN_NOT_PROPAGATE_NAN, win_h, 
										   win_w, ver_padd, 
										   hor_padd, ver_str, hor_str));
}

void maxpoolDescriptor(layerInfo* stpoolDesc){
    
    /* INPUT:
     *    stpool_desc = struct of all the maxpool descriptor
     * Description:
     *    This function fills the parameters of all the maxpool descriptors of VGG19
     */
     
     singlemaxpoolDescriptor(stpoolDesc->des_maxpool1_1);
     singlemaxpoolDescriptor(stpoolDesc->des_maxpool2_1);
     singlemaxpoolDescriptor(stpoolDesc->des_maxpool3_1);
     singlemaxpoolDescriptor(stpoolDesc->des_maxpool4_1);
     singlemaxpoolDescriptor(stpoolDesc->des_maxpool5_1);
}

void getsingleLayerDimensions(cudnnConvolutionDescriptor_t& conv_desc, cudnnFilterDescriptor_t& weight_desc, 
                              cudnnTensorDescriptor_t& in_desc, cudnnTensorDescriptor_t& out_desc, 
							  layerDim& out_dim){
								  
    /* INPUT:
     *    conv_desc = Convolution descriptor
	 *    weight_desc = Filter descriptor
	 *    in_desc = Input descriptor
	 * OUTPUT:
	 *    out_dim =  struct that stores the dimensions of the output layer
	 *    out_desc = Output descriptor filled with parameters
     * Description:
     *    1) Get the dimensions of the output of convolution operation for given convolution descriptor,filter descriptor and input descriptor.
	 *    2) Fills the output descriptor
     */
   
								  
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, 
	                                                 weight_desc, &(out_dim.n), 
													 &(out_dim.c), &(out_dim.h), 
													 &(out_dim.w)));
	
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
	
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, 
	                                      CUDNN_DATA_FLOAT, out_dim.n, 
										  out_dim.c, out_dim.h, out_dim.w));
	
}

void getsingleMaxpoolDimensions(cudnnPoolingDescriptor_t& pool_desc, cudnnTensorDescriptor_t& in_desc, 
                                cudnnTensorDescriptor_t& out_desc, layerDim& out_dim, float* output_data){
									
    /* INPUT:
     *    pool_desc = Maxpool descriptor
	 *    weight_desc = Filter descriptor
	 *    in_desc = Input descriptor
	 * OUTPUT:
	 *    out_dim =  struct that stores the dimensions of the output layer
	 *    out_desc = Output descriptor filled with parameters
     * Description:
     *    1) Get the dimensions of the output of maxpool operation for given convolution descriptor,filter descriptor and input descriptor.
	 *    2) Fills the output descriptor
     */
	 
    CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(pool_desc, in_desc, &(out_dim.n), &(out_dim.c), &(out_dim.h), &(out_dim.w)));
	CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
	CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
	                                      out_dim.n, out_dim.c, out_dim.h, out_dim.w));
    CUDA_CALL(cudaMalloc(&output_data, out_dim.n * out_dim.c * out_dim.h * out_dim.w * sizeof(float)));

}

void inputDimDesc(inputinfo* input){
	
    /* INPUT:
     *    input = struct of input descriptor and input dimention
     * Description:
     *    Fills the input descriptor with input dimensions, input data type and input data format 
     */
	 
    CUDNN_CALL(cudnnCreateTensorDescriptor(&(input->in_desc)));
    
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input->in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
                                      input->in_dim.n, input->in_dim.c, input->in_dim.h, input->in_dim.w));
}

void getAllLayerDimensions(layerInfo* p_layer, inputinfo* input, pData* output){
	
    /* INPUT:
	 *    p_layer = struct of all the filter descriptor, convolution descriptors, pool descriptors and output descriptors and activation descriptors 
     *    input = struct of input descriptor and input dimention
	 *    output = struct of pointers of all output layers
     * Description:
     *     1) Get the dimensions of the each layer and allocates the memory for each layer
	 *     2) Fills the output descriptor of each layer
     */
	
    //block -1
	//conv1_1
    getsingleLayerDimensions(p_layer->des_conv1_1, p_layer->fil_des_W1_1, 
	                         input->in_desc, p_layer->ten_des_conv1_1, 
							 p_layer->dim_conv1_1);
	
	CUDA_CALL(cudaMalloc(&(output->conv1_1), 
						 p_layer->dim_conv1_1.n * p_layer->dim_conv1_1.c * p_layer->dim_conv1_1.h * p_layer->dim_conv1_1.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv1_1_actv), 
						 p_layer->dim_conv1_1.n * p_layer->dim_conv1_1.c * p_layer->dim_conv1_1.h * p_layer->dim_conv1_1.w * sizeof(float)));
	
    
	//conv1_2
	getsingleLayerDimensions(p_layer->des_conv1_2, p_layer->fil_des_W1_2, 
	                         p_layer->ten_des_conv1_1, p_layer->ten_des_conv1_2, 
							 p_layer->dim_conv1_2);
	
	CUDA_CALL(cudaMalloc(&(output->conv1_2), 
						 p_layer->dim_conv1_2.n * p_layer->dim_conv1_2.c * p_layer->dim_conv1_2.h * p_layer->dim_conv1_2.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv1_2_actv), 
						 p_layer->dim_conv1_2.n * p_layer->dim_conv1_2.c * p_layer->dim_conv1_2.h * p_layer->dim_conv1_2.w * sizeof(float)));
	
    
	//maxpool1_1
    getsingleMaxpoolDimensions(p_layer->des_maxpool1_1, p_layer->ten_des_conv1_2, 
	                           p_layer->ten_des_maxpool1_1, p_layer->dim_maxpool1_1, 
							   output->maxpool1_1);
    CUDA_CALL(cudaMalloc(&(output->maxpool1_1), 
						 p_layer->dim_maxpool1_1.n * p_layer->dim_maxpool1_1.c * p_layer->dim_maxpool1_1.h * p_layer->dim_maxpool1_1.w * sizeof(float)));
	
    //block-2
	//conv2_1
    getsingleLayerDimensions(p_layer->des_conv2_1, p_layer->fil_des_W2_1, 
	                         p_layer->ten_des_maxpool1_1, p_layer->ten_des_conv2_1, 
							 p_layer->dim_conv2_1);
	CUDA_CALL(cudaMalloc(&(output->conv2_1), 
						 p_layer->dim_conv2_1.n * p_layer->dim_conv2_1.c * p_layer->dim_conv2_1.h * p_layer->dim_conv2_1.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv2_1_actv), 
						 p_layer->dim_conv2_1.n * p_layer->dim_conv2_1.c * p_layer->dim_conv2_1.h * p_layer->dim_conv2_1.w * sizeof(float)));
    
	//conv2_2
	getsingleLayerDimensions(p_layer->des_conv2_2, p_layer->fil_des_W2_2, 
	                         p_layer->ten_des_conv2_1, p_layer->ten_des_conv2_2, 
							 p_layer->dim_conv2_2);
	CUDA_CALL(cudaMalloc(&(output->conv2_2), 
						 p_layer->dim_conv2_2.n * p_layer->dim_conv2_2.c * p_layer->dim_conv2_2.h * p_layer->dim_conv2_2.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv2_2_actv), 
						 p_layer->dim_conv2_2.n * p_layer->dim_conv2_2.c * p_layer->dim_conv2_2.h * p_layer->dim_conv2_2.w * sizeof(float)));
    
	//maxpool2_1
    getsingleMaxpoolDimensions(p_layer->des_maxpool2_1, p_layer->ten_des_conv2_2, 
	                           p_layer->ten_des_maxpool2_1, p_layer->dim_maxpool2_1, 
							   output->maxpool2_1);
	CUDA_CALL(cudaMalloc(&(output->maxpool2_1), 
						 p_layer->dim_maxpool2_1.n * p_layer->dim_maxpool2_1.c * p_layer->dim_maxpool2_1.h * p_layer->dim_maxpool2_1.w * sizeof(float)));
    
	
    //block-3
	//conv3_1
    getsingleLayerDimensions(p_layer->des_conv3_1, p_layer->fil_des_W3_1, 
	                         p_layer->ten_des_maxpool2_1, p_layer->ten_des_conv3_1, 
							 p_layer->dim_conv3_1);
    CUDA_CALL(cudaMalloc(&(output->conv3_1), 
						 p_layer->dim_conv3_1.n * p_layer->dim_conv3_1.c * p_layer->dim_conv3_1.h * p_layer->dim_conv3_1.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv3_1_actv), 
						 p_layer->dim_conv3_1.n * p_layer->dim_conv3_1.c * p_layer->dim_conv3_1.h * p_layer->dim_conv3_1.w * sizeof(float)));
	
	//conv3_2
    getsingleLayerDimensions(p_layer->des_conv3_2, p_layer->fil_des_W3_2, 
	                         p_layer->ten_des_conv3_1, p_layer->ten_des_conv3_2, 
							 p_layer->dim_conv3_2);
	CUDA_CALL(cudaMalloc(&(output->conv3_2), 
						 p_layer->dim_conv3_2.n * p_layer->dim_conv3_2.c * p_layer->dim_conv3_2.h * p_layer->dim_conv3_2.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv3_2_actv), 
						 p_layer->dim_conv3_2.n * p_layer->dim_conv3_2.c * p_layer->dim_conv3_2.h * p_layer->dim_conv3_2.w * sizeof(float)));
    
	//conv3_3
    getsingleLayerDimensions(p_layer->des_conv3_3, p_layer->fil_des_W3_3, 
	                         p_layer->ten_des_conv3_2, p_layer->ten_des_conv3_3, 
							 p_layer->dim_conv3_3);
	CUDA_CALL(cudaMalloc(&(output->conv3_3), 
						 p_layer->dim_conv3_3.n * p_layer->dim_conv3_3.c * p_layer->dim_conv3_3.h * p_layer->dim_conv3_3.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv3_3_actv), 
						 p_layer->dim_conv3_3.n * p_layer->dim_conv3_3.c * p_layer->dim_conv3_3.h * p_layer->dim_conv3_3.w * sizeof(float)));
	
	//conv3_4
    getsingleLayerDimensions(p_layer->des_conv3_4, p_layer->fil_des_W3_4, 
	                         p_layer->ten_des_conv3_3, p_layer->ten_des_conv3_4, 
							 p_layer->dim_conv3_4);
	CUDA_CALL(cudaMalloc(&(output->conv3_4), 
						 p_layer->dim_conv3_4.n * p_layer->dim_conv3_4.c * p_layer->dim_conv3_4.h * p_layer->dim_conv3_4.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv3_4_actv), 
						 p_layer->dim_conv3_4.n * p_layer->dim_conv3_4.c * p_layer->dim_conv3_4.h * p_layer->dim_conv3_4.w * sizeof(float)));
    
	//maxpool3_1
    getsingleMaxpoolDimensions(p_layer->des_maxpool3_1, p_layer->ten_des_conv3_4, 
	                           p_layer->ten_des_maxpool3_1, p_layer->dim_maxpool3_1, 
							   output->maxpool3_1);
	CUDA_CALL(cudaMalloc(&(output->maxpool3_1), 
						 p_layer->dim_maxpool3_1.n * p_layer->dim_maxpool3_1.c * p_layer->dim_maxpool3_1.h * p_layer->dim_maxpool3_1.w * sizeof(float)));
    
	
    //block-4
	//conv4_1
    getsingleLayerDimensions(p_layer->des_conv4_1, p_layer->fil_des_W4_1, 
	                         p_layer->ten_des_maxpool3_1, p_layer->ten_des_conv4_1, 
							 p_layer->dim_conv4_1);
    CUDA_CALL(cudaMalloc(&(output->conv4_1), 
						 p_layer->dim_conv4_1.n * p_layer->dim_conv4_1.c * p_layer->dim_conv4_1.h * p_layer->dim_conv4_1.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv4_1_actv), 
						 p_layer->dim_conv4_1.n * p_layer->dim_conv4_1.c * p_layer->dim_conv4_1.h * p_layer->dim_conv4_1.w * sizeof(float)));
	
	//conv4_2
    getsingleLayerDimensions(p_layer->des_conv4_2, p_layer->fil_des_W4_2, 
	                         p_layer->ten_des_conv4_1, p_layer->ten_des_conv4_2, 
							 p_layer->dim_conv4_2);
	CUDA_CALL(cudaMalloc(&(output->conv4_2), 
						 p_layer->dim_conv4_2.n * p_layer->dim_conv4_2.c * p_layer->dim_conv4_2.h * p_layer->dim_conv4_2.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv4_2_actv), 
						 p_layer->dim_conv4_2.n * p_layer->dim_conv4_2.c * p_layer->dim_conv4_2.h * p_layer->dim_conv4_2.w * sizeof(float)));
	
	//conv4_3
    getsingleLayerDimensions(p_layer->des_conv4_3, p_layer->fil_des_W4_3, 
	                         p_layer->ten_des_conv4_2, p_layer->ten_des_conv4_3, 
							 p_layer->dim_conv4_3);
	CUDA_CALL(cudaMalloc(&(output->conv4_3), 
						 p_layer->dim_conv4_3.n * p_layer->dim_conv4_3.c * p_layer->dim_conv4_3.h * p_layer->dim_conv4_3.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv4_3_actv), 
						 p_layer->dim_conv4_3.n * p_layer->dim_conv4_3.c * p_layer->dim_conv4_3.h * p_layer->dim_conv4_3.w * sizeof(float)));
	
	//conv4_4
    getsingleLayerDimensions(p_layer->des_conv4_4, p_layer->fil_des_W4_4, 
	                         p_layer->ten_des_conv4_3, p_layer->ten_des_conv4_4, 
							 p_layer->dim_conv4_4);
	CUDA_CALL(cudaMalloc(&(output->conv4_4), 
						 p_layer->dim_conv4_4.n * p_layer->dim_conv4_4.c * p_layer->dim_conv4_4.h * p_layer->dim_conv4_4.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv4_4_actv), 
						 p_layer->dim_conv4_4.n * p_layer->dim_conv4_4.c * p_layer->dim_conv4_4.h * p_layer->dim_conv4_4.w * sizeof(float)));

    
	//maxpool4_1
    getsingleMaxpoolDimensions(p_layer->des_maxpool4_1, p_layer->ten_des_conv4_4, 
							   p_layer->ten_des_maxpool4_1, p_layer->dim_maxpool4_1, 
							   output->maxpool4_1);
	CUDA_CALL(cudaMalloc(&(output->maxpool4_1), 
						 p_layer->dim_maxpool4_1.n * p_layer->dim_maxpool4_1.c * p_layer->dim_maxpool4_1.h * p_layer->dim_maxpool4_1.w * sizeof(float)));
    

    //block-5
	//conv5_1
    getsingleLayerDimensions(p_layer->des_conv5_1, p_layer->fil_des_W5_1, 
	                         p_layer->ten_des_maxpool4_1, p_layer->ten_des_conv5_1, 
							 p_layer->dim_conv5_1);
    CUDA_CALL(cudaMalloc(&(output->conv5_1), 
						 p_layer->dim_conv5_1.n * p_layer->dim_conv5_1.c * p_layer->dim_conv5_1.h * p_layer->dim_conv5_1.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv5_1_actv), 
						 p_layer->dim_conv5_1.n * p_layer->dim_conv5_1.c * p_layer->dim_conv5_1.h * p_layer->dim_conv5_1.w * sizeof(float)));
	
	//conv5_2
    getsingleLayerDimensions(p_layer->des_conv5_2, p_layer->fil_des_W5_2, 
							 p_layer->ten_des_conv5_1, p_layer->ten_des_conv5_2, 
							 p_layer->dim_conv5_2);
	CUDA_CALL(cudaMalloc(&(output->conv5_2), 
						 p_layer->dim_conv5_2.n * p_layer->dim_conv5_2.c * p_layer->dim_conv5_2.h * p_layer->dim_conv5_2.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv5_2_actv), 
						 p_layer->dim_conv5_2.n * p_layer->dim_conv5_2.c * p_layer->dim_conv5_2.h * p_layer->dim_conv5_2.w * sizeof(float)));
	
	//conv5_3
    getsingleLayerDimensions(p_layer->des_conv5_3, p_layer->fil_des_W5_3, 
	                         p_layer->ten_des_conv5_2, p_layer->ten_des_conv5_3, 
							 p_layer->dim_conv5_3);
	CUDA_CALL(cudaMalloc(&(output->conv5_3), 
						 p_layer->dim_conv5_3.n * p_layer->dim_conv5_3.c * p_layer->dim_conv5_3.h * p_layer->dim_conv5_3.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv5_3_actv), 
						 p_layer->dim_conv5_3.n * p_layer->dim_conv5_3.c * p_layer->dim_conv5_3.h * p_layer->dim_conv5_3.w * sizeof(float)));
    
	//conv5_4
	
	getsingleLayerDimensions(p_layer->des_conv5_4, p_layer->fil_des_W5_4, 
	                         p_layer->ten_des_conv5_3, p_layer->ten_des_conv5_4, 
							 p_layer->dim_conv5_4);
	CUDA_CALL(cudaMalloc(&(output->conv5_4), 
						 p_layer->dim_conv5_4.n * p_layer->dim_conv5_4.c * p_layer->dim_conv5_4.h * p_layer->dim_conv5_4.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->conv5_4_actv), 
						 p_layer->dim_conv5_4.n * p_layer->dim_conv5_4.c * p_layer->dim_conv5_4.h * p_layer->dim_conv5_4.w * sizeof(float)));
						 
    //maxpool5_1
    getsingleMaxpoolDimensions(p_layer->des_maxpool5_1, p_layer->ten_des_conv5_4, 
							   p_layer->ten_des_maxpool5_1, p_layer->dim_maxpool5_1, 
							   output->maxpool5_1);
	CUDA_CALL(cudaMalloc(&(output->maxpool5_1), 
						 p_layer->dim_maxpool5_1.n * p_layer->dim_maxpool5_1.c * p_layer->dim_maxpool5_1.h * p_layer->dim_maxpool5_1.w * sizeof(float)));
	
	
	//FC-1
    
	getsingleLayerDimensions(p_layer->des_fc_1, p_layer->fil_des_FC_1, 
	                         p_layer->ten_des_maxpool5_1, p_layer->ten_des_FC_1, 
							 p_layer->dim_fc1);
    CUDA_CALL(cudaMalloc(&(output->fc1), 
						 p_layer->dim_fc1.n * p_layer->dim_fc1.c * p_layer->dim_fc1.h * p_layer->dim_fc1.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->fc1_actv), 
						 p_layer->dim_fc1.n * p_layer->dim_fc1.c * p_layer->dim_fc1.h * p_layer->dim_fc1.w * sizeof(float)));
	
	
	
	//FC-2
    getsingleLayerDimensions(p_layer->des_fc_2, p_layer->fil_des_FC_2, 
	                         p_layer->ten_des_FC_1, p_layer->ten_des_FC_2, 
							 p_layer->dim_fc2);
    CUDA_CALL(cudaMalloc(&(output->fc2), 
						 p_layer->dim_fc2.n * p_layer->dim_fc2.c * p_layer->dim_fc2.h * p_layer->dim_fc2.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->fc2_actv), 
						 p_layer->dim_fc2.n * p_layer->dim_fc2.c * p_layer->dim_fc2.h * p_layer->dim_fc2.w * sizeof(float)));
	
							 
	//FC-3
	getsingleLayerDimensions(p_layer->des_fc_3, p_layer->fil_des_FC_3, 
	                         p_layer->ten_des_FC_2, p_layer->ten_des_FC_3, 
							 p_layer->dim_fc3);
	CUDA_CALL(cudaMalloc(&(output->fc3), 
						 p_layer->dim_fc3.n * p_layer->dim_fc3.c * p_layer->dim_fc3.h * p_layer->dim_fc3.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(output->fc3_actv), 
						 p_layer->dim_fc3.n * p_layer->dim_fc3.c * p_layer->dim_fc3.h * p_layer->dim_fc3.w * sizeof(float)));
	
}

void setsingleAlgorithmWorkspace(cudnnHandle_t& cudnn, cudnnTensorDescriptor_t& in_desc, 
                                 cudnnFilterDescriptor_t& filt_desc, cudnnConvolutionDescriptor_t& conv_desc, 
								 cudnnTensorDescriptor_t& out_desc, cudnnConvolutionFwdAlgo_t* algo, 
								  size_t& ws_size){
	
    /* INPUT:
	 *    cudnn = cudnn handler
     *    in_desc = input descriptor
	 *    filt_desc = filter descriptor
	 *    conv_desc = convolution descriptor
	 *    out_desc = output descriptor
	 * OUTPUT:
	 *    algo = algorithm to perform the convolution
	 *    ws_size =  workspace size required to perform the convolution operation for given algorithm
     * Description:
     *     1) selects the algorithm which best for the given input descriptor,filter descriptor,convolution descriptor and output descriptor
	 *     2) gets the workspace size required to perform the convolution operation for given algorithm
     */
	
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc, 
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algo));
   
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, *algo, &ws_size));
   
}

void setallAlgorithmWorkspace(layerInfo* p_layer, inputinfo* input, cudnnHandle_t& cudnn){
	
    /* INPUT:
	 *    p_layer = struct of all the filter descriptor, convolution descriptors, pool descriptors and output descriptors and activation descriptors 
     *    input = struct of input descriptor and input dimention
	 *    cudnn = cudnn handler
     * Description:
     *     1) Gets the algorithm for the each layer.
	 *     2) allocates the workspace in which all the convolution layer operations can be performed.
     */
	    
    size_t  op_size1,op_size2;
	
		//block -1
    setsingleAlgorithmWorkspace(cudnn, input->in_desc, 
	                            p_layer->fil_des_W1_1, p_layer->des_conv1_1, 
								p_layer->ten_des_conv1_1, &(p_layer->conv1_1algo), 
								 op_size1);
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_conv1_1, 
								p_layer->fil_des_W1_2, p_layer->des_conv1_2, 
								p_layer->ten_des_conv1_2, &(p_layer->conv1_2algo), 
								 op_size2);
    op_size1 = (op_size1 > op_size2) ? op_size1 : op_size2;
	//block -2
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_maxpool1_1, 
								p_layer->fil_des_W2_1, p_layer->des_conv2_1, 
								p_layer->ten_des_conv2_1, &(p_layer->conv2_1algo), 
								 op_size2);
	op_size1 = (op_size1 > op_size2) ? op_size1 : op_size2;
								
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_conv2_1, 
	                            p_layer->fil_des_W2_2, p_layer->des_conv2_2, 
								p_layer->ten_des_conv2_2, &(p_layer->conv2_2algo), 
								 op_size2);
	op_size1 = (op_size1 > op_size2) ? op_size1 : op_size2;
    
	//block -3
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_maxpool2_1, 
	                            p_layer->fil_des_W3_1, p_layer->des_conv3_1, 
								p_layer->ten_des_conv3_1, &(p_layer->conv3_1algo), 
								 op_size2);
	op_size1 = (op_size1 > op_size2) ? op_size1 : op_size2;
    
	
	setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_conv3_1, 
	                            p_layer->fil_des_W3_2, p_layer->des_conv3_2, 
								p_layer->ten_des_conv3_2, &(p_layer->conv3_2algo), 
								 op_size2);
    op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
    
	
	setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_conv3_2, 
	                            p_layer->fil_des_W3_3, p_layer->des_conv3_3, 
								p_layer->ten_des_conv3_3, &(p_layer->conv3_3algo), 
								 op_size2);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	 
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_conv3_3, 
	                            p_layer->fil_des_W3_4, p_layer->des_conv3_4, 
								p_layer->ten_des_conv3_4, &(p_layer->conv3_4algo), 
								 op_size2);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
    
	//block-4
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_maxpool3_1, 
								p_layer->fil_des_W4_1, p_layer->des_conv4_1, 
								p_layer->ten_des_conv4_1, &(p_layer->conv4_1algo), 
								 op_size2);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	 
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_conv4_1, 
	                            p_layer->fil_des_W4_2, p_layer->des_conv4_2, 
								p_layer->ten_des_conv4_2, &(p_layer->conv4_2algo), 
								 op_size2);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	 
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_conv4_2, 
	                            p_layer->fil_des_W4_3, p_layer->des_conv4_3, 
								p_layer->ten_des_conv4_3, &(p_layer->conv4_3algo), 
								 op_size2);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	 
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_conv4_3, 
								p_layer->fil_des_W4_4, p_layer->des_conv4_4, 
								p_layer->ten_des_conv4_4, &(p_layer->conv4_4algo), 
								 op_size2);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	 
    
	//block-5
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_maxpool4_1, 
	                            p_layer->fil_des_W5_1, p_layer->des_conv5_1, 
								p_layer->ten_des_conv5_1, &(p_layer->conv5_1algo), 
								 op_size2);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	 
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_conv5_1, 
	                            p_layer->fil_des_W5_2, p_layer->des_conv5_2, 
								p_layer->ten_des_conv5_2, &(p_layer->conv5_2algo), 
								 op_size2);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	 
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_conv5_2, 
	                            p_layer->fil_des_W5_3, p_layer->des_conv5_3, 
								p_layer->ten_des_conv5_3, &(p_layer->conv5_3algo), 
								 op_size2);
								 
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_conv5_3, 
	                            p_layer->fil_des_W5_4, p_layer->des_conv5_4, 
								p_layer->ten_des_conv5_4, &(p_layer->conv5_4algo), 
								 op_size2);
								 
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	//Fully connected layers
	setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_maxpool5_1, 
	                            p_layer->fil_des_FC_1, p_layer->des_fc_1, 
								p_layer->ten_des_FC_1, &(p_layer->fc_1_algo), 
					            op_size2);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	 
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_FC_1, 
	                            p_layer->fil_des_FC_2, p_layer->des_fc_2, 
								p_layer->ten_des_FC_2, &(p_layer->fc_2_algo), 
					            op_size2);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	 
    setsingleAlgorithmWorkspace(cudnn, p_layer->ten_des_FC_2, 
	                            p_layer->fil_des_FC_3, p_layer->des_fc_3, 
								p_layer->ten_des_FC_3, &(p_layer->fc_3_algo), 
					            op_size2);
	op_size1 = op_size1 > op_size2? op_size1 : op_size2;
    p_layer->ws_size = op_size1;
	CUDA_CALL(cudaMalloc(&(p_layer->ws_data), p_layer->ws_size));
    }

void setActivation(cudnnActivationDescriptor_t& activate){
	
	/* INPUT:
	 *   activate = Activation descriptor
	 * Description:
	 *   This function sets the activation descriptor with ReLu activation
	 */ 
	//activation
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activate));
    CUDNN_CALL(cudnnSetActivationDescriptor(activate, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 255));
}

void setAllActivations(layerInfo* p_layer){
	
	/* INPUT:
	 *    p_layer = struct of all the filter descriptor, convolution descriptors, pool descriptors and output descriptors and activation descriptors 
     * Description:
     *     This function will set activations for all convolution layer,FC-1 and FC-2 with ReLu 
     */
	 setActivation(p_layer->act_conv1_1);
	 setActivation(p_layer->act_conv1_2);
	 setActivation(p_layer->act_conv2_1);
	 setActivation(p_layer->act_conv2_2);
	 setActivation(p_layer->act_conv3_1);
	 setActivation(p_layer->act_conv3_2);
	 setActivation(p_layer->act_conv3_3);
	 setActivation(p_layer->act_conv3_4);
	 setActivation(p_layer->act_conv4_1);
	 setActivation(p_layer->act_conv4_2);
	 setActivation(p_layer->act_conv4_3);
	 setActivation(p_layer->act_conv4_4);
	 setActivation(p_layer->act_conv5_1);
	 setActivation(p_layer->act_conv5_2);
	 setActivation(p_layer->act_conv5_3);
	 setActivation(p_layer->act_conv5_4);
	 setActivation(p_layer->act_FC_1);
	 setActivation(p_layer->act_FC_2);
}

void setAllLayers(layerInfo* p_layer, inputinfo* input, pData* output, cudnnHandle_t& cudnn){
	
	/* INPUT:
	 *    p_layer = struct of all the filter descriptor, convolution descriptors, pool descriptors and output descriptors and activation descriptors 
     *    input = struct of input descriptor and input dimention
	 *    output = struct of pointers of all output layers
	 *    cudnn = cudnn handler
     * Description:
     *     This function will perform the initial setup to perform the forward pass of the VGG19
     */
    
    inputDimDesc(input);
    cout<<"input descriptor done"<<endl;
    
    setWeightnBiasDescriptor(p_layer);
    cout<<"setweightbais descriptor done"<<endl;
    
    convDescriptor(p_layer);
    cout<<"conv descriptor done"<<endl;
    
    maxpoolDescriptor(p_layer);
    cout<<"maxpool descriptor done"<<endl;
    
    
    getAllLayerDimensions(p_layer, input, output);
    cout<<"dimensions of all layers done"<<endl;
    
    setallAlgorithmWorkspace(p_layer, input, cudnn);
    cout<<"all algo done"<<endl;
	
	setAllActivations(p_layer);
	cout<<"set all activations"<<endl;

}


void computeConvol(cudnnHandle_t& cudnn, cudnnTensorDescriptor_t& in_desc, 
                   float* input_data, cudnnFilterDescriptor_t& filt_desc, 
				   float* filt_data, cudnnConvolutionDescriptor_t& conv_desc, 
				   cudnnConvolutionFwdAlgo_t& algo, cudnnTensorDescriptor_t& out_desc, 
				   cudnnTensorDescriptor_t& bias_desc, cudnnActivationDescriptor_t& activate, 
				   float* ws_data, size_t& ws_size, 
				   float* out_conv, float* bias_data,
				   float* out_actv)
                   {
					   
	/* INPUT:
	 *    cudnn = cudnn handler
     *    in_desc = input descriptor
	 *    input_data =  pointer to the input image
	 *    filt_desc = filter descriptor
	 *    filt_data = pointer to the filter data
	 *    conv_desc = convolution descriptor
	 *    algo =  algorithm that should be used to perform the convolution operation
	 *    out_desc = output descriptor
	 *    ws_data =  pointer to the workspace
	 *    ws_size =  size of the workspace
	 *    bias_data = pointer to the bias data
	 * OUTPUT:
	 *    out_conv = pointer to the output of convolution operation[convolution + addition of bias]
	 *    out_actv = pointer to the output of activation
     * Description:
     *     This function performs the convolution of input data with filter and then adds with bias followed by ReLu activation
     */
	 
    float alpha = 1.f;
    float beta = 0.f;
    
	//connvolution
    CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, in_desc, input_data, filt_desc, filt_data, conv_desc, algo, ws_data, ws_size, &beta, out_desc, out_conv));
    //addiction of bias
    CUDNN_CALL(cudnnAddTensor(cudnn, &alpha, bias_desc, bias_data, &alpha, out_desc, out_conv));
    //activation
    CUDNN_CALL(cudnnActivationForward(cudnn, activate, &alpha, out_desc, out_conv, &beta, out_desc, out_actv));
    
}



void computeForwardpass(cudnnHandle_t& cudnn, layerInfo* p_layer, inputinfo* input, pData* p_layerdata, pW_n_B* p_weightdata)
{
    /* INPUT:
	 *    cudnn = cudnn handler
	 *    p_layer = struct of all the filter descriptor, convolution descriptors, pool descriptors and output descriptors and activation descriptors 
     *    input = struct of input descriptor and input dimention
	 *    p_layerdata = struct of pointers of all output layers
	 *    p_weightdata = struct of pointers of filter data and bias data
     * Description:
     *     This function will perform theforward pass of VGG19
     */
	 
    float alpha = 1.f;
    float beta = 0.f;
    
	//block -1
    //conv1_1
  
    cout<<"conv1_1"<<endl;
	computeConvol(cudnn, input->in_desc, p_layerdata->input,
                  p_layer->fil_des_W1_1, p_weightdata->conv1_1_w, 
				  p_layer->des_conv1_1, p_layer->conv1_1algo, 
				  p_layer->ten_des_conv1_1, p_layer->ten_des_B1_1, 
				  p_layer->act_conv1_1, p_layer->ws_data, 
				  p_layer->ws_size, p_layerdata->conv1_1, 
				  p_weightdata->conv1_1_b, p_layerdata->conv1_1_actv);
                  
	//conv1_2
	cout<<"conv1_2"<<endl;
    computeConvol(cudnn, p_layer->ten_des_conv1_1,
				  p_layerdata->conv1_1, p_layer->fil_des_W1_2,
                  p_weightdata->conv1_2_w, p_layer->des_conv1_2, 
				  p_layer->conv1_2algo, p_layer->ten_des_conv1_2, 
                  p_layer->ten_des_B1_2, p_layer->act_conv1_2, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv1_2, p_weightdata->conv1_2_b, 
				  p_layerdata->conv1_2_actv);
    
	cout<<"maxpool1_1"<<endl;
	//maxpool1_1
	
    CUDNN_CALL(cudnnPoolingForward(cudnn, p_layer->des_maxpool1_1,  
	                               &alpha, p_layer->ten_des_conv1_2, 
								   p_layerdata->conv1_2, &beta,  
								   p_layer->ten_des_maxpool1_1, p_layerdata->maxpool1_1));
    
    //block-2
	//conv2_1
	cout<<"conv2_1"<<endl;
    computeConvol(cudnn, p_layer->ten_des_maxpool1_1, 
			      p_layerdata->maxpool1_1, p_layer->fil_des_W2_1, 
				  p_weightdata->conv2_1_w, p_layer->des_conv2_1, 
				  p_layer->conv2_1algo, p_layer->ten_des_conv2_1, 
                  p_layer->ten_des_B2_1, p_layer->act_conv2_1, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv2_1, p_weightdata->conv2_1_b, 
				  p_layerdata->conv2_1_actv);
    
	//conv2_2
	cout<<"conv2_2"<<endl;
    computeConvol(cudnn, p_layer->ten_des_conv2_1, 
	              p_layerdata->conv2_1, p_layer->fil_des_W2_2, 
				  p_weightdata->conv2_2_w, p_layer->des_conv2_2, 
				  p_layer->conv2_2algo, p_layer->ten_des_conv2_2, 
                  p_layer->ten_des_B2_2, p_layer->act_conv2_2, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv2_2, p_weightdata->conv2_2_b, 
				  p_layerdata->conv2_2_actv);
    
	//maxpool2_1
	cout<<"maxpool2_1"<<endl;
    CUDNN_CALL(cudnnPoolingForward(cudnn, p_layer->des_maxpool2_1,  
	                               &alpha, p_layer->ten_des_conv2_2, 
								   p_layerdata->conv2_2, &beta, 
								   p_layer->ten_des_maxpool2_1, p_layerdata->maxpool2_1));
    
    //block-3
	//conv3_1
	cout<<"conv3_1"<<endl;
    computeConvol(cudnn, p_layer->ten_des_maxpool2_1,
                  p_layerdata->maxpool2_1, p_layer->fil_des_W3_1, 
				  p_weightdata->conv3_1_w, p_layer->des_conv3_1, 
				  p_layer->conv3_1algo, p_layer->ten_des_conv3_1, 
                  p_layer->ten_des_B3_1, p_layer->act_conv3_1, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv3_1, p_weightdata->conv3_1_b, 
				  p_layerdata->conv3_1_actv);
    
	//conv3_2
	cout<<"conv3_2"<<endl;
    computeConvol(cudnn, p_layer->ten_des_conv3_1, 
	              p_layerdata->conv3_1, p_layer->fil_des_W3_2, 
				  p_weightdata->conv3_2_w, p_layer->des_conv3_2, 
				  p_layer->conv3_2algo, p_layer->ten_des_conv3_2, 
                  p_layer->ten_des_B3_2, p_layer->act_conv3_2, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv3_2, p_weightdata->conv3_2_b, 
				  p_layerdata->conv3_2_actv);
    
	//conv3_3
	cout<<"conv3_3"<<endl;
    computeConvol(cudnn, p_layer->ten_des_conv3_2, 
	              p_layerdata->conv3_2, p_layer->fil_des_W3_3, 
				  p_weightdata->conv3_3_w, p_layer->des_conv3_3, 
				  p_layer->conv3_3algo, p_layer->ten_des_conv3_3, 
                  p_layer->ten_des_B3_3, p_layer->act_conv3_3, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv3_3, p_weightdata->conv3_3_b, 
				  p_layerdata->conv3_3_actv);
    
	//conv3_4
	cout<<"conv3_4"<<endl;
    computeConvol(cudnn, p_layer->ten_des_conv3_3, 
	              p_layerdata->conv3_3, p_layer->fil_des_W3_4, 
				  p_weightdata->conv3_4_w, p_layer->des_conv3_4, 
				  p_layer->conv3_4algo, p_layer->ten_des_conv3_4, 
                  p_layer->ten_des_B3_4, p_layer->act_conv3_4, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv3_4, p_weightdata->conv3_4_b, 
				  p_layerdata->conv3_4_actv);
    
	//maxpool3_1
	cout<<"maxpool3_1"<<endl;
    CUDNN_CALL(cudnnPoolingForward(cudnn, p_layer->des_maxpool3_1,  
	                               &alpha, p_layer->ten_des_conv3_4, 
								   p_layerdata->conv3_4, &beta,  \
								   p_layer->ten_des_maxpool3_1, p_layerdata->maxpool3_1));
   
    //block-4
	//conv4_1
	cout<<"conv4_1"<<endl;
    computeConvol(cudnn, p_layer->ten_des_maxpool3_1, 
	              p_layerdata->maxpool3_1, p_layer->fil_des_W4_1, 
				  p_weightdata->conv4_1_w, p_layer->des_conv4_1, 
				  p_layer->conv4_1algo, p_layer->ten_des_conv4_1, 
                  p_layer->ten_des_B4_1, p_layer->act_conv4_1, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv4_1, p_weightdata->conv4_1_b, 
				  p_layerdata->conv4_1_actv);
    
	//conv4_2
	cout<<"conv4_2"<<endl;
    computeConvol(cudnn, p_layer->ten_des_conv4_1, 
				  p_layerdata->conv4_1, p_layer->fil_des_W4_2, 
				  p_weightdata->conv4_2_w, p_layer->des_conv4_2, 
				  p_layer->conv4_2algo, p_layer->ten_des_conv4_2, 
                  p_layer->ten_des_B4_2, p_layer->act_conv4_2, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv4_2, p_weightdata->conv4_2_b, 
				  p_layerdata->conv4_2_actv);
    
	//conv4_3
	cout<<"conv4_3"<<endl;
    computeConvol(cudnn, p_layer->ten_des_conv4_2, 
	              p_layerdata->conv4_2, p_layer->fil_des_W4_3, 
				  p_weightdata->conv4_3_w, p_layer->des_conv4_3, 
				  p_layer->conv4_3algo, p_layer->ten_des_conv4_3, 
                  p_layer->ten_des_B4_3, p_layer->act_conv4_3, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv4_3, p_weightdata->conv4_3_b, 
				  p_layerdata->conv4_3_actv);
    
	//conv4_4
	cout<<"conv4_4"<<endl;
    computeConvol(cudnn, p_layer->ten_des_conv4_3, 
	              p_layerdata->conv4_3, p_layer->fil_des_W4_4, 
				  p_weightdata->conv4_4_w, p_layer->des_conv4_4, 
				  p_layer->conv4_4algo, p_layer->ten_des_conv4_4, 
                  p_layer->ten_des_B4_4, p_layer->act_conv4_4, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv4_4, p_weightdata->conv4_4_b, 
				  p_layerdata->conv4_4_actv);
    
	//maxpool4_1
	cout<<"maxpool4_1"<<endl;
    CUDNN_CALL(cudnnPoolingForward(cudnn, p_layer->des_maxpool4_1, 
	                               &alpha, p_layer->ten_des_conv4_4, 
								   p_layerdata->conv4_4, &beta, 
								   p_layer->ten_des_maxpool4_1, p_layerdata->maxpool4_1));
    
    //block-5
	//conv5_1
    cout<<"conv5_1"<<endl;
	
    computeConvol(cudnn, p_layer->ten_des_maxpool4_1,
                  p_layerdata->maxpool4_1, p_layer->fil_des_W5_1, 
				  p_weightdata->conv5_1_w, p_layer->des_conv5_1, 
				  p_layer->conv5_1algo, p_layer->ten_des_conv5_1, 
                  p_layer->ten_des_B5_1, p_layer->act_conv5_1, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv5_1, p_weightdata->conv5_1_b, 
				  p_layerdata->conv5_1_actv);
    //conv5_2
    cout<<"conv5_2"<<endl;
    computeConvol(cudnn, p_layer->ten_des_conv5_1, 
				  p_layerdata->conv5_1, p_layer->fil_des_W5_2, 
				  p_weightdata->conv5_2_w, p_layer->des_conv5_2, 
				  p_layer->conv5_2algo, p_layer->ten_des_conv5_2, 
                  p_layer->ten_des_B5_2, p_layer->act_conv5_2, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv5_2, p_weightdata->conv5_2_b, 
				  p_layerdata->conv5_2_actv);
    
	//conv5_3
    cout<<"conv5_3"<<endl;
    computeConvol(cudnn, p_layer->ten_des_conv5_2, 
	              p_layerdata->conv5_2, p_layer->fil_des_W5_3, 
				  p_weightdata->conv5_3_w, p_layer->des_conv5_3, 
				  p_layer->conv5_3algo, p_layer->ten_des_conv5_3, 
                  p_layer->ten_des_B5_3, p_layer->act_conv5_3, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv5_3, p_weightdata->conv5_3_b, 
				  p_layerdata->conv5_3_actv);
    
	//conv5_4
    cout<<"conv5_4"<<endl;
    computeConvol(cudnn, p_layer->ten_des_conv5_3, 
			      p_layerdata->conv5_3, p_layer->fil_des_W5_4, 
				  p_weightdata->conv5_4_w, p_layer->des_conv5_4, 
				  p_layer->conv5_4algo, p_layer->ten_des_conv5_4, 
				  p_layer->ten_des_B5_4, p_layer->act_conv5_4, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->conv5_4, p_weightdata->conv5_4_b, 
				  p_layerdata->conv5_4_actv);
	
    //maxpool5_1 
    cout<<"maxpool5_1"<<endl;			
    CUDNN_CALL(cudnnPoolingForward(cudnn, p_layer->des_maxpool5_1, 
	                               &alpha, p_layer->ten_des_conv5_1, 
								   p_layerdata->conv5_4, &beta, 
								   p_layer->ten_des_maxpool5_1, p_layerdata->maxpool5_1));
    
    //Fully connected layers
	//FC1
	cout<<"FC_1"<<endl;
	computeConvol(cudnn, p_layer->ten_des_maxpool5_1, 
	              p_layerdata->maxpool5_1, p_layer->fil_des_FC_1, 
				  p_weightdata->fc1_w, p_layer->des_fc_1, 
				  p_layer->fc_1_algo, p_layer->ten_des_FC_1, 
                  p_layer->ten_des_B_FC_1, p_layer->act_FC_1, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->fc1, p_weightdata->fc1_b, 
				  p_layerdata->fc1_actv);
				
	//FC2
	cout<<"FC_2"<<endl;
	computeConvol(cudnn, p_layer->ten_des_FC_1, 
	              p_layerdata->fc1, p_layer->fil_des_FC_2, 
				  p_weightdata->fc2_w, p_layer->des_fc_2, 
				  p_layer->fc_2_algo, p_layer->ten_des_FC_2, 
                  p_layer->ten_des_B_FC_2, p_layer->act_FC_2, 
				  p_layer->ws_data, p_layer->ws_size, 
				  p_layerdata->fc2, p_weightdata->fc2_b, 
				  p_layerdata->fc2_actv);
				
	//FC3
	cout<<"FC_3"<<endl;
	//multiplying with weights
	CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha,
                                       p_layer->ten_des_FC_2, p_layerdata->fc2, 
									   p_layer->fil_des_FC_3, p_weightdata->fc1_w, 
									   p_layer->des_fc_3, 
									   p_layer->fc_3_algo, p_layer->ws_data, 
									   p_layer->ws_size, &beta, 
									   p_layer->ten_des_FC_3, p_layerdata->fc3));
    
    //adding bias to FC-3
    CUDNN_CALL(cudnnAddTensor(cudnn, &alpha,
                              p_layer->ten_des_B_FC_3, p_weightdata->fc3_b, 
							  &alpha, p_layer->ten_des_FC_3, 
							  p_layerdata->fc3));
				

    CUDA_CALL(cudaMalloc(&(p_layerdata->out_data), input->in_dim.n * 1000));
    //softmax activation
    CUDNN_CALL(cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, 
	                               CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, 
								   p_layer->ten_des_FC_3, p_layerdata->fc3,
                                   &beta, p_layer->ten_des_FC_3, 
								   p_layerdata->out_data));
	cout<<"softmax done"<<endl;
   
}
__global__ void fillones(float* ones){
	int idx = blockIdx.x;
	ones[idx] = 1.f;
}

void setGradWeightAlgoWorkspace(cudnnHandle_t& cudnn, cudnnTensorDescriptor_t  xDesc,
								cudnnTensorDescriptor_t dyDesc,cudnnConvolutionDescriptor_t  convDesc,
				                cudnnConvolutionBwdFilterAlgo_t& algo,size_t& workSpaceSizeInBytes,
								cudnnFilterDescriptor_t dwDesc,size_t avail){
	/* INPUT:
	 *    cudnn = cudnn handler
     *    xDesc = input descriptor
	 *    dyDesc = descriptor of gradient of previous layer
	 *    conv_desc = convolution descriptor
	 *    dwDesc = filter descriptor
	 *    avail = available workspace to calculation
	 * OUTPUT:
	 *    algo = algorithm to calculate the gardient of weights
	 *    workSpaceSizeInBytes =  workspace size required to perform operation for given algorithm
     * Description:
     *     1) selects the algorithm which best for the given input descriptor,filter descriptor,convolution descriptor and output descriptor
	 *     2) gets the workspace size required to calculate the gradient of weights for given algorithm
     */
	
	CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn,xDesc,
	                                                      dyDesc,convDesc,
														  dwDesc,CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
														  avail,&(algo)));
	CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn,xDesc,
	                                                          dyDesc,convDesc,
															  dwDesc,algo,
															  &workSpaceSizeInBytes));
}

void setGradDataAlgoWorkspace(cudnnHandle_t& handle,cudnnFilterDescriptor_t wDesc,
                              cudnnTensorDescriptor_t dyDesc,cudnnConvolutionDescriptor_t  convDesc,
			                  cudnnConvolutionBwdDataAlgo_t& algo,size_t& workSpaceSizeInBytes, 
			                   cudnnTensorDescriptor_t dxDesc,size_t avail){
	
	/* INPUT:
	 *    cudnn = cudnn handler
	 *    wDesc = filter descriptor
	 *    dyDesc = descriptor of gradient of previous layer
	 *    conv_desc = convolution descriptor
	 *    dxDesc = input descriptor
	 *    avail = available workspace to calculation
	 * OUTPUT:
	 *    algo = algorithm to calculate the gardient of input data
	 *    workSpaceSizeInBytes =  workspace size required to perform operation for given algorithm
     * Description:
     *     1) selects the algorithm which best for the given input descriptor,filter descriptor,convolution descriptor and output descriptor
	 *     2) gets the workspace size required to calculate the gardient of input data for given algorithm
     */
	
	CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(handle, wDesc,
	                                                    dyDesc, convDesc,
														dxDesc,CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
														avail,&algo));
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc,
	                                                        dyDesc, convDesc,
															dxDesc, algo,
															&workSpaceSizeInBytes));
}

void backLayerDesc(cudnnHandle_t& cudnn,inputinfo* input,layerInfo* p_layer, backLayerInfo* bak_layer, 
                   pgradData* p_graddata,int num_class, int batch_size,string optimizer){
	
	
	/* INPUT:
	 *    cudnn = cudnn handler
	 *    input = struct of input descriptor and input dimention
	 *    p_layer = struct of all the filter descriptor, convolution descriptors, pool descriptors and output descriptors and activation descriptors 
	 *    
	 *    num_class: number of classes
	 *    batch_size: batch with which the layer is trained
     * OUTPUT:
     *    bak_layer = memory pointer to the grad_loss data
	 *    p_graddata = struct of pointers of all gradient of weights and bias
     * Description:
     *    allocates memory for the gradient of weights and bias and setup to perform the backpropagation of VGG19
     */ 
	
	cout<<"API backLayerDesc"<<endl;
	size_t  op_size1,op_size2,grad_size1,grad_size2;
	
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_fc3_data), num_class* batch_size * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_loss), num_class* batch_size * sizeof(float)))
	CUDA_CALL(cudaMalloc(&(bak_layer->loss_i), batch_size*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->ones_batch_size), batch_size*sizeof(float)));
    
	fillones<<<batch_size,1>>>(p_graddata->ones_batch_size);
	
	
	CUDNN_CALL(cudnnCreateFilterDescriptor(&(bak_layer->fil_ones)));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(bak_layer->fil_ones, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, 1,
                                          1, batch_size, 1));
	
	singleconvDescriptor(bak_layer->des_ones_conv,"ones");
	
	CUDNN_CALL(cudnnCreateTensorDescriptor(&(bak_layer->grad_ten_des_output)));
	CUDNN_CALL(cudnnSetTensor4dDescriptor(bak_layer->grad_ten_des_output, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
	                                      1, 1, batch_size, num_class));
	
	getsingleLayerDimensions(bak_layer->des_ones_conv, bak_layer->fil_ones, 
                              bak_layer->grad_ten_des_output, bak_layer->grad_ten_des_FC_3, 
							  bak_layer->grad_dim_fc3);
	
	setsingleAlgorithmWorkspace(cudnn, bak_layer->grad_ten_des_output, 
								bak_layer->fil_ones, bak_layer->des_ones_conv, 
								bak_layer-> grad_ten_des_FC_3,&(bak_layer->grad_fc_3_algo), 
								op_size1);
	
	cout<<"workspace for one gard : "<<op_size1<<endl;
	
	//Tensor descriptors for gradient of data
	
	
	//FC-3
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_FC_2,
							   p_layer->ten_des_FC_3, p_layer->des_fc_3,
							   bak_layer->fc_3_bw_fil_algo,op_size2,
							   p_layer->fil_des_FC_3,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	cout<<"p_layer->ws_size : "<<p_layer->ws_size<<endl;
	cout<<"workspace for FC3 weight : "<<op_size2<<endl;
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_FC_3,p_layer->ten_des_FC_3, 
	                         p_layer->des_fc_3, bak_layer->fc_3_bw_dt_algo, 
							 op_size2, p_layer->ten_des_FC_2,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	//FC-2
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_FC_1,
							   p_layer->ten_des_FC_2, p_layer->des_fc_2,
							   bak_layer->fc_2_bw_fil_algo,op_size2,
							   p_layer->fil_des_FC_2,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_FC_2,p_layer->ten_des_FC_2, 
	                         p_layer->des_fc_2, bak_layer->fc_2_bw_dt_algo, 
							 op_size2, p_layer->ten_des_FC_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	//FC-1
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_maxpool5_1,
							   p_layer->ten_des_FC_1, p_layer->des_fc_1,
							   bak_layer->fc_1_bw_fil_algo,op_size2,
							   p_layer->fil_des_FC_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_FC_1,p_layer->ten_des_FC_1, 
	                         p_layer->des_fc_1, bak_layer->fc_1_bw_dt_algo, 
							 op_size2, p_layer->ten_des_maxpool5_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;

	//conv5_4
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_conv5_3,
							   p_layer->ten_des_conv5_4, p_layer->des_conv5_4,
							   bak_layer->conv5_4bw_fil_algo,op_size2,
							   p_layer->fil_des_W5_4,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
		setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W5_4,p_layer->ten_des_conv5_4, 
	                         p_layer->des_conv5_4, bak_layer->conv5_4bw_dt_algo, 
							 op_size2, p_layer->ten_des_conv5_3,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	//conv5_3
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_conv5_2,
							   p_layer->ten_des_conv5_3, p_layer->des_conv5_3,
							   bak_layer->conv5_3bw_fil_algo,op_size2,
							   p_layer->fil_des_W5_3,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W5_3,p_layer->ten_des_conv5_3, 
	                         p_layer->des_conv5_3, bak_layer->conv5_3bw_dt_algo, 
							 op_size2, p_layer->ten_des_conv5_2,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	//conv5_2
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_conv5_1,
							   p_layer->ten_des_conv5_2, p_layer->des_conv5_2,
							   bak_layer->conv5_2bw_fil_algo,op_size2,
							   p_layer->fil_des_W5_2,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W5_2,p_layer->ten_des_conv5_2, 
	                         p_layer->des_conv5_2, bak_layer->conv5_2bw_dt_algo, 
							 op_size2, p_layer->ten_des_conv5_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	//conv5_1
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_maxpool4_1,
							   p_layer->ten_des_conv5_1, p_layer->des_conv5_1,
							   bak_layer->conv5_1bw_fil_algo,op_size2,
							   p_layer->fil_des_W5_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W5_1,p_layer->ten_des_conv5_1, 
	                         p_layer->des_conv5_1, bak_layer->conv5_1bw_dt_algo, 
							 op_size2, p_layer->ten_des_maxpool4_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;

	
	//conv4_4
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_conv4_3,
							   p_layer->ten_des_conv4_4, p_layer->des_conv4_4,
							   bak_layer->conv4_4bw_fil_algo,op_size2,
							   p_layer->fil_des_W4_4,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W4_4,p_layer->ten_des_conv4_4, 
	                         p_layer->des_conv4_4, bak_layer->conv4_4bw_dt_algo, 
							 op_size2, p_layer->ten_des_conv4_3,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	//conv4_3
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_conv4_2,
							   p_layer->ten_des_conv4_3, p_layer->des_conv4_3,
							   bak_layer->conv4_3bw_fil_algo,op_size2,
							   p_layer->fil_des_W4_3,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W4_3,p_layer->ten_des_conv4_3, 
	                         p_layer->des_conv4_3, bak_layer->conv4_3bw_dt_algo, 
							 op_size2, p_layer->ten_des_conv4_2,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	//conv4_2
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_conv4_1,
							   p_layer->ten_des_conv4_2, p_layer->des_conv4_2,
							   bak_layer->conv4_2bw_fil_algo,op_size2,
							   p_layer->fil_des_W4_2,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W4_2,p_layer->ten_des_conv4_2, 
	                         p_layer->des_conv4_2, bak_layer->conv4_2bw_dt_algo, 
							 op_size2, p_layer->ten_des_conv4_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	//conv4_1
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_maxpool3_1,
							   p_layer->ten_des_conv4_1, p_layer->des_conv4_1,
							   bak_layer->conv4_1bw_fil_algo,op_size2,
							   p_layer->fil_des_W4_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W4_1,p_layer->ten_des_conv4_1, 
	                         p_layer->des_conv4_1, bak_layer->conv4_1bw_dt_algo, 
							 op_size2, p_layer->ten_des_maxpool3_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;

	
	//conv3_4
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_conv3_3,
							   p_layer->ten_des_conv3_4, p_layer->des_conv3_4,
							   bak_layer->conv3_4bw_fil_algo,op_size2,
							   p_layer->fil_des_W3_4,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W3_4,p_layer->ten_des_conv3_4, 
	                         p_layer->des_conv3_4, bak_layer->conv3_4bw_dt_algo, 
							 op_size2, p_layer->ten_des_conv3_3,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	//conv3_3
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_conv3_2,
							   p_layer->ten_des_conv3_3, p_layer->des_conv3_3,
							   bak_layer->conv3_3bw_fil_algo,op_size2,
							   p_layer->fil_des_W3_3,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W3_3,p_layer->ten_des_conv3_3, 
	                         p_layer->des_conv3_3, bak_layer->conv3_3bw_dt_algo, 
							 op_size2, p_layer->ten_des_conv3_2,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	//conv3_2
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_conv3_1,
							   p_layer->ten_des_conv3_2, p_layer->des_conv3_2,
							   bak_layer->conv3_2bw_fil_algo,op_size2,
							   p_layer->fil_des_W3_2,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W3_2,p_layer->ten_des_conv3_2, 
	                         p_layer->des_conv3_2, bak_layer->conv3_2bw_dt_algo, 
							 op_size2, p_layer->ten_des_conv3_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	//conv3_1
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_maxpool2_1,
							   p_layer->ten_des_conv3_1, p_layer->des_conv3_1,
							   bak_layer->conv3_1bw_fil_algo,op_size2,
							   p_layer->fil_des_W3_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W3_1,p_layer->ten_des_conv3_1, 
	                         p_layer->des_conv3_1, bak_layer->conv3_1bw_dt_algo, 
							 op_size2, p_layer->ten_des_maxpool2_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;

	
	//conv2_2
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_conv2_1,
							   p_layer->ten_des_conv2_2, p_layer->des_conv2_2,
							   bak_layer->conv2_2bw_fil_algo,op_size2,
							   p_layer->fil_des_W2_2,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;

	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W2_2,p_layer->ten_des_conv2_2, 
	                         p_layer->des_conv2_2, bak_layer->conv2_2bw_dt_algo, 
							 op_size2, p_layer->ten_des_conv2_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;

	//conv2_1
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_maxpool1_1,
							   p_layer->ten_des_conv2_1, p_layer->des_conv2_1,
							   bak_layer->conv2_1bw_fil_algo,op_size2,
							   p_layer->fil_des_W2_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W2_1,p_layer->ten_des_conv2_1, 
	                         p_layer->des_conv2_1, bak_layer->conv2_1bw_dt_algo, 
							 op_size2, p_layer->ten_des_maxpool1_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	//conv1_2
	setGradWeightAlgoWorkspace(cudnn, p_layer->ten_des_conv1_1,
							   p_layer->ten_des_conv1_2, p_layer->des_conv1_2,
							   bak_layer->conv1_2bw_fil_algo,op_size2,
							   p_layer->fil_des_W1_2,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	
	
	setGradDataAlgoWorkspace(cudnn, p_layer->fil_des_W1_2,p_layer->ten_des_conv1_2, 
	                         p_layer->des_conv1_2, bak_layer->conv1_2bw_dt_algo, 
							 op_size2, p_layer->ten_des_conv1_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	//conv1_1
	setGradWeightAlgoWorkspace(cudnn, input->in_desc,
							   p_layer->ten_des_conv1_1, p_layer->des_conv1_1,
							   bak_layer->conv1_1bw_fil_algo,op_size2,
							   p_layer->fil_des_W1_1,p_layer->ws_size);
	op_size1 = op_size1 > op_size2 ? op_size1 : op_size2;
	cout<<"block -1 done"<<endl;
	
	grad_size1 = p_layer->dim_fc2.n * p_layer->dim_fc2.c * p_layer->dim_fc2.w * p_layer->dim_fc2.h * sizeof(float);
	grad_size2 = p_layer->dim_fc1.n * p_layer->dim_fc1.c * p_layer->dim_fc1.w * p_layer->dim_fc1.h * sizeof(float);
	grad_size1 = grad_size1 > grad_size2 ? grad_size1 :grad_size2;
	
	grad_size2 = p_layer->dim_maxpool5_1.n * p_layer->dim_maxpool5_1.c * p_layer->dim_maxpool5_1.w * p_layer->dim_maxpool5_1.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv5_4.n * p_layer->dim_conv5_4.c * p_layer->dim_conv5_4.w * p_layer->dim_conv5_4.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv5_3.n * p_layer->dim_conv5_3.c * p_layer->dim_conv5_3.w * p_layer->dim_conv5_3.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv5_2.n * p_layer->dim_conv5_2.c * p_layer->dim_conv5_2.w * p_layer->dim_conv5_2.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv5_1.n * p_layer->dim_conv5_1.c * p_layer->dim_conv5_1.w * p_layer->dim_conv5_1.h * sizeof(float);
	grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
	
	
    grad_size2 = p_layer->dim_maxpool4_1.n * p_layer->dim_maxpool4_1.c * p_layer->dim_maxpool4_1.w * p_layer->dim_maxpool4_1.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv4_4.n * p_layer->dim_conv4_4.c * p_layer->dim_conv4_4.w * p_layer->dim_conv4_4.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv4_3.n * p_layer->dim_conv4_3.c * p_layer->dim_conv4_3.w * p_layer->dim_conv4_3.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv4_2.n * p_layer->dim_conv4_2.c * p_layer->dim_conv4_2.w * p_layer->dim_conv4_2.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv4_1.n * p_layer->dim_conv4_1.c * p_layer->dim_conv4_1.w * p_layer->dim_conv4_1.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
	
	
    grad_size2 = p_layer->dim_maxpool3_1.n * p_layer->dim_maxpool3_1.c * p_layer->dim_maxpool3_1.w * p_layer->dim_maxpool3_1.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv3_4.n * p_layer->dim_conv3_4.c * p_layer->dim_conv3_4.w * p_layer->dim_conv3_4.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv3_3.n * p_layer->dim_conv3_3.c * p_layer->dim_conv3_3.w * p_layer->dim_conv3_3.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv3_2.n * p_layer->dim_conv3_2.c * p_layer->dim_conv3_2.w * p_layer->dim_conv3_2.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv3_1.n * p_layer->dim_conv3_1.c * p_layer->dim_conv3_1.w * p_layer->dim_conv3_1.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
	
	
    grad_size2 = p_layer->dim_maxpool2_1.n * p_layer->dim_maxpool2_1.c * p_layer->dim_maxpool2_1.w * p_layer->dim_maxpool2_1.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv2_2.n * p_layer->dim_conv2_2.c * p_layer->dim_conv2_2.w * p_layer->dim_conv2_2.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv2_1.n * p_layer->dim_conv2_1.c * p_layer->dim_conv2_1.w * p_layer->dim_conv2_1.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	

	
    grad_size2 = p_layer->dim_maxpool1_1.n * p_layer->dim_maxpool1_1.c * p_layer->dim_maxpool1_1.w * p_layer->dim_maxpool1_1.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv1_2.n * p_layer->dim_conv1_2.c * p_layer->dim_conv1_2.w * p_layer->dim_conv1_2.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;	
    
	grad_size2 = p_layer->dim_conv1_1.n * p_layer->dim_conv1_1.c * p_layer->dim_conv1_1.w * p_layer->dim_conv1_1.h * sizeof(float);
    grad_size1 = grad_size1 > grad_size2 ? grad_size1 : grad_size2;
	
	
	/*
	cout<<"set algo fc1 data"<<endl;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_fc2_data), 
	                       p_layer->dim_fc2.n * p_layer->dim_fc2.c * p_layer->dim_fc2.w * p_layer->dim_fc2.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_fc1_data), 
	                       p_layer->dim_fc1.n * p_layer->dim_fc1.c * p_layer->dim_fc1.w * p_layer->dim_fc1.h * sizeof(float)));
	cout<<"allocate fc grad data"<<endl;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_maxpool5_1_data), 
	                       p_layer->dim_maxpool5_1.n * p_layer->dim_maxpool5_1.c * p_layer->dim_maxpool5_1.w * p_layer->dim_maxpool5_1.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_4_data), 
	                       p_layer->dim_conv5_4.n * p_layer->dim_conv5_4.c * p_layer->dim_conv5_4.w * p_layer->dim_conv5_4.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_3_data), 
	                       p_layer->dim_conv5_3.n * p_layer->dim_conv5_3.c * p_layer->dim_conv5_3.w * p_layer->dim_conv5_3.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_2_data), 
	                       p_layer->dim_conv5_2.n * p_layer->dim_conv5_2.c * p_layer->dim_conv5_2.w * p_layer->dim_conv5_2.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_1_data), 
	                       p_layer->dim_conv5_1.n * p_layer->dim_conv5_1.c * p_layer->dim_conv5_1.w * p_layer->dim_conv5_1.h * sizeof(float)));
	
	cout<<"allocate block 5 grad data"<<endl;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_maxpool4_1_data), 
	                       p_layer->dim_maxpool4_1.n * p_layer->dim_maxpool4_1.c * p_layer->dim_maxpool4_1.w * p_layer->dim_maxpool4_1.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_4_data), 
	                       p_layer->dim_conv4_4.n * p_layer->dim_conv4_4.c * p_layer->dim_conv4_4.w * p_layer->dim_conv4_4.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_3_data), 
	                       p_layer->dim_conv4_3.n * p_layer->dim_conv4_3.c * p_layer->dim_conv4_3.w * p_layer->dim_conv4_3.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_2_data), 
	                       p_layer->dim_conv4_2.n * p_layer->dim_conv4_2.c * p_layer->dim_conv4_2.w * p_layer->dim_conv4_2.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_1_data), 
	                       p_layer->dim_conv4_1.n * p_layer->dim_conv4_1.c * p_layer->dim_conv4_1.w * p_layer->dim_conv4_1.h * sizeof(float)));
	
	cout<<"allocate block 4 grad data"<<endl;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_maxpool3_1_data), 
	                       p_layer->dim_maxpool3_1.n * p_layer->dim_maxpool3_1.c * p_layer->dim_maxpool3_1.w * p_layer->dim_maxpool3_1.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_4_data), 
	                       p_layer->dim_conv3_4.n * p_layer->dim_conv3_4.c * p_layer->dim_conv3_4.w * p_layer->dim_conv3_4.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_3_data), 
	                       p_layer->dim_conv3_3.n * p_layer->dim_conv3_3.c * p_layer->dim_conv3_3.w * p_layer->dim_conv3_3.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_2_data), 
	                       p_layer->dim_conv3_2.n * p_layer->dim_conv3_2.c * p_layer->dim_conv3_2.w * p_layer->dim_conv3_2.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_1_data), 
	                       p_layer->dim_conv3_1.n * p_layer->dim_conv3_1.c * p_layer->dim_conv3_1.w * p_layer->dim_conv3_1.h * sizeof(float)));
	
	cout<<"allocate block 3 grad data"<<endl;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_maxpool2_1_data), 
	                       p_layer->dim_maxpool2_1.n * p_layer->dim_maxpool2_1.c * p_layer->dim_maxpool2_1.w * p_layer->dim_maxpool2_1.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv2_2_data), 
	                       p_layer->dim_conv2_2.n * p_layer->dim_conv2_2.c * p_layer->dim_conv2_2.w * p_layer->dim_conv2_2.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv2_1_data), 
	                       p_layer->dim_conv2_1.n * p_layer->dim_conv2_1.c * p_layer->dim_conv2_1.w * p_layer->dim_conv2_1.h * sizeof(float)));
	
	cout<<"allocate block 2 grad data"<<endl;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_maxpool1_1_data), 
	                       p_layer->dim_maxpool1_1.n * p_layer->dim_maxpool1_1.c * p_layer->dim_maxpool1_1.w * p_layer->dim_maxpool1_1.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv1_2_data), 
	                       p_layer->dim_conv1_2.n * p_layer->dim_conv1_2.c * p_layer->dim_conv1_2.w * p_layer->dim_conv1_2.h * sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv1_1_data), 
	                       p_layer->dim_conv1_1.n * p_layer->dim_conv1_1.c * p_layer->dim_conv1_1.w * p_layer->dim_conv1_1.h * sizeof(float)));*/
	
	cout<<"allocate block 1 grad data"<<endl;
	int k=1000, c=4096, w=1, h=1;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_fc3_b), p_layer->dim_fc3.n*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_fc3_w), k*c*w*h*sizeof(float)));
	
	k=4096, c=4096, w=1, h=1;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_fc2_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_fc2_w), k*c*w*h*sizeof(float)));
	
	c=512;
	k=4096, w=7, h=7;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_fc1_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_fc1_w), k*c*w*h*sizeof(float)));
	
	cout<<"allocate fc grad weight and bias"<<endl;
	k=512, c=512, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_4_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_4_w), k*c*w*h*sizeof(float))); 
    k=512, c=512, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_3_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_3_w), k*c*w*h*sizeof(float)));
    k=512, c=512, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_2_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_2_w), k*c*w*h*sizeof(float)));
    k=512, c=512, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_1_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv5_1_w), k*c*w*h*sizeof(float)));
	
	cout<<"allocate block 5 grad weight and bias"<<endl;
	k=512, c=512, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_4_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_4_w), k*c*w*h*sizeof(float)));
    k=512, c=512, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_3_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_3_w), k*c*w*h*sizeof(float)));
    k=512, c=512, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_2_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_2_w), k*c*w*h*sizeof(float)));
    k=512, c=256, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_1_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv4_1_w), k*c*w*h*sizeof(float)));
	
	cout<<"allocate block 4 grad weight and bias"<<endl;
	k=256, c=256, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_4_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_4_w), k*c*w*h*sizeof(float)));
    k=256, c=256, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_3_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_3_w), k*c*w*h*sizeof(float)));
    k=256, c=256, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_2_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_2_w), k*c*w*h*sizeof(float)));
    k=256, c=128, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_1_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv3_1_w), k*c*w*h*sizeof(float)));
	
	cout<<"allocate block 3 grad weight and bias"<<endl;
	k=128, c=128, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv2_2_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv2_2_w), k*c*w*h*sizeof(float)));
    k=128, c=64, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv2_1_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv2_1_w), k*c*w*h*sizeof(float)));
    
	cout<<"allocate block 2 grad weight and bias"<<endl;
	k=64, c=64, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv1_2_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv1_2_w), k*c*w*h*sizeof(float)));
    k=64, c=3, w=3, h=3;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv1_1_b),k*sizeof(float)));
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_conv1_1_w), k*c*w*h*sizeof(float)));
	
	if(optimizer.compare("sgd_momentum")){
		k=1000, c=4096, w=1, h=1;
	
	    CUDA_CALL(cudaMalloc(&(p_graddata->vel_fc3_b), p_layer->dim_fc3.n*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_fc3_w), k*c*w*h*sizeof(float)));
	
		k=4096, c=4096, w=1, h=1;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_fc2_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_fc2_w), k*c*w*h*sizeof(float)));
	
		c=512;
		k=4096, w=7, h=7;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_fc1_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_fc1_w), k*c*w*h*sizeof(float)));
		
		cout<<"allocate fc vel weight and bias"<<endl;
		k=512, c=512, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv5_4_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv5_4_w), k*c*w*h*sizeof(float))); 
		k=512, c=512, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv5_3_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv5_3_w), k*c*w*h*sizeof(float)));
		k=512, c=512, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv5_2_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv5_2_w), k*c*w*h*sizeof(float)));
		k=512, c=512, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv5_1_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv5_1_w), k*c*w*h*sizeof(float)));
	
		cout<<"allocate block 5 vel weight and bias"<<endl;
		k=512, c=512, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv4_4_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv4_4_w), k*c*w*h*sizeof(float)));
		k=512, c=512, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv4_3_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv4_3_w), k*c*w*h*sizeof(float)));
		k=512, c=512, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv4_2_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv4_2_w), k*c*w*h*sizeof(float)));
		k=512, c=256, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv4_1_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv4_1_w), k*c*w*h*sizeof(float)));
	
		cout<<"allocate block 4 vel weight and bias"<<endl;
		k=256, c=256, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv3_4_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv3_4_w), k*c*w*h*sizeof(float)));
		k=256, c=256, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv3_3_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv3_3_w), k*c*w*h*sizeof(float)));
		k=256, c=256, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv3_2_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv3_2_w), k*c*w*h*sizeof(float)));
		k=256, c=128, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv3_1_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv3_1_w), k*c*w*h*sizeof(float)));
	
		cout<<"allocate block 3 vel weight and bias"<<endl;
		k=128, c=128, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv2_2_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv2_2_w), k*c*w*h*sizeof(float)));
		k=128, c=64, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv2_1_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv2_1_w), k*c*w*h*sizeof(float)));
    
		cout<<"allocate block 2 vel weight and bias"<<endl;
		k=64, c=64, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv1_2_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv1_2_w), k*c*w*h*sizeof(float)));
		k=64, c=3, w=3, h=3;
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv1_1_b),k*sizeof(float)));
		CUDA_CALL(cudaMalloc(&(p_graddata->vel_conv1_1_w), k*c*w*h*sizeof(float)));
	
	}
	
	cout<<"grad_size1 : "<<grad_size1<<endl;
	CUDA_CALL(cudaMalloc(&(p_graddata->grad_data), grad_size1));
	
	if(p_layer->ws_size < op_size1){
		//cout<<"p_layer->ws_size : "<<p_layer->ws_size<<endl;
		//cout<<"op_size1 : "<<op_size1<<endl;
		p_layer->ws_size = op_size1;
		//CUDA_CALL(cudaFree(p_layer->ws_data));
		//cout<<"p_layer->ws_size : "<<p_layer->ws_size<<endl;
		//cout<<"p_layer->ws_data : "<<p_layer->ws_data<<endl;
		CUDA_CALL(cudaMalloc(&(p_layer->ws_data), p_layer->ws_size));
		//cout<<"allocation done"<<endl;
	}
	cout<<"API backLayerDesc done"<<endl;
}


__global__ void calculateLoss(int*y, float* y_cap,float* loss_i,float* grad_in_soft,int batch_size){
	int tid = y[blockIdx.x] + blockIdx.x * blockDim.x;
	loss_i[blockIdx.x] = -logf(y_cap[tid]);
	grad_in_soft[tid] = -1.f;
	grad_in_soft[tid] /= batch_size;
	}

void crossEntropyLoss(int* y, float* y_cap, float* loss_i, float* grad_in_soft, int num_class, int batch_size){
    
	/* INPUT:
	 *    y = pointer actual output of the given images
     *    y_cap =  pointer to the predicted output 
	 *    loss_i = pointer to loss
	 *    
	 *    num_class: number of classes
	 *    batch_size: batch with which the layer is trained
     * OUTPUT:
     *    grad_in_soft = gradient of cross entropy loss and softmax
     * Description:
     *    calcuates the gradient of cross entropy and softmax combined
     */ 
	
	calculateLoss<<<batch_size,num_class>>>(y,y_cap,loss_i,y_cap,batch_size);
	
}

void cal_gradient(cudnnHandle_t& cudnn, layerInfo* p_layer, 
                  pData* p_layerdata, pW_n_B* p_weightdata,inputinfo* input,
				  backLayerInfo* bak_layer,pgradData* p_graddata, int num_class, 
				  int batch_size){
	/* INPUT:
	 *    cudnn: cudnn Handler
	 *    p_layer: struct that contains all layers information
	 *    p_layerdata: struct that contains all layers output data
	 *    p_weightdata: struct that contains weights and biases of the all
	 *                  convolution layers and fully connected layers
	 *    bak_layer: struct contains tensor descriptors of back propagation
	 *    num_class: number of classes
	 *    batch_size: batch with which the layer is trained
     * OUTPUT:
     *    bak_layer = memory pointer to the grad_loss data
     * Description:
     *    calculates the gradient for each layer
     */ 
	 
	float alpha = 1.f;
    float beta = 0.f;
	
	crossEntropyLoss(p_layerdata->y, p_layerdata->out_data, 
	                 bak_layer->loss_i, p_graddata->grad_loss, 
					 num_class, batch_size);
	

	//gradient of bias of FC3
	
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_FC_3, p_layerdata->out_data,
											&beta,p_layer->ten_des_B_FC_3, p_graddata->grad_fc3_b));
    cout<<"gradient of FC3 Bias"<<endl;

	
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_FC_2, p_layerdata->fc2,
											  p_layer->ten_des_FC_3, p_layerdata->out_data,
											  p_layer->des_fc_3,bak_layer->fc_3_bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_FC_3,
											  p_graddata->grad_fc3_w));
	cout<<"gradient of FC3 weight"<<endl;
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_FC_3,  p_weightdata->fc3_w,
											p_layer->ten_des_FC_3,  p_layerdata->out_data,
											p_layer->des_fc_3,bak_layer->fc_3_bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta,p_layer->ten_des_FC_2,p_graddata->grad_data));
	cout<<"gradient of FC2 data actv"<<endl;
	
	//gradient of activation of FC2
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_FC_2,
									   &alpha, p_layer->ten_des_FC_2,
									   p_layerdata->fc2_actv, p_layer->ten_des_FC_2,
									   p_graddata->grad_data, p_layer->ten_des_FC_2,
									   p_layerdata->fc2, &beta, p_layer->ten_des_FC_2,
									   p_graddata->grad_data));
	
	cout<<"gradient of FC2 data"<<endl;
	//gradient of bias of FC2
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_FC_2, p_graddata->grad_data,
											&beta,p_layer->ten_des_B_FC_2, p_graddata->grad_fc2_b));
											
	cout<<"gradient of FC2 bias"<<endl;
	
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_FC_1, p_layerdata->fc1,
											  p_layer->ten_des_FC_2, p_graddata->grad_data,
											  p_layer->des_fc_2,bak_layer->fc_2_bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_FC_2,
											  p_graddata->grad_fc2_w));
	cout<<"gradient of FC2 weight"<<endl;
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_FC_2,  p_weightdata->fc2_w,
											p_layer->ten_des_FC_2,  p_graddata->grad_data,
											p_layer->des_fc_2,bak_layer->fc_2_bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta,p_layer->ten_des_FC_1,p_graddata->grad_data));
	cout<<"gradient of FC1 data actv"<<endl;
	
	//gradient of activation of FC1
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_FC_1,
									   &alpha, p_layer->ten_des_FC_1,
									   p_layerdata->fc1_actv, p_layer->ten_des_FC_1,
									   p_graddata->grad_data, p_layer->ten_des_FC_1,
									   p_layerdata->fc1, &beta, p_layer->ten_des_FC_1,
									   p_graddata->grad_data));
	
	cout<<"gradient of FC1 data "<<endl;
	
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_FC_1, p_graddata->grad_data,
											&beta,p_layer->ten_des_B_FC_1, p_graddata->grad_fc1_b));
	cout<<"gradient of FC1 bias"<<endl;
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_maxpool5_1, p_layerdata->maxpool5_1,
											  p_layer->ten_des_FC_1, p_graddata->grad_data,
											  p_layer->des_fc_1,bak_layer->fc_1_bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_FC_1,
											  p_graddata->grad_fc1_w));
	cout<<"gradient of FC1 weight"<<endl;
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_FC_1,  p_weightdata->fc1_w,
											p_layer->ten_des_FC_1,  p_graddata->grad_data,
											p_layer->des_fc_1,bak_layer->fc_1_bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_maxpool5_1,p_graddata->grad_data));
	cout<<"gradient of maxpool5_1 data"<<endl;
	//gradient of maxpool5_1
	CUDNN_CALL(cudnnPoolingBackward(cudnn, p_layer->des_maxpool5_1, &alpha,
								    p_layer->ten_des_maxpool5_1, p_layerdata->maxpool5_1,
									p_layer->ten_des_maxpool5_1, p_graddata->grad_data,
									p_layer->ten_des_conv5_4, p_layerdata->conv5_4,
									&beta,
								    p_layer->ten_des_conv5_4, p_graddata->grad_data));
	cout<<"gardient of conv5_4 actv data"<<endl;
	
	//gradient of activation of conv5_4
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv5_4,
									   &alpha, p_layer->ten_des_conv5_4,
									   p_layerdata->conv5_4_actv, p_layer->ten_des_conv5_4,
									   p_graddata->grad_data, p_layer->ten_des_conv5_4,
									   p_layerdata->conv5_4, &beta, p_layer->ten_des_conv5_4,
									   p_graddata->grad_data));
	cout<<"gardient of conv5_4 data"<<endl;
	
	//gradient of bias of conv5_4
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv5_4, p_graddata->grad_data,
											&beta,p_layer->ten_des_B5_4, p_graddata->grad_conv5_4_b));
	cout<<"gradient of conv5_4 bias"<<endl;
											
	//gradient of weights of conv5_4
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_conv5_3, p_layerdata->conv5_3,
											  p_layer->ten_des_conv5_4, p_graddata->grad_data,
											  p_layer->des_conv5_4,bak_layer->conv5_4bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W5_4,
											  p_graddata->grad_conv5_4_w));
	cout<<"gradient of conv5_4 weight"<<endl;
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W5_4,  p_weightdata->conv5_4_w,
											p_layer->ten_des_conv5_4,  p_graddata->grad_data,
											p_layer->des_conv5_4,bak_layer->conv5_4bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_conv5_3,p_graddata->grad_data));
	cout<<"gradient of conv5_3 data actv"<<endl;
	
	//gradient of activation of conv5_3
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv5_3,
									   &alpha, p_layer->ten_des_conv5_3,
									   p_layerdata->conv5_3_actv, p_layer->ten_des_conv5_3,
									   p_graddata->grad_data, p_layer->ten_des_conv5_3,
									   p_layerdata->conv5_3, &beta, p_layer->ten_des_conv5_3,
									   p_graddata->grad_data));
	cout<<"gardient of conv5_3 data"<<endl;
	
	//gradient of bias of conv5_3
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv5_3, p_graddata->grad_data,
											&beta,p_layer->ten_des_B5_3, p_graddata->grad_conv5_3_b));
	cout<<"gradient of conv5_3 bias"<<endl;
										
	//gradient of weights of conv5_3
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_conv5_3, p_layerdata->conv5_3,
											  p_layer->ten_des_conv5_3, p_graddata->grad_data,
											  p_layer->des_conv5_3,bak_layer->conv5_3bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W5_3,
											  p_graddata->grad_conv5_3_w));
	cout<<"gradient of conv5_3 weight"<<endl;
	
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W5_3,  p_weightdata->conv5_3_w,
											p_layer->ten_des_conv5_3,  p_graddata->grad_data,
											p_layer->des_conv5_3,bak_layer->conv5_3bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_conv5_2,p_graddata->grad_data));
	cout<<"gradient of conv5_2 data actv"<<endl;
	
	 
	//gradient of activation of conv5_2
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv5_2,
									   &alpha, p_layer->ten_des_conv5_2,
									   p_layerdata->conv5_2_actv, p_layer->ten_des_conv5_2,
									   p_graddata->grad_data, p_layer->ten_des_conv5_2,
									   p_layerdata->conv5_2, &beta, p_layer->ten_des_conv5_2,
									   p_graddata->grad_data));
	cout<<"gardient of conv5_2 data"<<endl;
	
	//gradient of bias of conv5_2
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv5_2, p_graddata->grad_data,
											&beta,p_layer->ten_des_B5_2, p_graddata->grad_conv5_2_b));
	cout<<"gradient of conv5_2 bias"<<endl;
											
	//gradient of weights of conv5_2
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_conv5_2, p_layerdata->conv5_2,
											  p_layer->ten_des_conv5_2, p_graddata->grad_data,
											  p_layer->des_conv5_2,bak_layer->conv5_2bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W5_2,
											  p_graddata->grad_conv5_2_w));
	cout<<"gradient of conv5_2 weight"<<endl;
	
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W5_2,  p_weightdata->conv5_2_w,
											p_layer->ten_des_conv5_2,  p_graddata->grad_data,
											p_layer->des_conv5_2,bak_layer->conv5_2bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_conv5_1,p_graddata->grad_data));
	cout<<"gradient of conv5_1 data actv"<<endl;
	
	
    //gradient of activation of conv5_1
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv5_1,
									   &alpha, p_layer->ten_des_conv5_1,
									   p_layerdata->conv5_1_actv, p_layer->ten_des_conv5_1,
									   p_graddata->grad_data, p_layer->ten_des_conv5_1,
									   p_layerdata->conv5_1, &beta, p_layer->ten_des_conv5_1,
									   p_graddata->grad_data));
	cout<<"gardient of conv5_1 data"<<endl;
	
	//gradient of bias of conv5_1
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv5_1, p_graddata->grad_data,
											&beta,p_layer->ten_des_B5_1, p_graddata->grad_conv5_1_b));
	cout<<"gradient of conv5_1 bias"<<endl;
											
	//gradient of weights of conv5_1
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_maxpool4_1, p_layerdata->maxpool4_1,
											  p_layer->ten_des_conv5_1, p_graddata->grad_data,
											  p_layer->des_conv5_1,bak_layer->conv5_1bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W5_1,
											  p_graddata->grad_conv5_1_w));
	cout<<"gradient of conv5_1 weight"<<endl;
	
    CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W5_1,  p_weightdata->conv5_1_w,
											p_layer->ten_des_conv5_1,  p_graddata->grad_data,
											p_layer->des_conv5_1,bak_layer->conv5_1bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_maxpool4_1,p_graddata->grad_data));
	cout<<"gradient of maxpool4_1 "<<endl;
	
	//gradient of maxpool4_1
	CUDNN_CALL(cudnnPoolingBackward(cudnn, p_layer->des_maxpool4_1, &alpha,
								    p_layer->ten_des_maxpool4_1, p_layerdata->maxpool4_1,
									p_layer->ten_des_maxpool4_1, p_graddata->grad_data,
									p_layer->ten_des_conv4_4, p_layerdata->conv4_4,
									&beta,
								    p_layer->ten_des_conv4_4, p_graddata->grad_data));
	
	cout<<"gradient of conv4_4 actv data"<<endl;
	//gradient of activation of conv4_4
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv4_4,
									   &alpha, p_layer->ten_des_conv4_4,
									   p_layerdata->conv4_4_actv, p_layer->ten_des_conv4_4,
									   p_graddata->grad_data, p_layer->ten_des_conv4_4,
									   p_layerdata->conv4_4, &beta, p_layer->ten_des_conv4_4,
									   p_graddata->grad_data));
	cout<<"gardient of conv4_4 data"<<endl;
	
	//gradient of bias of conv4_4
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv4_4, p_graddata->grad_data,
											&beta,p_layer->ten_des_B4_4, p_graddata->grad_conv4_4_b));
	cout<<"gradient of conv4_4 bias"<<endl;
											
	//gradient of weights of conv4_4
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_conv4_3, p_layerdata->conv4_3,
											  p_layer->ten_des_conv4_4, p_graddata->grad_data,
											  p_layer->des_conv4_4,bak_layer->conv4_4bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W4_4,
											  p_graddata->grad_conv4_4_w));
	cout<<"gradient of conv4_4 weight"<<endl;
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W4_4,  p_weightdata->conv4_4_w,
											p_layer->ten_des_conv4_4,  p_graddata->grad_data,
											p_layer->des_conv4_4,bak_layer->conv4_4bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_conv4_3,p_graddata->grad_data));
	cout<<"gradient of conv4_3 data actv"<<endl;
	
	//gradient of activation of conv4_3
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv4_3,
									   &alpha, p_layer->ten_des_conv4_3,
									   p_layerdata->conv4_3_actv, p_layer->ten_des_conv4_3,
									   p_graddata->grad_data, p_layer->ten_des_conv4_3,
									   p_layerdata->conv4_3, &beta, p_layer->ten_des_conv4_3,
									   p_graddata->grad_data));
	cout<<"gardient of conv4_3 data"<<endl;
	
	//gradient of bias of conv4_3
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv4_3, p_graddata->grad_data,
											&beta,p_layer->ten_des_B4_3, p_graddata->grad_conv4_3_b));
	cout<<"gradient of conv4_3 bias"<<endl;
										
	//gradient of weights of conv4_3
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_conv4_3, p_layerdata->conv4_3,
											  p_layer->ten_des_conv4_3, p_graddata->grad_data,
											  p_layer->des_conv4_3,bak_layer->conv4_3bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W4_3,
											  p_graddata->grad_conv4_3_w));
	cout<<"gradient of conv4_3 weight"<<endl;
	
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W4_3,  p_weightdata->conv4_3_w,
											p_layer->ten_des_conv4_3,  p_graddata->grad_data,
											p_layer->des_conv4_3,bak_layer->conv4_3bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_conv4_2,p_graddata->grad_data));
	cout<<"gradient of conv4_2 data actv"<<endl;
	
	 
	//gradient of activation of conv4_2
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv4_2,
									   &alpha, p_layer->ten_des_conv4_2,
									   p_layerdata->conv4_2_actv, p_layer->ten_des_conv4_2,
									   p_graddata->grad_data, p_layer->ten_des_conv4_2,
									   p_layerdata->conv4_2, &beta, p_layer->ten_des_conv4_2,
									   p_graddata->grad_data));
	cout<<"gardient of conv4_2 data"<<endl;
	
	//gradient of bias of conv4_2
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv4_2, p_graddata->grad_data,
											&beta,p_layer->ten_des_B4_2, p_graddata->grad_conv4_2_b));
	cout<<"gradient of conv4_2 bias"<<endl;
											
	//gradient of weights of conv4_2
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_conv4_2, p_layerdata->conv4_2,
											  p_layer->ten_des_conv4_2, p_graddata->grad_data,
											  p_layer->des_conv4_2,bak_layer->conv4_2bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W4_2,
											  p_graddata->grad_conv4_2_w));
	cout<<"gradient of conv4_2 weight"<<endl;
	
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W4_2,  p_weightdata->conv4_2_w,
											p_layer->ten_des_conv4_2,  p_graddata->grad_data,
											p_layer->des_conv4_2,bak_layer->conv4_2bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_conv4_1,p_graddata->grad_data));
	cout<<"gradient of conv4_1 data actv"<<endl;
	
	
    //gradient of activation of conv4_1
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv4_1,
									   &alpha, p_layer->ten_des_conv4_1,
									   p_layerdata->conv4_1_actv, p_layer->ten_des_conv4_1,
									   p_graddata->grad_data, p_layer->ten_des_conv4_1,
									   p_layerdata->conv4_1, &beta, p_layer->ten_des_conv4_1,
									   p_graddata->grad_data));
	cout<<"gardient of conv4_1 data"<<endl;
	
	//gradient of bias of conv4_1
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv4_1, p_graddata->grad_data,
											&beta,p_layer->ten_des_B4_1, p_graddata->grad_conv4_1_b));
	cout<<"gradient of conv4_1 bias"<<endl;
											
	//gradient of weights of conv4_1
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_maxpool3_1, p_layerdata->maxpool3_1,
											  p_layer->ten_des_conv4_1, p_graddata->grad_data,
											  p_layer->des_conv4_1,bak_layer->conv4_1bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W4_1,
											  p_graddata->grad_conv4_1_w));
	cout<<"gradient of conv4_1 weight"<<endl;
	
    CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W4_1,  p_weightdata->conv4_1_w,
											p_layer->ten_des_conv4_1,  p_graddata->grad_data,
											p_layer->des_conv4_1,bak_layer->conv4_1bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_maxpool3_1,p_graddata->grad_data));
	
	cout<<"gradient of maxpool3_1"<<endl;
	//gradient of maxpool3_1
	CUDNN_CALL(cudnnPoolingBackward(cudnn, p_layer->des_maxpool3_1, &alpha,
								    p_layer->ten_des_maxpool3_1, p_layerdata->maxpool3_1,
									p_layer->ten_des_maxpool3_1, p_graddata->grad_data,
									p_layer->ten_des_conv3_4, p_layerdata->conv3_4,
									&beta,
								    p_layer->ten_des_conv3_4, p_graddata->grad_data));
	
	cout<<"gradient of conv3_4 actv "<<endl;
    
	//gradient of activation of conv3_4
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv3_4,
									   &alpha, p_layer->ten_des_conv3_4,
									   p_layerdata->conv3_4_actv, p_layer->ten_des_conv3_4,
									   p_graddata->grad_data, p_layer->ten_des_conv3_4,
									   p_layerdata->conv3_4, &beta, p_layer->ten_des_conv3_4,
									   p_graddata->grad_data));
	cout<<"gardient of conv3_4 data"<<endl;
	
	//gradient of bias of conv3_4
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv3_4, p_graddata->grad_data,
											&beta,p_layer->ten_des_B3_4, p_graddata->grad_conv3_4_b));
	cout<<"gradient of conv3_4 bias"<<endl;
											
	//gradient of weights of conv3_4
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_conv3_3, p_layerdata->conv3_3,
											  p_layer->ten_des_conv3_4, p_graddata->grad_data,
											  p_layer->des_conv3_4,bak_layer->conv3_4bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W3_4,
											  p_graddata->grad_conv3_3_w));
	cout<<"gradient of conv3_4 weight"<<endl;
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W3_4,  p_weightdata->conv3_4_w,
											p_layer->ten_des_conv3_4,  p_graddata->grad_data,
											p_layer->des_conv3_4,bak_layer->conv3_4bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_conv3_3,p_graddata->grad_data));
	cout<<"gradient of conv3_3 data actv"<<endl;
	
	//gradient of activation of conv3_3
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv3_3,
									   &alpha, p_layer->ten_des_conv3_3,
									   p_layerdata->conv3_3_actv, p_layer->ten_des_conv3_3,
									   p_graddata->grad_data, p_layer->ten_des_conv3_3,
									   p_layerdata->conv3_3, &beta, p_layer->ten_des_conv3_3,
									   p_graddata->grad_data));
	cout<<"gardient of conv3_3 data"<<endl;
	
	//gradient of bias of conv3_3
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv3_3, p_graddata->grad_data,
											&beta,p_layer->ten_des_B3_3, p_graddata->grad_conv3_3_b));
	cout<<"gradient of conv3_3 bias"<<endl;
										
	//gradient of weights of conv3_3
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_conv3_3, p_layerdata->conv3_3,
											  p_layer->ten_des_conv3_3, p_graddata->grad_data,
											  p_layer->des_conv3_3,bak_layer->conv3_3bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W3_3,
											  p_graddata->grad_conv3_3_w));
	cout<<"gradient of conv3_3 weight"<<endl;
	
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W3_3,  p_weightdata->conv3_3_w,
											p_layer->ten_des_conv3_3,  p_graddata->grad_data,
											p_layer->des_conv3_3,bak_layer->conv3_3bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_conv3_2,p_graddata->grad_data));
	cout<<"gradient of conv3_2 data actv"<<endl;
	
	 
	//gradient of activation of conv3_2
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv3_2,
									   &alpha, p_layer->ten_des_conv3_2,
									   p_layerdata->conv3_2_actv, p_layer->ten_des_conv3_2,
									   p_graddata->grad_data, p_layer->ten_des_conv3_2,
									   p_layerdata->conv3_2, &beta, p_layer->ten_des_conv3_2,
									   p_graddata->grad_data));
	cout<<"gardient of conv3_2 data"<<endl;
	
	//gradient of bias of conv3_2
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv3_2, p_graddata->grad_data,
											&beta,p_layer->ten_des_B3_2, p_graddata->grad_conv3_2_b));
	cout<<"gradient of conv3_2 bias"<<endl;
											
	//gradient of weights of conv3_2
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_conv3_2, p_layerdata->conv3_2,
											  p_layer->ten_des_conv3_2, p_graddata->grad_data,
											  p_layer->des_conv3_2,bak_layer->conv3_2bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W3_2,
											  p_graddata->grad_conv3_2_w));
	cout<<"gradient of conv3_2 weight"<<endl;
	
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W3_2,  p_weightdata->conv3_2_w,
											p_layer->ten_des_conv3_2,  p_graddata->grad_data,
											p_layer->des_conv3_2,bak_layer->conv3_2bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_conv3_1,p_graddata->grad_data));
	cout<<"gradient of conv3_1 data actv"<<endl;
	
	
    //gradient of activation of conv3_1
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv3_1,
									   &alpha, p_layer->ten_des_conv3_1,
									   p_layerdata->conv3_1_actv, p_layer->ten_des_conv3_1,
									   p_graddata->grad_data, p_layer->ten_des_conv3_1,
									   p_layerdata->conv3_1, &beta, p_layer->ten_des_conv3_1,
									   p_graddata->grad_data));
	cout<<"gardient of conv3_1 data"<<endl;
	
	//gradient of bias of conv3_1
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv3_1, p_graddata->grad_data,
											&beta,p_layer->ten_des_B3_1, p_graddata->grad_conv3_1_b));
	cout<<"gradient of conv3_1 bias"<<endl;
											
	//gradient of weights of conv3_1
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_maxpool2_1, p_layerdata->maxpool2_1,
											  p_layer->ten_des_conv3_1, p_graddata->grad_data,
											  p_layer->des_conv3_1,bak_layer->conv3_1bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W3_1,
											  p_graddata->grad_conv3_1_w));
	cout<<"gradient of conv3_1 weight"<<endl;
	
    CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W3_1,  p_weightdata->conv3_1_w,
											p_layer->ten_des_conv3_1,  p_graddata->grad_data,
											p_layer->des_conv3_1,bak_layer->conv3_1bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_maxpool2_1,p_graddata->grad_data));
	
	cout<<"gradient of maxpool2_1"<<endl;
	
	//gradient of maxpool2_1
	CUDNN_CALL(cudnnPoolingBackward(cudnn, p_layer->des_maxpool2_1, &alpha,
								    p_layer->ten_des_maxpool2_1, p_layerdata->maxpool2_1,
									p_layer->ten_des_maxpool2_1, p_graddata->grad_data,
									p_layer->ten_des_conv2_2, p_layerdata->conv2_2,
									&beta,
								    p_layer->ten_des_conv2_2, p_graddata->grad_data));

    cout<<"gradient of conv2_2 actv"<<endl;
    
	//gradient of activation of conv2_2
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv2_2,
									   &alpha, p_layer->ten_des_conv2_2,
									   p_layerdata->conv2_2_actv, p_layer->ten_des_conv2_2,
									   p_graddata->grad_data, p_layer->ten_des_conv2_2,
									   p_layerdata->conv2_2, &beta, p_layer->ten_des_conv2_2,
									   p_graddata->grad_data));
	cout<<"gardient of conv2_2 data"<<endl;
	
	//gradient of bias of conv2_2
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv2_2, p_graddata->grad_data,
											&beta,p_layer->ten_des_B2_2, p_graddata->grad_conv2_2_b));
	cout<<"gradient of conv2_2 bias"<<endl;
											
	//gradient of weights of conv2_2
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_conv2_2, p_layerdata->conv2_2,
											  p_layer->ten_des_conv2_2, p_graddata->grad_data,
											  p_layer->des_conv2_2,bak_layer->conv2_2bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W2_2,
											  p_graddata->grad_conv2_2_w));
	cout<<"gradient of conv2_2 weight"<<endl;
	
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W2_2,  p_weightdata->conv2_2_w,
											p_layer->ten_des_conv2_2,  p_graddata->grad_data,
											p_layer->des_conv2_2,bak_layer->conv2_2bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_conv2_1,p_graddata->grad_data));
	cout<<"gradient of conv2_1 data actv"<<endl;
	
	
    //gradient of activation of conv2_1
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv2_1,
									   &alpha, p_layer->ten_des_conv2_1,
									   p_layerdata->conv2_1_actv, p_layer->ten_des_conv2_1,
									   p_graddata->grad_data, p_layer->ten_des_conv2_1,
									   p_layerdata->conv2_1, &beta, p_layer->ten_des_conv2_1,
									   p_graddata->grad_data));
	cout<<"gardient of conv2_1 data"<<endl;
	
	//gradient of bias of conv2_1
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv2_1, p_graddata->grad_data,
											&beta,p_layer->ten_des_B2_1, p_graddata->grad_conv2_1_b));
	cout<<"gradient of conv2_1 bias"<<endl;
											
	//gradient of weights of conv2_1
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_maxpool1_1, p_layerdata->maxpool1_1,
											  p_layer->ten_des_conv2_1, p_graddata->grad_data,
											  p_layer->des_conv2_1,bak_layer->conv2_1bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W2_1,
											  p_graddata->grad_conv2_1_w));
	cout<<"gradient of conv2_1 weight"<<endl;
	
    CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W2_1,  p_weightdata->conv2_1_w,
											p_layer->ten_des_conv2_1,  p_graddata->grad_data,
											p_layer->des_conv2_1,bak_layer->conv2_1bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_maxpool1_1,p_graddata->grad_data));
	
	//gradient of maxpool1_1
	CUDNN_CALL(cudnnPoolingBackward(cudnn, p_layer->des_maxpool1_1, &alpha,
								    p_layer->ten_des_maxpool1_1, p_layerdata->maxpool1_1,
									p_layer->ten_des_maxpool1_1, p_graddata->grad_data,
									p_layer->ten_des_conv1_2, p_layerdata->conv1_2,
									&beta,
								    p_layer->ten_des_conv1_2, p_graddata->grad_data));
	//gradient of activation of conv1_2
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv1_2,
									   &alpha, p_layer->ten_des_conv1_2,
									   p_layerdata->conv1_2_actv, p_layer->ten_des_conv1_2,
									   p_graddata->grad_data, p_layer->ten_des_conv1_2,
									   p_layerdata->conv1_2, &beta, p_layer->ten_des_conv1_2,
									   p_graddata->grad_data));
	cout<<"gardient of conv1_2 data"<<endl;
	
	//gradient of bias of conv1_2
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv1_2, p_graddata->grad_data,
											&beta,p_layer->ten_des_B1_2, p_graddata->grad_conv1_2_b));
	cout<<"gradient of conv1_2 bias"<<endl;
											
	//gradient of weights of conv1_2
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, p_layer->ten_des_conv1_2, p_layerdata->conv1_2,
											  p_layer->ten_des_conv1_2, p_graddata->grad_data,
											  p_layer->des_conv1_2,bak_layer->conv1_2bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W1_2,
											  p_graddata->grad_conv1_2_w));
	cout<<"gradient of conv1_2 weight"<<endl;
	
	CUDNN_CALL(cudnnConvolutionBackwardData(cudnn, &alpha, p_layer->fil_des_W1_2,  p_weightdata->conv1_1_w,
											p_layer->ten_des_conv1_2,  p_graddata->grad_data,
											p_layer->des_conv1_2,bak_layer->conv1_2bw_dt_algo,
											p_layer->ws_data,p_layer->ws_size,
											&beta, p_layer->ten_des_conv1_1,p_graddata->grad_data));
	cout<<"gradient of conv1_1 data actv"<<endl;
	
	
    //gradient of activation of conv1_1
	CUDNN_CALL(cudnnActivationBackward(cudnn, p_layer->act_conv1_1,
									   &alpha, p_layer->ten_des_conv1_1,
									   p_layerdata->conv1_1_actv, p_layer->ten_des_conv1_1,
									   p_graddata->grad_data, p_layer->ten_des_conv1_1,
									   p_layerdata->conv1_1, &beta, p_layer->ten_des_conv1_1,
									   p_graddata->grad_data));
	cout<<"gardient of conv1_1 data"<<endl;
	
	//gradient of bias of conv1_1
	CUDNN_CALL(cudnnConvolutionBackwardBias(cudnn, &alpha,
	                                        p_layer->ten_des_conv1_1, p_graddata->grad_data,
											&beta,p_layer->ten_des_B1_1, p_graddata->grad_conv1_1_b));
	cout<<"gradient of conv1_1 bias"<<endl;
											
	//gradient of weights of conv1_1
	CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn, &alpha, input->in_desc, p_layerdata->input,
											  p_layer->ten_des_conv1_1, p_graddata->grad_data,
											  p_layer->des_conv1_1,bak_layer->conv1_1bw_fil_algo,
											  p_layer->ws_data,p_layer->ws_size,
											  &beta,p_layer->fil_des_W1_1,
											  p_graddata->grad_conv1_1_w));
	cout<<"gradient of conv1_1 weight"<<endl;
	
}

void sgd_momentum(cudnnHandle_t& cudnn,cudnnTensorDescriptor_t& ten_desc,float* vel,float* gradW,float* Weight,float rho,float learning_rate){
	
	/* INPUT:
	 *    cudnn = cudnn Handler
	 *    ten_desc = tensor descriptor of weight
	 *    vel = pointer to the velocity of the gradient of weight
	 *    gradW = pointer to gradient of weight
	 *    rho = momentum value
	 *    learning_rate = learning rate value
     * OUTPUT:
     *    Weight = pointer to updated weight
     * Description:
     *    calculates the sgd+momentum optimizer
	 * 
	 *    //////////////////////////////
	 *    // Formula:                 //
	 *    //////////////////////////////
	 *    // vw = rho*vw + gradw      //
	 *    //  w -= learning_rate*vw   //
	 *    //////////////////////////////
     */
	 
	

	 float alpha = 1.f;
	 float beta = rho;
	 CUDNN_CALL(cudnnAddTensor(cudnn,&alpha,ten_desc,gradW,&beta,ten_desc,vel));
	 alpha = -learning_rate;
	 beta = 1;
	 CUDNN_CALL(cudnnAddTensor(cudnn,&alpha,ten_desc,vel,&beta,ten_desc,Weight));
}

void sgd(cudnnHandle_t& cudnn,cudnnTensorDescriptor_t& ten_desc,float* gradW,float* Weight,float learning_rate){
	
	 /* INPUT:
	 *    cudnn = cudnn Handler
	 *    ten_desc = tensor descriptor of weight
	 *    gradW = pointer to gradient of weight
	 *    learning_rate = learning rate value
     * OUTPUT:
     *    Weight = pointer to updated weight
     * Description:
     *    calculates the sgd optimizer
	 * 
	 *    //////////////////////////////
	 *    // Formula:                 //
	 *    //////////////////////////////
	 *    // w -= learning_rate*gradw //
	 *    //////////////////////////////
     */
	
     float alpha = -learning_rate;
	 float beta = 1;
	 CUDNN_CALL(cudnnAddTensor(cudnn,&alpha,ten_desc,gradW,&beta,ten_desc,Weight));
}

void update_weight(cudnnHandle_t& cudnn,layerInfo* p_layer,pW_n_B* WeightData,pgradData* gradData,float rho,float learning_rate,string optimizer){
	
	/* INPUT:
	 *    cudnn = cudnn Handler
	 *    p_layer =  struct that contains all layers information
	 *    gradData = struct of pointers to gradient data
	 *    rho = momentum value
	 *    learning_rate = learning rate value
	 *    optimizer = optimizer string
     * OUTPUT:
     *    Weightdata = struct that contains weights and biases of the all
	 *                  convolution layers and fully connected layers
     * Description:
     *    Updates all the weights in the network with the specified optimizer
     */ 
	  
	 if(optimizer.compare("sgd_momentum")){
	 //conv1_1 
	 sgd_momentum(cudnn,p_layer->ten_des_W1_1,gradData->vel_conv1_1_w,gradData->grad_conv1_1_w,WeightData->conv1_1_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B1_1,gradData->vel_conv1_1_b,gradData->grad_conv1_1_b,WeightData->conv1_1_b,rho,learning_rate);
	 
	 //conv1_2 
	 sgd_momentum(cudnn,p_layer->ten_des_W1_2,gradData->vel_conv1_2_w,gradData->grad_conv1_2_w,WeightData->conv1_2_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B1_2,gradData->vel_conv1_2_b,gradData->grad_conv1_2_b,WeightData->conv1_2_b,rho,learning_rate);
	 
	 //conv2_1 
	 sgd_momentum(cudnn,p_layer->ten_des_W2_1,gradData->vel_conv2_1_w,gradData->grad_conv2_1_w,WeightData->conv2_1_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B2_1,gradData->vel_conv2_1_b,gradData->grad_conv2_1_b,WeightData->conv2_1_b,rho,learning_rate);
	 
	 //conv2_2 
	 sgd_momentum(cudnn,p_layer->ten_des_W2_2,gradData->vel_conv2_2_w,gradData->grad_conv2_2_w,WeightData->conv2_2_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B2_2,gradData->vel_conv2_2_b,gradData->grad_conv2_2_b,WeightData->conv2_2_b,rho,learning_rate);
	 
	 //conv3_1 
	 sgd_momentum(cudnn,p_layer->ten_des_W3_1,gradData->vel_conv3_1_w,gradData->grad_conv3_1_w,WeightData->conv3_1_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B3_1,gradData->vel_conv3_1_b,gradData->grad_conv3_1_b,WeightData->conv3_1_b,rho,learning_rate);
	 
	 //conv3_2 
	 sgd_momentum(cudnn,p_layer->ten_des_W3_2,gradData->vel_conv3_2_w,gradData->grad_conv3_2_w,WeightData->conv3_2_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B3_2,gradData->vel_conv3_2_b,gradData->grad_conv3_2_b,WeightData->conv3_2_b,rho,learning_rate);
	 
	 //conv3_3 
	 sgd_momentum(cudnn,p_layer->ten_des_W3_3,gradData->vel_conv3_3_w,gradData->grad_conv3_3_w,WeightData->conv3_3_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B3_3,gradData->vel_conv3_3_b,gradData->grad_conv3_3_b,WeightData->conv3_3_b,rho,learning_rate);
	 
	 //conv3_4 
	 sgd_momentum(cudnn,p_layer->ten_des_W3_4,gradData->vel_conv3_4_w,gradData->grad_conv3_4_w,WeightData->conv3_4_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B3_4,gradData->vel_conv3_4_b,gradData->grad_conv3_4_b,WeightData->conv3_4_b,rho,learning_rate);
	 
	 //conv4_1 
	 sgd_momentum(cudnn,p_layer->ten_des_W4_1,gradData->vel_conv4_1_w,gradData->grad_conv4_1_w,WeightData->conv4_1_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B4_1,gradData->vel_conv4_1_b,gradData->grad_conv4_1_b,WeightData->conv4_1_b,rho,learning_rate);
	 
	 //conv4_2 
	 sgd_momentum(cudnn,p_layer->ten_des_W4_2,gradData->vel_conv4_2_w,gradData->grad_conv4_2_w,WeightData->conv4_2_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B4_2,gradData->vel_conv4_2_b,gradData->grad_conv4_2_b,WeightData->conv4_2_b,rho,learning_rate);
	 
	 //conv4_3 
	 sgd_momentum(cudnn,p_layer->ten_des_W4_3,gradData->vel_conv4_3_w,gradData->grad_conv4_3_w,WeightData->conv4_3_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B4_3,gradData->vel_conv4_3_b,gradData->grad_conv4_3_b,WeightData->conv4_3_b,rho,learning_rate);
	 
	 //conv4_4 
	 sgd_momentum(cudnn,p_layer->ten_des_W4_4,gradData->vel_conv4_4_w,gradData->grad_conv4_4_w,WeightData->conv4_4_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B4_4,gradData->vel_conv4_4_b,gradData->grad_conv4_4_b,WeightData->conv4_4_b,rho,learning_rate);
	 
	 //conv5_1 
	 sgd_momentum(cudnn,p_layer->ten_des_W5_1,gradData->vel_conv5_1_w,gradData->grad_conv5_1_w,WeightData->conv5_1_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B5_1,gradData->vel_conv5_1_b,gradData->grad_conv5_1_b,WeightData->conv5_1_b,rho,learning_rate);
	 
	 //conv5_2 
	 sgd_momentum(cudnn,p_layer->ten_des_W5_2,gradData->vel_conv5_2_w,gradData->grad_conv5_2_w,WeightData->conv5_2_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B5_2,gradData->vel_conv5_2_b,gradData->grad_conv5_2_b,WeightData->conv5_2_b,rho,learning_rate);
	 
	 //conv5_3 
	 sgd_momentum(cudnn,p_layer->ten_des_W5_3,gradData->vel_conv5_3_w,gradData->grad_conv5_3_w,WeightData->conv5_3_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B5_3,gradData->vel_conv5_3_b,gradData->grad_conv5_3_b,WeightData->conv5_3_b,rho,learning_rate);
	 
	 //conv5_4 
	 sgd_momentum(cudnn,p_layer->ten_des_W5_4,gradData->vel_conv5_4_w,gradData->grad_conv5_4_w,WeightData->conv5_4_w,rho,learning_rate);
	 sgd_momentum(cudnn,p_layer->ten_des_B5_4,gradData->vel_conv5_4_b,gradData->grad_conv5_4_b,WeightData->conv5_4_b,rho,learning_rate);
	 
	 
	 }
	 else if(optimizer.compare("sgd")){
	 //conv1_1 
	 sgd(cudnn,p_layer->ten_des_W1_1,gradData->grad_conv1_1_w,WeightData->conv1_1_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B1_1,gradData->grad_conv1_1_b,WeightData->conv1_1_b,learning_rate);
	 
	 //conv1_2 
	 sgd(cudnn,p_layer->ten_des_W1_2,gradData->grad_conv1_2_w,WeightData->conv1_2_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B1_2,gradData->grad_conv1_2_b,WeightData->conv1_2_b,learning_rate);
	 
	 //conv2_1 
	 sgd(cudnn,p_layer->ten_des_W2_1,gradData->grad_conv2_1_w,WeightData->conv2_1_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B2_1,gradData->grad_conv2_1_b,WeightData->conv2_1_b,learning_rate);
	 
	 //conv2_2 
	 sgd(cudnn,p_layer->ten_des_W2_2,gradData->grad_conv2_2_w,WeightData->conv2_2_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B2_2,gradData->grad_conv2_2_b,WeightData->conv2_2_b,learning_rate);
	 
	 //conv3_1 
	 sgd(cudnn,p_layer->ten_des_W3_1,gradData->grad_conv3_1_w,WeightData->conv3_1_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B3_1,gradData->grad_conv3_1_b,WeightData->conv3_1_b,learning_rate);
	 
	 //conv3_2 
	 sgd(cudnn,p_layer->ten_des_W3_2,gradData->grad_conv3_2_w,WeightData->conv3_2_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B3_2,gradData->grad_conv3_2_b,WeightData->conv3_2_b,learning_rate);
	 
	 //conv3_3 
	 sgd(cudnn,p_layer->ten_des_W3_3,gradData->grad_conv3_3_w,WeightData->conv3_3_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B3_3,gradData->grad_conv3_3_b,WeightData->conv3_3_b,learning_rate);
	 
	 //conv3_4 
	 sgd(cudnn,p_layer->ten_des_W3_4,gradData->grad_conv3_4_w,WeightData->conv3_4_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B3_4,gradData->grad_conv3_4_b,WeightData->conv3_4_b,learning_rate);
	 
	 //conv4_1 
	 sgd(cudnn,p_layer->ten_des_W4_1,gradData->grad_conv4_1_w,WeightData->conv4_1_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B4_1,gradData->grad_conv4_1_b,WeightData->conv4_1_b,learning_rate);
	 
	 //conv4_2 
	 sgd(cudnn,p_layer->ten_des_W4_2,gradData->grad_conv4_2_w,WeightData->conv4_2_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B4_2,gradData->grad_conv4_2_b,WeightData->conv4_2_b,learning_rate);
	 
	 //conv4_3 
	 sgd(cudnn,p_layer->ten_des_W4_3,gradData->grad_conv4_3_w,WeightData->conv4_3_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B4_3,gradData->grad_conv4_3_b,WeightData->conv4_3_b,learning_rate);
	 
	 //conv4_4 
	 sgd(cudnn,p_layer->ten_des_W4_4,gradData->grad_conv4_4_w,WeightData->conv4_4_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B4_4,gradData->grad_conv4_4_b,WeightData->conv4_4_b,learning_rate);
	 
	 //conv5_1 
	 sgd(cudnn,p_layer->ten_des_W5_1,gradData->grad_conv5_1_w,WeightData->conv5_1_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B5_1,gradData->grad_conv5_1_b,WeightData->conv5_1_b,learning_rate);
	 
	 //conv5_2 
	 sgd(cudnn,p_layer->ten_des_W5_2,gradData->grad_conv5_2_w,WeightData->conv5_2_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B5_2,gradData->grad_conv5_2_b,WeightData->conv5_2_b,learning_rate);
	 
	 //conv5_3 
	 sgd(cudnn,p_layer->ten_des_W5_3,gradData->grad_conv5_3_w,WeightData->conv5_3_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B5_3,gradData->grad_conv5_3_b,WeightData->conv5_3_b,learning_rate);
	 
	 //conv5_4 
	 sgd(cudnn,p_layer->ten_des_W5_4,gradData->grad_conv5_4_w,WeightData->conv5_4_w,learning_rate);
	 sgd(cudnn,p_layer->ten_des_B5_4,gradData->grad_conv5_4_b,WeightData->conv5_4_b,learning_rate);
	 }
	 cout<<"weights updated"<<endl;
}

void printoutputdimensions(layerInfo* d_layer, inputinfo* d_input){
    
	/* INPUT:
	 *    d_layer = struct of all the filter descriptor, convolution descriptors, pool descriptors and output descriptors and activation descriptors 
	 *    d_input =  struct of input descriptor and input dimention
     * Description:
     *    Prints the output dimensions of each layer
     */
	 
    std::cout<<"input: ";
    std::cout<<d_input->in_dim.n<<" "<<d_input->in_dim.c<<" "<<d_input->in_dim.h<<" "<<d_input->in_dim.w<<std::endl;
    std::cout<<"conv1_1: ";
    std::cout<<d_layer->dim_conv1_1.n<<" "<<d_layer->dim_conv1_1.c<<" "<<d_layer->dim_conv1_1.h<<" "<<d_layer->dim_conv1_1.w<<std::endl;
    std::cout<<"conv1_2: ";
    std::cout<<d_layer->dim_conv1_2.n<<" "<<d_layer->dim_conv1_2.c<<" "<<d_layer->dim_conv1_2.h<<" "<<d_layer->dim_conv1_2.w<<std::endl;
    std::cout<<"pool1_1: ";
    std::cout<<d_layer->dim_maxpool1_1.n<<" "<<d_layer->dim_maxpool1_1.c<<" "<<d_layer->dim_maxpool1_1.h<<" "<<d_layer->dim_maxpool1_1.w<<std::endl;
    
    std::cout<<"conv2_1: ";
    std::cout<<d_layer->dim_conv2_1.n<<" "<<d_layer->dim_conv2_1.c<<" "<<d_layer->dim_conv2_1.h<<" "<<d_layer->dim_conv2_1.w<<std::endl;
    std::cout<<"conv2_2: ";
    std::cout<<d_layer->dim_conv2_2.n<<" "<<d_layer->dim_conv2_2.c<<" "<<d_layer->dim_conv2_2.h<<" "<<d_layer->dim_conv2_2.w<<std::endl;
    std::cout<<"pool2_1: ";
    std::cout<<d_layer->dim_maxpool2_1.n<<" "<<d_layer->dim_maxpool2_1.c<<" "<<d_layer->dim_maxpool2_1.h<<" "<<d_layer->dim_maxpool2_1.w<<std::endl;
    
    std::cout<<"conv3_1: ";
    std::cout<<d_layer->dim_conv3_1.n<<" "<<d_layer->dim_conv3_1.c<<" "<<d_layer->dim_conv3_1.h<<" "<<d_layer->dim_conv3_1.w<<std::endl;
    std::cout<<"conv3_2: ";
    std::cout<<d_layer->dim_conv3_2.n<<" "<<d_layer->dim_conv3_2.c<<" "<<d_layer->dim_conv3_2.h<<" "<<d_layer->dim_conv3_2.w<<std::endl;
    std::cout<<"conv3_3: ";
    std::cout<<d_layer->dim_conv3_3.n<<" "<<d_layer->dim_conv3_3.c<<" "<<d_layer->dim_conv3_3.h<<" "<<d_layer->dim_conv3_3.w<<std::endl;
    std::cout<<"conv3_4: ";
    std::cout<<d_layer->dim_conv3_4.n<<" "<<d_layer->dim_conv3_4.c<<" "<<d_layer->dim_conv3_4.h<<" "<<d_layer->dim_conv3_4.w<<std::endl;
    std::cout<<"pool3_1: ";
    std::cout<<d_layer->dim_maxpool3_1.n<<" "<<d_layer->dim_maxpool3_1.c<<" "<<d_layer->dim_maxpool3_1.h<<" "<<d_layer->dim_maxpool3_1.w<<std::endl;
    
    std::cout<<"conv4_1: ";
    std::cout<<d_layer->dim_conv4_1.n<<" "<<d_layer->dim_conv4_1.c<<" "<<d_layer->dim_conv4_1.h<<" "<<d_layer->dim_conv4_1.w<<std::endl;
    std::cout<<"conv4_2: ";
    std::cout<<d_layer->dim_conv4_2.n<<" "<<d_layer->dim_conv4_2.c<<" "<<d_layer->dim_conv4_2.h<<" "<<d_layer->dim_conv4_2.w<<std::endl;
    std::cout<<"conv4_3: ";
    std::cout<<d_layer->dim_conv4_3.n<<" "<<d_layer->dim_conv4_3.c<<" "<<d_layer->dim_conv4_3.h<<" "<<d_layer->dim_conv4_3.w<<std::endl;
    std::cout<<"conv4_4: ";
    std::cout<<d_layer->dim_conv4_4.n<<" "<<d_layer->dim_conv4_4.c<<" "<<d_layer->dim_conv4_4.h<<" "<<d_layer->dim_conv4_4.w<<std::endl;
    std::cout<<"pool4_1: ";
    std::cout<<d_layer->dim_maxpool4_1.n<<" "<<d_layer->dim_maxpool4_1.c<<" "<<d_layer->dim_maxpool4_1.h<<" "<<d_layer->dim_maxpool4_1.w<<std::endl;
    
    std::cout<<"conv5_1: ";
    std::cout<<d_layer->dim_conv5_1.n<<" "<<d_layer->dim_conv5_1.c<<" "<<d_layer->dim_conv5_1.h<<" "<<d_layer->dim_conv5_1.w<<std::endl;
    std::cout<<"conv5_2: ";
    std::cout<<d_layer->dim_conv5_2.n<<" "<<d_layer->dim_conv5_2.c<<" "<<d_layer->dim_conv5_2.h<<" "<<d_layer->dim_conv5_2.w<<std::endl;
    std::cout<<"conv5_3: ";
    std::cout<<d_layer->dim_conv5_3.n<<" "<<d_layer->dim_conv5_3.c<<" "<<d_layer->dim_conv5_3.h<<" "<<d_layer->dim_conv5_3.w<<std::endl;
    std::cout<<"conv5_4: ";
    std::cout<<d_layer->dim_conv5_4.n<<" "<<d_layer->dim_conv5_4.c<<" "<<d_layer->dim_conv5_4.h<<" "<<d_layer->dim_conv5_4.w<<std::endl;
    std::cout<<"pool5_1: ";
    std::cout<<d_layer->dim_maxpool5_1.n<<" "<<d_layer->dim_maxpool5_1.c<<" "<<d_layer->dim_maxpool5_1.h<<" "<<d_layer->dim_maxpool5_1.w<<std::endl;
	
	std::cout<<"fc_1: ";
    std::cout<<d_layer->dim_fc1.n<<" "<<d_layer->dim_fc1.c<<" "<<d_layer->dim_fc1.h<<" "<<d_layer->dim_fc1.w<<std::endl;
    std::cout<<"fc_2: ";
    std::cout<<d_layer->dim_fc2.n<<" "<<d_layer->dim_fc2.c<<" "<<d_layer->dim_fc2.h<<" "<<d_layer->dim_fc2.w<<std::endl;
    std::cout<<"fc_3: ";
    std::cout<<d_layer->dim_fc3.n<<" "<<d_layer->dim_fc3.c<<" "<<d_layer->dim_fc3.h<<" "<<d_layer->dim_fc3.w<<std::endl;
	
}


void destroy_layerinfo(layerInfo* layer_info){
	
	/* INPUT:
	 *    layer_info = struct of all the filter descriptor, convolution descriptors, pool descriptors and output descriptors and activation descriptors 
     * Description:
     *    destroys all the descriptors
     */
    
	CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv1_1));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv1_2));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv2_1));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv2_2));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv3_1));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv3_2));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv3_3));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv3_4));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv4_1));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv4_2));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv4_3));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv4_4));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv5_1));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv5_2));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv5_3));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_conv5_4));
    
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv1_1));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv1_2));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv2_1));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv2_2));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv3_1));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv3_2));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv3_3));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv3_4));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv4_1));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv4_2));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv4_3));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv4_4));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv5_1));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv5_2));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv5_3));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(layer_info->des_conv5_4));
    
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(layer_info->des_maxpool1_1));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(layer_info->des_maxpool2_1));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(layer_info->des_maxpool3_1));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(layer_info->des_maxpool4_1));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(layer_info->des_maxpool5_1));
    
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W1_1));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W1_2));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W2_1));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W2_2));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W3_1));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W3_2));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W3_3));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W3_4));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W4_1));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W4_2));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W4_3));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W4_4));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W5_1));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W5_2));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W5_3));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(layer_info->fil_des_W5_4));
    
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B1_1));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B1_2));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B2_1));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B2_2));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B3_1));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B3_2));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B3_3));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B3_4));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B4_1));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B4_2));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B4_3));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B4_4));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B5_1));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B5_2));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B5_3));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(layer_info->ten_des_B5_4));
}

void destroy_data(pData* output){
	
	/* INPUT:
	 *    output = struct of pointers to output of each layer
	 * Description:
     *    Free memory allocated to output of each layers
     */
    
	CUDA_CALL(cudaFree(output->conv1_1));
    CUDA_CALL(cudaFree(output->conv1_2));
    CUDA_CALL(cudaFree(output->maxpool1_1));
    CUDA_CALL(cudaFree(output->conv2_1));
    CUDA_CALL(cudaFree(output->conv2_2));
    CUDA_CALL(cudaFree(output->maxpool2_1));
    CUDA_CALL(cudaFree(output->conv3_1));
    CUDA_CALL(cudaFree(output->conv3_2));
    CUDA_CALL(cudaFree(output->conv3_3));
    CUDA_CALL(cudaFree(output->conv3_4));
    CUDA_CALL(cudaFree(output->maxpool3_1));
    CUDA_CALL(cudaFree(output->conv4_1));
    CUDA_CALL(cudaFree(output->conv4_2));
    CUDA_CALL(cudaFree(output->conv4_3));
    CUDA_CALL(cudaFree(output->conv4_4));
    CUDA_CALL(cudaFree(output->maxpool4_1));
    CUDA_CALL(cudaFree(output->conv5_1));
    CUDA_CALL(cudaFree(output->conv5_2));
    CUDA_CALL(cudaFree(output->conv5_3));
    CUDA_CALL(cudaFree(output->conv5_4));
    CUDA_CALL(cudaFree(output->maxpool5_1));
}

void destroy(layerInfo* layer_info, pData* output){
	
	/* INPUT:
	 *    layer_info = struct of all the filter descriptor, convolution descriptors, pool descriptors and output descriptors and activation descriptors 
	 *    output = struct of pointers to output of each layer
	 * Description:
     *    This function initiates the clean up work
     */
	 
    destroy_layerinfo(layer_info);
    free(layer_info);
    destroy_data(output);
    free(output);
}
