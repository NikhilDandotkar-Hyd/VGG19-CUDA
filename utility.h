#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include "vgg19_lib.h"
#include <bits/stdc++.h>
#include <fstream>
#include "image.h"

#define CONV_SIZE 3



float rand_uniform(float min, float max);

void InitializeRandomWeights( float *WeightDataptr, int out_channels, int in_channels, int height, int width);

void hostFCWeights_update (float* Fc_data, float* Fc_bias, FILE *fcp, int z);

void hostConvWeights_update( float* w_data, float* bias, FILE *fp, int z);

void read_FC_weights (string FCweight_file, pW_n_B* hostWnb_data,pW_n_B* weight_data);

void read_conv_weights(string weight_file, pW_n_B* hostWnb_data, pW_n_B* weight_data);

void InitializeBias(float *BiasDataptr, int out_channels);

void InitializeAllRandomWeights(pW_n_B* hostWnb_data, pW_n_B* weight_data);

image readTestImage(string Imagepath);

void copying_intermediate_reults(pData* output, pData* InterOut, layerInfo* layer_info);

