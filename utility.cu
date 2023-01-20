
#include "utility.h"

using namespace std;

int cshape[16][4] = { 
	{ 64, 3, CONV_SIZE, CONV_SIZE },    // 1
	{ 64, 64, CONV_SIZE, CONV_SIZE },   // 2
	{ 128, 64, CONV_SIZE, CONV_SIZE },  // 3
	{ 128, 128, CONV_SIZE, CONV_SIZE }, // 4
	{ 256, 128, CONV_SIZE, CONV_SIZE }, // 5
	{ 256, 256, CONV_SIZE, CONV_SIZE }, // 6 
	{ 256, 256, CONV_SIZE, CONV_SIZE }, // 7
    { 256, 256, CONV_SIZE, CONV_SIZE }, // 8
	{ 512, 256, CONV_SIZE, CONV_SIZE }, // 9
	{ 512, 512, CONV_SIZE, CONV_SIZE }, // 10
	{ 512, 512, CONV_SIZE, CONV_SIZE }, // 11
    { 512, 512, CONV_SIZE, CONV_SIZE }, // 12
	{ 512, 512, CONV_SIZE, CONV_SIZE }, // 13
	{ 512, 512, CONV_SIZE, CONV_SIZE }, // 14
	{ 512, 512, CONV_SIZE, CONV_SIZE }, // 15
    { 512, 512, CONV_SIZE, CONV_SIZE }  // 16
};

int fshape[3][2] = {
	{ 25088, 4096 },
	{ 4096, 4096 },
	{ 4096, 1000 }
};

float rand_uniform(float min, float max)
{
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand() / RAND_MAX * (max - min)) + min;
}


void InitializeRandomWeights( float *WeightDataptr, int out_channels, int in_channels, int height, int width)
{
	int WeightDataptr_Size = in_channels * height * width *out_channels; 
    
    float scale = sqrt(2. / (WeightDataptr_Size));
    
    //cout<<" allocated weights space"<<endl;
    for (int i = 0; i < WeightDataptr_Size; ++i) 
        WeightDataptr[i] = scale*rand_uniform(0, 1);
    
        
}


void InitializeBias(float *BiasDataptr, int out_channels)
{
   
    for (int i = 0 ; i < out_channels; i++){
        BiasDataptr[i] = 1.f;
    }
}

void InitializeAllRandomWeights(pW_n_B* hostWnb_data, pW_n_B* weight_data)
{
    cout<<"API InitializeAllRandomWeights"<<endl;
    int k=64, c=3, w=3, h=3;
    hostWnb_data->conv1_1_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv1_1_b, k);
    hostWnb_data->conv1_1_w = (float*) malloc( k*c*h*w * sizeof(float) );
    InitializeRandomWeights(hostWnb_data->conv1_1_w,k,c,h,w);
    CUDA_CALL(cudaMalloc(&weight_data->conv1_1_b, (k * sizeof(float))));
    CUDA_CALL(cudaMemcpy(weight_data->conv1_1_b, hostWnb_data->conv1_1_b, (k * sizeof(float)), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&weight_data->conv1_1_w, (k*c*h*w * sizeof(float))));
    CUDA_CALL(cudaMemcpy(weight_data->conv1_1_w, hostWnb_data->conv1_1_w, (k*c*h*w * sizeof(float)), cudaMemcpyHostToDevice));
    
    
    //conv1_2
    k=64, c=64, w=3, h=3;
    hostWnb_data->conv1_2_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv1_2_b, k);
    hostWnb_data->conv1_2_w = (float*) malloc( k*c*h*w * sizeof(float) );
    InitializeRandomWeights(hostWnb_data->conv1_2_w,k,c,h,w);
    cudaMalloc(&weight_data->conv1_2_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv1_2_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv1_2_b, hostWnb_data->conv1_2_b, k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv1_2_w, hostWnb_data->conv1_2_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
    

    //conv2_1
    k=128, c=64, w=3, h=3;
    hostWnb_data->conv2_1_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv2_1_b, k);
    hostWnb_data->conv2_1_w = (float*) malloc( k*c*h*w * sizeof(float) );
    InitializeRandomWeights(hostWnb_data->conv2_1_w,k,c,h,w);
    cudaMalloc(&weight_data->conv2_1_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv2_1_w, sizeof(float)*k*c*w*h);
    cudaMemcpy(weight_data->conv2_1_b, hostWnb_data->conv2_1_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv2_1_w, hostWnb_data->conv2_1_w, sizeof(float)*k*c*w*h, cudaMemcpyHostToDevice);
    
    
    //conv2_2
    k=128, c=128, w=3, h=3;
    hostWnb_data->conv2_2_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv2_2_b, k);
    hostWnb_data->conv2_2_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv2_2_w,k,c,h,w);
    cudaMalloc(&weight_data->conv2_2_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv2_2_w, sizeof(float)*k*c*w*h);
    cudaMemcpy(weight_data->conv2_2_b, hostWnb_data->conv2_2_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv2_2_w, hostWnb_data->conv2_2_w, sizeof(float)*k*c*w*h, cudaMemcpyHostToDevice);
    
    
    
    
    //h_conv3_1
    k=256, c=128, w=3, h=3;
    hostWnb_data->conv3_1_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv3_1_b, k);
    hostWnb_data->conv3_1_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv3_1_w,k,c,h,w);
    cudaMalloc(&weight_data->conv3_1_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv3_1_w, sizeof(float)*k*c*w*h);
    cudaMemcpy(weight_data->conv3_1_b, hostWnb_data->conv3_1_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv3_1_w, hostWnb_data->conv3_1_w, sizeof(float)*k*c*w*h, cudaMemcpyHostToDevice);
  

    
    //h_conv3_2
    k=256, c=256, w=3, h=3;
    hostWnb_data->conv3_2_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv3_2_b, k);
    hostWnb_data->conv3_2_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv3_2_w,k,c,h,w);
    cudaMalloc(&weight_data->conv3_2_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv3_2_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv3_2_b, hostWnb_data->conv3_2_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv3_2_w, hostWnb_data->conv3_2_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
    
    
    //h_conv3_3
    k=256, c=256, w=3, h=3;
    hostWnb_data->conv3_3_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv3_3_b, k);
    hostWnb_data->conv3_3_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv3_3_w,k,c,h,w);
    cudaMalloc(&weight_data->conv3_3_b, sizeof(float) * k);
    cudaMalloc(&weight_data->conv3_3_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv3_3_b, hostWnb_data->conv3_3_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv3_3_w, hostWnb_data->conv3_3_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
    
    //h_conv3_4
    k=256, c=256, w=3, h=3;
    hostWnb_data->conv3_4_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv3_4_b, k);
    hostWnb_data->conv3_4_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv3_4_w,k,c,h,w);
    cudaMalloc(&weight_data->conv3_4_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv3_4_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv3_4_b, hostWnb_data->conv3_4_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv3_4_w, hostWnb_data->conv3_4_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    

    //h_conv4_1
    k=512, c=256, w=3, h=3;
    hostWnb_data->conv4_1_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv4_1_b, k);
    hostWnb_data->conv4_1_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv4_1_w,k,c,h,w);
    cudaMalloc(&weight_data->conv4_1_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv4_1_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv4_1_b, hostWnb_data->conv4_1_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv4_1_w, hostWnb_data->conv4_1_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);

    
    //h_conv4_2
    k=512, c=512, w=3, h=3;
    hostWnb_data->conv4_2_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv4_2_b, k);
    hostWnb_data->conv4_2_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv4_2_w,k,c,h,w);
    cudaMalloc(&weight_data->conv4_2_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv4_2_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv4_2_b, hostWnb_data->conv4_2_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv4_2_w, hostWnb_data->conv4_2_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
    
    //h_conv4_3
    k=512, c=512, w=3, h=3;
    hostWnb_data->conv4_3_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv4_3_b, k);
    hostWnb_data->conv4_3_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv4_3_w,k,c,h,w);
    cudaMalloc(&weight_data->conv4_3_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv4_3_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv4_3_b, hostWnb_data->conv4_3_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv4_3_w, hostWnb_data->conv4_3_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
    
    //h_conv4_4
    k=512, c=512, w=3, h=3;
    hostWnb_data->conv4_4_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv4_4_b, k);
    hostWnb_data->conv4_4_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv4_4_w,k,c,h,w);
    cudaMalloc(&weight_data->conv4_4_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv4_4_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv4_4_b, hostWnb_data->conv4_4_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv4_4_w, hostWnb_data->conv4_4_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
    
    //h_conv5_1
    k=512, c=512, w=3, h=3;
    hostWnb_data->conv5_1_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv5_1_b, k);
    hostWnb_data->conv5_1_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv5_1_w,k,c,h,w);
    cudaMalloc(&weight_data->conv5_1_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv5_1_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv5_1_b, hostWnb_data->conv5_1_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv5_1_w, hostWnb_data->conv5_1_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
    
    //h_conv5_2
    k=512, c=512, w=3, h=3;
    hostWnb_data->conv5_2_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv5_2_b, k);
    hostWnb_data->conv5_2_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv5_2_w,k,c,h,w);
    cudaMalloc(&weight_data->conv5_2_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv5_2_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv5_2_b, hostWnb_data->conv5_2_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv5_2_w, hostWnb_data->conv5_2_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    

    //h_conv5_3
    k=512, c=512, w=3, h=3;
    hostWnb_data->conv5_3_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv5_3_b, k);
    hostWnb_data->conv5_3_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv5_3_w,k,c,h,w);
    cudaMalloc(&weight_data->conv5_3_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv5_3_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv5_3_b, hostWnb_data->conv5_3_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv5_3_w, hostWnb_data->conv5_3_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
    
    //h_conv5_4
    k=512, c=512, w=3, h=3;
    hostWnb_data->conv5_4_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->conv5_4_b, k);
    hostWnb_data->conv5_4_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->conv5_4_w,k,c,h,w);
    cudaMalloc(&weight_data->conv5_4_b, sizeof(float)*k);
    cudaMalloc(&weight_data->conv5_4_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->conv5_4_b, hostWnb_data->conv5_4_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->conv5_4_w, hostWnb_data->conv5_4_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
	
	//Fully Connected Layer 1
	c=512;
	k=4096, w=7, h=7;
    hostWnb_data->fc1_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->fc1_b, k);
    hostWnb_data->fc1_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->fc1_w,k,c,h,w);
    cudaMalloc(&weight_data->fc1_b, sizeof(float)*k);
    cudaMalloc(&weight_data->fc1_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->fc1_b, hostWnb_data->fc1_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->fc1_w, hostWnb_data->fc1_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
	
	//Fully Connected Layer 2
    k=4096, c=4096, w=1, h=1;
    hostWnb_data->fc2_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->fc2_b, k);
    hostWnb_data->fc2_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->fc2_w,k,c,h,w);
    cudaMalloc(&weight_data->fc2_b, sizeof(float)*k);
    cudaMalloc(&weight_data->fc2_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->fc2_b, hostWnb_data->fc2_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->fc2_w, hostWnb_data->fc2_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
    //Fully Connected Layer 3
    k=1000, c=4096, w=1, h=1;
    hostWnb_data->fc3_b = (float*) malloc(k * sizeof(float));
    InitializeBias(hostWnb_data->fc3_b, k);
    hostWnb_data->fc3_w = (float*) malloc( k*c*h*w * sizeof(float));
    InitializeRandomWeights(hostWnb_data->fc3_w,k,c,h,w);
    cudaMalloc(&weight_data->fc3_b, sizeof(float)*k);
    cudaMalloc(&weight_data->fc3_w, sizeof(float)*c * h * w *k);
    cudaMemcpy(weight_data->fc3_b, hostWnb_data->fc3_b, sizeof(float)*k, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data->fc3_w, hostWnb_data->fc3_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
}

void hostFCWeights_update (float* Fc_data, float* Fc_bias, FILE *fcp, int z)
{
    int i, j;
    float fval;
    
        for (i = 0; i < fshape[z][0]; i++) 
            {
                for (j = 0; j < fshape[z][1]; j++) 
                {
                    fscanf(fcp, "%f", &fval);
                    Fc_data[j+ fshape[z][1]* i] = fval;
                    
                }
            }
        for (i = 0; i < fshape[z][1]; i++) 
            {
                fscanf(fcp, "%f", &fval);
                Fc_bias[i] = fval;
            }
    
}

void hostConvWeights_update( float* w_data, float* bias, FILE *fp, int z)
{
        float dval;
        int i, j, m, l;
        //int idx =0;
        //printf("Read conv block %d weights\n", z);
        for (i = 0; i < cshape[z][0]; i++) 
        {
            for (j = 0; j < cshape[z][1]; j++) 
            {
                for (m = 0; m < cshape[z][2]; m++) 
                {
                    for (l = 0; l < cshape[z][3]; l++) 
                    {
                        fscanf(fp, "%f", &dval);
                        //cout << "hostWnb_data->conv1_1_w weights are loading" << endl;
                        w_data[l + cshape[z][3]*m + cshape[z][3]*cshape[z][2]*j+ cshape[z][3]*cshape[z][2]*cshape[z][1]*i] = dval;

                        //idx++;
                    }
                }
            }
        }
        
        for (i = 0; i < cshape[z][0]; i++) 
        {

            fscanf(fp, "%f", &dval);
            bias[i]  =  dval;

        }
}

void read_FC_weights (string FCweight_file, pW_n_B* hostWnb_data,pW_n_B* weight_data)
{
    cout << "Reading FC Layers weights" << endl;
    int z;
    FILE *fcp;
    char char_array[FCweight_file.length() + 1]; 
	strcpy(char_array, FCweight_file.c_str()); 
	
    fcp = fopen(char_array, "r");
    if (fcp == NULL) 
    {
		cout<<"File %s absent\n"<<FCweight_file<<endl;
        exit(1);
    }
    
    for (z = 0; z < 3; z++) 
    {
        if(z==0)
        {
            int c=512;
            int k=4096, w=7, h=7;
            hostWnb_data->fc1_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->fc1_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostFCWeights_update(hostWnb_data->fc1_w, hostWnb_data->fc1_b, fcp, z);
            cudaMalloc(&weight_data->fc1_b, sizeof(float)*k);
            cudaMalloc(&weight_data->fc1_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->fc1_b, hostWnb_data->fc1_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->fc1_w, hostWnb_data->fc1_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "FC-1 Weights Read Done" << endl;
        }
        if(z==1)
        {
            int k=4096, c=4096, w=1, h=1;
            hostWnb_data->fc2_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->fc2_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostFCWeights_update(hostWnb_data->fc2_w, hostWnb_data->fc2_b, fcp, z);
            cudaMalloc(&weight_data->fc2_b, sizeof(float)*k);
            cudaMalloc(&weight_data->fc2_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->fc2_b, hostWnb_data->fc2_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->fc2_w, hostWnb_data->fc2_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "FC-2 Weights Read Done" << endl;
        }
        if(z==2)
        {
            int k=1000, c=4096, w=1, h=1;
            hostWnb_data->fc3_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->fc3_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostFCWeights_update(hostWnb_data->fc3_w, hostWnb_data->fc3_b, fcp, z);
            cudaMalloc(&weight_data->fc3_b, sizeof(float)*k);
            cudaMalloc(&weight_data->fc3_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->fc3_b, hostWnb_data->fc3_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->fc3_w, hostWnb_data->fc3_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "FC-3 Weights Read Done" << endl;
        }

    }
    fclose(fcp);
}

void read_conv_weights(string weight_file, pW_n_B* hostWnb_data, pW_n_B* weight_data)
{
    cout << "Reading Conv Layers Weights" << endl;
    //float dval;
    int convs = 16;
    //int i, j, m, l, z;
    int z;
    FILE *fp;
    int total_convs_read = 0;
	
	char char_array[weight_file.length() + 1]; 
	strcpy(char_array, weight_file.c_str()); 
	
    fp = fopen(char_array, "r");
    if (fp == NULL) 
    {
		cout<<"File %s absent\n"<<weight_file<<endl;
        exit(1);
    }
    
    for (z = 0; z < 16; z++)
    {
        if (total_convs_read >= convs && convs != -1)
            break;
        if(z==0)
        {
            int k=64, c=3, w=3, h=3;
            hostWnb_data->conv1_1_w = (float*) malloc( k*c*h*w * sizeof(float) );
            hostWnb_data->conv1_1_b = (float*) malloc(k * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv1_1_w, hostWnb_data->conv1_1_b, fp, z);
            for(int i=0;i<w*h;i++)
                cout<<"weight : "<<hostWnb_data->conv1_1_w[i]<<endl;
            for(int i=0;i<w*h;i++)
                cout<<"bias : "<<hostWnb_data->conv1_1_b[i]<<endl;
            CUDA_CALL(cudaMalloc(&weight_data->conv1_1_b, (k * sizeof(float))));
            CUDA_CALL(cudaMemcpy(weight_data->conv1_1_b, hostWnb_data->conv1_1_b, (k * sizeof(float)), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMalloc(&weight_data->conv1_1_w, (k*c*h*w * sizeof(float))));
            CUDA_CALL(cudaMemcpy(weight_data->conv1_1_w, hostWnb_data->conv1_1_w, (k*c*h*w * sizeof(float)), cudaMemcpyHostToDevice));
            cout << "hostWnb_data->conv1_1_w weights are loading done" << endl;
        }
        else if(z==1)
        {
            int k=64, c=64, w=3, h=3;
            hostWnb_data->conv1_2_w = (float*) malloc( k*c*h*w * sizeof(float) );
            hostWnb_data->conv1_2_b = (float*) malloc(k * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv1_2_w, hostWnb_data->conv1_2_b, fp, z);
            cudaMalloc(&weight_data->conv1_2_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv1_2_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv1_2_b, hostWnb_data->conv1_2_b, k*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv1_2_w, hostWnb_data->conv1_2_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv1_2_w weights are loading done" << endl;
        }
        else if (z == 2)
        {
            int k=128, c=64, w=3, h=3;
            hostWnb_data->conv2_1_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv2_1_w = (float*) malloc( k*c*h*w * sizeof(float) );
            hostConvWeights_update(hostWnb_data->conv2_1_w, hostWnb_data->conv2_1_b, fp, z);
            cudaMalloc(&weight_data->conv2_1_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv2_1_w, sizeof(float)*k*c*w*h);
            cudaMemcpy(weight_data->conv2_1_b, hostWnb_data->conv2_1_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv2_1_w, hostWnb_data->conv2_1_w, sizeof(float)*k*c*w*h, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv2_1_w weights are loading done" << endl;
        }
        else if (z == 3)
        {
            int k=128, c=128, w=3, h=3;
            hostWnb_data->conv2_2_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv2_2_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv2_2_w, hostWnb_data->conv2_2_b, fp, z);
            cudaMalloc(&weight_data->conv2_2_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv2_2_w, sizeof(float)*k*c*w*h);
            cudaMemcpy(weight_data->conv2_2_b, hostWnb_data->conv2_2_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv2_2_w, hostWnb_data->conv2_2_w, sizeof(float)*k*c*w*h, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv2_2_w weights are loading done" << endl;

        }
        else if (z == 4)
        {
            int k=256, c=128, w=3, h=3;
            hostWnb_data->conv3_1_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv3_1_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv3_1_w, hostWnb_data->conv3_1_b, fp, z);
            cudaMalloc(&weight_data->conv3_1_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv3_1_w, sizeof(float)*k*c*w*h);
            cudaMemcpy(weight_data->conv3_1_b, hostWnb_data->conv3_1_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv3_1_w, hostWnb_data->conv3_1_w, sizeof(float)*k*c*w*h, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv3_1_w weights are loading done" << endl;
        }
        else if (z == 5)
        {
            int k=256, c=256, w=3, h=3;
            hostWnb_data->conv3_2_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv3_2_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv3_2_w, hostWnb_data->conv3_2_b, fp, z);
            cudaMalloc(&weight_data->conv3_2_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv3_2_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv3_2_b, hostWnb_data->conv3_2_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv3_2_w, hostWnb_data->conv3_2_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv3_2_w weights are loading done" << endl;
        }
        else if (z == 6)
        {
            int k=256, c=256, w=3, h=3;
            hostWnb_data->conv3_3_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv3_3_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv3_3_w, hostWnb_data->conv3_3_b, fp, z);
            cudaMalloc(&weight_data->conv3_3_b, sizeof(float) * k);
            cudaMalloc(&weight_data->conv3_3_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv3_3_b, hostWnb_data->conv3_3_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv3_3_w, hostWnb_data->conv3_3_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv3_3_w weights are loading done" << endl;
        }
        else if (z == 7)
        {
            int k=256, c=256, w=3, h=3;
            hostWnb_data->conv3_4_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv3_4_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv3_4_w, hostWnb_data->conv3_4_b, fp, z);
            cudaMalloc(&weight_data->conv3_4_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv3_4_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv3_4_b, hostWnb_data->conv3_4_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv3_4_w, hostWnb_data->conv3_4_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv3_4_w weights are loading done" << endl;
        }
        else if (z == 8)
        {
            
            int k=512, c=256, w=3, h=3;
            hostWnb_data->conv4_1_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv4_1_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv4_1_w, hostWnb_data->conv4_1_b, fp, z);
            cudaMalloc(&weight_data->conv4_1_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv4_1_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv4_1_b, hostWnb_data->conv4_1_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv4_1_w, hostWnb_data->conv4_1_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv4_1_w weights are loading done" << endl;
            
            
        }
        else if (z == 9)
        {
            
            int k=512, c=512, w=3, h=3;
            hostWnb_data->conv4_2_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv4_2_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv4_2_w, hostWnb_data->conv4_2_b, fp, z);
            cudaMalloc(&weight_data->conv4_2_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv4_2_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv4_2_b, hostWnb_data->conv4_2_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv4_2_w, hostWnb_data->conv4_2_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv4_2_w weights are loading done" << endl;
        }
        else if (z == 10)
        {
            
            int k=512, c=512, w=3, h=3;
            hostWnb_data->conv4_3_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv4_3_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv4_3_w, hostWnb_data->conv4_3_b, fp, z);
            cudaMalloc(&weight_data->conv4_3_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv4_3_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv4_3_b, hostWnb_data->conv4_3_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv4_3_w, hostWnb_data->conv4_3_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv4_3_w weights are loading done" << endl;
        }
        else if (z == 11)
        {
            int k=512, c=512, w=3, h=3;
            hostWnb_data->conv4_4_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv4_4_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv4_4_w, hostWnb_data->conv4_4_b, fp, z);
            cudaMalloc(&weight_data->conv4_4_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv4_4_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv4_4_b, hostWnb_data->conv4_4_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv4_4_w, hostWnb_data->conv4_4_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
    
            cout << "hostWnb_data->conv4_4_w weights are loading done" << endl;
        }
        else if (z == 12)
        {
            int k=512, c=512, w=3, h=3;
            hostWnb_data->conv5_1_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv5_1_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv5_1_w, hostWnb_data->conv5_1_b, fp, z);
            cudaMalloc(&weight_data->conv5_1_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv5_1_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv5_1_b, hostWnb_data->conv5_1_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv5_1_w, hostWnb_data->conv5_1_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv5_1_w weights are loading done" << endl;
        }
        else if (z == 13)
        {
            int k=512, c=512, w=3, h=3;
            hostWnb_data->conv5_2_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv5_2_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv5_2_w, hostWnb_data->conv5_2_b, fp, z);
            cudaMalloc(&weight_data->conv5_2_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv5_2_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv5_2_b, hostWnb_data->conv5_2_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv5_2_w, hostWnb_data->conv5_2_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv5_2_w weights are loading done" << endl;
        }
        else if (z == 14)
        {
            int k=512, c=512, w=3, h=3;
            hostWnb_data->conv5_3_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv5_3_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv5_3_w, hostWnb_data->conv5_3_b, fp, z);
            cudaMalloc(&weight_data->conv5_3_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv5_3_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv5_3_b, hostWnb_data->conv5_3_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv5_3_w, hostWnb_data->conv5_3_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv5_3_w weights are loading done" << endl;
        }
        else if (z == 15)
        {
            int k=512, c=512, w=3, h=3;
            hostWnb_data->conv5_4_b = (float*) malloc(k * sizeof(float));
            hostWnb_data->conv5_4_w = (float*) malloc( k*c*h*w * sizeof(float));
            hostConvWeights_update(hostWnb_data->conv5_4_w, hostWnb_data->conv5_4_b, fp, z);
            cudaMalloc(&weight_data->conv5_4_b, sizeof(float)*k);
            cudaMalloc(&weight_data->conv5_4_w, sizeof(float)*c * h * w *k);
            cudaMemcpy(weight_data->conv5_4_b, hostWnb_data->conv5_4_b, sizeof(float)*k, cudaMemcpyHostToDevice);
            cudaMemcpy(weight_data->conv5_4_w, hostWnb_data->conv5_4_w, sizeof(float)*c * h * w *k, cudaMemcpyHostToDevice);
            cout << "hostWnb_data->conv5_4_w weights are loading done" << endl;
        }
        
        total_convs_read += 1;
    }
    fclose(fp);
}

image readTestImage(string Imagepath)
{
    
    image image_t;
    image_t = load_image(Imagepath, 224, 224, 3);
    return image_t;
    
}

void copying_intermediate_reults(pData* output, pData* InterOut, layerInfo* layer_info)
{
    int conv1_1_out_dim = (layer_info->dim_conv1_1.n * layer_info->dim_conv1_1.c * layer_info->dim_conv1_1.h * layer_info->dim_conv1_1.w);
    cout << "conv1_1_out_dim >>>>>>> :" << layer_info->dim_conv1_1.n <<" <>" << layer_info->dim_conv1_1.c << "<>" << layer_info->dim_conv1_1.h << "<>" << layer_info->dim_conv1_1.w << endl;
    
    //int conv1_1_out_dim = 1 * 64 * 224 * 224;
    InterOut->conv1_1 = (float *)malloc(1 * 64 * 224 * 224 *sizeof(float));
    
    cudaMemcpy(InterOut->conv1_1, output->conv1_1, 1 * 64 * 224 * 224 *sizeof(float), cudaMemcpyDeviceToHost);
    ofstream file;
    file.open ("conv1_1.txt");
    if( !file ) 
    { 
      cerr << "Error: file could not be opened" << endl;
      exit(1);
    }
    for(int i =0; i < conv1_1_out_dim; i++ )
        file << InterOut->conv1_1[i] << endl;
        
    file.close();
}