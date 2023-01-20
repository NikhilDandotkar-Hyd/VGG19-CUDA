
#include <iostream>
#include <stdexcept>
#include <cudnn.h>
#include "utility.h"

#define BATCH_SIZE 32
#define CHANNELS 3
#define WIDTH 224
#define HEIGHT 224
#define CLASSES 1000
using namespace std;
	
bool train = false,inference = false;


void vgg19_Init(cudnnHandle_t cudnn , inputinfo* input_info, layerInfo* layer_info, pData* output, pW_n_B* weight_data)
{
 
        cout<<"Initializing Graph ..."<<endl;
        input_info->in_dim.n = BATCH_SIZE;
        input_info->in_dim.c = CHANNELS;
        input_info->in_dim.h = HEIGHT;
        input_info->in_dim.w = WIDTH;

        setAllLayers(layer_info,input_info,output,cudnn);
        
    
}

void vgg19_inference(string Imagepath,cudnnHandle_t& cudnn , 
                 inputinfo* input_info, layerInfo* layer_info, pData* output, pW_n_B* weight_data, pW_n_B* hostWnb_data, pData* InterOut)
{
    // Listing All Fills in that Folder
    ListFiles("/raid/Dataset/ILSVRC2017/ILSVRC/Data/CLS-LOC/train/", "", [](const std::string &path) {
        AllFileNames.push_back(path);
        });
    cout << "Listing Test Directory done " << endl;
    
    // Shuffle the Files 
    random_shuffle ( AllFileNames.begin(), AllFileNames.end());
    cout << "Suhuffle The Data done" << endl;
    
    image ImagesList;
    int Imagesize = 1*CHANNELS*HEIGHT*WIDTH *sizeof(float);
    cudaMalloc(&(output->input), Imagesize);
    
    //host out put varible to store 
    float* h_output;
    float prob[BATCH_SIZE][CLASSES];
    int Maxprob_value[BATCH_SIZE];
    h_output = (float *)malloc(BATCH_SIZE*CLASSES*sizeof(float));
    
    //Loading the Weights from text file 
    //read_conv_weights("/home/sunil/Conv_weights_txt.txt", hostWnb_data, weight_data);
    //read_FC_weights("/home/sunil/FC_weights_txt.txt",hostWnb_data, weight_data);
    
    // Reading the 1 Batch of Images  
    //ImagesList = GetBatch(0,32,"/raid/Dataset/ILSVRC2017/ILSVRC/Data/CLS-LOC/train/");
    ImagesList = readTestImage(Imagepath);
    for(int i=0;i<224;i++)
       cout<<"image data : "<<ImagesList.data[i]<<endl;

    // Saving the labels for Testing 
    int Testlabels[labelList.size()];
    copy(labelList.begin(), labelList.end(), Testlabels);
    cout << "32 Batch Array of Labels" << endl;
        
    // Copying Test Images to Device 
    cudaMemcpy(output->input,  ImagesList.data, CHANNELS*WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice);
    
    // Forward Pass Invoking
    computeForwardpass(cudnn, layer_info, input_info, output, weight_data);
     
    copying_intermediate_reults(output, InterOut, layer_info);

    
    // Copying Device to Host pointer 
    cudaMemcpy(h_output, output->out_data, CLASSES*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Saving the Host data to 2D array 
    for(int i =0 ; i < BATCH_SIZE ; i++)
        for (int j = 0; j < CLASSES ; j++)
            prob[i][j] = h_output[i*CLASSES+j];
                
    //int n = sizeof(prob) / sizeof(prob[0]);
    
    // Finding the Maximum Probability value in the Output layer and storing it Pridected array
    string Folder[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) 
    {
  
        Maxprob_value[i] = distance(prob[i], max_element(prob[i], prob[i] + CLASSES));
 
    }
    
    
    // Finding the Common Elements in the Testarray and Predicted Values  
    int n1 = sizeof(Testlabels) / sizeof(Testlabels[0]); 
    int n2 = sizeof(Maxprob_value) / sizeof(Maxprob_value[0]); 
  
    sort(Testlabels, Testlabels + n1); 
    sort(Maxprob_value, Maxprob_value + n2); 
    
    vector<int> v(n1 + n2); 
    vector<int>::iterator it, st; 
    it = set_intersection(Testlabels, Testlabels + n1, Maxprob_value,  Maxprob_value + n2, v.begin()); 
    cout << "\nCommon elements:\n"; 
    for (st = v.begin(); st != it; ++st) 
        cout << *st << ", "; 
    cout << '\n'; 
  
  

        //Maxprob_value = distance(prob, max_element(prob, prob + n));
        //cout << "\n Max Element Index = " << Maxprob_value << endl;
               
        /*for (auto &it : LabelMap) 
        {
            if (it.second == Maxprob_value[i]) 
            {
                Folder[i] = it.first;
                break; 
            }
        }*/
    free(ImagesList.data);
    
}

void vgg19_train(string mainDirectory, int size, cudnnHandle_t cudnn , 
                 inputinfo* input_info, layerInfo* layer_info, backLayerInfo* bak_layer_info, 
				 pData* layer_data, pW_n_B* weight_data, pgradData* grad_weight,
				 int epoch,string optimizer) 
{
    
    ListFiles(mainDirectory, "", [](const std::string &path) {
        AllFileNames.push_back(path);
        });
    cout << "Listing Train Directory done " << endl;
    random_shuffle ( AllFileNames.begin(), AllFileNames.end());
    cout << "Suhuffle The Data done" << endl;
    
    int index = 0; 
    int id = 0;
    float rho = 0.9;
	float learning_rate = 0.01;
    
    image ImagesList;
    int Imagesize = BATCH_SIZE * CHANNELS * HEIGHT * WIDTH * sizeof(float);
    ImagesList.c = CHANNELS;
    ImagesList.h = HEIGHT;
    ImagesList.w = WIDTH;
    ImagesList.bs = BATCH_SIZE;

    cudaMalloc(&(layer_data->input), Imagesize);
	cudaMalloc(&(layer_data->y),ImagesList.bs * sizeof(float));


    
	backLayerDesc(cudnn,input_info,layer_info, bak_layer_info, grad_weight,CLASSES, BATCH_SIZE,optimizer);
    int itr = 0;
	
   
	for (int ii = 0; ii<epoch ;ii++){// Loop for # of epoch
		
		while(true)//for(int jj = 0; jj<10; jj++)// Loop for passing all the batch 
		{
			/*for testing purpose consider 10 batchs only
		     * */
			 if(index>=AllFileNames.size())
				 {
					 break;
				 }
			ImagesList = GetBatch(index,size,"/raid/Dataset/ILSVRC2017/ILSVRC/Data/CLS-LOC/train/");
			
			
			int labels[labelList.size()];
			std::copy(labelList.begin(), labelList.end(), labels);
			
			
			
			cudaMemcpy(layer_data->input,  ImagesList.data, Imagesize, cudaMemcpyHostToDevice);
			cudaMemcpy(layer_data->y,  labels, ImagesList.bs*sizeof(float), cudaMemcpyHostToDevice);
			
			cout << "Image data malloc Done" << endl;
			cout<<"=========================== Iteration ["<<itr<<"]=========================="<<endl;
			itr++;
			computeForwardpass(cudnn, layer_info, input_info, layer_data, weight_data);
			cout << "computeForwardpass Done" << endl;
			
			/*****************************************************
			 * * calculating gradient
			 * ***************************************************/
			cal_gradient(cudnn, layer_info, 
                         layer_data, weight_data,
						 input_info,bak_layer_info,
						 grad_weight, CLASSES, BATCH_SIZE);
			/*****************************************************
			 * * updating weights
			 * ***************************************************/
			update_weight(cudnn,layer_info,weight_data,grad_weight,rho,learning_rate,optimizer);
			 
			  index = index + size;
			  id = id +1;
			  free(ImagesList.data);
		}//end of all batch		
	}//end of all epoch
	
	/*********************************************************
	 * * code to write weights in file
	 ********************************************************/	
}
    

int main(int argc, char **argv)
{
  
	string weight_path;
	string test_img_path;
	string dataset_path;
	string weight_type;
	string mode;
	string optimizer;
	
	dataset_path = "/raid/Dataset/ILSVRC2017/ILSVRC/Data/CLS-LOC/train/";

	
	int epoch = 1;
	try{
	   string mode = argv[1];
	   weight_type = argv[2];
	   if(mode.compare("train") == 0){
		   train = true;
		   if(weight_type.compare("random") == 0){
			   epoch = atoi(argv[3]);
			   optimizer = argv[4];
		   }
		   else if(weight_type.compare("load") == 0){
			   epoch = atoi(argv[4]);
			   optimizer = argv[5];
		   }
	   }
	   else if(mode.compare("predict") == 0){
		   inference = true;
		   weight_path = argv[3];
		   string file_type = argv[4];
		   test_img_path = argv[5];
	   }
	   else{
		   cout<<"Invalid command line"<<endl;
		   std::exit(1); 
	   }
	}
	catch(int e){
		cout<<"Please enter valid commands"<<endl;
	}
	
	cout<<"epoch : "<<epoch<<endl;
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));
    
    inputinfo* input_info;
    input_info = (inputinfo*)malloc(sizeof(inputinfo));
    
    layerInfo* layer_info;
    layer_info = (layerInfo*)malloc(sizeof(layerInfo));
    
    pData* layer_data;
    layer_data = (pData*)malloc(sizeof(pData));
        
    pW_n_B* weight_data;
    weight_data = (pW_n_B*)malloc(sizeof(pW_n_B));

    pW_n_B* hostWnb_data;
    hostWnb_data = (pW_n_B*)malloc(sizeof(pW_n_B));
	
	pData* InterOut;
    InterOut = (pData*)malloc(sizeof(pData));
	
    
    /****************************************************
	* Initializing the Graph
	* **************************************************/
	cout<<"================== Graph ==================="<<endl;
	vgg19_Init(cudnn, input_info, layer_info, layer_data, weight_data);
	printoutputdimensions(layer_info, input_info);
	cout<<"============================================"<<endl;
    
	if(weight_type.compare("random") == 0){
	
    InitializeAllRandomWeights(hostWnb_data,weight_data);
    cout << "Initialization of Random Weights and Bias is done." << endl;
	}
	else if(weight_type.compare("load") == 0)
	{
		/****************************************************
		 * loading data from the file
		 * **************************************************/
		read_conv_weights("/raid/AI_COMP/Conv_weights_txt.txt", hostWnb_data, weight_data);
        read_FC_weights("/raid/AI_COMP/FC_weights_txt.txt",hostWnb_data, weight_data);
		
	}
    
    if(train){
		 
		 backLayerInfo* bak_layer_info;
		 bak_layer_info = (backLayerInfo*)malloc(sizeof(backLayerInfo));
		 
		 pgradData* grad_weight;
		 grad_weight = (pgradData*)malloc(sizeof(pgradData));
		 
         vgg19_train(dataset_path, BATCH_SIZE, cudnn, input_info, 
		             layer_info,bak_layer_info,
					 layer_data,weight_data,grad_weight,
					 epoch,optimizer);
		
		
	}
	else if (inference){
	/*********************************************************
	 * inference 
	 * ********************************************************/
        vgg19_inference(test_img_path, cudnn, input_info, layer_info, layer_data, weight_data,hostWnb_data, InterOut);
    }
	
	cout<<"done."<<endl;
    destroy(layer_info,layer_data);
	cudnnDestroy(cudnn);
	return 0;
}
