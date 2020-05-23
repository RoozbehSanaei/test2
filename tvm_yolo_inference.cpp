#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "tvm/runtime/module.h"
#include "tvm/runtime/registry.h"
#include "tvm/runtime/packed_func.h"
#include <chrono>
#include <fstream>

using namespace std;

double duration;

void getBox(float out[3][6][32][32], int anchors[12],int maskNum, int location[3], int scale, int img_size, float *coordinateBox)
{
    float bx = (location[2] + out[location[0]][0][location[1]][location[2]]) / scale;
    float by = (location[1] + out[location[0]][1][location[1]][location[2]]) / scale;
    float baseW = out[location[0]][2][location[1]][location[2]];
    float baseH = out[location[0]][3][location[1]][location[2]];
    float expW = exp(baseW);
    float expH = exp(baseH);
    float bw = expW * anchors[2*maskNum] / img_size;
    float bh = expH * anchors[2*maskNum+1] / img_size;
    coordinateBox[0] = bx;
    coordinateBox[1] = by;
    coordinateBox[2] = bw;
    coordinateBox[3] = bh;
}

void getBox(float out[3][6][16][16], int anchors[12],int maskNum, int location[3], int scale, int img_size, float *coordinateBox)
{
    float bx = (location[2] + out[location[0]][0][location[1]][location[2]]) / scale;
    float by = (location[1] + out[location[0]][1][location[1]][location[2]]) / scale;
    float baseW = out[location[0]][2][location[1]][location[2]];
    float baseH = out[location[0]][3][location[1]][location[2]];
    float expW = exp(baseW);
    float expH = exp(baseH);
    float bw = expW * anchors[2*maskNum] / img_size;
    float bh = expH * anchors[2*maskNum+1] / img_size;
    coordinateBox[0] = bx;
    coordinateBox[1] = by;
    coordinateBox[2] = bw;
    coordinateBox[3] = bh;
}

class FR_MFN_Deploy{
    
    private:
        void * handle;
    
    public:
        FR_MFN_Deploy(std::string modelFolder)
        {
    
            tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(modelFolder + "/deploy.so");
            std::ifstream json_in(modelFolder + "/deploy.json");
            std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
            json_in.close();
    
            int device_type = kDLCPU;
            int device_id = 0;
            // get global function module for graph runtime
            tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
            this->handle = new tvm::runtime::Module(mod);
    
            //load param
            std::ifstream params_in(modelFolder + "/deploy.params", std::ios::binary);
            std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
            params_in.close();
    
            TVMByteArray params_arr;
            params_arr.data = params_data.c_str();
            params_arr.size = params_data.length();

            tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
            load_params(params_arr);
        }
      
        void forward(cv::Mat inputImage, float (&out1)[3][6][32][32], float (&out2)[3][6][16][16])
        {


            cv::Mat tensor = cv::dnn::blobFromImage(inputImage,1.0/255.0,cv::Size(512,512),cv::Scalar(0),false,false);
            
            //std::cout<<tensor.channels()<<std::endl;
            //std::cout<<tensor.at<uchar>(0,0).size()<<std::endl;
            int numberOfDims = tensor.dims;
            //std::cout<<numberOfDims<<" dimensions,"<<tensor.size[numberOfDims-4]<<','<<tensor.size[numberOfDims-3]<<','<<tensor.size[numberOfDims-2]<<','<<tensor.size[numberOfDims-1]<<std::endl;

            DLTensor* input;
            constexpr int dtype_code = kDLFloat;
            constexpr int dtype_bits = 32;
            constexpr int dtype_lanes = 1;
            constexpr int device_type = kDLCPU;
            constexpr int device_id = 0;
            constexpr int in_ndim = 4;
            const int64_t in_shape[in_ndim] = {1, 1,512, 512};
            int64_t out_shape[4] = {1,18,32,32};
            int64_t out_shape1[4] = {1,18,16,16};
            // const int64_t in_shape[in_ndim] = {1, 256,256, 1};
            // int64_t out_shape[4] = {1,256,256,1};

            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayCopyFromBytes(input,tensor.data,512*512*4);


            tvm::runtime::Module* mod = (tvm::runtime::Module*)handle;
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);

            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();

            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            //tvm::runtime::NDArray res = get_output(0);
            //tvm::runtime::NDArray res;
            DLTensor* res;
            DLTensor* res1;
            int out_ndim = 4;

            TVMArrayAlloc(out_shape,out_ndim,dtype_code,dtype_bits,dtype_lanes,device_type,device_id, &res);
            TVMArrayAlloc(out_shape1,out_ndim,dtype_code,dtype_bits,dtype_lanes,device_type,device_id, &res1);
            get_output(0,res);
            get_output(4,res1);                       
            

            memcpy(out1, res->data, 3*6*32*32*4);
            memcpy(out2, res1->data,3*6*16*16*4);


            TVMArrayFree(input);
            TVMArrayFree(res);
    }

};


float* find_coordinates(std::string fileName, float *coordinateBox){
    cv::Mat A;
    A = cv::imread(fileName,0);
    A.convertTo(A, CV_32FC1);

    std::cout << fileName << std::endl;
    float out1[3][6][32][32] = {0};
    float out2[3][6][16][16] = {0};
    int activeDataLoc[3];
    float maxScore = -999;
    float thresh = 0.25;
    bool isBig = false;
    bool isSmall = false;
    int firstMaskArray[3] = {0,1,2};
    int secondMaskArray[3] = {3,4,5};
    int anchorArray[12] = {14, 14, 16, 16, 18, 18, 21, 21, 26, 26, 35, 35};
    float objectness = 0;
    FR_MFN_Deploy deploy("./tvm_models/darknet");

    auto start = std::chrono::steady_clock::now();
    deploy.forward(A, out1, out2);

    // std::cout<<"Out of tvm"<<std::endl;

    for(int i=0;i<3;i++)
    {
        for(int j=0;j<32;j++)
        {
            for(int k=0;k<32;k++)
            {
                if(out1[i][4][j][k] > maxScore)
                {
                    maxScore = out1[i][4][j][k];
                    activeDataLoc[0] = i;
                    activeDataLoc[1] = j;
                    activeDataLoc[2] = k;
                    isBig = true;
                }   
            }
        }
    }

    for(int i=0;i<3;i++)
    {
        for(int j=0;j<16;j++)
        {
            for(int k=0;k<16;k++)
            {
                if(out2[i][4][j][k] > maxScore)
                {
                    maxScore = out2[i][4][j][k];
                    activeDataLoc[0] = i;
                    activeDataLoc[1] = j;
                    activeDataLoc[2] = k;
                    isSmall = true;
                    isBig = false;
                }
            }
        }
    }

    // std::cout<<"Out of heads"<<std::endl;

    if(maxScore < thresh)
    {

	    auto end = chrono::steady_clock::now();
        return 0;
    }

    // std::cout<<"Out of thresh"<<std::endl;
    // std::cout<<isBig<<std::endl;
    // std::cout<<isSmall<<std::endl;
    if(isBig)
    {
        getBox(out1,anchorArray,secondMaskArray[activeDataLoc[0]],activeDataLoc,32,512,coordinateBox);
        coordinateBox[4] = maxScore;
        
        auto end = chrono::steady_clock::now();
        return coordinateBox;
    }
    else if(isSmall)
    {
        getBox(out2,anchorArray,firstMaskArray[activeDataLoc[0]],activeDataLoc,16,512,coordinateBox);

        // std::cout<<"Out of getbox"<<std::endl;
        // std::cout<<maxScore<<std::endl;
        coordinateBox[4] = maxScore;
        // std::cout<<"Before return"<<std::endl;
        
        auto end = chrono::steady_clock::now();
        duration += chrono::duration_cast<chrono::milliseconds>(end - start).count();
        return coordinateBox;
    }
} 


int main(){
    duration = 0;
    std::string fileName = "examples/IMG3.jpg";
    float coordinateBox[5] = {0};
    for (size_t i = 0; i < 1; i++)
    {
        find_coordinates(fileName, coordinateBox);
        std::cout<<duration/(i+1)<<" ms"<<std::endl;
    }

    std::cout<<"After return"<<std::endl;
    std::cout<<sizeof(coordinateBox)/4<<std::endl;
    for (size_t i = 0; i < sizeof(coordinateBox)/4; i++)
    {
        std::cout<<coordinateBox[i]<<std::endl;
    }
    

    return 0;
}
