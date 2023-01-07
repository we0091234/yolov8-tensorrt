#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "include/utils.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#define MAX_OBJECTS 1024
#define NUM_BOX_ELEMENT 7
#define MAX_IMAGE_INPUT_SIZE_THRESH 5000 * 5000


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.25

using namespace nvinfer1;

static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int NUM_CLASSES = 80;  //80类


const char* INPUT_BLOB_NAME = "images"; //onnx 输入  名字
const char* OUTPUT_BLOB_NAME = "output0"; //onnx 输出 名字
static Logger gLogger;

struct affine_matrix  //前处理仿射变换矩阵和逆矩阵
{
    float i2d[6];   //仿射变换正矩阵
    float d2i[6];   //仿射变换逆矩阵

};

struct bbox 
{
     float x1,x2,y1,y2;
     float score;
     int label;
};

void preprocess_kernel_img(uint8_t* src, int src_width, int src_height,
                           float* dst, int dst_width, int dst_height,
                           float *d2i,cudaStream_t stream);       //letter_box 前处理 用cuda

 void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes,float confidence_threshold, 
        float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream
    );                                           //后处理 cuda

void nms_kernel_invoker(
        float* parray, float nms_threshold, int max_objects, cudaStream_t stream
    );                            //nums cuda

void affine_project(float *d2i,float x,float y ,float *ox,float *oy)  //通过仿射变换逆矩阵返回原图上的坐标
{
  *ox = x*d2i[0]+y*d2i[1]+d2i[2];
  *oy = x*d2i[3]+y*d2i[4]+d2i[5];
}

void get_affine_martrix(affine_matrix &afmt,cv::Size &to,cv::Size &from)  //计算放射变换的正矩阵和逆矩阵
{
    float scale= std::min(to.width/(float)from.width,to.height/(float)from.height);
    afmt.i2d[0]=scale;
    afmt.i2d[1]=0;
    afmt.i2d[2]=(-scale*from.width+to.width)*0.5;
    afmt.i2d[3]=0;
    afmt.i2d[4]=scale;
    afmt.i2d[5]=(-scale*from.height+to.height)*0.5;
    cv::Mat  cv_i2d(2,3,CV_32F,afmt.i2d);
    cv::Mat  cv_d2i(2,3,CV_32F,afmt.d2i);
    cv::invertAffineTransform(cv_i2d,cv_d2i);         //通过opencv获取仿射变换逆矩阵
    memcpy(afmt.d2i,cv_d2i.ptr<float>(0),sizeof(afmt.d2i));
}

int main(int argc, char** argv)
 {
    cudaSetDevice(DEVICE);
    char *trtModelStreamDet{nullptr};
    size_t size{0};
    const std::string engine_file_path {argv[1]};  
    std::ifstream file(engine_file_path, std::ios::binary);

    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStreamDet = new char[size];
        assert(trtModelStreamDet);
        file.read(trtModelStreamDet, size);
        file.close();
    }

   

    //det模型trt初始化
    IRuntime* runtime_det = createInferRuntime(gLogger);
    assert(runtime_det != nullptr);
    ICudaEngine* engine_det = runtime_det->deserializeCudaEngine(trtModelStreamDet, size);
    assert(engine_det != nullptr); 
    IExecutionContext* context_det = engine_det->createExecutionContext();
    assert(context_det != nullptr);
    delete[] trtModelStreamDet;

  

    float *buffers[2];
    const int inputIndex = engine_det->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine_det->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device

    float *affine_matrix_d2i_host= nullptr;  
    float *affine_matrix_d2i_device=nullptr;

    float *decode_ptr_host=nullptr;
    float *decode_ptr_device=nullptr;

    CHECK(cudaMallocHost(&affine_matrix_d2i_host,sizeof(float)*6));  //仿射变换逆矩阵数组长度是6
    CHECK(cudaMalloc(&affine_matrix_d2i_device,sizeof(float)*6)) ; 

    decode_ptr_host = new float[1+MAX_OBJECTS*NUM_BOX_ELEMENT];

    CHECK(cudaMalloc(&decode_ptr_device,sizeof(float)*(1+MAX_OBJECTS*NUM_BOX_ELEMENT)));

    
   

    auto out_dims = engine_det->getBindingDimensions(1);
    auto output_size = 1;
    int OUTPUT_CANDIDATES = out_dims.d[1];

       for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }


    CHECK(cudaMalloc((void**)&buffers[inputIndex],  3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc((void**)&buffers[outputIndex], output_size * sizeof(float)));


     // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    // prepare input data cache in pinned memory 
    CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

   




    // std::string imgPath ="/mnt/Gpan/Mydata/pytorchPorject/Chinese_license_plate_detection_recognition/imgs";
    std::string input_image_path=argv[2];
     std::string imgPath=argv[2];
    std::vector<std::string> imagList;
    std::vector<std::string>fileType{"jpg","png"};
    readFileList(const_cast<char *>(imgPath.c_str()),imagList,fileType);
    double sumTime = 0;
    int index = 0;
    cv::Size to(INPUT_W,INPUT_H);
    for (auto &input_image_path:imagList) 
    {
        affine_matrix afmt;
        cv::Mat img = cv::imread(input_image_path);
        double begin_time = cv::getTickCount();
        cv::Size from(img.cols,img.rows);
        get_affine_martrix(afmt,to,from);

        memcpy(affine_matrix_d2i_host,afmt.d2i,sizeof(afmt.d2i));
        CHECK(cudaMemcpyAsync(affine_matrix_d2i_device,affine_matrix_d2i_host,sizeof(afmt.d2i),cudaMemcpyHostToDevice,stream));
        
        float *buffer_idx = (float*)buffers[inputIndex];
        size_t size_image = img.cols * img.rows * 3;
        size_t size_image_dst = INPUT_H * INPUT_W * 3;
        memcpy(img_host, img.data, size_image);

        CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
        preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H,affine_matrix_d2i_device, stream);
        double time_pre = cv::getTickCount();
        double time_pre_=(time_pre-begin_time)/cv::getTickFrequency()*1000;
        (*context_det).enqueueV2((void**)buffers, stream, nullptr);
        float *predict = buffers[outputIndex];
        CHECK(cudaMemsetAsync(decode_ptr_device,0,sizeof(int),stream));
        decode_kernel_invoker(predict,OUTPUT_CANDIDATES,NUM_CLASSES,BBOX_CONF_THRESH,affine_matrix_d2i_device,decode_ptr_device,MAX_OBJECTS,stream);
        nms_kernel_invoker(decode_ptr_device, NMS_THRESH, MAX_OBJECTS, stream);//cuda nms
        CHECK(cudaMemcpyAsync(decode_ptr_host,decode_ptr_device,sizeof(float)*(1+MAX_OBJECTS*NUM_BOX_ELEMENT),cudaMemcpyDeviceToHost,stream));
        cudaStreamSynchronize(stream);
      
        std::cout<<input_image_path<<" ";

        std::vector<bbox> boxes;
               
        int boxes_count=0;
        int count = std::min((int)*decode_ptr_host,MAX_OBJECTS);
        

         for (int i = 0; i<count;i++)
        {
           int basic_pos = 1+i*NUM_BOX_ELEMENT;
           int keep_flag= decode_ptr_host[basic_pos+6];
           if (keep_flag==1)
           {
             boxes_count+=1;
             bbox  box;
             box.x1 =  decode_ptr_host[basic_pos+0];
             box.y1 =  decode_ptr_host[basic_pos+1];
             box.x2 =  decode_ptr_host[basic_pos+2];
             box.y2 =  decode_ptr_host[basic_pos+3];
             box.score=decode_ptr_host[basic_pos+4];
             box.label=decode_ptr_host[basic_pos+5];
             boxes.push_back(box);
           }
        }

         for (int i = 0; i<boxes_count; i++)
        {
            cv::Rect roi_area(boxes[i].x1,boxes[i].y1,boxes[i].x2-boxes[i].x1,boxes[i].y2-boxes[i].y1);
            cv::rectangle(img, roi_area, cv::Scalar(0,255,0), 2);
              cv::putText(img, std::to_string((int)boxes[i].label), cv::Point(boxes[i].x1, boxes[i].y1 - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }

          double end_time = cv::getTickCount();
          auto time_gap = (end_time-begin_time)/cv::getTickFrequency()*1000;
        std::cout<<"  time_gap: "<<time_gap<<"ms ";
         if (index)
            {
                sumTime+=time_gap;
            }
        std::cout<<std::endl;
        index+=1;

        int pos = input_image_path.find_last_of("/");
        std::string image_name = input_image_path.substr(pos+1);
        cv::imwrite(image_name,img);
    }

   
 
    // destroy the engine
    std::cout<<"averageTime:"<<(sumTime/(imagList.size()-1))<<"ms"<<std::endl;
    context_det->destroy();
    engine_det->destroy();
    runtime_det->destroy();
 
    cudaStreamDestroy(stream);
    CHECK(cudaFree(img_device));
    CHECK(cudaFreeHost(img_host));
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaFree(decode_ptr_device));
    CHECK(cudaFree(affine_matrix_d2i_device));
    CHECK(cudaFreeHost(affine_matrix_d2i_host));

    delete [] decode_ptr_host;
    return 0;
}
