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
#include "preprocess.h"
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

struct Object
{
    cv::Rect_<float> rect; 
    int label;
    float prob;
};


static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}



int find_max(float *prob,int num) //找到类别
{
    int max= 0;
    for(int i=1; i<num; i++)
    {
        if (prob[max]<prob[i])
         max = i;
    }

    return max;

}


static void generate_yolo_proposals(float *feat_blob, float prob_threshold,
                                     std::vector<Object> &objects,int OUTPUT_CANDIDATES) {
  const int num_class = NUM_CLASSES;  
  const int ckpt=0  ; 

  const int num_anchors = OUTPUT_CANDIDATES;

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    // const int basic_pos = anchor_idx * (num_class + 5 + 1);
    // float box_objectness = feat_blob[basic_pos + 4];

    // int cls_id = feat_blob[basic_pos + 5];
    // float score = feat_blob[basic_pos + 5 + 1 + cls_id];
    // score *= box_objectness;


    const int basic_pos = anchor_idx * (num_class + 4 + ckpt); //5代表 x,y,w,h,object_score  ckpt代表5个关键点 每个关键点3个数据
    // float box_objectness = feat_blob[basic_pos + 4];

    // int cls_id = find_max(&feat_blob[basic_pos +5+ckpt],num_class);   //找到类别v5
    int cls_id = find_max(&feat_blob[basic_pos +4],num_class);   //v7
    // float score = feat_blob[basic_pos + 5 +8 + cls_id]; //v5
    float score = feat_blob[basic_pos + 4 + cls_id];  //v7
    // score *= box_objectness; 


    if (score > prob_threshold) {
      // yolox/models/yolo_head.py decode logic
      float x_center = feat_blob[basic_pos + 0];
      float y_center = feat_blob[basic_pos + 1];
      float w = feat_blob[basic_pos + 2];
      float h = feat_blob[basic_pos + 3];
      float x0 = x_center - w * 0.5f;
      float y0 = y_center - h * 0.5f;
      
    //   float *landmarks=&feat_blob[basic_pos +5]; //v5
    float *landmarks=&feat_blob[basic_pos +5+num_class];

      Object obj;
      obj.rect.x = x0;
      obj.rect.y = y0;
      obj.rect.width = w;
      obj.rect.height = h;
      obj.label = cls_id;
      obj.prob = score;
      objects.push_back(obj);
    }
  }
}

static void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h,int OUTPUT_CANDIDATES,int top,int left) {
        std::vector<Object> proposals;
        generate_yolo_proposals(prob,  BBOX_CONF_THRESH, proposals,OUTPUT_CANDIDATES);
        qsort_descent_inplace(proposals);
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);
        int count = picked.size();
        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x-left) / scale;
            float y0 = (objects[i].rect.y-top) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width-left) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height-top) / scale;
    
            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}

const float color_list[5][3] =
{
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {0, 255, 255},
    {255,255,0},
};


void doInference_cu(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize,int OUTPUT_SIZE) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
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

   

    static float* prob = new float[output_size];


    // std::string imgPath ="/mnt/Gpan/Mydata/pytorchPorject/Chinese_license_plate_detection_recognition/imgs";
    std::string input_image_path=argv[2];
     std::string imgPath=argv[2];
    std::vector<std::string> imagList;
    std::vector<std::string>fileType{"jpg","png"};
    readFileList(const_cast<char *>(imgPath.c_str()),imagList,fileType);
    double sumTime = 0;
    int index = 0;
    for (auto &input_image_path:imagList) 
    {
        
        cv::Mat img = cv::imread(input_image_path);
          double begin_time = cv::getTickCount();
         float *buffer_idx = (float*)buffers[inputIndex];
        size_t size_image = img.cols * img.rows * 3;
        size_t size_image_dst = INPUT_H * INPUT_W * 3;
        memcpy(img_host, img.data, size_image);
       
        CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
        preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);
        double time_pre = cv::getTickCount();
        double time_pre_=(time_pre-begin_time)/cv::getTickFrequency()*1000;
        // std::cout<<"preprocessing time is "<<time_pre_<<" ms"<<std::endl;
      
        doInference_cu(*context_det,stream, (void**)buffers,prob,1,output_size);

        float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
        int unpad_w = r * img.cols;
        int unpad_h = r * img.rows;
        int left = (INPUT_W-unpad_w)/2;
        int top = (INPUT_H-unpad_h)/2; 
        int img_w = img.cols;
        int img_h = img.rows;
        float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
        
        std::vector<Object> objects;
        decode_outputs(prob, objects, scale, img_w, img_h,OUTPUT_CANDIDATES,top,left);
        std::cout<<input_image_path<<" ";
        
        for (int i = 0; i<objects.size(); i++)
        {
            cv::rectangle(img, objects[i].rect, cv::Scalar(0,255,0), 2);
            cv::putText(img, std::to_string((int)objects[i].label), cv::Point(objects[i].rect.x, objects[i].rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
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
    delete [] prob;
    return 0;
}
