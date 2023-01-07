# yolov8 TensorRT

## onnx model

pytorch model is from here  [https://github.com/ultralytics/assets/releases/tag/v0.0.0](https://github.com/ultralytics/assets/releases/tag/v0.0.0)

[yolov8 _onnx_model](https://pan.baidu.com/s/13JqhFB1uWhqzz_zSHAeO7Q)   提取码：1tpm

## How to Run, yolov8n as example

1. Modify the tensorrt cuda opencv path in CMakeLists.txt

   ```
   #cuda 
   include_directories(/mnt/Gu/softWare/cuda-11.0/targets/x86_64-linux/include)
   link_directories(/mnt/Gu/softWare/cuda-11.0/targets/x86_64-linux/lib)

   #tensorrt 
   include_directories(/mnt/Gpan/tensorRT/TensorRT-8.2.0.6/include/)
   link_directories(/mnt/Gpan/tensorRT/TensorRT-8.2.0.6/lib/)
   ```
2. build

   ```
   1. mkdir build
   2. cd build
   3. cmake ..
   4. make

   ```
3. onnx  to tensorrt model

   ```
   ./onnx2trt/onnx2trt  ../onnx_model/yolov8n.onnx ./yolov8n.trt  1

   ```
4. inference

   ```
   ./yolov8 ./yolov8n.trt  ../samples/
   ```

   The results are saved in the build folder.

   ![image](result/zidane.jpg)

## contact

Tencent qq group:  871797331
