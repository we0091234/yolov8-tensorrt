# yolov8 TensorRT

The Pytorch implementation is [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).

## onnx model

step1. install yolov8

```
 pip install ultralytics
```

step2. download yolov8 model from [https://github.com/ultralytics/assets/releases](https://github.com/ultralytics/assets/releases)

step3. convert yolov8 model  to onnx

```
yolo mode=export model=yolov8n.pt format=onnx simplify=True
```

or you can download  onnx model from here [z16b](https://pan.baidu.com/s/1KzJ3-15LrPnWjavnqeWsTg)

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
