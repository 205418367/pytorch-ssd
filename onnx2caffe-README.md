## pytorch-ssd
该教程将使用onnx2caffe工具完成下面三件事情：
1. 如何把pth模型转成caffe模型 
2. 如何在训练好的模型中添加和删除层 
3. 测试修改后caffe模型的正确性 

### Dependencies
* Caffe 
* Python3.6 
* Pytorch1.2 
* CUDA10.0 + CUDNN7.6 

### 一、pth模型转成caffe模型

#### 1.设置convert_to_onnx.py中的net_type和model_path
```
net_type = "RFB"  # or slim
model_path = f"models/onnx/RFB-test.onnx"
```

#### 2.pth 模型转onnx模型
```
$ cd pytorch-ssd/
$ python convert_to_onnx.py
```
此时，在models/onnx/目录下生成一个RFB-test.onnx模型

#### 3.onnx模型转caffe模型
~~~
$ pip install onnx-simplifier
$ cd pytorch-ssd/models/onnx/
$ python -m onnxsim RFB-test.onnx RFB-test-simplifier.onnx
$ cd pytorch-ssd/onnx2caffe/
# 设置 convertCaffe.py中相关路径
    onnx_path = "../models/onnx/RFB-test-simplifier.onnx"
    prototxt_path = "./models/RFB-320.prototxt"
    caffemodel_path = "./models/RFB-320.caffemodel"
$ python convertCaffe.py
~~~
此时，pytorch-ssd/onnx2caffe/models目录下生成caffe模型

### 二、训练好的模型中添加和删除层

#### 1. 以在caffe模型中添加softmax层为例
```
在pytorch-ssd/vision/ssd/ssd.py 94行添加
confidences = F.softmax(confidences, dim=2)
```
然后在按照 pth模型转成caffe模型 的步骤生成caffe模型
此时，可以得到包含softmax层的caffe模型

#### 2.以在caffe模型中删除softmax层为例
```
在pytorch-ssd/vision/ssd/ssd.py 94行注释
confidences = F.softmax(confidences, dim=2)
```
然后在按照 pth模型转成caffe模型 的步骤生成caffe模型
此时，可以得到无softmax层的caffe模型

### 三、测试修改后caffe模型的正确性

生成caffe模型后，修改pytorch-ssd/onnx2caffe/caffe_inference.py中caffemodel和prototxt路径
```
python caffe_inference.py
```
如果运行正常，即可得知caffe模型是正确的！
