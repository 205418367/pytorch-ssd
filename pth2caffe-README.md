## pytorch-ssd
该教程将使用pth2caffe工具完成下面三件事情：
1. 如何把pth模型转成caffe模型 
2. 如何在训练好的模型中添加和删除层 
3. 测试修改后caffe模型的正确性 

### Dependencies
* Caffe 
* Python3.6 
* Pytorch1.2 
* CUDA10.0 + CUDNN7.6 

### pth模型转成caffe模型 
修改convert_to_caffe.py，具体看代码
```
python convert_to_caffe.py
```
此时，在pth2caffe/models/目录下生成caffe模型
添加层和caffe模型测试与onnx2caffe工具过程相似
