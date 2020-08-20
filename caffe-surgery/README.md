# caffe-surgery

## Notification
1. 使用该工具前必须用自己编译的caffe.proto替换Caffe目录的caffe.proto并重新编译 
```
protoc --python_out ./ caffe.proto 
```
2. 或者用caffe/python/caffe/proto/caffe_pb2.py替换
3. 通过caffemodel生成prototxt可能会导致信息缺失，尤其是那些没有参数的层
4. prototxt是存放索引、caffemodel是存放参数，在prototxt可以添加无参数的层

## Reference
* [Caffe](https://github.com/xxradon/PytorchToCaffe/tree/master/Caffe)
