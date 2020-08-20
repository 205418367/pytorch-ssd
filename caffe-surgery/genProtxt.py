"""
通过caffemodel生成prototxt可能导致信息缺失
"""
from Caffe import caffe_net

def main():
    net=caffe_net.Caffemodel(caffemodel)
    net.save_prototxt(prototxt)

if __name__=="__main__":
    caffemodel="/home/lichen/project/yolov5/pth2caffe/models/yolo5.caffemodel"
    prototxt="/home/lichen/project/yolov5/pth2caffe/models/yolo5.prototxt"
    main()
