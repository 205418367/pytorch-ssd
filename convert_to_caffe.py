import sys
import torch
from torch.autograd import Variable
from torchvision.models.alexnet import alexnet
from pth2caffe import pytorch_to_caffe

import sys
import torch.onnx
from vision.ssd.config.fd_config import define_img_size

input_img_size = 320
define_img_size(input_img_size)
#from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd

name='ultra-ssd'
net_type = "slim"
def main():
    if net_type == 'slim':
        model_path = "models/pretrained/version-slim-320.pth"
        net = create_mb_tiny_fd(2, is_test=True)
    elif net_type == 'RFB':
        model_path = "models/pretrained/version-RFB-320.pth"
        net = create_Mb_Tiny_RFB_fd(2, is_test=True)
    else:
        print("unsupport network type.")
        sys.exit(1)

    net.load(model_path)
    net.eval()
    net.to("cpu")

    dummy_input=torch.ones([1,3,240,320])
    pytorch_to_caffe.trans_net(net,dummy_input,name)
    pytorch_to_caffe.save_prototxt('pth2caffe/models/{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('pth2caffe/models/{}.caffemodel'.format(name))

if __name__=='__main__':
    main()
