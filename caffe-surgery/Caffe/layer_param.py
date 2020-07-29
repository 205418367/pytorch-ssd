from __future__ import absolute_import
from . import caffe_pb2 as pb
import numpy as np

def pair_process(item,strict_one=True):
    if hasattr(item,'__iter__'):
        for i in item:
            if i!=item[0]:
                if strict_one:
                    raise ValueError("number in item {} must be the same".format(item))
                else:
                    print("IMPORTANT WARNING: number in item {} must be the same".format(item))
        return item[0]
    return item

def pair_reduce(item):
    if hasattr(item,'__iter__'):
        for i in item:
            if i!=item[0]:
                return item
        return [item[0]]
    return [item]

class Layer_param():
    def __init__(self,name='',type='',top='',bottom='',phase=0):
        self.param=pb.LayerParameter()
        self.name=self.param.name=name
        self.type=self.param.type=type
        self.phase=self.param.phase=phase
       
        self.top=self.param.top
        self.top.extend([top])
        self.bottom=self.param.bottom
        self.bottom.extend([bottom])
  
    def fc_param(self, num_output, weight_filler='xavier', bias_filler='constant',has_bias=True):
        if self.type != 'InnerProduct':
            raise TypeError('the layer type must be InnerProduct if you want set fc param')
        fc_param = pb.InnerProductParameter()
        fc_param.num_output = num_output
        fc_param.weight_filler.type = weight_filler
        fc_param.bias_term = has_bias
        if has_bias:
            fc_param.bias_filler.type = bias_filler
        self.param.inner_product_param.CopyFrom(fc_param)

    def conv_param(self,kernel_h,kernel_w,num_output,stride_h,stride_w,bias_term=False,dilation=1,pad_h=1,pad_w=1,group=1):
        if self.type not in ['Convolution','Deconvolution']:
            raise TypeError('the layer type must be Convolution or Deconvolution if you want set conv param')
        conv_param=pb.ConvolutionParameter()
        conv_param.bias_term=bias_term
        conv_param.dilation.extend([dilation])
        conv_param.kernel_h=kernel_h
        conv_param.kernel_w=kernel_w
        conv_param.num_output=num_output
        conv_param.pad_h=pad_h
        conv_param.pad_w=pad_w
        conv_param.stride_h=stride_h
        conv_param.stride_w=stride_w
        conv_param.group=group
        self.param.convolution_param.CopyFrom(conv_param)

    def softmax_param(self, num_axis):
        if self.type not in ['Softmax']:
            raise TypeError('the layer type must be Softmax if you want set Softmax param')
        softmax_param=pb.SoftmaxParameter()
        softmax_param.axis=num_axis  
        self.param.softmax_param.CopyFrom(softmax_param)

    def permute_param(self):
        if self.type not in ['Permute']:
            raise TypeError('the layer type must be Permute if you want set Permute param')
        permute_param=pb.PermuteParameter()  
        self.param.permute_param.CopyFrom(permute_param)

    def reshape_param(self, shape):
        if self.type not in ['Reshape']:
            raise TypeError('the layer type must be Reshape if you want set Reshape param')
        resh_param=pb.ReshapeParameter()
        self.param.reshape_param.CopyFrom(resh_param)

    def norm_param(self, eps):
        """
        add a conv_param layer if you spec the layer type "Convolution"
        Args:
            num_output: a int
            kernel_size: int list
            stride: a int list
            weight_filler_type: the weight filer type
            bias_filler_type: the bias filler type
        Returns:
        """
        l2norm_param = pb.NormalizeParameter()
        l2norm_param.across_spatial = False
        l2norm_param.channel_shared = False
        l2norm_param.eps = eps
        self.param.norm_param.CopyFrom(l2norm_param)

    def pool_param(self,type='MAX',kernel_size=2,stride=2,pad=None, ceil_mode = True):
        pool_param=pb.PoolingParameter()
        pool_param.pool=pool_param.PoolMethod.Value(type)
        pool_param.kernel_size=pair_process(kernel_size)
        pool_param.stride=pair_process(stride)
        pool_param.ceil_mode=ceil_mode
        if pad:
            if isinstance(pad,tuple):
                pool_param.pad_h = pad[0]
                pool_param.pad_w = pad[1]
            else:
                pool_param.pad=pad
        self.param.pooling_param.CopyFrom(pool_param)

    def batch_norm_param(self,use_global_stats=0,moving_average_fraction=None,eps=None):
        bn_param=pb.BatchNormParameter()
        bn_param.use_global_stats=use_global_stats
        if moving_average_fraction:
            bn_param.moving_average_fraction=moving_average_fraction
        if eps:
            bn_param.eps = eps
        self.param.batch_norm_param.CopyFrom(bn_param)

    def upsample_param(self,size=None, scale_factor=None):
        upsample_param=pb.UpsampleParameter()
        if scale_factor:
            if isinstance(scale_factor,int):
                upsample_param.scale = scale_factor
            else:
                upsample_param.scale_h = scale_factor[0]
                upsample_param.scale_w = scale_factor[1]
        if size:
            if isinstance(size,int):
                upsample_param.upsample_h = size
            else:
                upsample_param.upsample_h = size[0] * scale_factor
                upsample_param.\
                    upsample_w = size[1] * scale_factor
        self.param.upsample_param.CopyFrom(upsample_param)

    def add_data(self,*args):
        del self.param.blobs[:]
        for data in args:
            new_blob = self.param.blobs.add()
            for dim in data.shape:
                new_blob.shape.dim.append(dim)
            new_blob.data.extend(data.flatten().astype(float))

    def set_params_by_dict(self,dic):
        pass

    def copy_from(self,layer_param):
        pass

def set_enum(param,key,value):
    setattr(param,key,param.Value(value))
