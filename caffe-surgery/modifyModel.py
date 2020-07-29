'''
实现的功能有：
    1.在caffemodel中添加新的层和参数
    2.修改caffemodel中任意层的参数
    3.删除caffemodel中任意层及其参数
'''
from Caffe import caffe_net
from Caffe import layer_param
import numpy as np
import json
import argparse
import os,sys
              
  
def add_new_layer(json_dict):
    currOPTop     = json_dict["currOPTop"]
    layerType    = json_dict["layerType"]
    prevOPName  = json_dict["prevOPName"]
    nextOPName = json_dict["nextOPName"]
    
    net=caffe_net.Caffemodel(caffemodel_file)
    layer = net.get_layer_by_name(prevOPName)
    prevOPBottom = layer.top[0] 
    layer_params = layer_param.Layer_param(currOPName,layerType,currOPTop,prevOPBottom,phase=1)
    
    if layerType.endswith("Convolution"):
       ke_h = json_dict["kernel_h"]
       ke_w = json_dict["kernel_w"]
       num_out = json_dict["num_output"]
       str_h = json_dict["stride_h"]
       str_w = json_dict["stride_w"]
       group = json_dict["group"]
       bias = json_dict["bias_term"]
       dila = json_dict["dilation"]
       p_h = json_dict["pad_h"]
       pad_w = json_dict["pad_w"]
       weight  = np.array(json_dict["weight"]).astype(np.float32)
    
       # 卷积层后面必须是BN层
       layerOP = net.get_layer_by_name(nextOPName)
       assert layerOP.type.endswith("BatchNorm")
       for blob in layerOP.blobs:
           shape=blob.shape.dim
           assert num_out == shape[0]
           break
       layer_params.conv_param(kernel_h=ke_h,kernel_w=ke_w,num_output=num_out,stride_h=str_h,stride_w=str_w)
       net.add_layer_with_data(layer_params, datas=weight, bottom=prevOPBottom)
    elif layerType.endswith("Softmax"):
       axis = json_dict["num_axis"]
       weight = json_dict["weight"]
       layer_params.softmax_param(num_axis=axis)
       net.add_layer_with_data(layer_params, datas=weight, bottom=prevOPBottom)
    elif layerType.endswith("Permute"):
       weight = json_dict["weight"]
       layer_params.permute_param()
       net.add_layer_with_data(layer_params, datas=weight, bottom=prevOPBottom)
    elif layerType.endswith("Reshape"):
       weight = json_dict["weight"]
       shape = json_dict["shape"]
       layer_params.reshape_param(shape=shape)
       net.add_layer_with_data(layer_params, datas=weight, bottom=prevOPBottom)

    if nextOPName != "":
       layer = net.get_layer_by_name(nextOPName)
       layer.bottom[0] = currOPTop 
    return net


def set_layer_data(json_dict):
    weight  = np.array(json_dict["weight"]).astype(np.float32)
    net=caffe_net.Caffemodel(caffemodel_file)
    net.set_layer_data(name, weight)
    return net


def remove_layer_by_name(json_dict):
    prevOPName = json_dict["prevOPName"]
    nextOPName = json_dict["nextOPName"]
    net=caffe_net.Caffemodel(caffemodel_file)
    prevlayer = net.get_layer_by_name(prevOPName)
    prevOPTop = prevlayer.top[0]
    net.remove_layer_by_name(currOPName)
    
    if nextOPName != "":
       layer = net.get_layer_by_name(nextOPName)
       layer.bottom[0] = prevOPTop 
    return net


def main(args):
    jsonFile = open(args.input, 'r', encoding='utf-8')
    json_dict = json.load(jsonFile)
    global caffemodel_file,prototxt_file,changed_caffemodel,currOPName
    modifierType    = json_dict["modifierType"]
    caffemodel_file    = json_dict["input_caffemodel"]
    prototxt_file      = json_dict["output_prototxt"]
    changed_caffemodel = json_dict["output_caffemodel"]
    currOPName    = json_dict["currOPName"]

    if modifierType.endswith("add_new_layer"):
       net = add_new_layer(json_dict)
    elif modifierType.endswith("remove_layer_by_name"):
       net = remove_layer_by_name(json_dict)
    elif modifierType.endswith("set_layer_data"):
       net = set_layer_data(json_dict)

    net.save_prototxt(prototxt_file)
    net.save(changed_caffemodel)
 

if __name__=='__main__':
   parser = argparse.ArgumentParser(description="modeify caffemodel")
   parser.add_argument("--input", default="", type=str, required=True)
   args = parser.parse_args()
   main(args)
   print("#### succ ####")
        






