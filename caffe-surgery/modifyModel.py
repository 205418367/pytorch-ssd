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
                
def add_new_layer(json_dict):
    top     = json_dict["top"]
    layer    = json_dict["type"]
    bottom  = json_dict["bottom"]
    nextOP = json_dict["nextOP"]
    
    net=caffe_net.Caffemodel(caffemodel_file)
    layer_params = layer_param.Layer_param(name,layer,top,bottom,phase=1)
    
    if layer.endswith("Convolution"):
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
    
       layerOP = net.get_layer_by_name(nextOP)
       for blob in layerOP.blobs:
           shape=blob.shape.dim
           assert layerOP.type.endswith("Scale")
           assert num_out == shape[0]
       layer_params.conv_param(kernel_h=ke_h,kernel_w=ke_w,num_output=num_out,stride_h=str_h,stride_w=str_w)
       net.add_layer_with_data(layer_params, datas=weight, bottom=bottom)
    elif layer.endswith("Softmax"):
       axis = json_dict["num_axis"]
       weight = json_dict["weight"]
       layer_params.softmax_param(num_axis=axis)
       net.add_layer_with_data(layer_params, datas=weight, bottom=bottom)

    if nextOP != "":
       layer = net.get_layer_by_name(nextOP)
       layer.bottom[0] = top
    return net


def set_layer_data(json_dict):
    weight  = np.array(json_dict["weight"]).astype(np.float32)
    net=caffe_net.Caffemodel(caffemodel_file)
    net.set_layer_data(name, weight)
    return net


def remove_layer_by_name(json_dict):
    top     = json_dict["top"]
    nextOP = json_dict["nextOP"]
    net=caffe_net.Caffemodel(caffemodel_file)
    net.remove_layer_by_name(name)
    if nextOP != "":
       layer = net.get_layer_by_name(nextOP)
       layer.bottom[0] = top
    return net


def parser_args():
    parser = argparse.ArgumentParser(description="modeify caffemodel")
    parser.add_argument("--input", default="", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    jsonFile = open(args.input, 'r', encoding='utf-8')
    json_dict = json.load(jsonFile)
    global caffemodel_file,prototxt_file,changed_caffemodel,name
    modifier    = json_dict["modifier"]
    caffemodel_file    = json_dict["input_caffemodel"]
    prototxt_file      = json_dict["output_prototxt"]
    changed_caffemodel = json_dict["output_caffemodel"]
    name    = json_dict["name"]

    if modifier.endswith("add_new_layer"):
       net = add_new_layer(json_dict)
    elif modifier.endswith("remove_layer_by_name"):
       net = remove_layer_by_name(json_dict)
    elif modifier.endswith("set_layer_data"):
       net = set_layer_data(json_dict)

    net.save_prototxt(prototxt_file)
    net.save(changed_caffemodel)
 

if __name__=='__main__':
    args = parser_args()
    main(args)
    print("#### success done! ####")






